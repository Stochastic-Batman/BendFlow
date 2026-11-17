import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bendflow.config import load_math_config
from bendflow.data import load_images
from bendflow.fields import ExactField
from bendflow.networks import load_network
from bendflow.posterior import EmpiricalPosterior
from bendflow.samplers import BendFlowEuler, ScheduleEuler, UniformEuler

# Categorical slots 1 to 4 of the validated palette; the markers carry identity as well, since slots 3 and 4 have low contrast.
METHODS = {
    "uniform": ("uniform Euler", "#2a78d6", "o"),
    "bendflow": ("BendFlow, config settings", "#eb6834", "s"),
    "bendflow_tuned": ("BendFlow, tuned $h_0$, $h_{\\min}$", "#1baf7a", "^"),
    "schedule": ("fixed schedule from mean profile", "#eda100", "D"),
}
FIELD_LABELS = {"exact": "exact field", "unet": "U-Net", "dit": "DiT"}
INK, GRID = "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description="Per-sample adaptive steps against a fixed schedule built from the mean curvature profile (paper, Section 7.3).")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--fields", nargs="+", default=["exact", "unet", "dit"], choices=["exact", "unet", "dit"])
    p.add_argument("--nfe", nargs="+", type=int, default=[10, 20, 40], help="target numbers of field evaluations")
    p.add_argument("--num-samples", type=int, default=32, help="evaluation samples")
    p.add_argument("--calib-samples", type=int, default=32, help="calibration samples (schedule and settings)")
    p.add_argument("--seed", type=int, default=1, help="evaluation noise; 1 gives the same samples as sampler_comparison.py")
    p.add_argument("--calib-seed", type=int, default=2, help="calibration noise, disjoint from the evaluation noise")
    p.add_argument("--ref-steps", type=int, default=1000, help="uniform Euler steps of the reference solution")
    p.add_argument("--profile-steps", type=int, default=200, help="uniform Euler steps used to measure the mean profile")
    p.add_argument("--h0-grid", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    p.add_argument("--h-min-grid", nargs="+", type=float, default=[0.001, 0.005, 0.02])
    p.add_argument("--bisect-iters", type=int, default=14, help="bisection steps on log10(tol) per target NFE")
    p.add_argument("--batch", type=int, default=8, help="samples evaluated together (memory)")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--replot", action="store_true", help="only redraw the figure from the saved summary.json")
    return p.parse_args()


def rms(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=-1))


# Grid with n steps that puts equal mass of rho(t) = sqrt(profile(t) / (1 - t)) in every step, the step rule of
# Algorithm 1 with the mean profile in place of the per-sample estimate. profile[k] describes t_k = k / P.
def schedule_from_profile(profile: np.ndarray, n: int) -> np.ndarray:
    p = len(profile) + 1
    edges = np.linspace(0.0, 1.0, p + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    prof = np.append(profile, profile[-1])
    rho = np.sqrt(np.maximum(prof, 0.0) / (1.0 - mids))
    rho = np.maximum(rho, 1e-6 * rho.max())  # keeps the cumulative mass strictly increasing
    mass = np.concatenate([[0.0], np.cumsum(rho * np.diff(edges))])
    ts = np.interp(np.linspace(0.0, mass[-1], n + 1), mass, edges)
    ts[0], ts[-1] = 0.0, 1.0
    return ts


def plot(path, summary):
    fields = list(summary["fields"])
    fig, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 3.2), squeeze=False)
    for ax, name in zip(axes[0], fields):
        rows = summary["fields"][name]["rows"]
        for m, (label, color, marker) in METHODS.items():
            pts = [(r[m]["mean_nfe"], r[m]["rms_err_mean"]) for r in rows if m in r]
            if pts:
                ax.plot(*zip(*pts), color=color, lw=2, marker=marker, ms=6, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mean NFE", color=INK)
        ax.set_title(f"{FIELD_LABELS[name]}: mean RMS error to reference", color=INK, fontsize=10)
        ax.grid(True, color=GRID, lw=0.6, which="both")
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    jax.config.update("jax_enable_x64", True)
    out = Path(args.out or f"results/schedule_baseline_{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        plot(out / "error_vs_nfe", json.loads((out / "summary.json").read_text()))
        print(f"figure redrawn in {out}/")
        return

    start = time.perf_counter()
    cfg = load_math_config().sampler
    raw = load_images(args.dataset)
    data = jnp.asarray(raw, dtype=jnp.float64)
    n, d = raw.shape
    x_eval = jax.random.normal(jax.random.key(args.seed), (args.num_samples, d), jnp.float64)
    x_cal = jax.random.normal(jax.random.key(args.calib_seed), (args.calib_samples, d), jnp.float64)
    print(f"{args.dataset}: N={n} d={d} eval={args.num_samples} calib={args.calib_samples} nfe={args.nfe} fields={args.fields} devices={jax.devices()}", flush=True)

    # One compiled function per sampler structure; the settings and the grid are traced arrays.
    batched = eqx.filter_jit(lambda sampler, field, xs: jax.lax.map(lambda x: sampler(field, x), xs, batch_size=args.batch))

    def run(sampler, field, xs):
        tr = batched(sampler, field, xs)
        return tr, np.asarray(tr.xs[:, -1]), np.asarray(tr.nfe)

    def bendflow(tol, h_min, h0):
        return BendFlowEuler(jnp.asarray(tol), jnp.asarray(h_min), jnp.asarray(cfg.h_max), jnp.asarray(h0), cfg.max_steps)

    # Smallest tol with mean NFE at most target, by bisection on log10(tol). Returns (tol, samples, nfe) or None.
    def tune_tol(field, xs, target, h_min, h0):
        lo, hi, best = -8.0, 3.0, None
        for _ in range(args.bisect_iters):
            mid = 0.5 * (lo + hi)
            _, s, nfe = run(bendflow(10.0**mid, h_min, h0), field, xs)
            if nfe.mean() <= target:
                hi, best = mid, (10.0**mid, s, nfe)
            else:
                lo = mid
        return best

    def nearest(samples):
        return np.array([int(jnp.argmin(jnp.sum((data - s) ** 2, axis=1))) for s in samples])

    summary = {"args": vars(args), "sampler_config": {"h_min": cfg.h_min, "h_max": cfg.h_max, "h0": cfg.h0, "max_steps": cfg.max_steps}, "n": n, "d": d, "fields": {}}
    for name in args.fields:
        field = ExactField(EmpiricalPosterior(data)) if name == "exact" else load_network(name, args.dataset, args.checkpoints)[0]
        _, ref, _ = run(UniformEuler(args.ref_steps), field, x_eval)
        _, half, _ = run(UniformEuler(args.ref_steps // 2), field, x_eval)
        _, ref_cal, _ = run(UniformEuler(args.ref_steps), field, x_cal)
        ref_err = rms(ref, half)
        ref_near = nearest(ref) if name == "exact" else None

        # Mean estimated ||mdot|| over the calibration samples, on a fine uniform grid.
        tr, _, _ = run(UniformEuler(args.profile_steps), field, x_cal)
        est = np.linalg.norm(np.diff(np.asarray(tr.x1_hats), axis=1), axis=2) * args.profile_steps
        profile = est.mean(axis=0)
        print(f"{name}: references and profile done, reference self-error median {np.median(ref_err):.2e}, {time.perf_counter() - start:.0f}s", flush=True)

        def describe(samples, nfe, base=None, ref=ref, ref_near=ref_near):
            err = rms(samples, ref)
            row = {"mean_nfe": float(nfe.mean()), "nfe_min": int(nfe.min()), "nfe_max": int(nfe.max()), "rms_err_median": float(np.median(err)), "rms_err_mean": float(err.mean())}
            if base is not None:
                row["better_than_uniform_fraction"] = float(np.mean(err < base))
            if ref_near is not None:
                row["same_image_as_reference"] = float(np.mean(nearest(samples) == ref_near))
            return row, err

        rows = []
        for target in args.nfe:
            _, uni, uni_nfe = run(UniformEuler(target), field, x_eval)
            row = {"target_nfe": target}
            row["uniform"], uni_err = describe(uni, uni_nfe)

            found = tune_tol(field, x_eval, target, cfg.h_min, cfg.h0)
            if found is not None:
                row["bendflow"], _ = describe(found[1], found[2], uni_err)
                row["bendflow"].update(tol=found[0], h0=cfg.h0, h_min=cfg.h_min)

            # Settings chosen on the calibration samples by mean error, then only tol is matched to the NFE on the evaluation samples.
            best = None
            for h0 in args.h0_grid:
                for h_min in args.h_min_grid:
                    cal = tune_tol(field, x_cal, target, h_min, h0)
                    if cal is not None:
                        score = float(rms(cal[1], ref_cal).mean())
                        if best is None or score < best[0]:
                            best = (score, h0, h_min)
            if best is not None:
                _, h0, h_min = best
                found = tune_tol(field, x_eval, target, h_min, h0)
                if found is not None:
                    row["bendflow_tuned"], _ = describe(found[1], found[2], uni_err)
                    row["bendflow_tuned"].update(tol=found[0], h0=h0, h_min=h_min, calib_rms_err_mean=best[0])

            ts = schedule_from_profile(profile, target)
            _, sch, sch_nfe = run(ScheduleEuler(jnp.asarray(ts)), field, x_eval)
            row["schedule"], _ = describe(sch, sch_nfe, uni_err)
            row["schedule"]["grid"] = [round(float(t), 4) for t in ts]
            rows.append(row)
            print(f"{name} nfe {target}: " + ", ".join(f"{m} {row[m]['rms_err_mean']:.4f}" for m in METHODS if m in row) + f", {time.perf_counter() - start:.0f}s", flush=True)
        summary["fields"][name] = {"reference_self_error_median": float(np.median(ref_err)), "reference_self_error_max": float(ref_err.max()), "profile": profile.tolist(), "rows": rows}

    summary["wall_time_s"] = time.perf_counter() - start
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(out / "error_vs_nfe", summary)

    print("--- summary ---")
    print("mean RMS error to reference (share of samples better than uniform; same image as reference for the exact field)")
    for name, f in summary["fields"].items():
        print(f"{name}: reference self-error median {f['reference_self_error_median']:.2e}, max {f['reference_self_error_max']:.2e}")
        for r in f["rows"]:
            parts = []
            for m in METHODS:
                if m not in r:
                    parts.append(f"{m} n/a")
                    continue
                c = r[m]
                extra = [f"better {c['better_than_uniform_fraction']:.2f}"] if "better_than_uniform_fraction" in c else []
                if "same_image_as_reference" in c:
                    extra.append(f"same img {c['same_image_as_reference']:.2f}")
                if m == "bendflow_tuned":
                    extra.append(f"h0 {c['h0']}, h_min {c['h_min']}")
                parts.append(f"{m} {c['rms_err_mean']:.4f}" + (f" ({', '.join(extra)})" if extra else ""))
            print(f"  NFE {r['target_nfe']}: " + " | ".join(parts))
    print(f"wall time {summary['wall_time_s']:.0f}s, saved to {out}/")


if __name__ == "__main__":
    main()
