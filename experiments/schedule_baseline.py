import argparse
import json
import time

import matplotlib
import matplotlib.ticker
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bendflow.experiment import add_runtime_args, as_numpy, experiment_device, initial_samples, result_directory
from bendflow.paths import GaussianPath
from bendflow.plotting import save_figure, science_figure

# Categorical slots 1 to 4 of the validated palette; the markers carry identity as well, since slots 3 and 4 have low contrast.
METHODS = {
    "uniform": ("uniform", "#2a78d6", "o"),
    "bendflow": ("BendFlow, config settings", "#eb6834", "s"),
    "bendflow_tuned": ("BendFlow, tuned settings", "#1baf7a", "^"),
    "schedule": ("fixed schedule from mean profile", "#eda100", "D"),
}
FIELD_LABELS = {"exact": "exact field", "unet": "U-Net", "dit": "DiT"}


def parse_args():
    p = argparse.ArgumentParser(description="Per-sample adaptive steps against a fixed schedule built from the mean curvature profile (paper, Section 7.3).")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--fields", nargs="+", default=["exact", "unet", "dit"], choices=["exact", "unet", "dit"])
    p.add_argument("--nfe", nargs="+", type=int, default=[10, 20, 40], help="target numbers of field evaluations")
    p.add_argument("--num-samples", type=int, default=32, help="evaluation samples")
    p.add_argument("--calib-samples", type=int, default=32, help="calibration samples (schedule and settings)")
    p.add_argument("--seed", type=int, default=1, help="evaluation noise; 1 gives the same samples as sampler_comparison.py")
    p.add_argument("--calib-seed", type=int, default=2, help="calibration noise, disjoint from the evaluation noise")
    p.add_argument("--ref-steps", type=int, default=1000, help="uniform steps of the reference solution")
    p.add_argument("--profile-steps", type=int, default=200, help="uniform steps used to measure the mean profile")
    p.add_argument("--h0-grid", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    p.add_argument("--h-min-grid", nargs="+", type=float, default=[0.001, 0.005, 0.02])
    p.add_argument("--h-max-grid", nargs="+", type=float, default=[0.25], help="0.25 is the config value; larger values leave room to adapt at NFE 5")
    p.add_argument("--bisect-iters", type=int, default=14, help="bisection steps on log10(tol) per target NFE")
    p.add_argument("--batch", type=int, default=8, help="samples evaluated together (memory)")
    p.add_argument("--checkpoints", default=None)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--replot", action="store_true", help="only redraw the figure from the saved summary.json")
    add_runtime_args(p)
    return p.parse_args()


def rms(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=-1))


# Equalize the mass of sqrt(profile(t) * sigma(t) * gamma_dot(t)) across the grid.
# The mean profile replaces the per-sample estimate in Algorithm 1.
def schedule_from_profile(profile: np.ndarray, n: int, path: GaussianPath = GaussianPath.LINEAR) -> np.ndarray:
    p = len(profile) + 1
    edges = np.linspace(0.0, 1.0, p + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    prof = np.append(profile, profile[-1])
    scale = as_numpy(path.velocity_scale(torch.from_numpy(mids))).clip(min=0)
    rho = np.sqrt(np.maximum(prof, 0.0) * scale)
    if not rho.any():
        return np.linspace(0, 1, n + 1)
    rho = np.maximum(rho, 1e-6 * rho.max())  # keeps the cumulative mass strictly increasing
    mass = np.concatenate([[0.0], np.cumsum(rho * np.diff(edges))])
    ts = np.interp(np.linspace(0.0, mass[-1], n + 1), mass, edges)
    ts[0], ts[-1] = 0.0, 1.0
    return ts


@science_figure
def plot(path, summary):
    fields = list(summary["fields"])
    fig, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 3.2), squeeze=False)
    solver = GaussianPath(summary.get("args", {}).get("schedule", "linear")).solver
    for ax, name in zip(axes[0], fields):
        rows = summary["fields"][name]["rows"]
        for m, (label, color, marker) in METHODS.items():
            if m == "uniform":
                label = f"uniform {solver}"
            pts = [(r[m]["mean_nfe"], r[m]["rms_err_mean"]) for r in rows if m in r]
            if pts:
                ax.plot(*zip(*pts), color=color, lw=2, marker=marker, ms=6, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        nfes = sorted({r["target_nfe"] for r in rows})
        ax.set_xticks(nfes, [str(v) for v in nfes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("mean NFE")
        ax.set_title(f"{FIELD_LABELS[name]}: mean RMS error to reference", fontsize=10)
    axes[0][0].legend(frameon=False, fontsize=8)
    save_figure(fig, path)


def main():
    args = parse_args()
    out = result_directory(args, "schedule_baseline")
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        plot(out / "error_vs_nfe", json.loads((out / "summary.json").read_text()))
        print(f"figure redrawn in {out}/")
        return

    from bendflow.config import load_math_config
    from bendflow.data import load_images
    from bendflow.fields import ExactField
    from bendflow.networks import load_network
    from bendflow.posterior import EmpiricalPosterior
    from bendflow.samplers import BendFlowEuler, ScheduleEuler, UniformEuler, run_batches

    device = experiment_device(args.device)
    start = time.perf_counter()
    cfg = load_math_config().sampler
    raw = load_images(args.dataset)
    data = torch.as_tensor(raw, dtype=torch.float64, device=device)
    n, d = raw.shape
    x_eval = initial_samples(data, args.num_samples, args.seed, args.schedule)
    x_cal = initial_samples(data, args.calib_samples, args.calib_seed, args.schedule)
    print(f"{args.dataset}: N={n} d={d} eval={args.num_samples} calib={args.calib_samples} nfe={args.nfe} fields={args.fields} device={device}", flush=True)

    def run(sampler, field, xs):
        tr = run_batches(sampler, field, xs, args.batch)
        return tr, as_numpy(tr.sample), as_numpy(tr.nfe)

    def bendflow(tol, h_min, h0, h_max):
        return BendFlowEuler(tol, h_min, h_max, h0, cfg.max_steps, args.schedule)

    # Smallest tol with mean NFE at most target, by bisection on log10(tol). Returns (tol, samples, nfe) or None.
    def tune_tol(field, xs, target, h_min, h0, h_max):
        lo, hi, best = -8.0, 3.0, None
        for _ in range(args.bisect_iters):
            mid = 0.5 * (lo + hi)
            _, s, nfe = run(bendflow(10.0**mid, h_min, h0, h_max), field, xs)
            if nfe.mean() <= target:
                hi, best = mid, (10.0**mid, s, nfe)
            else:
                lo = mid
        return best

    def nearest(samples):
        return np.array([int((data - s).square().sum(dim=-1).argmin()) for s in torch.as_tensor(samples, device=device)])

    summary = {"backend": "pytorch", "args": vars(args), "sampler_config": {"h_min": cfg.h_min, "h_max": cfg.h_max, "h0": cfg.h0, "max_steps": cfg.max_steps}, "n": n, "d": d, "fields": {}}
    for name in args.fields:
        field = ExactField(EmpiricalPosterior(data, args.schedule)) if name == "exact" else load_network(name, args.dataset, args.checkpoints, device=device, path=args.schedule)[0]
        _, ref, _ = run(UniformEuler(args.ref_steps, args.schedule), field, x_eval)
        _, half, _ = run(UniformEuler(args.ref_steps // 2, args.schedule), field, x_eval)
        _, ref_cal, _ = run(UniformEuler(args.ref_steps, args.schedule), field, x_cal)
        ref_err = rms(ref, half)
        ref_near = nearest(ref) if name == "exact" else None

        # Mean estimated ||mdot|| over the calibration samples, on a fine uniform grid.
        tr, _, _ = run(UniformEuler(args.profile_steps, args.schedule), field, x_cal)
        est = np.linalg.norm(np.diff(as_numpy(tr.x1_hats), axis=1), axis=2) * args.profile_steps
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
            _, uni, uni_nfe = run(UniformEuler(target, args.schedule), field, x_eval)
            row = {"target_nfe": target}
            row["uniform"], uni_err = describe(uni, uni_nfe)

            found = tune_tol(field, x_eval, target, cfg.h_min, cfg.h0, cfg.h_max)
            if found is not None:
                row["bendflow"], _ = describe(found[1], found[2], uni_err)
                row["bendflow"].update(tol=found[0], h0=cfg.h0, h_min=cfg.h_min, h_max=cfg.h_max)

            # Settings chosen on the calibration samples by mean error, then only tol is matched to the NFE on the evaluation samples.
            best = None
            for h0 in args.h0_grid:
                for h_min in args.h_min_grid:
                    for h_max in args.h_max_grid:
                        cal = tune_tol(field, x_cal, target, h_min, h0, h_max)
                        if cal is not None:
                            score = float(rms(cal[1], ref_cal).mean())
                            if best is None or score < best[0]:
                                best = (score, h0, h_min, h_max)
            if best is not None:
                _, h0, h_min, h_max = best
                found = tune_tol(field, x_eval, target, h_min, h0, h_max)
                if found is not None:
                    row["bendflow_tuned"], _ = describe(found[1], found[2], uni_err)
                    row["bendflow_tuned"].update(tol=found[0], h0=h0, h_min=h_min, h_max=h_max, calib_rms_err_mean=best[0])

            ts = schedule_from_profile(profile, target, args.schedule)
            _, sch, sch_nfe = run(ScheduleEuler(ts, args.schedule), field, x_eval)
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
                    extra.append(f"h0 {c['h0']}, h_min {c['h_min']}, h_max {c.get('h_max', 0.25)}")
                parts.append(f"{m} {c['rms_err_mean']:.4f}" + (f" ({', '.join(extra)})" if extra else ""))
            print(f"  NFE {r['target_nfe']}: " + " | ".join(parts))
    print(f"wall time {summary['wall_time_s']:.0f}s, saved to {out}/")


if __name__ == "__main__":
    main()
