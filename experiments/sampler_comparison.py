import argparse
import json
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bendflow.experiment import add_runtime_args, as_numpy, experiment_device, initial_samples, result_directory
from bendflow.paths import GaussianPath
from bendflow.plotting import save_figure, science_figure

# Uniform, BendFlow: categorical slots 1 and 2 of the validated palette.
COLORS = {"uniform": "#2a78d6", "bendflow": "#eb6834"}
FIELD_LABELS = {"exact": "exact field", "unet": "U-Net", "dit": "DiT"}


def parse_args():
    p = argparse.ArgumentParser(description="Curvature-adaptive against uniform stepping at matched NFE (paper, Section 7.3).")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--fields", nargs="+", default=["exact", "unet", "dit"], choices=["exact", "unet", "dit"])
    p.add_argument("--nfe", nargs="+", type=int, default=[10, 20, 40], help="target numbers of field evaluations")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--ref-steps", type=int, default=1000, help="uniform steps of the reference solution")
    p.add_argument("--bisect-iters", type=int, default=20, help="bisection steps on log10(tol) per target NFE")
    p.add_argument("--batch", type=int, default=8, help="samples evaluated together (memory)")
    p.add_argument("--seed", type=int, default=1, help="differs from the other experiments so the samples are fresh")
    p.add_argument("--checkpoints", default=None)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--replot", action="store_true", help="only redraw the figure from the saved summary.json")
    add_runtime_args(p)
    return p.parse_args()


def rms(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=-1))


@science_figure
def plot(path, summary, fields):
    fig, axes = plt.subplots(1, len(fields), figsize=(3.6 * len(fields), 3.0), squeeze=False)
    solver = GaussianPath(summary.get("args", {}).get("schedule", "linear")).solver
    for ax, name in zip(axes[0], fields):
        rows = summary["fields"][name]["rows"]
        for s in ("uniform", "bendflow"):
            label = f"uniform {solver}" if s == "uniform" else "BendFlow (Alg. 1)"
            ax.plot([r[f"{s}_mean_nfe"] for r in rows], [r[f"{s}_rms_err_median"] for r in rows], color=COLORS[s], lw=2, marker="o", ms=5, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("mean NFE")
        ax.set_title(f"{FIELD_LABELS[name]}: RMS error to reference", fontsize=10)
    axes[0][0].legend(frameon=False, fontsize=9)
    save_figure(fig, path)


def main():
    args = parse_args()
    out = result_directory(args, "sampler_comparison")
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        summary = json.loads((out / "summary.json").read_text())
        plot(out / "error_vs_nfe", summary, list(summary["fields"]))
        print(f"figure redrawn in {out}/")
        return

    from bendflow.config import load_math_config
    from bendflow.data import load_images
    from bendflow.fields import ExactField
    from bendflow.networks import load_network
    from bendflow.posterior import EmpiricalPosterior
    from bendflow.samplers import BendFlowEuler, UniformEuler, run_batches

    device = experiment_device(args.device)
    start = time.perf_counter()
    cfg = load_math_config().sampler
    raw = load_images(args.dataset)
    data = torch.as_tensor(raw, dtype=torch.float64, device=device)
    n, d = raw.shape
    x0s = initial_samples(data, args.num_samples, args.seed, args.schedule)
    print(f"{args.dataset}: N={n} d={d} samples={args.num_samples} nfe={args.nfe} fields={args.fields} device={device}", flush=True)

    def run(sampler, field):
        result = run_batches(sampler, field, x0s, args.batch)
        return as_numpy(result.sample), as_numpy(result.nfe)

    def nearest(samples):
        return np.array([int((data - s).square().sum(dim=-1).argmin()) for s in torch.as_tensor(samples, device=device)])

    summary = {"backend": "pytorch", "args": vars(args), "sampler_config": {"h_min": cfg.h_min, "h_max": cfg.h_max, "h0": cfg.h0, "max_steps": cfg.max_steps}, "n": n, "d": d, "fields": {}}
    for name in args.fields:
        field = ExactField(EmpiricalPosterior(data, args.schedule)) if name == "exact" else load_network(name, args.dataset, args.checkpoints, device=device, path=args.schedule)[0]
        ref, _ = run(UniformEuler(args.ref_steps, args.schedule), field)
        half, _ = run(UniformEuler(args.ref_steps // 2, args.schedule), field)
        # For a first-order solver, ref - half estimates the error of ref itself.
        ref_err = rms(ref, half)
        ref_near = nearest(ref) if name == "exact" else None
        print(f"{name}: reference done, its own error (vs half the steps) median {np.median(ref_err):.2e}, {time.perf_counter() - start:.0f}s", flush=True)

        rows = []
        for target in args.nfe:
            uni, uni_nfe = run(UniformEuler(target, args.schedule), field)
            # Smallest tol whose mean NFE is at most the target, by bisection on log10(tol).
            lo, hi, best = -8.0, 3.0, None
            for _ in range(args.bisect_iters):
                mid = 0.5 * (lo + hi)
                bf, bf_nfe = run(BendFlowEuler(10.0**mid, cfg.h_min, cfg.h_max, cfg.h0, cfg.max_steps, args.schedule), field)
                if bf_nfe.mean() <= target:
                    hi, best = mid, (mid, bf, bf_nfe)
                else:
                    lo = mid
            if best is None:
                print(f"{name} nfe {target}: no tol in [1e-8, 1e3] reaches mean NFE <= {target}", flush=True)
                continue
            log_tol, bf, bf_nfe = best
            uni_err, bf_err = rms(uni, ref), rms(bf, ref)
            row = {
                "target_nfe": target,
                "tol": 10.0**log_tol,
                "uniform_mean_nfe": float(uni_nfe.mean()),
                "bendflow_mean_nfe": float(bf_nfe.mean()),
                "bendflow_nfe_min": int(bf_nfe.min()),
                "bendflow_nfe_max": int(bf_nfe.max()),
                "uniform_rms_err_median": float(np.median(uni_err)),
                "bendflow_rms_err_median": float(np.median(bf_err)),
                "uniform_rms_err_mean": float(uni_err.mean()),
                "bendflow_rms_err_mean": float(bf_err.mean()),
                "bendflow_better_fraction": float(np.mean(bf_err < uni_err)),
            }
            if ref_near is not None:
                row["uniform_same_image_as_reference"] = float(np.mean(nearest(uni) == ref_near))
                row["bendflow_same_image_as_reference"] = float(np.mean(nearest(bf) == ref_near))
            rows.append(row)
            print(f"{name} nfe {target}: {json.dumps(row)}, {time.perf_counter() - start:.0f}s", flush=True)
        summary["fields"][name] = {"reference_self_error_median": float(np.median(ref_err)), "reference_self_error_max": float(ref_err.max()), "rows": rows}

    summary["wall_time_s"] = time.perf_counter() - start
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(out / "error_vs_nfe", summary, list(summary["fields"]))

    print("--- summary ---")
    for name, f in summary["fields"].items():
        print(f"{name}: reference self-error median {f['reference_self_error_median']:.2e}, max {f['reference_self_error_max']:.2e}")
        for r in f["rows"]:
            same = f", same image as ref uniform {r['uniform_same_image_as_reference']:.2f} bendflow {r['bendflow_same_image_as_reference']:.2f}" if "uniform_same_image_as_reference" in r else ""
            print(
                f"  NFE {r['target_nfe']}: uniform err {r['uniform_rms_err_median']:.4f} | bendflow err {r['bendflow_rms_err_median']:.4f} (mean NFE {r['bendflow_mean_nfe']:.1f}, range {r['bendflow_nfe_min']}-{r['bendflow_nfe_max']}, tol {r['tol']:.2e}), bendflow better on {r['bendflow_better_fraction']:.2f}{same}"
            )
    print(f"wall time {summary['wall_time_s']:.0f}s, saved to {out}/")


if __name__ == "__main__":
    main()
