import argparse
import json
import time
from functools import partial

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bendflow.experiment import add_runtime_args, as_numpy, experiment_device, generator, initial_samples, result_directory
from bendflow.plotting import save_figure, science_figure

# A point counts as undecided while the largest posterior weight is below this (same as exact_field.py).
UNDECIDED = 0.99
# Exact field, U-Net, DiT: categorical slots 1 to 3 of the validated palette.
COLORS = {"exact": "#2a78d6", "unet": "#eb6834", "dit": "#1baf7a"}
LABELS = {"exact": "exact field", "unet": "U-Net", "dit": "DiT"}


def parse_args():
    p = argparse.ArgumentParser(description="Trained U-Net and DiT against the exact field (paper, Section 7.2).")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--models", nargs="+", default=["unet", "dit"], choices=["unet", "dit"])
    p.add_argument("--num-traj", type=int, default=8, help="trajectories per field")
    p.add_argument("--steps", type=int, default=100, help="uniform Euler steps per trajectory")
    p.add_argument("--eig-traj", type=int, default=4, help="trajectories with Jacobian eigenvalues")
    p.add_argument("--eig-every", type=int, default=5, help="evaluate eigenvalues every n-th step")
    p.add_argument("--seed", type=int, default=0, help="same seed as exact_field.py gives the same starting noise")
    p.add_argument("--checkpoints", default=None)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--replot", action="store_true", help="only redraw the figures from the saved curves.npz")
    add_runtime_args(p)
    return p.parse_args()


def first_peak_time(ts, curve) -> float:
    return float(ts[int(np.nanargmax(curve))])


def stats(a) -> dict:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


# Zeros become gaps on a log axis.
def positive(a):
    return np.where(a > 0, a, np.nan)


@science_figure
def plot_mdot(path, ts, curves, n_show):
    names = [k for k in curves if k != "exact"]
    fig, axes = plt.subplots(len(names), n_show, figsize=(3.0 * n_show, 2.7 * len(names)), sharex=True, sharey=True, squeeze=False)
    top = max(float(np.nanmax(c[k][:n_show])) for c in curves.values() for k in ("mdot", "mdot_est") if k in c)
    for r, name in enumerate(names):
        for j in range(n_show):
            ax = axes[r][j]
            ax.plot(ts, positive(curves["exact"]["mdot"][j]), color=COLORS["exact"], lw=2, label="exact field, closed form")
            ax.plot(ts, positive(curves[name]["mdot"][j]), color=COLORS[name], lw=2, label=f"{LABELS[name]}, autodiff")
            ax.plot(ts[:-1], positive(curves[name]["mdot_est"][j]), color=COLORS[name], lw=1.5, ls="--", label=f"{LABELS[name]}, estimator (18)")
            ax.set_yscale("log")
            ax.set_title(f"{LABELS[name]}, trajectory {j}", fontsize=10)
            if r == len(names) - 1:
                ax.set_xlabel("$t$")
        axes[r][0].set_ylabel(r"$\|\dot m\|$")
        axes[r][0].legend(frameon=False, fontsize=8)
    # After the posterior commits, the exact ||mdot|| falls towards 0; cut the axis four decades below the peak.
    axes[0][0].set_ylim(top * 1e-4, top * 3)
    save_figure(fig, path)


@science_figure
def plot_denoiser_error(path, ts, curves):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for name, c in curves.items():
        if name == "exact":
            continue
        ax.plot(ts, np.median(c["denoiser_err"], axis=0), color=COLORS[name], lw=2, label=LABELS[name])
    ax.set_xlabel("$t$")
    ax.set_title(r"RMS of $\hat x_1 - m$ per pixel (median over trajectories)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, path)


@science_figure
def plot_jacobian(path, eig_ts, curves):
    names = [k for k in curves if k != "exact"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.0), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        c = curves[name]
        for j in range(c["eig_net"].shape[0]):
            ax.plot(eig_ts, c["eig_exact"][j], color=COLORS["exact"], lw=1.5, marker="o", ms=4, label="exact upper end (5)" if j == 0 else None)
            ax.plot(eig_ts, c["eig_net"][j], color=COLORS[name], lw=1.5, marker="s", ms=4, label=f"{LABELS[name]}, top eigenvalue" if j == 0 else None)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("$t$")
        ax.set_title(f"{LABELS[name]}: top eigenvalue of the Jacobian", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path)


def draw_all(out, ts, eig_steps, curves, n_show):
    plot_mdot(out / "mdot_networks", ts, curves, n_show)
    plot_denoiser_error(out / "denoiser_error", ts, curves)
    plot_jacobian(out / "jacobian_top_eig", ts[eig_steps], curves)


def main():
    args = parse_args()
    out = result_directory(args, "trained_networks")
    out.mkdir(parents=True, exist_ok=True)
    n_show = min(4, args.num_traj)

    if args.replot:
        z = np.load(out / "curves.npz")
        curves = {}
        for key in z.files:
            if "/" in key:
                name, field = key.split("/", 1)
                curves.setdefault(name, {})[field] = z[key]
        draw_all(out, z["ts"], z["eig_steps"], curves, n_show)
        print(f"figures redrawn in {out}/")
        return

    from bendflow.config import load_math_config
    from bendflow.curvature import jacobian_top_eig, mdot_autodiff, mdot_closed_form
    from bendflow.data import load_images
    from bendflow.fields import ExactField
    from bendflow.networks import load_network
    from bendflow.posterior import EmpiricalPosterior
    from bendflow.samplers import UniformEuler

    device = experiment_device(args.device)
    start = time.perf_counter()
    cfg = load_math_config()
    raw = load_images(args.dataset)
    post = EmpiricalPosterior(torch.as_tensor(raw, dtype=torch.float64, device=device), args.schedule)
    exact = ExactField(post)
    n, d = raw.shape
    rng = generator(device, args.seed + 1)
    x0s = initial_samples(post.data, args.num_traj, args.seed, args.schedule)
    eig_steps = np.arange(0, args.steps, args.eig_every)
    e_traj = min(args.eig_traj, args.num_traj)
    print(f"{args.dataset}: N={n} d={d} device={device} traj={args.num_traj} steps={args.steps} models={args.models}", flush=True)

    def exact_point(x, t):
        w, m = post.weights_and_mean(x, t)
        return torch.linalg.vector_norm(mdot_closed_form(post, x, t, (w, m)), dim=-1), w.amax(dim=-1)

    def network_point(net, x, t, x1_hat):
        w, m = post.weights_and_mean(x, t)
        err = torch.linalg.vector_norm(x1_hat - m, dim=-1) / np.sqrt(d)
        return (
            err,
            torch.linalg.vector_norm(mdot_autodiff(net, x, t, args.schedule), dim=-1),
            torch.linalg.vector_norm(mdot_closed_form(post, x, t, (w, m)), dim=-1),
            w.amax(dim=-1),
        )

    def measure(fn, tr, *extra):
        columns = (tr.xs[:, :-1].reshape(-1, d), tr.ts[:, :-1].reshape(-1), *extra)
        values = [fn(*batch) for batch in zip(*(column.split(8) for column in columns))]
        return tuple(as_numpy(torch.cat(parts)).reshape(args.num_traj, args.steps) for parts in zip(*values))

    sampler = UniformEuler(args.steps, args.schedule)
    traj = sampler(exact, x0s)
    ts = as_numpy(traj.ts[0, :-1])
    mdot_ex, wmax_ex = measure(exact_point, traj)
    end_nearest_ex = [int((post.data - x).square().sum(dim=-1).argmin()) for x in traj.sample]
    curves = {"exact": {"mdot": mdot_ex, "w_max": wmax_ex}}
    summary = {
        "backend": "pytorch",
        "args": vars(args),
        "n": n,
        "d": d,
        "undecided_threshold_w_max": UNDECIDED,
        "exact": {"t_peak_mdot": [first_peak_time(ts, mdot_ex[j]) for j in range(args.num_traj)], "endpoint_nearest_index": end_nearest_ex},
    }
    print(f"exact field done, {time.perf_counter() - start:.0f}s", flush=True)

    for name in args.models:
        net, meta = load_network(name, args.dataset, args.checkpoints, device=device, path=args.schedule)
        tr = sampler(net, x0s)
        xs = tr.xs[:, :-1]
        err, mdot_net, mdot_ex_here, wmax_here = measure(partial(network_point, net), tr, tr.x1_hats.reshape(-1, d))
        # Estimator (18): difference of consecutive denoised predictions; entry k uses the interval [t_k, t_{k + 1}].
        x1h = as_numpy(tr.x1_hats)
        mdot_est = np.linalg.norm(np.diff(x1h, axis=1), axis=2) * args.steps

        # Top eigenvalue of the symmetric part of the network Jacobian, and the upper end of (5) at the same point.
        eig_net, eig_res, eig_exact = (np.zeros((e_traj, len(eig_steps))) for _ in range(3))
        for j in range(e_traj):
            for i, k in enumerate(eig_steps):
                x, t = xs[j, k], tr.ts[j, k]
                mu, r = jacobian_top_eig(net, x, t, rng, cfg.spectral.power_iters)
                lam, _ = post.lambda_max(x, t, rng, cfg.spectral.power_iters)
                eig_net[j, i], eig_res[j, i] = float(mu), float(r) / max(abs(float(mu)), 1e-12)
                alpha, sigma, _, sigma_dot = args.schedule.coefficients(t)
                eig_exact[j, i] = float(args.schedule.velocity_scale(t) * alpha / sigma.square() * lam + sigma_dot / sigma)

        near = [torch.linalg.vector_norm(post.data - x, dim=-1) for x in tr.sample]
        near_dist = [float(a.min()) / np.sqrt(d) for a in near]
        near_idx = [int(a.argmin()) for a in near]
        undecided = wmax_here < UNDECIDED
        val = meta.get("val_loss", [])
        summary[name] = {
            "params": meta.get("params"),
            "best_val_loss": min(val) if val else None,
            "epochs_run": len(meta.get("train_loss", [])),
            "denoiser_rms_err_undecided": stats(err[undecided]),
            "denoiser_rms_err_decided": stats(err[~undecided]),
            "denoiser_rms_err_median_by_t": {f"{ts[k]:.2f}": float(np.median(err[:, k])) for k in range(0, args.steps, 10)},
            "mdot_net_over_exact_same_point_undecided": stats(mdot_net[undecided] / mdot_ex_here[undecided]),
            "jacobian_top_eig_minus_exact_upper_end": stats(eig_net - eig_exact),
            "jacobian_points_above_exact_upper_end": int((eig_net > eig_exact).sum()),
            "jacobian_points": int(eig_net.size),
            "jacobian_power_iteration_max_rel_residual": float(eig_res.max()),
            "per_trajectory": [
                {
                    "t_peak_mdot_exact_traj": summary["exact"]["t_peak_mdot"][j],
                    "t_peak_mdot_net_autodiff": first_peak_time(ts, mdot_net[j]),
                    "t_peak_mdot_net_estimator": first_peak_time(ts[:-1], mdot_est[j]),
                    "sample_nearest_rms_dist": near_dist[j],
                    "sample_nearest_index": near_idx[j],
                    "same_image_as_exact_endpoint": near_idx[j] == end_nearest_ex[j],
                }
                for j in range(args.num_traj)
            ],
        }
        curves[name] = {
            "mdot": mdot_net,
            "mdot_est": mdot_est,
            "denoiser_err": err,
            "mdot_exact_same_point": mdot_ex_here,
            "w_max": wmax_here,
            "eig_net": eig_net,
            "eig_exact": eig_exact,
            "eig_rel_residual": eig_res,
        }
        print(f"{name} done, {time.perf_counter() - start:.0f}s", flush=True)

    summary["wall_time_s"] = time.perf_counter() - start
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / "curves.npz", ts=ts, eig_steps=eig_steps, **{f"{k}/{f}": v for k, c in curves.items() for f, v in c.items()})
    draw_all(out, ts, eig_steps, curves, n_show)

    print("--- summary ---")
    print("exact t_peak_mdot:", summary["exact"]["t_peak_mdot"])
    for name in args.models:
        s = summary[name]
        print(f"{name}: params {s['params']}, best val loss {s['best_val_loss']}, epochs {s['epochs_run']}")
        print(f"  denoiser RMS err undecided {json.dumps(s['denoiser_rms_err_undecided'])}")
        print(f"  denoiser RMS err decided   {json.dumps(s['denoiser_rms_err_decided'])}")
        print(f"  denoiser RMS err by t      {json.dumps(s['denoiser_rms_err_median_by_t'])}")
        print(f"  ||mdot_net|| / ||mdot_exact|| undecided {json.dumps(s['mdot_net_over_exact_same_point_undecided'])}")
        print(
            f"  jacobian: {s['jacobian_points_above_exact_upper_end']}/{s['jacobian_points']} above exact upper end, net - exact {json.dumps(s['jacobian_top_eig_minus_exact_upper_end'])}, max rel residual {s['jacobian_power_iteration_max_rel_residual']:.2e}"
        )
        for j, r in enumerate(s["per_trajectory"]):
            print(f"  traj {j}: {json.dumps(r)}")
    print(f"wall time {summary['wall_time_s']:.0f}s, saved to {out}/")


if __name__ == "__main__":
    main()
