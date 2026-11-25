import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bendflow.plotting import save_figure, science_figure

# A point counts as undecided while the largest posterior weight is below this.
UNDECIDED = 0.99
BLUE, ORANGE = "#2a78d6", "#eb6834"


def parse_args():
    p = argparse.ArgumentParser(description="Exact field on image data (paper, Section 7.1).")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--num-traj", type=int, default=8, help="trajectories to follow")
    p.add_argument("--steps", type=int, default=100, help="uniform Euler steps per trajectory")
    p.add_argument("--bound-traj", type=int, default=4, help="trajectories with bounds")
    p.add_argument("--bound-every", type=int, default=5, help="evaluate bounds every n-th step")
    p.add_argument("--check-points", type=int, default=5, help="points for the identity checks")
    p.add_argument(
        "--div-points",
        type=int,
        default=2,
        help="points for the divergence check (d jvps each)",
    )
    p.add_argument("--div-batch", type=int, default=8, help="jvps evaluated together in the divergence check (lower it if the GPU runs out of memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--float32", action="store_true", help="single precision (default: double)")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--replot", action="store_true", help="only redraw the figures from the saved curves.npz")
    return p.parse_args()


# Relative errors of the closed forms against autodiff and central differences.
def identity_checks(post, field, points, key, div_points, div_batch):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from bendflow.curvature import divergence_closed_form, mdot_autodiff, mdot_closed_form

    def rel_err(a, b):
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    dtype = post.data.dtype
    h = float(jnp.finfo(dtype).eps) ** (1 / 3)
    d = post.data.shape[1]

    @eqx.filter_jit
    def jac_check(x, t, v):
        closed = t / (1 - t) ** 2 * post.cov_matvec(x, t, v)
        auto = jax.jvp(lambda y: post.mean(y, t), (x,), (v,))[1]
        fd = (post.mean(x + h * v, t) - post.mean(x - h * v, t)) / (2 * h)
        return closed, auto, fd

    @eqx.filter_jit
    def mdot_check(x, t):
        u = field(x, t)
        m = lambda s: post.mean(x + s * u, t + s)
        return (
            mdot_closed_form(post, x, t),
            mdot_autodiff(field, x, t),
            (m(h) - m(-h)) / (2 * h),
        )

    @eqx.filter_jit
    def div_check(x, t):
        diag = jax.lax.map(
            lambda e: e @ jax.jvp(lambda y: field(y, t), (x,), (e,))[1],
            jnp.eye(d, dtype=dtype),
            batch_size=div_batch,
        )
        return divergence_closed_form(post, x, t), jnp.sum(diag)

    rows = []
    for i, (x, t) in enumerate(points):
        v = jax.random.normal(jax.random.fold_in(key, i), (d,), dtype)
        jc, ja, jf = jac_check(x, t, v / jnp.linalg.norm(v))
        mc, ma, mf = mdot_check(x, t)
        row = {
            "t": float(t),
            "jacobian_vs_autodiff": rel_err(jc, ja),
            "jacobian_vs_finite_diff": rel_err(jc, jf),
            "mdot_vs_autodiff": rel_err(mc, ma),
            "mdot_vs_finite_diff": rel_err(mc, mf),
        }
        if i < div_points:
            dc, da = div_check(x, t)
            row["divergence_vs_autodiff"] = float(jnp.abs(dc - da) / jnp.abs(da))
        rows.append(row)
    return rows


@science_figure
def plot_mdot_and_bound(path, ts, mdot, bound_steps, bound):
    n = bound.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharey=True, squeeze=False)
    for j, ax in enumerate(axes[0]):
        ax.plot(
            ts,
            np.where(mdot[j] > 0, mdot[j], np.nan),
            color=BLUE,
            lw=2,
            label=r"$\|\dot m\|$",
        )
        ax.plot(
            ts[bound_steps],
            bound[j],
            color=ORANGE,
            lw=2,
            ls="--",
            marker="o",
            ms=4,
            label="bound (Thm. 2)",
        )
        ax.set_yscale("log")
        ax.set_xlabel("$t$")
        ax.set_title(f"trajectory {j}", fontsize=10)
    # After the posterior commits, ||mdot|| underflows towards 0; cut the axis six decades below its peak.
    axes[0][0].set_ylim(bottom=float(np.max(mdot)) * 1e-6)
    axes[0][0].legend(frameon=False, fontsize=9)
    save_figure(fig, path)


@science_figure
def plot_active_set(path, ts, n_active, d_active):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for j in range(n_active.shape[0]):
        axes[0].plot(ts, n_active[j], color=BLUE, lw=1.5, alpha=0.7)
        axes[1].plot(ts, d_active[j], color=BLUE, lw=1.5, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_title("active-set size $|A|$ (one line per trajectory)", fontsize=10)
    axes[1].set_title("active diameter $D_A$", fontsize=10)
    for ax in axes:
        ax.set_xlabel("$t$")
    save_figure(fig, path)


def main():
    args = parse_args()
    out = Path(args.out or f"results/exact_field_{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        c = np.load(out / "curves.npz")
        plot_mdot_and_bound(out / "mdot_and_bound", c["ts"], c["mdot"][: len(c["mdot_bound"])], c["bound_steps"], c["mdot_bound"])
        plot_active_set(out / "active_set", c["ts"][c["bound_steps"]], c["n_active"], c["d_active"])
        print(f"figures redrawn in {out}/")
        return

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from bendflow.config import load_math_config
    from bendflow.curvature import local_bound, mdot_closed_form
    from bendflow.data import load_images
    from bendflow.fields import ExactField
    from bendflow.posterior import EmpiricalPosterior, diameter
    from bendflow.samplers import UniformEuler

    if not args.float32:
        jax.config.update("jax_enable_x64", True)
    start = time.perf_counter()
    cfg = load_math_config()

    raw = load_images(args.dataset)
    post = EmpiricalPosterior(jnp.asarray(raw, dtype=jnp.float32 if args.float32 else jnp.float64))
    field = ExactField(post)
    n, d = raw.shape
    dtype = post.data.dtype
    k_x0, k_lam, k_check, k_pick = jax.random.split(jax.random.key(args.seed), 4)
    print(
        f"{args.dataset}: N={n} d={d} dtype={dtype} traj={args.num_traj} steps={args.steps}",
        flush=True,
    )

    # Trajectories of the exact field on a fine uniform grid.
    sampler = UniformEuler(args.steps)
    x0s = jax.random.normal(k_x0, (args.num_traj, d), dtype)
    traj = eqx.filter_jit(lambda xs: jax.lax.map(lambda x: sampler(field, x), xs))(x0s)
    ts = np.asarray(traj.ts[0, :-1])  # times at which the field was evaluated
    xs = traj.xs[:, :-1]  # (traj, steps, d)
    print(f"trajectories done, {time.perf_counter() - start:.0f}s", flush=True)

    # (iii) ||mdot|| and how decided the posterior is, at every step.
    @eqx.filter_jit
    def per_point(x, t):
        w = jnp.exp(post.log_weights(x, t))
        mdot = jnp.linalg.norm(mdot_closed_form(post, x, t))
        return mdot, jnp.max(w), 1.0 / jnp.sum(w**2), jnp.argmax(w)

    flat = jax.lax.map(
        lambda a: per_point(*a),
        (xs.reshape(-1, d), jnp.tile(traj.ts[0, :-1], args.num_traj)),
    )
    mdot, w_max, n_eff, nearest = (np.asarray(a).reshape(args.num_traj, args.steps) for a in flat)
    undecided = w_max < UNDECIDED
    print(f"mdot along trajectories done, {time.perf_counter() - start:.0f}s", flush=True)

    # (ii) local bounds and the active set.
    d_all = diameter(raw, cfg.numerics.diameter_chunk)
    lam_fn = eqx.filter_jit(lambda x, t, k: post.lambda_max(x, t, k, cfg.spectral.power_iters))
    kappa_fn = eqx.filter_jit(lambda x, t: jnp.linalg.norm(post.kappa(x, t)))
    steps = np.arange(0, args.steps, args.bound_every)
    b_traj = min(args.bound_traj, args.num_traj)
    names = (
        "lam",
        "lam_residual",
        "lam_bound",
        "kappa",
        "kappa_bound",
        "mdot_bound",
        "d_active",
        "n_active",
    )
    b = {name: np.zeros((b_traj, len(steps))) for name in names}
    for j in range(b_traj):
        for i, k in enumerate(steps):
            x, t = xs[j, k], traj.ts[j, k]
            bound = local_bound(post, x, t, cfg.active_set.delta, d_all, cfg.numerics.diameter_chunk)
            b["lam"][j, i], b["lam_residual"][j, i] = lam_fn(x, t, jax.random.fold_in(k_lam, j * args.steps + int(k)))
            b["kappa"][j, i] = kappa_fn(x, t)
            b["lam_bound"][j, i], b["kappa_bound"][j, i] = bound.lam, bound.kappa
            b["mdot_bound"][j, i], b["d_active"][j, i] = bound.mdot, bound.d_active
            b["n_active"][j, i] = bound.n_active
    print(f"bounds done, {time.perf_counter() - start:.0f}s", flush=True)

    # (i) identities at random undecided points of the trajectories. Where the
    # posterior has collapsed onto one image, C = kappa = 0 and both sides of
    # every identity vanish, so a relative error says nothing there.
    cand = np.argwhere(undecided[:, 1:]) + [0, 1]
    if len(cand) == 0:
        raise SystemExit("no undecided trajectory point; increase --steps")
    chosen = jax.random.choice(k_pick, len(cand), (min(args.check_points, len(cand)),), replace=False)
    points = [(xs[j, k], traj.ts[j, k]) for j, k in cand[np.asarray(chosen)]]
    checks = identity_checks(post, field, points, k_check, args.div_points, args.div_batch)
    print(f"identity checks done, {time.perf_counter() - start:.0f}s", flush=True)

    def ratio_stats(bound, actual):
        ok = undecided[:b_traj][:, steps] & (actual > 0)
        ratio = bound[ok] / actual[ok]
        return {
            "min": float(ratio.min()),
            "median": float(np.median(ratio)),
            "max": float(ratio.max()),
        }

    peak = mdot.argmax(axis=1)
    rows = np.arange(args.num_traj)
    commit = [(int(np.argmax(w_max[j] > 0.5)) if (w_max[j] > 0.5).any() else None) for j in rows]
    summary = {
        "args": vars(args),
        "math_config": {
            "delta": cfg.active_set.delta,
            "power_iters": cfg.spectral.power_iters,
        },
        "n": n,
        "d": d,
        "dtype": str(dtype),
        "diameter_D": d_all,
        "identity_checks": checks,
        "identity_max_rel_err": {k: max(c[k] for c in checks if k in c) for k in checks[0] if k != "t"},
        "undecided_threshold_w_max": UNDECIDED,
        "lambda_max_max_rel_residual_undecided": float(b["lam_residual"][undecided[:b_traj][:, steps]].max(initial=0.0)),
        "bound_points_undecided": int(undecided[:b_traj][:, steps].sum()),
        "bound_points_total": int(b_traj * len(steps)),
        "bound_over_actual_undecided": {
            "lambda_max": ratio_stats(b["lam_bound"], b["lam"]),
            "kappa": ratio_stats(b["kappa_bound"], b["kappa"]),
            "mdot": ratio_stats(b["mdot_bound"], mdot[:b_traj][:, steps]),
        },
        "bound_violations": int((b["lam"] > b["lam_bound"]).sum() + (b["kappa"] > b["kappa_bound"]).sum() + (mdot[:b_traj][:, steps] > b["mdot_bound"]).sum()),
        "per_trajectory": [
            {
                "t_peak_mdot": float(ts[peak[j]]),
                "peak_mdot": float(mdot[j, peak[j]]),
                "n_eff_at_peak": float(n_eff[j, peak[j]]),
                "w_max_at_peak": float(w_max[j, peak[j]]),
                "t_w_max_above_half": None if commit[j] is None else float(ts[commit[j]]),
                "final_nearest_index": int(nearest[j, -1]),
            }
            for j in rows
        ],
        "active_set_size_by_t": {f"{ts[k]:.2f}": b["n_active"][:, i].tolist() for i, k in enumerate(steps)},
        "wall_time_s": time.perf_counter() - start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out / "curves.npz",
        ts=ts,
        mdot=mdot,
        w_max=w_max,
        n_eff=n_eff,
        nearest=nearest,
        bound_steps=steps,
        **b,
    )
    plot_mdot_and_bound(out / "mdot_and_bound", ts, mdot[:b_traj], steps, b["mdot_bound"])
    plot_active_set(out / "active_set", ts[steps], b["n_active"], b["d_active"])

    print("--- summary ---")
    print("identity max rel err:", json.dumps(summary["identity_max_rel_err"]))
    print(
        f"bound / actual on {summary['bound_points_undecided']} undecided points:",
        json.dumps(summary["bound_over_actual_undecided"]),
    )
    print(
        "lambda_max power iteration, max relative residual:",
        summary["lambda_max_max_rel_residual_undecided"],
    )
    print("bound violations:", summary["bound_violations"], "| D =", round(d_all, 3))
    for j, r in enumerate(summary["per_trajectory"]):
        print(f"traj {j}:", json.dumps(r))
    print(f"wall time {summary['wall_time_s']:.0f}s, saved to {out}/")


if __name__ == "__main__":
    main()
