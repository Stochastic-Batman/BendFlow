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
from bendflow.data import image_shape
from bendflow.networks import load_network
from bendflow.samplers import BendFlowEuler, ScheduleEuler, UniformEuler

ROWS = {
    "reference": "reference (1000 steps)",
    "uniform": "uniform Euler",
    "bendflow": "BendFlow, config",
    "bendflow_tuned": "BendFlow, tuned",
    "schedule": "fixed schedule",
}
FIELD_LABELS = {"unet": "U-Net", "dit": "DiT"}
INK = "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description="Sample grids of the trained networks under the four samplers at a few NFE (paper, Section 7.3).")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--models", nargs="+", default=["unet", "dit"], choices=["unet", "dit"])
    p.add_argument("--nfe", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=3, help="noise for the grids, disjoint from evaluation (1) and calibration (2)")
    p.add_argument("--ref-steps", type=int, default=1000)
    p.add_argument("--settings", default=None, help="schedule_baseline output whose tol, h0, h_min, h_max and grids are reused")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default=None, help="output directory")
    return p.parse_args()


def rms(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=-1))


# (k, d) samples in [-1, 1] -> one (H, k * W, C) strip in [0, 1] with a 1-pixel white gap between images.
def strip(samples, shape):
    c, h, w = shape
    imgs = np.clip((np.asarray(samples).reshape(-1, c, h, w).transpose(0, 2, 3, 1) + 1.0) / 2.0, 0.0, 1.0)
    out = np.ones((h, len(imgs) * (w + 1) - 1, c))
    for i, im in enumerate(imgs):
        out[:, i * (w + 1) : i * (w + 1) + w] = im
    return out


def plot(path, title, panels, shape):
    nfes = list(panels)
    fig, axes = plt.subplots(len(ROWS), len(nfes), figsize=(3.4 * len(nfes), 0.75 * len(ROWS) + 0.6), squeeze=False)
    gray = shape[0] == 1
    for j, nfe in enumerate(nfes):
        for i, m in enumerate(ROWS):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            cell = panels[nfe].get(m)
            if cell is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color=INK, fontsize=8, transform=ax.transAxes)
                continue
            img = strip(cell["samples"], shape)
            ax.imshow(img[..., 0] if gray else img, cmap="gray" if gray else None, vmin=0.0, vmax=1.0, interpolation="nearest")
            if m != "reference":
                ax.set_title(f"NFE {cell['mean_nfe']:.1f}, RMS err {cell['rms_err_mean']:.3f}", color=INK, fontsize=7, pad=2)
            if j == 0:
                ax.set_ylabel(ROWS[m], color=INK, fontsize=8, rotation=0, ha="right", va="center")
        axes[0][j].set_title(f"target NFE {nfe}", color=INK, fontsize=9, pad=4)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    jax.config.update("jax_enable_x64", True)
    out = Path(args.out or f"results/sample_grids_{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    settings_dir = Path(args.settings or f"results/schedule_baseline_{args.dataset}_n128")
    settings = json.loads((settings_dir / "summary.json").read_text())
    start = time.perf_counter()
    cfg = load_math_config().sampler
    shape = image_shape(args.dataset)
    d = int(np.prod(shape))
    x0 = jax.random.normal(jax.random.key(args.seed), (args.num_images, d), jnp.float64)
    print(f"{args.dataset}: images={args.num_images} nfe={args.nfe} models={args.models} settings={settings_dir} devices={jax.devices()}", flush=True)

    batched = eqx.filter_jit(lambda sampler, field, xs: jax.vmap(lambda x: sampler(field, x))(xs))

    def run(sampler, field):
        tr = batched(sampler, field, x0)
        return np.asarray(tr.xs[:, -1]), np.asarray(tr.nfe)

    summary = {"args": vars(args), "settings": str(settings_dir), "models": {}}
    for name in args.models:
        field = load_network(name, args.dataset, args.checkpoints)[0]
        rows = {r["target_nfe"]: r for r in settings["fields"][name]["rows"]}
        ref, _ = run(UniformEuler(args.ref_steps), field)
        panels, table = {}, {}
        for nfe in args.nfe:
            row = rows.get(nfe)
            if row is None:
                print(f"{name}: NFE {nfe} is not in {settings_dir}; skipped", flush=True)
                continue
            samplers = {"uniform": UniformEuler(nfe)}
            for m in ("bendflow", "bendflow_tuned"):
                if m in row:
                    c = row[m]
                    samplers[m] = BendFlowEuler(jnp.asarray(c["tol"]), jnp.asarray(c["h_min"]), jnp.asarray(c.get("h_max", cfg.h_max)), jnp.asarray(c["h0"]), cfg.max_steps)
            samplers["schedule"] = ScheduleEuler(jnp.asarray(row["schedule"]["grid"]))
            panels[nfe] = {"reference": {"samples": ref}}
            table[nfe] = {}
            for m, sampler in samplers.items():
                s, k = run(sampler, field)
                cell = {"mean_nfe": float(k.mean()), "rms_err_mean": float(rms(s, ref).mean())}
                panels[nfe][m] = {"samples": s, **cell}
                table[nfe][m] = cell
        plot(out / f"grid_{name}", f"{args.dataset}, {FIELD_LABELS[name]}: same noise under each sampler", panels, shape)
        summary["models"][name] = table
        print(f"{name} done, {time.perf_counter() - start:.0f}s", flush=True)

    summary["wall_time_s"] = time.perf_counter() - start
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("--- summary ---")
    print("mean RMS error of the grid images to their 1000-step reference (mean NFE)")
    for name, table in summary["models"].items():
        for nfe, cells in table.items():
            print(f"{name} NFE {nfe}: " + " | ".join(f"{m} {c['rms_err_mean']:.4f} ({c['mean_nfe']:.1f})" for m, c in cells.items()))
    print(f"wall time {summary['wall_time_s']:.0f}s, saved to {out}/")


if __name__ == "__main__":
    main()
