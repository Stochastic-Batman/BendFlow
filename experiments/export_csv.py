"""Export a reproduction run's JSON summaries to tidy CSVs.

The cluster has no LaTeX, so figures made there cannot use the paper's TeX
styling. This script pulls the numbers out of the summaries instead, so the
figures can be rebuilt locally where TeX is installed.

Usage:
    python experiments/export_csv.py --run results/pytorch/linear
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SAMPLERS = ("uniform", "bendflow", "bendflow_tuned", "schedule")

FIELDS = {
    "sampler_comparison": [
        "dataset", "field", "target_nfe", "sampler", "mean_nfe", "nfe_min",
        "nfe_max", "rms_err_mean", "rms_err_median", "better_than_uniform_fraction", "tol",
    ],
    "schedule_baseline": [
        "dataset", "variant", "field", "target_nfe", "sampler", "mean_nfe",
        "nfe_min", "nfe_max", "rms_err_mean", "rms_err_median",
        "better_than_uniform_fraction", "same_image_as_reference", "tol",
        "h0", "h_min", "h_max", "calib_rms_err_mean",
    ],
    "sample_grids": ["dataset", "model", "nfe", "sampler", "mean_nfe", "rms_err_mean"],
    "exact_field_identity": [
        "dataset", "t", "jacobian_vs_autodiff", "jacobian_vs_finite_diff",
        "mdot_vs_autodiff", "mdot_vs_finite_diff", "divergence_vs_autodiff",
    ],
    "exact_field_bounds": ["dataset", "metric", "min", "median", "max"],
    "exact_field_trajectories": [
        "dataset", "traj", "t_peak_mdot", "peak_mdot", "n_eff_at_peak",
        "w_max_at_peak", "t_w_max_above_half", "final_nearest_index",
    ],
    "trained_networks": [
        "dataset", "field", "params", "best_val_loss", "epochs_run",
        "denoiser_rms_err_undecided", "denoiser_rms_err_decided",
        "jacobian_points_above_exact_upper_end", "jacobian_points",
        "jacobian_power_iteration_max_rel_residual",
    ],
    "training_history": ["dataset", "model", "epoch", "train_loss", "val_loss"],
}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: {len(rows)} rows")


def summaries(results_dir: Path):
    for child in sorted(results_dir.iterdir()):
        summary = child / "summary.json"
        if summary.is_file():
            yield child.name, json.loads(summary.read_text())


def split_dataset(name: str, prefix: str) -> tuple[str, str]:
    rest = name[len(prefix):]
    variant = ""
    if rest.endswith("_n128"):
        rest, variant = rest[: -len("_n128")], "n128"
    return rest, variant


def export(results_dir: Path, checkpoints_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {key: [] for key in FIELDS}

    for name, data in summaries(results_dir):
        if name.startswith("sampler_comparison_"):
            dataset, _ = split_dataset(name, "sampler_comparison_")
            for field, f in data["fields"].items():
                for row in f["rows"]:
                    base = {"dataset": dataset, "field": field, "target_nfe": row["target_nfe"]}
                    rows["sampler_comparison"].append({
                        **base, "sampler": "uniform",
                        "mean_nfe": row.get("uniform_mean_nfe"),
                        "nfe_min": row["target_nfe"], "nfe_max": row["target_nfe"],
                        "rms_err_mean": row.get("uniform_rms_err_mean"),
                        "rms_err_median": row.get("uniform_rms_err_median"),
                    })
                    rows["sampler_comparison"].append({
                        **base, "sampler": "bendflow",
                        "mean_nfe": row.get("bendflow_mean_nfe"),
                        "nfe_min": row.get("bendflow_nfe_min"),
                        "nfe_max": row.get("bendflow_nfe_max"),
                        "rms_err_mean": row.get("bendflow_rms_err_mean"),
                        "rms_err_median": row.get("bendflow_rms_err_median"),
                        "better_than_uniform_fraction": row.get("bendflow_better_fraction"),
                        "tol": row.get("tol"),
                    })

        elif name.startswith("schedule_baseline_"):
            dataset, variant = split_dataset(name, "schedule_baseline_")
            for field, f in data["fields"].items():
                for row in f["rows"]:
                    for sampler in SAMPLERS:
                        stats = row.get(sampler) or {}
                        rows["schedule_baseline"].append({
                            "dataset": dataset, "variant": variant, "field": field,
                            "target_nfe": row["target_nfe"], "sampler": sampler,
                            "mean_nfe": stats.get("mean_nfe"),
                            "nfe_min": stats.get("nfe_min"), "nfe_max": stats.get("nfe_max"),
                            "rms_err_mean": stats.get("rms_err_mean"),
                            "rms_err_median": stats.get("rms_err_median"),
                            "better_than_uniform_fraction": stats.get("better_than_uniform_fraction"),
                            "same_image_as_reference": stats.get("same_image_as_reference"),
                            "tol": stats.get("tol"), "h0": stats.get("h0"),
                            "h_min": stats.get("h_min"), "h_max": stats.get("h_max"),
                            "calib_rms_err_mean": stats.get("calib_rms_err_mean"),
                        })

        elif name.startswith("sample_grids_"):
            dataset, _ = split_dataset(name, "sample_grids_")
            for model, by_nfe in (data.get("models") or {}).items():
                for nfe, by_sampler in by_nfe.items():
                    for sampler, stats in by_sampler.items():
                        rows["sample_grids"].append({
                            "dataset": dataset, "model": model, "nfe": nfe,
                            "sampler": sampler, "mean_nfe": stats.get("mean_nfe"),
                            "rms_err_mean": stats.get("rms_err_mean"),
                        })

        elif name.startswith("exact_field_"):
            dataset, _ = split_dataset(name, "exact_field_")
            for check in data.get("identity_checks", []):
                rows["exact_field_identity"].append({"dataset": dataset, **check})
            for metric, stats in (data.get("bound_over_actual_undecided") or {}).items():
                rows["exact_field_bounds"].append({"dataset": dataset, "metric": metric, **stats})
            for index, traj in enumerate(data.get("per_trajectory", [])):
                rows["exact_field_trajectories"].append({"dataset": dataset, "traj": index, **traj})

        elif name.startswith("trained_networks_"):
            dataset, _ = split_dataset(name, "trained_networks_")
            for field in ("exact", "unet", "dit"):
                stats = data.get(field) or {}
                rows["trained_networks"].append({
                    "dataset": dataset, "field": field,
                    **{k: v for k, v in stats.items() if not isinstance(v, (list, dict))},
                })

    for name in sorted(checkpoints_dir.glob("*.json")):
        history = json.loads(name.read_text())
        dataset, _, model = name.stem.rpartition("_")
        for epoch, (train, val) in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
            rows["training_history"].append({
                "dataset": dataset, "model": model, "epoch": epoch,
                "train_loss": train, "val_loss": val,
            })

    for key, fields in FIELDS.items():
        write_csv(out_dir / f"{key}.csv", rows[key], fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run root with results/ and checkpoints/")
    parser.add_argument("--out", type=Path, default=None, help="output directory, defaults to <run>/csv")
    args = parser.parse_args()

    run = args.run
    results_dir = run / "results" if (run / "results").is_dir() else run
    checkpoints_dir = run / "checkpoints"
    export(results_dir, checkpoints_dir, args.out or run / "csv")


if __name__ == "__main__":
    main()
