import argparse
import json
from pathlib import Path

DATASETS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
MODELS = {"unet": "U-Net", "dit": "DiT"}


def main():
    p = argparse.ArgumentParser(description="Summarise the training runs saved by experiments/train.py as LaTeX table rows.")
    p.add_argument("--dir", default="results/training", help="folder with the <dataset>_<model>.json files")
    args = p.parse_args()
    for ds, ds_name in DATASETS.items():
        for model, model_name in MODELS.items():
            path = Path(args.dir) / f"{ds}_{model}.json"
            if not path.exists():
                continue
            h = json.loads(path.read_text())
            best = h["best_epoch"]
            row = [
                ds_name,
                model_name,
                f"${h['params']:,}$".replace(",", "{,}"),
                str(len(h["train_loss"])),
                str(best),
                f"${h['train_loss'][best - 1]:.4f}$",
                f"${min(h['val_loss']):.4f}$",
                f"${h['wall_time'] / 60:.0f}$",
            ]
            print(" & ".join(row) + r"\\")


if __name__ == "__main__":
    main()
