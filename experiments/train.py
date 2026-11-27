import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from bendflow.config import load_networks_config
from bendflow.data import image_shape, load_images
from bendflow.experiment import add_runtime_args, experiment_device
from bendflow.networks import build_model
from bendflow.train import save_model, train


def main():
    p = argparse.ArgumentParser(description="Train a U-Net or a DiT with conditional flow matching.")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--model", required=True, choices=["unet", "dit"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="override the config (timing runs)")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first n images (timing runs)",
    )
    p.add_argument("--out", default=None)
    add_runtime_args(p)
    args = p.parse_args()

    device = experiment_device(args.device)
    shape = image_shape(args.dataset)
    model, cfg = build_model(args.model, shape, load_networks_config(), args.seed)
    model.to(device)
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
    data = load_images(args.dataset)[: args.limit]
    n_params = sum(param.numel() for param in model.parameters())
    print(
        f"{args.dataset} {args.model}: N={len(data)} params={n_params} device={device}",
        flush=True,
    )
    print("train config:", json.dumps(dataclasses.asdict(cfg)), flush=True)

    start = time.perf_counter()
    model, history = train(model, data, shape, cfg, args.seed, log=lambda s: print(s, flush=True), path=args.schedule)

    suffix = "" if args.limit is None and args.epochs is None else "_timing"
    path = Path(args.out or f"checkpoints/{args.schedule}") / f"{args.dataset}_{args.model}{suffix}.pt"
    save_model(path, model)
    history.update(backend="pytorch", framework_version=str(torch.__version__), args=vars(args), train_config=dataclasses.asdict(cfg), params=n_params)
    path.with_suffix(".json").write_text(json.dumps(history, indent=2))

    print("--- summary ---")
    print(f"epochs run {len(history['train_loss'])}, best epoch {history['best_epoch']}, stopped early {history['stopped_early']}")
    print(f"final train loss {history['train_loss'][-1]:.5f}, best val loss {min(history['val_loss'], default=float('nan')):.5f}")
    print(f"wall time {time.perf_counter() - start:.0f}s, checkpoint {path}")


if __name__ == "__main__":
    main()
