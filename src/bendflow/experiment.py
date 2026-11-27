import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from bendflow.paths import GaussianPath


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule", choices=list(GaussianPath), type=GaussianPath, default=GaussianPath.LINEAR)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")


def experiment_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    return device


def result_directory(args: argparse.Namespace, experiment: str) -> Path:
    if args.out:
        return Path(args.out)
    name = f"{experiment}_{args.dataset}"
    target = Path("results/pytorch") / args.schedule / name
    if getattr(args, "replot", False) and args.schedule is GaussianPath.LINEAR and not target.is_dir():
        return Path("results") / name
    return target


def generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def initial_samples(data: Tensor, count: int, seed: int, path: GaussianPath) -> Tensor:
    # Draw on CPU so the same seed and dataset give the same start on CPU and CUDA.
    rng = generator(torch.device("cpu"), seed)
    noise = torch.randn((count, data.shape[1]), dtype=data.dtype, generator=rng).to(data.device)
    if path is not GaussianPath.VE:
        return noise
    targets = data[torch.randint(len(data), (count,), generator=rng).to(data.device)]
    return path.interpolant(targets, noise, noise.new_zeros(count))[0]


def as_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
