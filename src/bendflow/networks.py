import dataclasses
import json
from pathlib import Path

import torch

from bendflow.config import NetworksConfig, load_networks_config
from bendflow.data import image_shape
from bendflow.fields import NetworkField
from bendflow.models import DiT, UNet
from bendflow.paths import GaussianPath
from bendflow.train import load_model


def build_model(name: str, shape: tuple[int, int, int], cfg: NetworksConfig, seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if name == "unet":
            return UNet(shape, **dataclasses.asdict(cfg.unet)).float(), cfg.unet_train
        if name == "dit":
            return DiT(shape, **dataclasses.asdict(cfg.dit)).float(), cfg.dit_train
    raise ValueError(f"unknown model {name!r}")


def load_network(name: str, dataset: str, ckpt_dir: str | None = None, *, device="cpu", path: GaussianPath = GaussianPath.LINEAR) -> tuple[NetworkField, dict]:
    shape = image_shape(dataset)
    model, _ = build_model(name, shape, load_networks_config(), 0)
    checkpoint = Path(ckpt_dir or f"checkpoints/{path}") / f"{dataset}_{name}.pt"
    meta_path = checkpoint.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    checkpoint_path = meta.get("args", {}).get("schedule", GaussianPath.LINEAR)
    if checkpoint_path != path:
        raise ValueError(f"{checkpoint} was trained with schedule {checkpoint_path!r}, expected {path!r}")
    return NetworkField(load_model(checkpoint, model).to(device).requires_grad_(False), shape), meta
