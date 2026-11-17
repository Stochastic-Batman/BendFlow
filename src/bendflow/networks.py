import dataclasses
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from bendflow.config import NetworksConfig, load_networks_config
from bendflow.data import image_shape
from bendflow.fields import NetworkField
from bendflow.models import DiT, UNet
from bendflow.train import load_model


# Returns the model named "unet" or "dit" built from the networks config, and its training configuration.
def build_model(name: str, shape: tuple[int, int, int], cfg: NetworksConfig, key):
    if name == "unet":
        return UNet(shape, key=key, **dataclasses.asdict(cfg.unet)), cfg.unet_train
    if name == "dit":
        return DiT(shape, key=key, **dataclasses.asdict(cfg.dit)), cfg.dit_train
    raise ValueError(f"unknown model {name!r}")


# A float32 network used as a field on vectors of any float dtype (float64 when x64 is on).
class CastField(eqx.Module):
    field: NetworkField

    def __call__(self, x, t):
        return self.field(x.astype(jnp.float32), jnp.asarray(t, jnp.float32)).astype(x.dtype)


# Loads checkpoints/<dataset>_<name>.eqx as a field, with the metadata written next to it by experiments/train.py.
def load_network(name: str, dataset: str, ckpt_dir: str = "checkpoints") -> tuple[CastField, dict]:
    shape = image_shape(dataset)
    like, _ = build_model(name, shape, load_networks_config(), jax.random.key(0))
    like = jax.tree.map(lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, like)
    path = Path(ckpt_dir) / f"{dataset}_{name}.eqx"
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return CastField(NetworkField(load_model(path, like), shape)), meta
