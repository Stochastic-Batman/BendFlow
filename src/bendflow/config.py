import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class ActiveSetConfig:
    delta: float

    def __post_init__(self):
        _check(0.0 < self.delta < 1.0, "active_set.delta must lie in (0, 1)")


@dataclass(frozen=True)
class SamplerConfig:
    tol: float
    h_min: float
    h_max: float
    h0: float
    max_steps: int

    def __post_init__(self):
        _check(self.tol > 0.0, "sampler.tol must be positive")
        _check(0.0 < self.h_min <= self.h_max, "sampler needs 0 < h_min <= h_max")
        _check(self.h0 > 0.0, "sampler.h0 must be positive")
        # Every step after the first is at least h_min long, so Algorithm 1 never
        # needs more steps than this; the padded buffer must not cut it short.
        needed = math.ceil(1.0 / self.h_min) + 1
        _check(
            self.max_steps >= needed,
            f"sampler.max_steps must be at least ceil(1 / h_min) + 1 = {needed}",
        )


@dataclass(frozen=True)
class SpectralConfig:
    power_iters: int

    def __post_init__(self):
        _check(self.power_iters >= 1, "spectral.power_iters must be at least 1")


@dataclass(frozen=True)
class NumericsConfig:
    diameter_chunk: int

    def __post_init__(self):
        _check(self.diameter_chunk >= 1, "numerics.diameter_chunk must be at least 1")


@dataclass(frozen=True)
class MathConfig:
    active_set: ActiveSetConfig
    sampler: SamplerConfig
    spectral: SpectralConfig
    numerics: NumericsConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    lr_schedule: str
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    ema_decay: float
    val_fraction: float
    early_stopping: bool
    patience: int
    min_delta: float

    def __post_init__(self):
        _check(
            self.epochs >= 1 and self.batch_size >= 1,
            "train needs epochs, batch_size >= 1",
        )
        _check(
            self.lr_schedule in ("cosine", "constant"),
            "train.lr_schedule: cosine or constant",
        )
        _check(0.0 <= self.ema_decay < 1.0, "train.ema_decay must lie in [0, 1)")
        _check(0.0 <= self.val_fraction < 1.0, "train.val_fraction must lie in [0, 1)")
        _check(
            not self.early_stopping or self.val_fraction > 0.0,
            "train.early_stopping needs val_fraction > 0",
        )


@dataclass(frozen=True)
class UNetConfig:
    base_channels: int
    channel_mults: tuple[int, ...]
    num_res_blocks: int
    time_dim: int
    groups: int
    attention: bool

    def __post_init__(self):
        object.__setattr__(self, "channel_mults", tuple(self.channel_mults))
        _check(
            self.base_channels % self.groups == 0,
            "unet.groups must divide base_channels",
        )
        _check(self.time_dim % 2 == 0, "unet.time_dim must be even")


@dataclass(frozen=True)
class DiTConfig:
    patch_size: int
    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float

    def __post_init__(self):
        _check(
            self.hidden_dim % self.num_heads == 0,
            "dit.num_heads must divide hidden_dim",
        )
        _check(self.hidden_dim % 2 == 0, "dit.hidden_dim must be even")


@dataclass(frozen=True)
class NetworksConfig:
    unet: UNetConfig
    unet_train: TrainConfig
    dit: DiTConfig
    dit_train: TrainConfig


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build(cls, table: dict, where: str):
    expected = {f.name for f in fields(cls)}
    unknown, missing = set(table) - expected, expected - set(table)
    if unknown or missing:
        raise ValueError(f"[{where}]: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return cls(**table)


def _read(path: Path, sections: set[str]) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    if set(raw) != sections:
        raise ValueError(f"{path}: expected the tables {sorted(sections)}, found {sorted(raw)}")
    return raw


def load_math_config(path: str | Path | None = None) -> MathConfig:
    raw = _read(Path(path or CONFIG_DIR / "math.toml"), {f.name for f in fields(MathConfig)})
    return MathConfig(
        active_set=_build(ActiveSetConfig, raw["active_set"], "active_set"),
        sampler=_build(SamplerConfig, raw["sampler"], "sampler"),
        spectral=_build(SpectralConfig, raw["spectral"], "spectral"),
        numerics=_build(NumericsConfig, raw["numerics"], "numerics"),
    )


def load_networks_config(path: str | Path | None = None) -> NetworksConfig:
    raw = _read(Path(path or CONFIG_DIR / "networks.toml"), {"unet", "dit"})
    unet, dit = dict(raw["unet"]), dict(raw["dit"])
    unet_train, dit_train = unet.pop("train", {}), dit.pop("train", {})
    return NetworksConfig(
        unet=_build(UNetConfig, unet, "unet"),
        unet_train=_build(TrainConfig, unet_train, "unet.train"),
        dit=_build(DiTConfig, dit, "dit"),
        dit_train=_build(TrainConfig, dit_train, "dit.train"),
    )
