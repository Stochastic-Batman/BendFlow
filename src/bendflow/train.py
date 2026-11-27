import copy
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from bendflow.config import TrainConfig
from bendflow.experiment import generator
from bendflow.paths import GaussianPath


def cfm_loss(model, x1: Tensor, rng: torch.Generator, path: GaussianPath = GaussianPath.LINEAR) -> Tensor:
    t = torch.rand(len(x1), dtype=x1.dtype, device=x1.device, generator=rng)
    eps = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=rng)
    inputs, velocity = path.interpolant(x1, eps, t)
    return mse_loss(model(inputs, t), velocity)


def _learning_rate(cfg: TrainConfig, step: int, total_steps: int) -> float:
    warmup = min(cfg.warmup_steps, total_steps - 1)
    if step < warmup:
        return cfg.learning_rate * step / warmup
    if cfg.lr_schedule == "cosine":
        progress = (step - warmup) / (total_steps - warmup)
        return cfg.learning_rate * (1 + math.cos(math.pi * progress)) / 2
    return cfg.learning_rate


def train(model, data, image_shape: tuple[int, int, int], cfg: TrainConfig, seed: int, log=print, *, path: GaussianPath = GaussianPath.LINEAR):
    device = next(model.parameters()).device
    data = torch.as_tensor(data, device=device, dtype=next(model.parameters()).dtype).reshape(-1, *image_shape)
    rng = generator(device, seed)
    val_rng = generator(device, seed + 1)
    perm = torch.randperm(len(data), generator=rng, device=device)
    n_val = int(cfg.val_fraction * len(data))
    val, train_data = data[perm[:n_val]], data[perm[n_val:]]
    steps_per_epoch = len(train_data) // cfg.batch_size
    if steps_per_epoch == 0:
        raise ValueError("batch_size is larger than the training split")
    if cfg.epochs < 1:
        raise ValueError("epochs must be positive")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema_decay > 0 else None
    averaged = ema if ema is not None else model

    @torch.no_grad()
    def validate(candidate):
        # Replay the validation draws at each epoch, including the reference batch convention.
        candidate.eval()
        val_rng.manual_seed(seed + 1)
        n_batches = max(n_val // cfg.batch_size, 1)
        chunks = val[: n_batches * cfg.batch_size].split(cfg.batch_size)
        return sum(float(cfm_loss(candidate, x, val_rng, path)) * len(x) for x in chunks) / sum(map(len, chunks))

    history = {"train_loss": [], "val_loss": [], "best_epoch": None, "stopped_early": False}
    best_val, best_weights, bad_epochs = float("inf"), None, 0
    total_steps = cfg.epochs * steps_per_epoch
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(train_data), generator=rng, device=device)
        total = torch.zeros((), device=device)
        for i, batch_ids in enumerate(order[: steps_per_epoch * cfg.batch_size].split(cfg.batch_size)):
            opt.param_groups[0]["lr"] = _learning_rate(cfg, (epoch - 1) * steps_per_epoch + i, total_steps)
            opt.zero_grad(set_to_none=True)
            loss = cfm_loss(model, train_data[batch_ids], rng, path)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for average, param in zip(ema.parameters(), model.parameters()):
                        average.lerp_(param, 1 - cfg.ema_decay)
            total += loss.detach()
        history["train_loss"].append(float(total) / steps_per_epoch)
        line = f"epoch {epoch}/{cfg.epochs} train {history['train_loss'][-1]:.5f}"
        if n_val > 0:
            val_loss = validate(averaged)
            history["val_loss"].append(val_loss)
            line += f" val {val_loss:.5f}"
            if val_loss < best_val - cfg.min_delta:
                best_val, bad_epochs = val_loss, 0
                best_weights = copy.deepcopy(averaged.state_dict())
                history["best_epoch"] = epoch
            else:
                bad_epochs += 1
        log(f"{line} time {time.perf_counter() - start:.1f}s")
        if cfg.early_stopping and bad_epochs >= cfg.patience:
            history["stopped_early"] = True
            log(f"early stopping at epoch {epoch}, best epoch {history['best_epoch']}")
            break

    history["wall_time"] = time.perf_counter() - start
    if best_weights is not None:
        averaged.load_state_dict(best_weights)
    return averaged.eval(), history


def save_model(path: str | Path, model) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, like):
    like.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return like.eval()
