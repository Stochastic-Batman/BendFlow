import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from bendflow.config import TrainConfig


# MSE to the conditional target x_1 - eps on the path x_t = t x_1 + (1 - t) eps. x1 is a batch (B, C, H, W).
def cfm_loss(model, x1: Array, key: Array) -> Array:
    k_t, k_eps = jax.random.split(key)
    t = jax.random.uniform(k_t, (x1.shape[0],), x1.dtype)
    eps = jax.random.normal(k_eps, x1.shape, x1.dtype)
    tb = t[:, None, None, None]
    pred = jax.vmap(model)(tb * x1 + (1.0 - tb) * eps, t)
    return jnp.mean((pred - (x1 - eps)) ** 2)


def _optimizer(cfg: TrainConfig, total_steps: int) -> optax.GradientTransformation:
    warmup = min(cfg.warmup_steps, max(total_steps - 1, 0))
    if cfg.lr_schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(0.0, cfg.learning_rate, warmup, total_steps)
    else:
        schedule = optax.warmup_constant_schedule(0.0, cfg.learning_rate, warmup)
    parts = [optax.clip_by_global_norm(cfg.grad_clip)] if cfg.grad_clip > 0 else []
    return optax.chain(*parts, optax.adamw(schedule, weight_decay=cfg.weight_decay))


# Trains on data of shape (N, d) and returns (model, history). The model carries the EMA weights if ema_decay > 0,
# and those of the best validation epoch if a validation split is used.
def train(model, data: Array, image_shape: tuple[int, int, int], cfg: TrainConfig, key: Array, log=print):
    k_split, k_val, key = jax.random.split(key, 3)
    data = jnp.asarray(data).reshape(-1, *image_shape)
    perm = jax.random.permutation(k_split, data.shape[0])
    n_val = int(cfg.val_fraction * data.shape[0])
    val, train_data = data[perm[:n_val]], data[perm[n_val:]]
    steps_per_epoch = train_data.shape[0] // cfg.batch_size
    if steps_per_epoch == 0:
        raise ValueError("batch_size is larger than the training split")

    opt = _optimizer(cfg, cfg.epochs * steps_per_epoch)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = opt.init(params)
    use_ema = cfg.ema_decay > 0
    ema = params

    @eqx.filter_jit
    def step(model, ema, opt_state, batch, key):
        loss, grads = eqx.filter_value_and_grad(cfm_loss)(model, batch, key)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = opt.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        if use_ema:
            new = eqx.filter(model, eqx.is_inexact_array)
            ema = jax.tree.map(lambda e, p: cfg.ema_decay * e + (1.0 - cfg.ema_decay) * p, ema, new)
        return model, ema, opt_state, loss

    eval_loss = eqx.filter_jit(cfm_loss)

    def averaged(model):
        return eqx.combine(ema, model) if use_ema else model

    def validate(model):
        # Same key every epoch: the same t and noise, so epochs are comparable.
        keys = jax.random.split(k_val, max(n_val // cfg.batch_size, 1))
        chunks = [val[i * cfg.batch_size : (i + 1) * cfg.batch_size] for i in range(len(keys))]
        return sum(float(eval_loss(model, c, k)) * len(c) for c, k in zip(chunks, keys)) / sum(len(c) for c in chunks)

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "stopped_early": False,
    }
    best_val, best_model, bad_epochs = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        key, k_perm, k_steps = jax.random.split(key, 3)
        order = jax.random.permutation(k_perm, train_data.shape[0])
        total = 0.0
        for i, k in enumerate(jax.random.split(k_steps, steps_per_epoch)):
            batch = train_data[order[i * cfg.batch_size : (i + 1) * cfg.batch_size]]
            model, ema, opt_state, loss = step(model, ema, opt_state, batch, k)
            total += float(loss)
        history["train_loss"].append(total / steps_per_epoch)
        line = f"epoch {epoch}/{cfg.epochs} train {history['train_loss'][-1]:.5f}"

        if n_val > 0:
            val_loss = validate(averaged(model))
            history["val_loss"].append(val_loss)
            line += f" val {val_loss:.5f}"
            if val_loss < best_val - cfg.min_delta:
                best_val, best_model, bad_epochs = val_loss, averaged(model), 0
                history["best_epoch"] = epoch
            else:
                bad_epochs += 1
        log(f"{line} time {time.perf_counter() - start:.1f}s")
        if cfg.early_stopping and bad_epochs >= cfg.patience:
            history["stopped_early"] = True
            log(f"early stopping at epoch {epoch}, best epoch {history['best_epoch']}")
            break

    history["wall_time"] = time.perf_counter() - start
    return (best_model if best_model is not None else averaged(model)), history


def save_model(path: str | Path, model) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(Path(path), model)


# Loads weights into like, a model built with the same configuration.
def load_model(path: str | Path, like):
    return eqx.tree_deserialise_leaves(Path(path), like)
