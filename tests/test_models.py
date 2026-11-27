import dataclasses

import numpy as np
import pytest
import torch
from conftest import normal, rng
from torch.func import jacfwd
from torch.nn.attention import SDPBackend, sdpa_kernel

from bendflow.config import TrainConfig, load_networks_config
from bendflow.curvature import jacobian_top_eig, mdot_autodiff
from bendflow.fields import NetworkField
from bendflow.metrics import endpoint_error, nearest_training_distance
from bendflow.models import DiT, UNet
from bendflow.models.dit import rope_rotate, rope_tables
from bendflow.samplers import BendFlowEuler
from bendflow.train import cfm_loss, load_model, save_model, train

SHAPE = (1, 8, 8)
UNET = {
    "base_channels": 8,
    "channel_mults": (1, 2),
    "num_res_blocks": 1,
    "time_dim": 16,
    "groups": 4,
    "attention": True,
}
DIT = {"patch_size": 4, "hidden_dim": 16, "depth": 2, "num_heads": 2, "mlp_ratio": 2.0}


# Replaces the zero-initialised output layers, so the output depends on the input.
@torch.no_grad()
def randomise(model, key):
    for param in model.parameters():
        param.add_(0.1 * torch.randn(param.shape, generator=key))
    return model


def build(name, shape=SHAPE):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return UNet(shape, **UNET) if name == "unet" else DiT(shape, **DIT)


@pytest.mark.parametrize("name", ["unet", "dit"])
@torch.no_grad()
def test_output_shape_and_zero_init(name):
    model = build(name)
    x = normal(1, SHAPE)
    out = model(x, torch.as_tensor(0.3))
    assert out.shape == SHAPE
    np.testing.assert_array_equal(out, 0.0)  # output layers start at zero
    out = randomise(model, rng(2))(x, torch.as_tensor(0.3))
    assert torch.all(torch.isfinite(out)) and torch.any(out != 0)


def test_unet_handles_odd_resolutions():
    shape = (3, 14, 14)  # 14 -> 7 -> 4
    model = UNet(shape, **{**UNET, "channel_mults": (1, 2, 2)})
    assert model(torch.zeros(shape), torch.as_tensor(0.5)).shape == shape


def test_dit_patchify_roundtrip():
    model = DiT((3, 8, 12), **DIT)
    x = normal(1, (3, 8, 12))
    tokens = model.patchify(x)
    assert tokens.shape == (6, 48)
    np.testing.assert_array_equal(tokens[1].reshape(4, 4, 3), x[:, :4, 4:8].permute(1, 2, 0))
    np.testing.assert_array_equal(model.unpatchify(tokens), x)


# After the rotation, the inner product of a query and a key depends only on the row and column offsets.
def test_rope_is_relative_and_norm_preserving():
    gh, gw, heads, dim = 3, 4, 2, 8
    cos, sin = rope_tables(gh, gw, dim)
    assert cos.shape == sin.shape == (gh * gw, dim // 2)
    q, k = normal(0, (2, 1, heads, dim))
    rq = rope_rotate(torch.broadcast_to(q, (gh * gw, heads, dim)), cos, sin)
    rk = rope_rotate(torch.broadcast_to(k, (gh * gw, heads, dim)), cos, sin)
    np.testing.assert_allclose(torch.linalg.norm(rq, axis=-1), torch.broadcast_to(torch.linalg.norm(q, axis=-1), (gh * gw, heads)), rtol=1e-12)
    np.testing.assert_allclose(rq[0], q[0], rtol=1e-12)  # row 0, column 0: no rotation
    logits = torch.einsum("ihd,jhd->hij", rq, rk).reshape(heads, gh, gw, gh, gw)
    for r, c in [(0, 0), (1, 2), (-1, 1), (2, -3)]:
        same = [logits[:, i, j, i + r, j + c] for i in range(gh) for j in range(gw) if 0 <= i + r < gh and 0 <= j + c < gw]
        np.testing.assert_allclose(torch.stack(same), torch.broadcast_to(same[0], (len(same), heads)), rtol=1e-10, atol=1e-12)
    assert not torch.allclose(logits[:, 0, 0, 0, 1], logits[:, 0, 0, 1, 0])


# Without positions the DiT would be equivariant to permuting patches; with RoPE it is not.
def test_dit_uses_patch_positions():
    model = randomise(build("dit"), rng(2))
    x = normal(1, SHAPE)
    swapped = torch.concatenate([x[:, :, 4:], x[:, :, :4]], axis=2)
    out, out_swapped = model(x, torch.as_tensor(0.3)), model(swapped, torch.as_tensor(0.3))
    assert not torch.allclose(torch.concatenate([out_swapped[:, :, 4:], out_swapped[:, :, :4]], axis=2), out, atol=1e-6)


def test_dit_rejects_head_dim_not_multiple_of_4():
    with pytest.raises(ValueError, match="multiple of 4"):
        DiT(SHAPE, **{**DIT, "hidden_dim": 12})


@pytest.mark.parametrize("name", ["unet", "dit"])
def test_models_from_config_files(name):
    cfg = load_networks_config()
    shape = (1, 28, 28)
    if name == "unet":
        model = UNet(shape, **dataclasses.asdict(cfg.unet))
    else:
        model = DiT(shape, **dataclasses.asdict(cfg.dit))
    assert model(torch.zeros(shape), torch.as_tensor(0.5)).shape == shape


@pytest.mark.parametrize("name", ["unet", "dit"])
@torch.no_grad()
def test_network_field_works_with_samplers_and_curvature(name):
    field = NetworkField(randomise(build(name), rng(2)), SHAPE)
    x = normal(3, (64,))
    t = torch.as_tensor(0.4)
    assert field(x, t).shape == (64,)

    # mdot_autodiff against a central difference along the direction (v, 1).
    # The time embedding oscillates at frequencies up to 1000, so the step must be small.
    v, h = field(x, t), 1e-7
    m = lambda s: (x + s * v) + (1 - (t + s)) * field(x + s * v, t + s)
    np.testing.assert_allclose(mdot_autodiff(field, x, t), (m(h) - m(-h)) / (2 * h), rtol=1e-5, atol=1e-7)

    with sdpa_kernel(SDPBackend.MATH):
        jac = jacfwd(field)(x, t)
    top = torch.linalg.eigvalsh((jac + jac.T) / 2)[-1]
    est, _ = jacobian_top_eig(field, x, t, rng(4), 400, shift=-torch.linalg.eigvalsh((jac + jac.T) / 2)[0])
    np.testing.assert_allclose(est, top, rtol=1e-4, atol=1e-6)

    traj = BendFlowEuler(tol=1e-2, h_min=0.02, h_max=0.25, h0=0.05, max_steps=64)(field, x)
    assert traj.ts[-1] == 1.0 and torch.all(torch.isfinite(traj.sample))


def train_config(**overrides):
    base = {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "lr_schedule": "cosine",
        "warmup_steps": 2,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "ema_decay": 0.9,
        "val_fraction": 0.2,
        "early_stopping": True,
        "patience": 2,
        "min_delta": 0.0,
    }
    return TrainConfig(**{**base, **overrides})


# A model that outputs 0 has loss E||x1 - eps||^2 / d = mean(x1^2) + 1.
@torch.no_grad()
def test_cfm_loss_at_zero_init_is_target_second_moment():
    x1 = torch.full((4096, *SHAPE), 0.5)
    loss = cfm_loss(build("dit"), x1, rng(0))
    np.testing.assert_allclose(loss, 0.25 + 1.0, rtol=0.01)


@pytest.mark.parametrize("name", ["unet", "dit"])
def test_training_reduces_loss(name, tmp_path):
    data = torch.tile(2 * torch.rand((8, 64), generator=rng(5)) - 1, (20, 1))
    lines = []
    model, history = train(
        build(name),
        data,
        SHAPE,
        train_config(epochs=12, ema_decay=0.0, early_stopping=False),
        6,
        lines.append,
    )
    assert len(history["train_loss"]) == len(history["val_loss"]) == len(lines) == 12
    # The validation loss uses the same t and noise every epoch, so it is comparable.
    assert history["val_loss"][-1] < history["val_loss"][0]
    assert history["val_loss"][history["best_epoch"] - 1] == min(history["val_loss"])
    assert history["wall_time"] > 0

    save_model(tmp_path / "ckpt" / "model.pt", model)
    loaded = load_model(tmp_path / "ckpt" / "model.pt", build(name))
    x = normal(7, SHAPE)
    torch.testing.assert_close(loaded(x, torch.as_tensor(0.5)), model(x, torch.as_tensor(0.5)), rtol=0, atol=0)


def test_early_stopping_and_no_validation():
    data = 2 * torch.rand((80, 64), generator=rng(5)) - 1
    # min_delta so large that no epoch after the first counts as an improvement.
    cfg = train_config(epochs=10, patience=2, min_delta=1e9)
    _, history = train(build("dit"), data, SHAPE, cfg, 6, lambda _: None)
    assert history["stopped_early"] and len(history["train_loss"]) == 3 and history["best_epoch"] == 1

    cfg = train_config(val_fraction=0.0, early_stopping=False)
    _, history = train(build("dit"), data, SHAPE, cfg, 6, lambda _: None)
    assert history["val_loss"] == [] and len(history["train_loss"]) == 3 and not history["stopped_early"]


def test_metrics():
    data = torch.eye(4)
    dist, i = nearest_training_distance(torch.tensor([0.0, 0.9, 0.0, 0.0]), data)
    assert int(i) == 1
    np.testing.assert_allclose(dist, 0.1, rtol=1e-6)
    np.testing.assert_allclose(endpoint_error(data[0], data[1]), np.sqrt(2.0), rtol=1e-6)


# load_network rebuilds the model from the config, loads the weights and keeps float32 inside a float64 field.
@torch.no_grad()
def test_load_network_roundtrip(tmp_path):
    from bendflow.config import load_networks_config
    from bendflow.data import image_shape
    from bendflow.networks import build_model, load_network

    shape = image_shape("mnist")
    model, _ = build_model("unet", shape, load_networks_config(), 0)
    model = randomise(model, rng(3)).float()
    save_model(tmp_path / "mnist_unet.pt", model)
    field, meta = load_network("unet", "mnist", str(tmp_path))
    assert meta == {}
    x = normal(4, (28 * 28,))
    out = field(x, torch.as_tensor(0.3))
    assert out.dtype == x.dtype
    np.testing.assert_allclose(out, model(x.reshape(shape).float(), torch.as_tensor(0.3, dtype=torch.float32)).reshape(-1), rtol=1e-5, atol=1e-6)
