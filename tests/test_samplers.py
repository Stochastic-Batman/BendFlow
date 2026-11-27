import dataclasses

import numpy as np
import pytest
import torch
from conftest import normal

from bendflow.config import load_math_config
from bendflow.curvature import mdot_closed_form
from bendflow.fields import ExactField
from bendflow.paths import GaussianPath
from bendflow.posterior import EmpiricalPosterior
from bendflow.samplers import BendFlowEuler, ScheduleEuler, UniformEuler

D = 5


@pytest.fixture
def field(posterior):
    return ExactField(posterior)


@pytest.fixture
def x0():
    return normal(7, (D,))


def test_uniform_euler_matches_python_loop(field, x0):
    traj = UniformEuler(8)(field, x0)
    x = x0
    for k in range(8):
        t = torch.as_tensor(k / 8)
        v = field(x, t)
        np.testing.assert_allclose(traj.xs[k], x, rtol=1e-12)
        np.testing.assert_allclose(traj.x1_hats[k], x + (1 - t) * v, rtol=1e-12)
        x = x + v / 8
    np.testing.assert_allclose(traj.sample, x, rtol=1e-12)
    np.testing.assert_allclose(traj.ts, torch.linspace(0, 1, 9), rtol=1e-12)
    assert int(traj.nfe) == 8


# Replays Algorithm 1 in plain Python and compares it with the sampler.
def test_bendflow_follows_algorithm_1(field, x0):
    s = BendFlowEuler(tol=1e-2, h_min=0.01, h_max=0.2, h0=0.05, max_steps=128)
    traj = s(field, x0)

    t, x, k, h_prev, x1_prev, hs = 0.0, x0, 0, None, None, []
    while t < 1:
        v = field(x, torch.as_tensor(t))
        x1 = x + (1 - t) * v
        if k >= 1:
            mdot = torch.linalg.norm((x1 - x1_prev) / h_prev)
            h = float(torch.clip(torch.sqrt(2 * (1 - t) * s.tol / mdot), s.h_min, s.h_max))
        else:
            h = s.h0
        h = min(h, 1 - t)
        np.testing.assert_allclose(traj.xs[k], x, rtol=1e-9)
        np.testing.assert_allclose(traj.ts[k], t, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(traj.x1_hats[k], x1, rtol=1e-9)
        x, t, k, h_prev, x1_prev = (
            x + h * v,
            (1.0 if h >= 1 - t else t + h),
            k + 1,
            h,
            x1,
        )
        hs.append(h)

    assert int(traj.nfe) == k < s.max_steps
    assert len(set(np.round(hs, 6))) > 3  # the step size really adapts
    np.testing.assert_allclose(traj.sample, x, rtol=1e-9)
    # Padding: states repeat the sample, times are 1.
    np.testing.assert_allclose(traj.xs[k:], torch.broadcast_to(x, (s.max_steps + 1 - k, D)), rtol=1e-9)
    assert torch.all(traj.ts[k:] == 1.0)


def test_bendflow_step_sizes_respect_clipping(field, x0):
    s = BendFlowEuler(tol=1e-3, h_min=0.02, h_max=0.1, h0=0.05, max_steps=64)
    traj = s(field, x0)
    n = int(traj.nfe)
    hs = torch.diff(traj.ts[: n + 1])
    assert traj.ts[n] == 1.0 and torch.all(hs > 0)
    assert torch.all(hs[1:-1] >= s.h_min - 1e-12) and torch.all(hs[1:] <= s.h_max + 1e-12)


def test_bendflow_reaches_t1_when_max_steps_is_hit(field, x0):
    traj = BendFlowEuler(tol=1e-6, h_min=0.01, h_max=0.01, h0=0.01, max_steps=10)(field, x0)
    assert int(traj.nfe) == 10 and traj.ts[-1] == 1.0
    assert torch.all(torch.isfinite(traj.xs))


# One training image: mdot = 0, so every step after the first is h_max.
def test_bendflow_zero_curvature_takes_largest_steps():
    from bendflow.posterior import EmpiricalPosterior

    data = normal(0, (1, D))
    s = BendFlowEuler(tol=1e-2, h_min=0.01, h_max=0.3, h0=0.1, max_steps=32)
    traj = s(ExactField(EmpiricalPosterior(data)), normal(1, (D,)))
    np.testing.assert_allclose(traj.ts[:5], [0.0, 0.1, 0.4, 0.7, 1.0], rtol=1e-12)
    assert int(traj.nfe) == 4
    np.testing.assert_allclose(traj.sample, data[0], rtol=1e-9)


def test_samplers_batch_and_config(field):
    cfg = load_math_config().sampler
    sampler = BendFlowEuler(**dataclasses.asdict(cfg))
    x0s = normal(3, (4, D))
    batch = sampler(field, x0s)
    assert batch.xs.shape == (4, cfg.max_steps + 1, D) and batch.nfe.shape == (4,)
    single = sampler(field, x0s[2])
    assert int(batch.nfe[2]) == int(single.nfe)
    np.testing.assert_allclose(batch.xs[2], single.xs, rtol=1e-8, atol=1e-10)


# Proposition 2: mdot_hat_k = mdot(t_{k-1}) + O(h) on the exact field.
def test_mdot_estimator_is_first_order(posterior, field, x0):

    def error(n):
        traj = UniformEuler(n)(field, x0)
        k = n // 2  # estimate between t = 0.5 - 1/n and t = 0.5
        est = (traj.x1_hats[k] - traj.x1_hats[k - 1]) * n
        return torch.linalg.norm(est - mdot_closed_form(posterior, traj.xs[k - 1], traj.ts[k - 1]))

    e1, e2, e3 = error(200), error(400), error(800)
    assert e3 < e2 < e1
    np.testing.assert_allclose(e2 / e3, 2.0, rtol=0.2)


def test_schedule_euler_on_a_uniform_grid_matches_uniform_euler(field, x0):
    a = UniformEuler(16)(field, x0)
    b = ScheduleEuler(torch.linspace(0.0, 1.0, 17))(field, x0)
    np.testing.assert_allclose(b.xs, a.xs, rtol=1e-12)
    np.testing.assert_allclose(b.x1_hats, a.x1_hats, rtol=1e-12)
    assert int(b.nfe) == 16


def test_schedule_euler_follows_its_grid(field, x0):
    ts = torch.tensor([0.0, 0.1, 0.15, 0.5, 0.9, 1.0])
    traj = ScheduleEuler(ts)(field, x0)
    x = x0
    for k in range(5):
        x = x + (ts[k + 1] - ts[k]) * field(x, ts[k])
    np.testing.assert_allclose(traj.sample, x, rtol=1e-12)
    np.testing.assert_allclose(traj.ts, ts, rtol=1e-12)


# Scalar tensors and Python numbers can be used for sampler settings.
def test_bendflow_accepts_array_settings(field, x0):
    run = lambda s, x: s(field, x)
    a = run(BendFlowEuler(torch.as_tensor(1e-2), torch.as_tensor(0.01), torch.as_tensor(0.2), torch.as_tensor(0.05), 128), x0)
    b = BendFlowEuler(1e-2, 0.01, 0.2, 0.05, 128)(field, x0)
    np.testing.assert_allclose(a.xs, b.xs, rtol=1e-12)
    assert int(a.nfe) == int(b.nfe)


@pytest.mark.parametrize("path", [GaussianPath.COSINE, GaussianPath.VE])
def test_ddim_follows_single_image_path_at_every_grid_point(x0, path):
    clean = normal(9, x0.shape)
    post = EmpiricalPosterior(clean.unsqueeze(0), path)
    t0 = x0.new_zeros(())
    field = ExactField(post)
    initial, _ = path.interpolant(clean, x0, t0)
    for sampler in (UniformEuler(12, path), BendFlowEuler(0.01, 0.01, 0.2, 0.05, 64, path)):
        tr = sampler(field, initial)
        alpha, sigma, _, _ = path.coefficients(tr.ts)
        expected = alpha[:, None] * clean + sigma[:, None] * x0
        torch.testing.assert_close(tr.xs, expected)
        torch.testing.assert_close(tr.sample, clean)
