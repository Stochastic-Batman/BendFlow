import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bendflow.config import load_math_config
from bendflow.curvature import mdot_closed_form
from bendflow.fields import ExactField
from bendflow.samplers import BendFlowEuler, ScheduleEuler, UniformEuler

D = 5


@pytest.fixture
def field(posterior):
    return ExactField(posterior)


@pytest.fixture
def x0():
    return jax.random.normal(jax.random.key(7), (D,))


def test_uniform_euler_matches_python_loop(field, x0):
    traj = UniformEuler(8)(field, x0)
    x = x0
    for k in range(8):
        t = jnp.asarray(k / 8)
        v = field(x, t)
        np.testing.assert_allclose(traj.xs[k], x, rtol=1e-12)
        np.testing.assert_allclose(traj.x1_hats[k], x + (1 - t) * v, rtol=1e-12)
        x = x + v / 8
    np.testing.assert_allclose(traj.sample, x, rtol=1e-12)
    np.testing.assert_allclose(traj.ts, jnp.linspace(0, 1, 9), rtol=1e-12)
    assert int(traj.nfe) == 8


# Replays Algorithm 1 in plain Python and compares it with the jitted sampler.
def test_bendflow_follows_algorithm_1(field, x0):
    s = BendFlowEuler(tol=1e-2, h_min=0.01, h_max=0.2, h0=0.05, max_steps=128)
    traj = eqx.filter_jit(s)(field, x0)

    t, x, k, h_prev, x1_prev, hs = 0.0, x0, 0, None, None, []
    while t < 1:
        v = field(x, jnp.asarray(t))
        x1 = x + (1 - t) * v
        if k >= 1:
            mdot = jnp.linalg.norm((x1 - x1_prev) / h_prev)
            h = float(jnp.clip(jnp.sqrt(2 * (1 - t) * s.tol / mdot), s.h_min, s.h_max))
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
    np.testing.assert_allclose(traj.xs[k:], jnp.broadcast_to(x, (s.max_steps + 1 - k, D)), rtol=1e-9)
    assert jnp.all(traj.ts[k:] == 1.0)


def test_bendflow_step_sizes_respect_clipping(field, x0):
    s = BendFlowEuler(tol=1e-3, h_min=0.02, h_max=0.1, h0=0.05, max_steps=64)
    traj = s(field, x0)
    n = int(traj.nfe)
    hs = jnp.diff(traj.ts[: n + 1])
    assert traj.ts[n] == 1.0 and jnp.all(hs > 0)
    assert jnp.all(hs[1:-1] >= s.h_min - 1e-12) and jnp.all(hs[1:] <= s.h_max + 1e-12)


def test_bendflow_reaches_t1_when_max_steps_is_hit(field, x0):
    traj = BendFlowEuler(tol=1e-6, h_min=0.01, h_max=0.01, h0=0.01, max_steps=10)(field, x0)
    assert int(traj.nfe) == 10 and traj.ts[-1] == 1.0
    assert jnp.all(jnp.isfinite(traj.xs))


# One training image: mdot = 0, so every step after the first is h_max.
def test_bendflow_zero_curvature_takes_largest_steps():
    from bendflow.posterior import EmpiricalPosterior

    data = jax.random.normal(jax.random.key(0), (1, D))
    s = BendFlowEuler(tol=1e-2, h_min=0.01, h_max=0.3, h0=0.1, max_steps=32)
    traj = s(ExactField(EmpiricalPosterior(data)), jax.random.normal(jax.random.key(1), (D,)))
    np.testing.assert_allclose(traj.ts[:5], [0.0, 0.1, 0.4, 0.7, 1.0], rtol=1e-12)
    assert int(traj.nfe) == 4
    np.testing.assert_allclose(traj.sample, data[0], rtol=1e-9)


def test_samplers_vmap_and_config(field):
    cfg = load_math_config().sampler
    sampler = BendFlowEuler(**dataclasses.asdict(cfg))
    x0s = jax.random.normal(jax.random.key(3), (4, D))
    batch = eqx.filter_jit(jax.vmap(lambda x: sampler(field, x)))(x0s)
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
        return jnp.linalg.norm(est - mdot_closed_form(posterior, traj.xs[k - 1], traj.ts[k - 1]))

    e1, e2, e3 = error(200), error(400), error(800)
    assert e3 < e2 < e1
    np.testing.assert_allclose(e2 / e3, 2.0, rtol=0.2)


def test_schedule_euler_on_a_uniform_grid_matches_uniform_euler(field, x0):
    a = UniformEuler(16)(field, x0)
    b = ScheduleEuler(jnp.linspace(0.0, 1.0, 17))(field, x0)
    np.testing.assert_allclose(b.xs, a.xs, rtol=1e-12)
    np.testing.assert_allclose(b.x1_hats, a.x1_hats, rtol=1e-12)
    assert int(b.nfe) == 16


def test_schedule_euler_follows_its_grid(field, x0):
    ts = jnp.array([0.0, 0.1, 0.15, 0.5, 0.9, 1.0])
    traj = ScheduleEuler(ts)(field, x0)
    x = x0
    for k in range(5):
        x = x + (ts[k + 1] - ts[k]) * field(x, ts[k])
    np.testing.assert_allclose(traj.sample, x, rtol=1e-12)
    np.testing.assert_allclose(traj.ts, ts, rtol=1e-12)


# Sampler settings passed as arrays are traced, so a jitted run does not recompile when tol changes.
def test_bendflow_accepts_array_settings(field, x0):
    run = eqx.filter_jit(lambda s, x: s(field, x))
    a = run(BendFlowEuler(jnp.asarray(1e-2), jnp.asarray(0.01), jnp.asarray(0.2), jnp.asarray(0.05), 128), x0)
    b = BendFlowEuler(1e-2, 0.01, 0.2, 0.05, 128)(field, x0)
    np.testing.assert_allclose(a.xs, b.xs, rtol=1e-12)
    assert int(a.nfe) == int(b.nfe)
