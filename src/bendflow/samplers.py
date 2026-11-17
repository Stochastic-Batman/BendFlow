from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


# Padded record of one run. Only the first nfe steps are real: xs[nfe] is the sample and later entries repeat it,
# later ts are 1 and later x1_hats repeat the last denoised prediction.
class Trajectory(NamedTuple):
    xs: Array  # (S + 1, d) states x_0, ..., x_nfe
    ts: Array  # (S + 1,)   times  t_0 = 0, ..., t_nfe = 1
    x1_hats: Array  # (S, d) denoised predictions x_k + (1 - t_k) v_k
    nfe: Array  # number of field evaluations

    @property
    def sample(self) -> Array:
        return self.xs[-1]


class UniformEuler(eqx.Module):
    num_steps: int

    def __call__(self, field, x0: Array) -> Trajectory:
        n = self.num_steps
        h = 1.0 / n

        def step(x, k):
            t = k * h
            v = field(x, jnp.asarray(t, x0.dtype))
            x_next = x + h * v
            return x_next, (x_next, x + (1.0 - t) * v)

        _, (xs, x1_hats) = jax.lax.scan(step, x0, jnp.arange(n))
        xs = jnp.concatenate([x0[None], xs])
        ts = jnp.linspace(0.0, 1.0, n + 1, dtype=x0.dtype)
        return Trajectory(xs, ts, x1_hats, jnp.asarray(n))


# Algorithm 1. max_steps only sizes the padded buffers; the config loader requires max_steps >= ceil(1 / h_min) + 1.
# If it is built by hand with a smaller value, the last step jumps to t = 1.
class BendFlowEuler(eqx.Module):
    tol: float
    h_min: float
    h_max: float
    h0: float
    max_steps: int

    def __call__(self, field, x0: Array) -> Trajectory:
        n, dtype = self.max_steps, x0.dtype
        zero, one = jnp.zeros((), dtype), jnp.ones((), dtype)

        def cond(state):
            k, t = state[0], state[1]
            return (t < 1.0) & (k < n)

        def body(state):
            k, t, x, x1_prev, h_prev, xs, ts, x1_hats = state
            v = field(x, t)
            x1 = x + (1.0 - t) * v
            mdot_norm = jnp.linalg.norm((x1 - x1_prev) / h_prev)
            h_adapt = jnp.clip(jnp.sqrt(2.0 * (1.0 - t) * self.tol / mdot_norm), self.h_min, self.h_max)
            h = jnp.where(k >= 1, h_adapt, self.h0)
            last = (h >= 1.0 - t) | (k == n - 1)
            h = jnp.where(last, 1.0 - t, h)
            x_next = x + h * v
            t_next = jnp.where(last, one, t + h)
            return (
                k + 1,
                t_next,
                x_next,
                x1,
                h,
                xs.at[k + 1].set(x_next),
                ts.at[k + 1].set(t_next),
                x1_hats.at[k].set(x1),
            )

        init = (
            jnp.asarray(0),
            zero,
            x0,
            jnp.zeros_like(x0),
            one,
            jnp.zeros((n + 1, *x0.shape), dtype).at[0].set(x0),
            jnp.ones((n + 1,), dtype).at[0].set(0.0),
            jnp.zeros((n, *x0.shape), dtype),
        )
        k, _, x, x1, _, xs, ts, x1_hats = jax.lax.while_loop(cond, body, init)
        xs = jnp.where((jnp.arange(n + 1) <= k)[:, None], xs, x)
        x1_hats = jnp.where((jnp.arange(n) < k)[:, None], x1_hats, x1)
        return Trajectory(xs, ts, x1_hats, k)


# Euler on a fixed time grid ts = (0, t_1, ..., 1), the same for every sample. ts is an array, so one compiled
# sampler serves every grid of the same length.
class ScheduleEuler(eqx.Module):
    ts: Array

    def __call__(self, field, x0: Array) -> Trajectory:
        ts = jnp.asarray(self.ts, x0.dtype)

        def step(x, k):
            t, h = ts[k], ts[k + 1] - ts[k]
            v = field(x, t)
            x_next = x + h * v
            return x_next, (x_next, x + (1.0 - t) * v)

        n = ts.shape[0] - 1
        _, (xs, x1_hats) = jax.lax.scan(step, x0, jnp.arange(n))
        return Trajectory(jnp.concatenate([x0[None], xs]), ts, x1_hats, jnp.asarray(n))
