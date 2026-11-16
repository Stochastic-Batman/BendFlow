import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


# Posterior over training images for one sample x of shape (d,) and a scalar t in [0, 1); batch with jax.vmap.
# C is never built, it is only available through matrix-vector products.
class EmpiricalPosterior(eqx.Module):
    data: Array  # (N, d) training set

    # log w_i with w_i proportional to exp(-||x - t x_i||^2 / 2(1-t)^2)
    def log_weights(self, x: Array, t: Array) -> Array:
        g = -jnp.sum((x - t * self.data) ** 2, axis=1) / (2.0 * (1.0 - t) ** 2)
        return g - jax.nn.logsumexp(g)

    def weights_and_mean(self, x: Array, t: Array) -> tuple[Array, Array]:
        w = jnp.exp(self.log_weights(x, t))
        return w, w @ self.data

    def mean(self, x: Array, t: Array) -> Array:
        return self.weights_and_mean(x, t)[1]

    # C v with C = sum_i w_i (x_i - m)(x_i - m)^T
    def cov_matvec(self, x: Array, t: Array, v: Array) -> Array:
        return self.cov_matvec_from(*self.weights_and_mean(x, t), v)

    # tr C = sum_i w_i ||x_i - m||^2
    def trace_cov(self, x: Array, t: Array) -> Array:
        w, m = self.weights_and_mean(x, t)
        return w @ self._sq_dev(m)

    # kappa = sum_i w_i ||x_i - m||^2 (x_i - m)
    def kappa(self, x: Array, t: Array) -> Array:
        return self.kappa_from(*self.weights_and_mean(x, t))

    # Largest eigenvalue of C by power iteration, returned as (rho, residual). rho approaches it from below,
    # residual = ||C v - rho v|| / rho, and some eigenvalue of C lies within rho * residual of rho.
    def lambda_max(self, x: Array, t: Array, key: Array, num_iters: int) -> tuple[Array, Array]:
        w, m = self.weights_and_mean(x, t)
        rho, res = power_iteration(lambda v: self.cov_matvec_from(w, m, v), x, key, num_iters)
        return rho, res / jnp.where(rho > 0, rho, 1.0)

    # Boolean mask of A = {i : g_i >= g_istar - log(N / delta)}
    def active_set(self, x: Array, t: Array, delta: float) -> Array:
        logw = self.log_weights(x, t)
        return logw >= jnp.max(logw) - jnp.log(self.data.shape[0] / delta)

    # The *_from variants take precomputed weights and mean, so that several
    # moments at the same (x, t) share one pass over the training set.

    def cov_matvec_from(self, w: Array, m: Array, v: Array) -> Array:
        wp = w * (self.data @ v - m @ v)
        return wp @ self.data - jnp.sum(wp) * m

    def kappa_from(self, w: Array, m: Array) -> Array:
        ws = w * self._sq_dev(m)
        return ws @ self.data - jnp.sum(ws) * m

    def _sq_dev(self, m: Array) -> Array:
        return jnp.sum((self.data - m) ** 2, axis=1)


# Returns (rho, ||M v - rho v||): the Rayleigh quotient and its absolute residual after num_iters
# power iterations from a random start. Both are 0 for the zero operator.
def power_iteration(matvec, like: Array, key: Array, num_iters: int) -> tuple[Array, Array]:

    def normalise(u):
        n = jnp.linalg.norm(u)
        return u / jnp.where(n > 0, n, 1.0)

    v0 = normalise(jax.random.normal(key, like.shape, like.dtype))
    v = jax.lax.fori_loop(0, num_iters, lambda _, v: normalise(matvec(v)), v0)
    mv = matvec(v)
    rho = v @ mv
    return rho, jnp.linalg.norm(mv - rho * v)


# Largest pairwise distance between the rows, computed in blocks of chunk_size rows. NumPy, not jittable, O(N^2 d).
def diameter(points, chunk_size: int = 1024) -> float:
    p = np.asarray(points)
    sq = np.sum(p * p, axis=1)
    best = 0.0
    for start in range(0, p.shape[0], chunk_size):
        block = p[start : start + chunk_size]
        d2 = sq[start : start + chunk_size, None] + sq[None, :] - 2.0 * (block @ p.T)
        best = max(best, float(d2.max()))
    return float(np.sqrt(max(best, 0.0)))
