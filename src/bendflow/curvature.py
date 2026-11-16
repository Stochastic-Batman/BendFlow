from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bendflow.posterior import EmpiricalPosterior, diameter, power_iteration


# mdot = C eps_hat / (1-t)^2 - t kappa / (1-t)^3 (Theorem 1)
def mdot_closed_form(posterior: EmpiricalPosterior, x: Array, t: Array) -> Array:
    r = 1.0 - t
    w, m = posterior.weights_and_mean(x, t)
    eps_hat = (x - t * m) / r
    return posterior.cov_matvec_from(w, m, eps_hat) / r**2 - t * posterior.kappa_from(w, m) / r**3


# V_t = tr(C) / (1-t)^2
def velocity_variance(posterior: EmpiricalPosterior, x: Array, t: Array) -> Array:
    return posterior.trace_cov(x, t) / (1.0 - t) ** 2


# div u_t = (t V_t - d) / (1-t)
def divergence_closed_form(posterior: EmpiricalPosterior, x: Array, t: Array) -> Array:
    return (t * velocity_variance(posterior, x, t) - x.shape[0]) / (1.0 - t)


class LocalBound(NamedTuple):
    lam: float  # bound on lambda_max(C)
    kappa: float  # bound on ||kappa||
    mdot: float  # bound on ||mdot||
    d_active: float  # D_A
    n_active: int  # |A|


# Bounds of Theorem 2. d_all is the diameter D of the whole training set, computed once with diameter.
# Not jittable: the active set has a data-dependent size.
def local_bound(posterior: EmpiricalPosterior, x: Array, t: Array, delta: float, d_all: float, chunk_size: int = 1024) -> LocalBound:
    mask = np.asarray(posterior.active_set(x, t, delta))
    if mask.all():
        d_active = d_all
    else:
        d_active = diameter(np.asarray(posterior.data)[mask], chunk_size)
    r = 1.0 - float(t)
    eps_norm = float(jnp.linalg.norm(x - t * posterior.mean(x, t))) / r
    lam = d_active**2 / 4.0 + delta * d_all**2
    kap = (d_active + delta * d_all) ** 3 + delta * d_all**3
    mdot = lam * eps_norm / r**2 + float(t) * kap / r**3
    return LocalBound(lam, kap, mdot, d_active, int(mask.sum()))


# mdot along the trajectory of field: one jvp of (x, t) -> x + (1-t) field(x, t) with tangent (field(x, t), 1)
def mdot_autodiff(field, x: Array, t: Array) -> Array:
    t = jnp.asarray(t, x.dtype)
    _, out = jax.jvp(
        lambda x, t: x + (1.0 - t) * field(x, t),
        (x, t),
        (field(x, t), jnp.ones_like(t)),
    )
    return out


# Top eigenvalue of the symmetric part of the Jacobian of field, by power iteration on J + shift I (default shift 1/(1-t)).
# Returns (eigenvalue, absolute residual). symmetrize=False skips the symmetrisation for the exact field, where J is symmetric.
def jacobian_top_eig(field, x: Array, t: Array, key: Array, num_iters: int, shift: Array | None = None, symmetrize: bool = True) -> tuple[Array, Array]:
    t = jnp.asarray(t, x.dtype)
    shift = 1.0 / (1.0 - t) if shift is None else shift
    _, jvp = jax.linearize(lambda y: field(y, t), x)
    vjp = jax.linear_transpose(jvp, x)

    def matvec(v):
        jv = jvp(v)
        if symmetrize:
            jv = 0.5 * (jv + vjp(v)[0])
        return jv + shift * v

    rho, residual = power_iteration(matvec, x, key, num_iters)
    return rho - shift, residual
