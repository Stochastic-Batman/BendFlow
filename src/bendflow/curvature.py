from typing import NamedTuple

import torch
from torch import Tensor
from torch.func import jvp, vjp
from torch.nn.attention import SDPBackend, sdpa_kernel

from bendflow.paths import GaussianPath
from bendflow.posterior import EmpiricalPosterior, diameter, power_iteration


def mdot_closed_form(posterior: EmpiricalPosterior, x: Tensor, t: Tensor, moments=None) -> Tensor:
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    alpha, sigma, _, _ = posterior.path.coefficients(t)
    gamma = alpha / sigma
    gamma_dot = posterior.path.velocity_scale(t) / sigma
    w, m = posterior.weights_and_mean(x, t) if moments is None else moments
    eps_hat = (x - alpha[..., None] * m) / sigma[..., None]
    return gamma_dot[..., None] * (posterior.cov_matvec_from(w, m, eps_hat) - gamma[..., None] * posterior.kappa_from(w, m))


def velocity_variance(posterior: EmpiricalPosterior, x: Tensor, t: Tensor) -> Tensor:
    return posterior.trace_cov(x, t) * posterior.path.velocity_scale(t).square()


def divergence_closed_form(posterior: EmpiricalPosterior, x: Tensor, t: Tensor) -> Tensor:
    alpha, sigma, _, sigma_dot = posterior.path.coefficients(t)
    return (sigma_dot / sigma) * x.shape[-1] + (alpha / sigma.square()) * posterior.path.velocity_scale(t) * posterior.trace_cov(x, t)


class LocalBound(NamedTuple):
    lam: float
    kappa: float
    mdot: float
    d_active: float
    n_active: int


def local_bound(posterior: EmpiricalPosterior, x: Tensor, t: Tensor, delta: float, d_all: float, chunk_size: int = 1024) -> LocalBound:
    mask = posterior.active_set(x, t, delta)
    n_active = int(mask.sum())
    d_active = d_all if n_active == len(mask) else diameter(posterior.data[mask], chunk_size)
    alpha, sigma, _, _ = posterior.path.coefficients(t)
    eps_norm = float(torch.linalg.vector_norm(x - alpha * posterior.mean(x, t)) / sigma)
    lam = d_active**2 / 4 + delta * d_all**2
    kap = (d_active + delta * d_all) ** 3 + delta * d_all**3
    mdot = float((posterior.path.velocity_scale(t) / sigma).abs() * (lam * eps_norm + (alpha / sigma).abs() * kap))
    return LocalBound(lam, kap, mdot, d_active, n_active)


@torch.no_grad()
def mdot_autodiff(field, x: Tensor, t: Tensor, path: GaussianPath = GaussianPath.LINEAR) -> Tensor:
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    # Fused CUDA attention kernels do not implement forward-mode autodiff.
    with sdpa_kernel(SDPBackend.MATH):
        _, out = jvp(lambda y, s: path.clean_prediction(y, field(y, s), s), (x, t), (field(x, t), torch.ones_like(t)))
    return out


@torch.no_grad()
def jacobian_top_eig(field, x: Tensor, t: Tensor, rng: torch.Generator, num_iters: int, shift: Tensor | None = None, symmetrize: bool = True):
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if shift is None:
        shift = 1 / (1 - t)

    def velocity(y):
        return field(y, t)

    with sdpa_kernel(SDPBackend.MATH):
        if symmetrize:
            _, transpose = vjp(velocity, x)

        def matvec(v):
            tangent = jvp(velocity, (x,), (v,))[1]
            if symmetrize:
                tangent = 0.5 * (tangent + transpose(v)[0])
            return tangent + shift * v

        rho, residual = power_iteration(matvec, x, rng, num_iters)
    return rho - shift, residual
