import math

import numpy as np
import torch
from torch import Tensor

from bendflow.paths import GaussianPath


class EmpiricalPosterior:
    """Posterior on the training set, with matrix-free covariance operations."""

    def __init__(self, data: Tensor, path: GaussianPath = GaussianPath.LINEAR):
        self.data = data
        self.path = path
        self.squared_norm = data.square().sum(dim=-1)

    def log_weights(self, x: Tensor, t: Tensor) -> Tensor:
        alpha, sigma, _, _ = self.path.coefficients(torch.as_tensor(t, dtype=x.dtype, device=x.device))
        # Drop the common -||x||² term to avoid materializing (batch, N, d).
        a = alpha[..., None]
        logits = (a * (x @ self.data.T) - 0.5 * a.square() * self.squared_norm) / sigma[..., None].square()
        return torch.log_softmax(logits, dim=-1)

    def weights_and_mean(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        w = self.log_weights(x, t).exp()
        return w, w @ self.data

    def mean(self, x: Tensor, t: Tensor) -> Tensor:
        return self.weights_and_mean(x, t)[1]

    def cov_matvec(self, x: Tensor, t: Tensor, v: Tensor) -> Tensor:
        return self.cov_matvec_from(*self.weights_and_mean(x, t), v)

    def trace_cov(self, x: Tensor, t: Tensor) -> Tensor:
        w, m = self.weights_and_mean(x, t)
        if m.ndim == 1:
            # Avoid subtracting nearly equal norms after posterior commitment.
            return w @ (self.data - m).square().sum(dim=-1)
        return (w * self._sq_dev(m)).sum(dim=-1)

    def kappa(self, x: Tensor, t: Tensor) -> Tensor:
        return self.kappa_from(*self.weights_and_mean(x, t))

    def lambda_max(self, x: Tensor, t: Tensor, rng: torch.Generator, num_iters: int) -> tuple[Tensor, Tensor]:
        w, m = self.weights_and_mean(x, t)
        rho, res = power_iteration(lambda v: self.cov_matvec_from(w, m, v), x, rng, num_iters)
        return rho, res / torch.where(rho > 0, rho, 1.0)

    def active_set(self, x: Tensor, t: Tensor, delta: float) -> Tensor:
        logw = self.log_weights(x, t)
        return logw >= logw.amax(dim=-1, keepdim=True) - math.log(len(self.data) / delta)

    def cov_matvec_from(self, w: Tensor, m: Tensor, v: Tensor) -> Tensor:
        wp = w * (v @ self.data.T - (m * v).sum(dim=-1, keepdim=True))
        return wp @ self.data - wp.sum(dim=-1, keepdim=True) * m

    def kappa_from(self, w: Tensor, m: Tensor) -> Tensor:
        ws = w * self._sq_dev(m)
        return ws @ self.data - ws.sum(dim=-1, keepdim=True) * m

    def _sq_dev(self, m: Tensor) -> Tensor:
        return (self.squared_norm - 2 * (m @ self.data.T) + m.square().sum(dim=-1, keepdim=True)).clamp_min(0)


def power_iteration(matvec, like: Tensor, rng: torch.Generator, num_iters: int) -> tuple[Tensor, Tensor]:
    def normalize(u):
        norm = torch.linalg.vector_norm(u)
        return u / torch.where(norm > 0, norm, 1.0)

    v = normalize(torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=rng))
    for _ in range(num_iters):
        v = normalize(matvec(v))
    mv = matvec(v)
    rho = v @ mv
    return rho, torch.linalg.vector_norm(mv - rho * v)


def diameter(points, chunk_size: int = 1024) -> float:
    """Largest pairwise Euclidean distance, computed in blocks of rows."""
    p = torch.as_tensor(points)
    sq = p.square().sum(dim=1)
    best = 0.0
    for start in range(0, len(p), chunk_size):
        block = p[start : start + chunk_size]
        d2 = sq[start : start + chunk_size, None] + sq[None, :] - 2 * (block @ p.T)
        best = max(best, float(d2.max()))
    return float(np.sqrt(best))
