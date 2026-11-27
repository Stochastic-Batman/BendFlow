from dataclasses import dataclass
from itertools import pairwise
from typing import NamedTuple

import torch
from torch import Tensor

from bendflow.paths import GaussianPath


class Trajectory(NamedTuple):
    # Time is the penultimate dimension of xs and x1_hats, with an optional batch dimension.
    xs: Tensor
    ts: Tensor
    x1_hats: Tensor
    nfe: Tensor

    @property
    def sample(self) -> Tensor:
        return self.xs[..., -1, :]


@dataclass(frozen=True)
class ScheduleEuler:
    """DDIM on a fixed grid, equivalent to Euler on the linear path."""

    ts: Tensor
    path: GaussianPath = GaussianPath.LINEAR

    @torch.no_grad()
    def __call__(self, field, x0: Tensor) -> Trajectory:
        grid = torch.as_tensor(self.ts, dtype=x0.dtype, device=x0.device)
        x, states, predictions = x0, [x0], []
        for t, next_t in pairwise(grid):
            v = field(x, t)
            clean = self.path.clean_prediction(x, v, t)
            x = self.path.step(x, v, clean, t, next_t)
            states.append(x)
            predictions.append(clean)
        nfe = torch.full(x0.shape[:-1], len(predictions), dtype=torch.long, device=x0.device)
        return Trajectory(torch.stack(states, dim=-2), grid.expand(*x0.shape[:-1], -1), torch.stack(predictions, dim=-2), nfe)


@dataclass(frozen=True)
class UniformEuler:
    num_steps: int
    path: GaussianPath = GaussianPath.LINEAR

    def __call__(self, field, x0: Tensor) -> Trajectory:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        grid = torch.linspace(0, 1, self.num_steps + 1, dtype=x0.dtype, device=x0.device)
        return ScheduleEuler(grid, self.path)(field, x0)


@dataclass(frozen=True)
class BendFlowEuler:
    """Curvature-adaptive DDIM, equivalent to adaptive Euler on the linear path."""

    tol: float
    h_min: float
    h_max: float
    h0: float
    max_steps: int
    path: GaussianPath = GaussianPath.LINEAR

    @torch.no_grad()
    def __call__(self, field, x0: Tensor) -> Trajectory:
        unbatched = x0.ndim == 1
        x = x0.unsqueeze(0) if unbatched else x0
        batch, dim = x.shape
        t = x.new_zeros(batch)
        nfe = torch.zeros(batch, dtype=torch.long, device=x.device)
        clean_prev = torch.zeros_like(x)
        h_prev = torch.ones_like(t)
        states = x.new_empty(batch, self.max_steps + 1, dim)
        times = x.new_ones(batch, self.max_steps + 1)
        predictions = x.new_empty(batch, self.max_steps, dim)
        states[:, 0], times[:, 0] = x, t

        for k in range(self.max_steps):
            active = t < 1
            if not active.any():
                states[:, k + 1 :] = x[:, None]
                predictions[:, k:] = clean_prev[:, None]
                break
            # Finished rows are kept in the batch for stable kernel shapes.
            eval_t = torch.where(active, t, 0)
            v = field(x[0], eval_t[0]).unsqueeze(0) if unbatched else field(x, eval_t)
            clean = self.path.clean_prediction(x, v, eval_t)
            if k == 0:
                h = torch.full_like(t, float(self.h0))
            else:
                rate = torch.linalg.vector_norm(clean - clean_prev, dim=-1) / h_prev
                h = (2 * self.tol / (self.path.velocity_scale(eval_t).abs() * rate)).sqrt().clamp(self.h_min, self.h_max)
            remaining = 1 - t
            next_t = torch.where((h >= remaining) | (k == self.max_steps - 1), 1, t + h)
            x = torch.where(active[:, None], self.path.step(x, v, clean, eval_t, next_t), x)
            clean_prev = torch.where(active[:, None], clean, clean_prev)
            h_prev = torch.where(active, next_t - t, h_prev)
            t = next_t
            nfe += active
            states[:, k + 1], times[:, k + 1], predictions[:, k] = x, t, clean_prev

        if unbatched:
            return Trajectory(states[0], times[0], predictions[0], nfe[0])
        return Trajectory(states, times, predictions, nfe)


def run_batches(sampler, field, xs: Tensor, size: int) -> Trajectory:
    runs = [sampler(field, batch) for batch in xs.split(size)]
    return Trajectory(*(torch.cat(parts, dim=0) for parts in zip(*runs)))
