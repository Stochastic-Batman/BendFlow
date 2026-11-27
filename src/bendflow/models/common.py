import math

import torch
from torch import Tensor, nn
from torch.nn.functional import silu


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = 1000 * t[..., None] * freqs
    return torch.cat([args.cos(), args.sin()], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t: Tensor) -> Tensor:
        return self.lin2(silu(self.lin1(timestep_embedding(t, self.dim))))


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer
