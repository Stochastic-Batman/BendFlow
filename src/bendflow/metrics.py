import torch
from torch import Tensor


def endpoint_error(x: Tensor, x_ref: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x - x_ref)


def nearest_training_distance(x: Tensor, data: Tensor) -> tuple[Tensor, Tensor]:
    d2 = (data - x).square().sum(dim=1)
    i = d2.argmin()
    return d2[i].sqrt(), i
