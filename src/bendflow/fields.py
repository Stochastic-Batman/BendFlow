import torch
from torch import Tensor, nn

from bendflow.paths import GaussianPath
from bendflow.posterior import EmpiricalPosterior


class ExactField:
    def __init__(self, posterior: EmpiricalPosterior):
        self.posterior = posterior
        self.path = posterior.path

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        m = self.posterior.mean(x, t)
        if self.path is GaussianPath.LINEAR:
            return (m - x) / (1 - t)[..., None]
        _, sigma, _, sigma_dot = self.path.coefficients(t)
        return self.path.velocity_scale(t)[..., None] * m + (sigma_dot / sigma)[..., None] * x


class NetworkField:
    def __init__(self, model: nn.Module, image_shape: tuple[int, int, int]):
        self.model = model
        self.image_shape = image_shape

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        dtype = next(self.model.parameters()).dtype
        images = x.reshape(*x.shape[:-1], *self.image_shape).to(dtype)
        t = torch.as_tensor(t, dtype=dtype, device=x.device)
        return self.model(images, t).reshape(x.shape).to(x.dtype)
