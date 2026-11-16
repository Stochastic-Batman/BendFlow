import equinox as eqx
from jax import Array

from bendflow.posterior import EmpiricalPosterior


# Marginal field of a finite training set, u_t(x) = (m - x) / (1 - t). A field maps one flat sample (x, t) to v.
class ExactField(eqx.Module):
    posterior: EmpiricalPosterior

    def __call__(self, x: Array, t: Array) -> Array:
        return (self.posterior.mean(x, t) - x) / (1.0 - t)


# Wraps an image model (image, t) -> image as a field on flat vectors.
class NetworkField(eqx.Module):
    model: eqx.Module
    image_shape: tuple[int, int, int]  # (channels, height, width)

    def __call__(self, x: Array, t: Array) -> Array:
        return self.model(x.reshape(self.image_shape), t).reshape(x.shape)
