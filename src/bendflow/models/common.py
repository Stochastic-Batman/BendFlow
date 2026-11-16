import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


# Sinusoidal embedding of a scalar t in [0, 1], scaled to [0, 1000].
def timestep_embedding(t: Array, dim: int, max_period: float = 10000.0) -> Array:
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = 1000.0 * t * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)])


# Scalar time -> conditioning vector of width dim.
class TimeMLP(eqx.Module):
    dim: int
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __init__(self, dim: int, *, key: Array):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.lin1 = eqx.nn.Linear(dim, dim, key=k1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=k2)

    def __call__(self, t: Array) -> Array:
        return self.lin2(jax.nn.silu(self.lin1(timestep_embedding(t, self.dim))))


# Copy of a linear or convolutional layer with weight and bias set to zero.
def zero_init(layer):
    return eqx.tree_at(lambda m: (m.weight, m.bias), layer, replace_fn=jnp.zeros_like)
