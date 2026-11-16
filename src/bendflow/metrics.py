import jax.numpy as jnp
from jax import Array


# Distance between a sample and a reference endpoint started from the same noise.
def endpoint_error(x: Array, x_ref: Array) -> Array:
    return jnp.linalg.norm(x - x_ref)


# Distance to the nearest row of data, and its index.
def nearest_training_distance(x: Array, data: Array) -> tuple[Array, Array]:
    d2 = jnp.sum((data - x) ** 2, axis=1)
    i = jnp.argmin(d2)
    return jnp.sqrt(d2[i]), i
