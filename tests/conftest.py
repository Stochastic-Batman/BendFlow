import jax
import pytest

# The identities are checked against autodiff at tight tolerances.
jax.config.update("jax_enable_x64", True)

from bendflow.posterior import EmpiricalPosterior

N, D = 20, 5


@pytest.fixture
def posterior():
    return EmpiricalPosterior(jax.random.normal(jax.random.key(0), (N, D)))


@pytest.fixture(params=[0.15, 0.5, 0.8])
# A point (x, t) on the path of a training image.
def point(request, posterior):
    t = request.param
    k_i, k_eps = jax.random.split(jax.random.key(int(100 * t)))
    x1 = posterior.data[jax.random.randint(k_i, (), 0, N)]
    return t * x1 + (1 - t) * jax.random.normal(k_eps, (D,)), jax.numpy.asarray(t)
