import jax
import jax.numpy as jnp
import numpy as np

from bendflow.curvature import (
    divergence_closed_form,
    jacobian_top_eig,
    mdot_autodiff,
    mdot_closed_form,
)
from bendflow.fields import ExactField
from bendflow.posterior import EmpiricalPosterior
from bendflow.samplers import UniformEuler


def dense_cov(posterior, x, t):
    return jax.vmap(lambda v: posterior.cov_matvec(x, t, v))(jnp.eye(x.shape[0])).T


def test_field_is_posterior_average_of_conditional_fields(posterior, point):
    x, t = point
    w = jnp.exp(posterior.log_weights(x, t))
    expected = w @ ((posterior.data - x) / (1 - t))
    np.testing.assert_allclose(ExactField(posterior)(x, t), expected, rtol=1e-10)


def test_jacobian_of_denoiser(posterior, point):
    x, t = point
    jac = jax.jacfwd(posterior.mean)(x, t)
    np.testing.assert_allclose(jac, t / (1 - t) ** 2 * dense_cov(posterior, x, t), rtol=1e-8, atol=1e-11)


def test_jacobian_of_field(posterior, point):
    x, t = point
    jac = jax.jacfwd(ExactField(posterior))(x, t)
    expected = (t / (1 - t) ** 2 * dense_cov(posterior, x, t) - jnp.eye(x.shape[0])) / (1 - t)
    np.testing.assert_allclose(jac, expected, rtol=1e-8, atol=1e-11)


def test_divergence(posterior, point):
    x, t = point
    div = jnp.trace(jax.jacfwd(ExactField(posterior))(x, t))
    np.testing.assert_allclose(divergence_closed_form(posterior, x, t), div, rtol=1e-8)


def test_jacobian_top_eig(posterior, point):
    x, t = point
    field = ExactField(posterior)
    top = jnp.linalg.eigvalsh(jax.jacfwd(field)(x, t))[-1]
    lam, _ = posterior.lambda_max(x, t, jax.random.key(0), 2000)
    np.testing.assert_allclose(top, (t * lam / (1 - t) ** 2 - 1) / (1 - t), rtol=1e-6, atol=1e-8)
    for symmetrize in (True, False):
        est, residual = jacobian_top_eig(field, x, t, jax.random.key(1), 2000, symmetrize=symmetrize)
        np.testing.assert_allclose(est, top, rtol=1e-6, atol=1e-8)
        assert residual < 1e-5


def test_mdot_closed_form_matches_autodiff(posterior, point):
    x, t = point
    closed = mdot_closed_form(posterior, x, t)
    np.testing.assert_allclose(mdot_autodiff(ExactField(posterior), x, t), closed, rtol=1e-8, atol=1e-11)


# x'' = d/dt u_t(x_t) along the trajectory equals mdot / (1 - t).
def test_acceleration(posterior, point):
    x, t = point
    field = ExactField(posterior)
    _, acc = jax.jvp(field, (x, t), (field(x, t), jnp.ones_like(t)))
    np.testing.assert_allclose(acc, mdot_closed_form(posterior, x, t) / (1 - t), rtol=1e-8, atol=1e-11)


def test_single_image_gives_straight_path():
    data = jax.random.normal(jax.random.key(0), (1, 4))
    post = EmpiricalPosterior(data)
    x, t = jax.random.normal(jax.random.key(1), (4,)), jnp.asarray(0.4)
    np.testing.assert_allclose(mdot_closed_form(post, x, t), 0.0, atol=1e-12)
    np.testing.assert_allclose(UniformEuler(1)(ExactField(post), x).sample, data[0], rtol=1e-12)


# x(t+h) - (x + h u) = h^2 / (2 (1-t)) mdot + O(h^3)
def test_euler_local_error(posterior, point):
    x, t = point
    field = ExactField(posterior)
    leading = mdot_closed_form(posterior, x, t) / (2 * (1 - t))

    def remainder(h):
        # Reference x(t + h) by 2000 RK4 steps.
        def rk4(y, s):
            dt = h / 2000
            k1 = field(y, s)
            k2 = field(y + dt / 2 * k1, s + dt / 2)
            k3 = field(y + dt / 2 * k2, s + dt / 2)
            k4 = field(y + dt * k3, s + dt)
            return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

        exact, _ = jax.lax.scan(rk4, x, t + h * jnp.arange(2000) / 2000)
        return jnp.linalg.norm(exact - (x + h * field(x, t)) - h**2 * leading)

    r1, r2 = remainder(2e-3), remainder(1e-3)
    assert r1 < 0.05 * jnp.linalg.norm((2e-3) ** 2 * leading)
    np.testing.assert_allclose(r1 / r2, 8.0, rtol=0.1)
