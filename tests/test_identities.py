import numpy as np
import pytest
import torch
from conftest import normal, rng
from torch.func import jacfwd, jvp, vmap

from bendflow.curvature import (
    divergence_closed_form,
    jacobian_top_eig,
    mdot_autodiff,
    mdot_closed_form,
)
from bendflow.fields import ExactField
from bendflow.paths import GaussianPath
from bendflow.posterior import EmpiricalPosterior
from bendflow.samplers import UniformEuler


def dense_cov(posterior, x, t):
    return vmap(lambda v: posterior.cov_matvec(x, t, v))(torch.eye(x.shape[0])).T


def test_field_is_posterior_average_of_conditional_fields(posterior, point):
    x, t = point
    w = torch.exp(posterior.log_weights(x, t))
    expected = w @ ((posterior.data - x) / (1 - t))
    np.testing.assert_allclose(ExactField(posterior)(x, t), expected, rtol=1e-10)


def test_jacobian_of_denoiser(posterior, point):
    x, t = point
    jac = jacfwd(posterior.mean)(x, t)
    np.testing.assert_allclose(jac, t / (1 - t) ** 2 * dense_cov(posterior, x, t), rtol=1e-8, atol=1e-11)


def test_jacobian_of_field(posterior, point):
    x, t = point
    jac = jacfwd(ExactField(posterior))(x, t)
    expected = (t / (1 - t) ** 2 * dense_cov(posterior, x, t) - torch.eye(x.shape[0])) / (1 - t)
    np.testing.assert_allclose(jac, expected, rtol=1e-8, atol=1e-11)


def test_divergence(posterior, point):
    x, t = point
    div = torch.trace(jacfwd(ExactField(posterior))(x, t))
    np.testing.assert_allclose(divergence_closed_form(posterior, x, t), div, rtol=1e-8)


def test_jacobian_top_eig(posterior, point):
    x, t = point
    field = ExactField(posterior)
    top = torch.linalg.eigvalsh(jacfwd(field)(x, t))[-1]
    lam, _ = posterior.lambda_max(x, t, rng(0), 2000)
    np.testing.assert_allclose(top, (t * lam / (1 - t) ** 2 - 1) / (1 - t), rtol=1e-6, atol=1e-8)
    for symmetrize in (True, False):
        est, residual = jacobian_top_eig(field, x, t, rng(1), 2000, symmetrize=symmetrize)
        np.testing.assert_allclose(est, top, rtol=1e-6, atol=1e-8)
        assert residual < 1e-5


def test_mdot_closed_form_matches_autodiff(posterior, point):
    x, t = point
    closed = mdot_closed_form(posterior, x, t)
    np.testing.assert_allclose(mdot_autodiff(ExactField(posterior), x, t), closed, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("path", [GaussianPath.COSINE, GaussianPath.VE])
def test_nonlinear_gaussian_identities(posterior, point, path):
    x, t = point
    post = EmpiricalPosterior(posterior.data, path)
    field = ExactField(post)
    alpha, sigma, alpha_dot, sigma_dot = path.coefficients(t)
    noise = (x - alpha * post.data) / sigma
    expected = post.log_weights(x, t).exp() @ (alpha_dot * post.data + sigma_dot * noise)
    torch.testing.assert_close(field(x, t), expected)
    torch.testing.assert_close(mdot_autodiff(field, x, t, path), mdot_closed_form(post, x, t))
    torch.testing.assert_close(divergence_closed_form(post, x, t), jacfwd(field)(x, t).trace())


# x'' = d/dt u_t(x_t) along the trajectory equals mdot / (1 - t).
def test_acceleration(posterior, point):
    x, t = point
    field = ExactField(posterior)
    _, acc = jvp(field, (x, t), (field(x, t), torch.ones_like(t)))
    np.testing.assert_allclose(acc, mdot_closed_form(posterior, x, t) / (1 - t), rtol=1e-8, atol=1e-11)


def test_single_image_gives_straight_path():
    data = normal(0, (1, 4))
    post = EmpiricalPosterior(data)
    x, t = normal(1, (4,)), torch.as_tensor(0.4)
    np.testing.assert_allclose(mdot_closed_form(post, x, t), 0.0, atol=1e-12)
    np.testing.assert_allclose(UniformEuler(1)(ExactField(post), x).sample, data[0], rtol=1e-12)


# x(t+h) - (x + h u) = h^2 / (2 (1-t)) mdot + O(h^3)
def test_euler_local_error(posterior, point):
    _, t = point
    # Probe a noncommitted point so the leading term is above roundoff.
    posterior = EmpiricalPosterior(posterior.data[:2])
    gap = posterior.data[0] - posterior.data[1]
    x = t * posterior.data.mean(dim=0) + 0.3 * (1 - t).square() * gap / (t * gap.square().sum())
    field = ExactField(posterior)
    leading = mdot_closed_form(posterior, x, t) / (2 * (1 - t))

    def remainder(h):
        # 40 RK4 substeps make the reference error negligible at these step sizes.
        def rk4(y, s, dt):
            k1 = field(y, s)
            k2 = field(y + dt / 2 * k1, s + dt / 2)
            k3 = field(y + dt / 2 * k2, s + dt / 2)
            k4 = field(y + dt * k3, s + dt)
            return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        dt = h / 40
        exact = x
        for j in range(40):
            exact = rk4(exact, t + j * dt, dt)
        return torch.linalg.norm(exact - (x + h * field(x, t)) - h**2 * leading)

    h = 2e-4
    r1, r2 = remainder(h), remainder(h / 2)
    assert r1 < 0.05 * torch.linalg.norm(h**2 * leading)
    np.testing.assert_allclose(r1 / r2, 8.0, rtol=0.1)
