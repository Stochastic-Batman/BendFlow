import numpy as np
import torch
from conftest import normal, rng

from bendflow.curvature import local_bound, mdot_closed_form
from bendflow.posterior import EmpiricalPosterior, diameter


# m, C and kappa written out exactly as in the paper, with C built densely.
def explicit_moments(posterior, x, t):
    data = posterior.data
    g = -torch.sum((x - t * data) ** 2, axis=1) / (2 * (1 - t) ** 2)
    w = torch.exp(g - g.max())
    w = w / w.sum()
    m = w @ data
    dev = data - m
    C = torch.einsum("i,ij,ik->jk", w, dev, dev)
    kappa = torch.einsum("i,i,ij->j", w, torch.sum(dev**2, axis=1), dev)
    return w, m, C, kappa


def test_moments_match_definitions(posterior, point):
    x, t = point
    w, m, C, kappa = explicit_moments(posterior, x, t)
    v = normal(1, x.shape)
    np.testing.assert_allclose(torch.exp(posterior.log_weights(x, t)), w, rtol=1e-10)
    np.testing.assert_allclose(posterior.mean(x, t), m, rtol=1e-10)
    np.testing.assert_allclose(posterior.cov_matvec(x, t, v), C @ v, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(posterior.trace_cov(x, t), torch.trace(C), rtol=1e-10)
    np.testing.assert_allclose(posterior.kappa(x, t), kappa, rtol=1e-9, atol=1e-12)


def test_lambda_max_matches_eigvalsh(posterior, point):
    x, t = point
    C = explicit_moments(posterior, x, t)[2]
    lam, residual = posterior.lambda_max(x, t, rng(2), 2000)
    assert residual < 1e-6
    np.testing.assert_allclose(lam, torch.linalg.eigvalsh(C)[-1], rtol=1e-6)


def test_log_weights_finite_near_t1(posterior):
    x, t = posterior.data[3] + 1e-4, torch.as_tensor(1 - 1e-6)
    logw = posterior.log_weights(x, t)
    assert torch.all(torch.isfinite(logw[3:4])) and not torch.any(torch.isnan(logw))
    np.testing.assert_allclose(posterior.mean(x, t), posterior.data[3], rtol=1e-10)


def test_tail_weight_at_most_delta(posterior, point):
    x, t = point
    for delta in (0.3, 0.05, 1e-3):
        mask = posterior.active_set(x, t, delta)
        w = torch.exp(posterior.log_weights(x, t))
        assert mask[torch.argmax(w)]
        assert torch.sum(torch.where(mask, 0.0, w)) <= delta


def test_active_set_geometric_form(posterior, point):
    x, t = point
    delta, n = 0.05, posterior.data.shape[0]
    d2 = torch.sum((x / t - posterior.data) ** 2, axis=1)
    expected = d2 <= d2.min() + 2 * ((1 - t) / t) ** 2 * np.log(n / delta)
    assert torch.equal(posterior.active_set(x, t, delta), expected)


def test_diameter(posterior):
    data = np.asarray(posterior.data)
    brute = max(np.linalg.norm(a - b) for a in data for b in data)
    np.testing.assert_allclose(diameter(data, chunk_size=7), brute, rtol=1e-10)


def test_local_bounds_hold(posterior, point):
    x, t = point
    _w, _m, C, kappa = explicit_moments(posterior, x, t)
    d_all = diameter(posterior.data)
    for delta in (0.3, 0.05, 1e-3):
        b = local_bound(posterior, x, t, delta, d_all)
        assert b.d_active <= d_all and 1 <= b.n_active <= posterior.data.shape[0]
        assert torch.linalg.eigvalsh(C)[-1] <= b.lam
        assert torch.linalg.norm(kappa) <= b.kappa
        assert torch.linalg.norm(mdot_closed_form(posterior, x, t)) <= b.mdot


# Tightness paragraph: at x = t (x1 + x2)/2, lambda_max(C) = D_A^2 / 4 and kappa = 0.
def test_two_point_tightness():
    data = normal(4, (2, 6))
    post = EmpiricalPosterior(data)
    t = torch.as_tensor(0.6)
    x = t * data.mean(axis=0)
    gap = torch.linalg.norm(data[0] - data[1])
    np.testing.assert_allclose(torch.exp(post.log_weights(x, t)), [0.5, 0.5], rtol=1e-12)
    lam, _ = post.lambda_max(x, t, rng(5), 50)
    np.testing.assert_allclose(lam, gap**2 / 4, rtol=1e-10)
    np.testing.assert_allclose(post.kappa(x, t), 0.0, atol=1e-12)
    np.testing.assert_allclose(local_bound(post, x, t, 0.01, float(gap)).d_active, float(gap), rtol=1e-12)
    # Interpretation paragraph: C eps_hat = 0 at the symmetric point, so mdot = 0.
    np.testing.assert_allclose(mdot_closed_form(post, x, t), 0.0, atol=1e-10)


def test_lambda_max_residual_reports_convergence(posterior, point):
    x, t = point
    C = explicit_moments(posterior, x, t)[2]
    rho, residual = posterior.lambda_max(x, t, rng(2), 3)
    # The residual is ||C v - rho v|| / rho for the vector the iteration ended on,
    # and some eigenvalue of C lies within rho * residual of rho.
    eigs = torch.linalg.eigvalsh(C)
    assert torch.min(torch.abs(eigs - rho)) <= rho * residual * (1 + 1e-9)
    assert rho <= eigs[-1] * (1 + 1e-12)
    assert posterior.lambda_max(x, t, rng(2), 2000)[1] < residual


def test_lambda_max_of_zero_covariance():
    post = EmpiricalPosterior(torch.ones((1, 4)))
    rho, residual = post.lambda_max(torch.zeros(4), torch.as_tensor(0.5), rng(0), 5)
    assert rho == 0.0 and residual == 0.0
