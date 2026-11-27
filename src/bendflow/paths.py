"""Noise-to-data Gaussian paths used for training and sampling."""

from enum import StrEnum
from math import expm1, log, pi

import torch
from torch import Tensor


class GaussianPath(StrEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    VE = "ve"

    @property
    def solver(self) -> str:
        return "Euler" if self is GaussianPath.LINEAR else "DDIM"

    def coefficients(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return alpha, sigma, and their time derivatives."""
        if self is GaussianPath.LINEAR:
            unit = torch.ones_like(t)
            return t, 1 - t, unit, -unit
        if self is GaussianPath.COSINE:
            omega = pi / 2
            alpha = torch.sin(omega * t)
            sigma = torch.sin(omega * (1 - t))
            return alpha, sigma, omega * sigma, -omega * alpha
        # Exponential noise, shifted to hit zero at the clean endpoint.
        rate = log(10.0)
        sigma = 10 * torch.expm1(rate * (1 - t)) / expm1(rate)
        return torch.ones_like(t), sigma, torch.zeros_like(t), -rate * (sigma + 10 / expm1(rate))

    def interpolant(self, clean: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        alpha, sigma, alpha_dot, sigma_dot = self.coefficients(t)
        shape = t.shape + (1,) * (clean.ndim - t.ndim)
        return alpha.reshape(shape) * clean + sigma.reshape(shape) * noise, alpha_dot.reshape(shape) * clean + sigma_dot.reshape(shape) * noise

    def velocity_scale(self, t: Tensor) -> Tensor:
        """Return sigma times the derivative of alpha / sigma."""
        if self is GaussianPath.LINEAR:
            return 1 / (1 - t)
        alpha, sigma, alpha_dot, sigma_dot = self.coefficients(t)
        return alpha_dot - alpha * sigma_dot / sigma

    def clean_prediction(self, x: Tensor, v: Tensor, t: Tensor) -> Tensor:
        if self is GaussianPath.LINEAR:
            return x + (1 - t)[..., None] * v
        _, sigma, _, sigma_dot = self.coefficients(t)
        return (v - (sigma_dot / sigma)[..., None] * x) / self.velocity_scale(t)[..., None]

    def step(self, x: Tensor, v: Tensor, clean: Tensor, t: Tensor, next_t: Tensor) -> Tensor:
        if self is GaussianPath.LINEAR:
            return x + (next_t - t)[..., None] * v
        alpha, sigma, _, _ = self.coefficients(t)
        next_alpha, next_sigma, _, _ = self.coefficients(next_t)
        ratio = next_sigma / sigma
        return ratio[..., None] * x + (next_alpha - ratio * alpha)[..., None] * clean
