"""Gaussian membership bank: lift CS-z features into Low/Mid/High in [0, 1].

This is the real-valued front-end of the notebook CFS layer
(amplitude only; the complex phase is an optional later ablation).
"""

from __future__ import annotations

import numpy as np

from .constants import MF_INIT_CENTERS, MF_INIT_SIGMA, N_FEATURES, N_MFS


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = x / max(float(temperature), 1e-6)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def gaussian_mf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Elementwise Gaussian membership. x, mu, sigma broadcastable."""
    s = np.clip(sigma, 1e-4, None)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


class GaussianBank:
    """Per-feature K Gaussians. Parameters are plain arrays (Adam/DE later)."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_mfs: int = N_MFS,
        centers: tuple[float, ...] = MF_INIT_CENTERS,
        sigma: float = MF_INIT_SIGMA,
    ) -> None:
        if n_mfs != len(centers):
            raise ValueError("n_mfs must match centers")
        self.n_features = int(n_features)
        self.n_mfs = int(n_mfs)
        self.mu = np.tile(np.asarray(centers, dtype=np.float64), (n_features, 1))
        self.log_sigma = np.full((n_features, n_mfs), np.log(sigma), dtype=np.float64)

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.log_sigma)

    def lift(self, x: np.ndarray) -> np.ndarray:
        """x: (..., F) → memberships (..., F, K)."""
        x = np.asarray(x, dtype=np.float64)
        mu = self.mu.reshape((1,) * (x.ndim - 1) + self.mu.shape)
        sig = self.sigma.reshape((1,) * (x.ndim - 1) + self.sigma.shape)
        return gaussian_mf(x[..., :, None], mu, sig)

    def n_params(self) -> int:
        return int(self.mu.size + self.log_sigma.size)
