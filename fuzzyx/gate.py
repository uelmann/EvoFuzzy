"""Market-guided feature gate (MASTER / TFT variable-selection style).

A market token (BTC + cross-section aggregates) produces a softmax over
features. Softmax temperature < 1 sharpens toward on/off, matching the
notebook's x_null mask without a hard binary search.
"""

from __future__ import annotations

import numpy as np

from .constants import GATE_TEMPERATURE, N_FEATURES
from .membership import softmax


class MarketGate:
    def __init__(
        self,
        n_market: int,
        n_features: int = N_FEATURES,
        temperature: float = GATE_TEMPERATURE,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = int(n_features)
        self.temperature = float(temperature)
        scale = 1.0 / max(n_market, 1) ** 0.5
        self.W = rng.normal(0.0, scale, size=(n_features, n_market))
        self.b = np.zeros(n_features, dtype=np.float64)

    def alpha(self, market: np.ndarray) -> np.ndarray:
        """market: (..., M) → α (..., F) summing to 1, then scaled by F."""
        m = np.asarray(market, dtype=np.float64)
        logits = m @ self.W.T + self.b
        a = softmax(logits, axis=-1, temperature=self.temperature)
        return a * self.n_features

    def apply(self, memberships: np.ndarray, market: np.ndarray) -> np.ndarray:
        """memberships (..., N, F, K), market (..., M) → gated memberships."""
        a = self.alpha(market)
        return memberships * a[..., None, :, None]

    def n_params(self) -> int:
        return int(self.W.size + self.b.size)
