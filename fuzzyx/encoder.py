"""Cross-section heads: DeepSets (default) and optional 1-layer attention.

Assets are a SET, not a sequence. DeepSets + CS residual is the
sample-efficient default (~250 weekly dates). The transformer is the
MASTER-style ablation: same API, no positional encoding.
"""

from __future__ import annotations

import numpy as np

from .constants import D_MODEL, FLAT_INIT_BIAS, N_HEADS
from .membership import softmax


def _split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    *lead, n, d = x.shape
    dh = d // n_heads
    return x.reshape(*lead, n, n_heads, dh).transpose(*range(len(lead)), -2, -3, -1)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    # (..., H, N, Dh) → (..., N, H*Dh)
    *lead, h, n, dh = x.shape
    return x.transpose(*range(len(lead)), -2, -3, -1).reshape(*lead, n, h * dh)


def _masked_mean(h: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return h.mean(axis=-2)
    m = mask.astype(np.float64)[..., None]
    return (h * m).sum(axis=-2) / np.clip(m.sum(axis=-2), 1.0, None)


class DeepSetsEncoder:
    """Permutation-equivariant CS residual: score_i = MLP([h_i, c, h_i−c, m]).

    c is the masked mean over names. This is the GKX-style / DeepSets
    inductive bias: each name is scored relative to today's cross-section.
    """

    def __init__(self, d_in: int, d_market: int, d_model: int = D_MODEL, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.d_model = int(d_model)
        s = 1.0 / max(d_in, 1) ** 0.5
        self.W_in = rng.normal(0.0, s, size=(d_in, d_model))
        self.b_in = np.zeros(d_model)
        d_z = 3 * d_model + int(d_market)
        s2 = 1.0 / max(d_z, 1) ** 0.5
        self.W_ff = rng.normal(0.0, s2, size=(d_z, d_model))
        self.b_ff = np.zeros(d_model)
        self.W_out = rng.normal(0.0, d_model**-0.5, size=(d_model, 3))
        self.b_out = np.zeros(3)
        self.b_out[2] = float(FLAT_INIT_BIAS)

    def encode(self, tokens: np.ndarray, market: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        h = np.tanh(np.asarray(tokens, dtype=np.float64) @ self.W_in + self.b_in)
        c = _masked_mean(h, mask)
        m = np.asarray(market, dtype=np.float64)
        ones = np.ones(h.shape[:-1] + (1,), dtype=np.float64)
        z = np.concatenate([h, c[..., None, :] * ones, h - c[..., None, :], m[..., None, :] * ones], axis=-1)
        hid = np.tanh(z @ self.W_ff + self.b_ff)
        return hid @ self.W_out + self.b_out

    def n_params(self) -> int:
        return int(
            self.W_in.size
            + self.b_in.size
            + self.W_ff.size
            + self.b_ff.size
            + self.W_out.size
            + self.b_out.size
        )


class CrossSectionEncoder:
    def __init__(
        self,
        d_in: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        seed: int = 0,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        rng = np.random.default_rng(seed)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        s = 1.0 / max(d_in, 1) ** 0.5
        self.W_in = rng.normal(0.0, s, size=(d_in, d_model))
        self.b_in = np.zeros(d_model)
        s2 = 1.0 / d_model**0.5
        self.W_q = rng.normal(0.0, s2, size=(d_model, d_model))
        self.W_k = rng.normal(0.0, s2, size=(d_model, d_model))
        self.W_v = rng.normal(0.0, s2, size=(d_model, d_model))
        self.W_o = rng.normal(0.0, s2, size=(d_model, d_model))
        self.W_ff1 = rng.normal(0.0, s2, size=(d_model, 4 * d_model))
        self.b_ff1 = np.zeros(4 * d_model)
        self.W_ff2 = rng.normal(0.0, (4 * d_model) ** -0.5, size=(4 * d_model, d_model))
        self.b_ff2 = np.zeros(d_model)
        self.W_out = rng.normal(0.0, s2, size=(d_model, 3))
        self.b_out = np.zeros(3)
        self.b_out[2] = float(FLAT_INIT_BIAS)

    def _attn(self, h: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        q = _split_heads(h @ self.W_q, self.n_heads)
        k = _split_heads(h @ self.W_k, self.n_heads)
        v = _split_heads(h @ self.W_v, self.n_heads)
        dh = self.d_model // self.n_heads
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / dh**0.5
        if mask is not None:
            # mask: (..., N) True = keep. prepend True for market token.
            keep = np.concatenate([np.ones(mask.shape[:-1] + (1,), dtype=bool), mask.astype(bool)], axis=-1)
            scores = np.where(keep[..., None, None, :], scores, -1e9)
        w = softmax(scores, axis=-1)
        ctx = _merge_heads(np.matmul(w, v))
        return ctx @ self.W_o

    def encode(self, tokens: np.ndarray, market: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """tokens (..., N, D_in), market (..., M_in) with M_in == D_in after projection prep.

        Caller already concatenates market features into the same D_in as tokens
        via a market embedding of shape (..., D_in).
        """
        m = np.asarray(market, dtype=np.float64)[..., None, :]
        x = np.concatenate([m, np.asarray(tokens, dtype=np.float64)], axis=-2)
        h = x @ self.W_in + self.b_in
        h = h + self._attn(h, mask)
        ff = np.tanh(h @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        h = h + ff
        asset = h[..., 1:, :]
        return asset @ self.W_out + self.b_out

    def n_params(self) -> int:
        return int(
            self.W_in.size
            + self.b_in.size
            + self.W_q.size
            + self.W_k.size
            + self.W_v.size
            + self.W_o.size
            + self.W_ff1.size
            + self.b_ff1.size
            + self.W_ff2.size
            + self.b_ff2.size
            + self.W_out.size
            + self.b_out.size
        )
