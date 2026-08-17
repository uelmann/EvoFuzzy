"""FuzzyX: membership → market gate → AND/OR rules → cross-section head.

Forward is date-batched: all N assets of a day are scored together.
Train uses soft positions; eval uses argmax → {+1, 0, −1}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    D_MODEL,
    ENCODER,
    FEATURE_COLS,
    N_FEATURES,
    N_HEADS,
    N_MFS,
    N_RULES,
    POS_TEMPERATURE,
    SEED,
)
from .encoder import CrossSectionEncoder, DeepSetsEncoder
from .gate import MarketGate
from .membership import GaussianBank, softmax
from .rules import RuleBank


def market_token(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Cross-section aggregates used as the MASTER market vector.

    x: (..., N, F). Returns (..., F*2) = [masked mean, masked std].
    """
    x = np.asarray(x, dtype=np.float64)
    if mask is None:
        mu = x.mean(axis=-2)
        sd = x.std(axis=-2)
    else:
        m = mask.astype(np.float64)[..., None]
        denom = np.clip(m.sum(axis=-2), 1.0, None)
        mu = (x * m).sum(axis=-2) / denom
        var = ((x - mu[..., None, :]) ** 2 * m).sum(axis=-2) / denom
        sd = np.sqrt(np.clip(var, 0.0, None))
    return np.concatenate([mu, sd], axis=-1)


def soft_positions(logits: np.ndarray, temperature: float = POS_TEMPERATURE) -> np.ndarray:
    """P(long) − P(short). v1c: notebook-style signed score."""
    p = softmax(logits, axis=-1, temperature=temperature)
    return p[..., 0] - p[..., 1]


def hard_positions(logits: np.ndarray) -> np.ndarray:
    """argmax over {LONG, SHORT, FLAT} → {+1, −1, 0}."""
    idx = np.argmax(logits, axis=-1)
    out = np.zeros(idx.shape, dtype=np.float64)
    out[idx == 0] = 1.0
    out[idx == 1] = -1.0
    return out


@dataclass
class FuzzyXForward:
    memberships: np.ndarray
    gated: np.ndarray
    firings: np.ndarray
    rule_scores: np.ndarray
    logits: np.ndarray
    soft_pos: np.ndarray
    hard_pos: np.ndarray
    alpha: np.ndarray


class FuzzyX:
    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_mfs: int = N_MFS,
        n_rules: int = N_RULES,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        encoder: str = ENCODER,
        seed: int = SEED,
    ) -> None:
        self.n_features = int(n_features)
        self.encoder_kind = str(encoder)
        self.mfs = GaussianBank(n_features=n_features, n_mfs=n_mfs)
        self.rules = RuleBank(n_features=n_features, n_mfs=n_mfs, n_rules=n_rules, seed=seed)
        n_market = n_features * 2
        self.gate = MarketGate(n_market=n_market, n_features=n_features, seed=seed + 1)
        # token = raw A0 features + flattened gated MFs + rule firings + rule scores
        d_in = n_features + n_features * n_mfs + n_rules + 3
        if self.encoder_kind == "deepsets":
            self.enc = DeepSetsEncoder(d_in=d_in, d_market=n_market, d_model=d_model, seed=seed + 2)
            self.W_mkt = None
        elif self.encoder_kind == "xsec":
            self.enc = CrossSectionEncoder(d_in=d_in, d_model=d_model, n_heads=n_heads, seed=seed + 2)
            self.W_mkt = np.random.default_rng(seed + 3).normal(0.0, d_in**-0.5, size=(n_market, d_in))
        else:
            raise ValueError(f"encoder must be 'deepsets' or 'xsec', got {encoder!r}")

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> FuzzyXForward:
        """x: (..., N, F) CS-z features. mask: (..., N) True = investable."""
        x = np.asarray(x, dtype=np.float64)
        mkt = market_token(x, mask)
        memb = self.mfs.lift(x)
        gated = self.gate.apply(memb, mkt)
        firings, rule_scores = self.rules.fire(gated)
        tokens = np.concatenate(
            [x, gated.reshape(*x.shape[:-1], -1), firings, rule_scores],
            axis=-1,
        )
        mkt_tok = mkt if self.W_mkt is None else mkt @ self.W_mkt
        logits = self.enc.encode(tokens, mkt_tok, mask=mask)
        if mask is not None:
            logits = np.where(mask[..., None], logits, np.array([0.0, 0.0, 8.0]))
        return FuzzyXForward(
            memberships=memb,
            gated=gated,
            firings=firings,
            rule_scores=rule_scores,
            logits=logits,
            soft_pos=soft_positions(logits),
            hard_pos=hard_positions(logits),
            alpha=self.gate.alpha(mkt),
        )

    def n_params(self) -> int:
        return (
            self.mfs.n_params()
            + self.rules.n_params()
            + self.gate.n_params()
            + self.enc.n_params()
            + (0 if self.W_mkt is None else int(self.W_mkt.size))
        )

    def rule_sheet(self) -> list[str]:
        return self.rules.describe(FEATURE_COLS)
