"""Notebook-style path loss: trend × (1 − maxDD) × (1 − DD duration).

v1b: occupancy floors are diagnostics only. The core/1e5 nuke is off
(OCC_NUKE=False). Training pay-to-play lives in torch_loss, not here.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    GROSS_LIMIT,
    MAXDD_CAP,
    OCC_LONG_MIN,
    OCC_NUKE,
    OCC_PENALTY,
    OCC_SHORT_MIN,
    OCC_TRADED_MIN,
    SLIPPAGE_BPS,
    TAKER_FEE_BPS,
)


def max_dd_path(returns: np.ndarray) -> tuple[float, float, float]:
    """returns: (T,) net portfolio returns. Returns (maxdd, ddur, equity_end)."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    equity = np.cumprod(1.0 + r)
    if equity.size == 0:
        return 0.0, 0.0, 1.0
    peak = np.maximum.accumulate(equity)
    dd = equity / np.clip(peak, 1e-12, None) - 1.0
    maxdd = float(min(-dd.min() if dd.size else 0.0, MAXDD_CAP))
    underwater = dd < -1e-12
    if underwater.any():
        # longest consecutive underwater run / T
        padded = np.concatenate([[False], underwater, [False]])
        diffs = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)
        longest = int((ends - starts).max()) if starts.size else 0
        ddur = longest / max(len(r), 1)
    else:
        ddur = 0.0
    return maxdd, float(ddur), float(equity[-1])


def trend_corr(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    if eq.size < 3 or np.std(eq) < 1e-12:
        return 0.0
    t = np.linspace(0.0, 1.0, eq.size)
    c = np.corrcoef(eq, t)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def occupancy(positions: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """positions (T, N) in {−1,0,+1} or soft. Returns (long_frac, short_frac, traded_frac)."""
    p = np.asarray(positions, dtype=np.float64)
    if mask is not None:
        m = mask.astype(bool)
        p = np.where(m, p, 0.0)
        denom = max(float(m.sum()), 1.0)
    else:
        denom = max(float(p.size), 1.0)
    long_frac = float(np.sum(p > 0.05) / denom)
    short_frac = float(np.sum(p < -0.05) / denom)
    traded_frac = float(np.sum(np.abs(p) > 0.05) / denom)
    return long_frac, short_frac, traded_frac


def portfolio_returns(
    positions: np.ndarray,
    asset_ret: np.ndarray,
    mask: np.ndarray | None = None,
    prev_positions: np.ndarray | None = None,
    fee_bps: float = TAKER_FEE_BPS,
    slip_bps: float = SLIPPAGE_BPS,
    gross_limit: float = GROSS_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-gross among active names. positions/asset_ret: (T, N).

    Returns (net_port_ret (T,), weights (T, N)).
    """
    p = np.asarray(positions, dtype=np.float64)
    r = np.asarray(asset_ret, dtype=np.float64)
    if mask is not None:
        p = np.where(mask.astype(bool), p, 0.0)
        r = np.where(mask.astype(bool), r, 0.0)
    # v1b: never lever dust up to unit gross. w = p / max(Σ|p|, 1).
    gross = np.sum(np.abs(p), axis=-1, keepdims=True)
    w = p / np.maximum(gross, 1.0) * gross_limit
    gross_pnl = np.sum(w * r, axis=-1)
    if prev_positions is None:
        prev = np.zeros_like(w)
        prev[1:] = w[:-1]
    else:
        prev = np.asarray(prev_positions, dtype=np.float64)
    dpos = w - prev
    cost = np.sum(np.abs(dpos), axis=-1) * (fee_bps + slip_bps) / 1e4
    return gross_pnl - cost, w


def path_loss(
    positions: np.ndarray,
    asset_ret: np.ndarray,
    mask: np.ndarray | None = None,
    turn_lambda: float = 0.05,
    bias_lambda: float = 0.05,
) -> dict[str, float]:
    """Scalar loss plus diagnostics. Lower is better (we return −core)."""
    port, w = portfolio_returns(positions, asset_ret, mask=mask)
    equity = np.cumprod(1.0 + port)
    maxdd, ddur, _ = max_dd_path(port)
    trend = trend_corr(equity)
    core = trend * (1.0 - maxdd) * (1.0 - ddur)
    long_f, short_f, traded_f = occupancy(positions, mask)
    if OCC_NUKE and (
        traded_f < OCC_TRADED_MIN or short_f < OCC_SHORT_MIN or long_f < OCC_LONG_MIN
    ):
        core = core / OCC_PENALTY
    turn = float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=-1))) if w.shape[0] > 1 else 0.0
    bias = float(np.abs(np.mean(w)))
    loss = -core + turn_lambda * turn + bias_lambda * bias
    return {
        "loss": float(loss),
        "core": float(core),
        "trend": float(trend),
        "maxdd": float(maxdd),
        "ddur": float(ddur),
        "long_frac": long_f,
        "short_frac": short_f,
        "traded_frac": traded_f,
        "turnover": turn,
        "bias": bias,
        "ann_mean": float(np.mean(port) * 365.0),
        "mean_pnl": float(np.mean(port)),
    }
