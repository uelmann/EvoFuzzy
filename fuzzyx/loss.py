"""v1e path loss: −corr(wealth, t) · (1 + cumRet[-1]).

wealth = cumprod(1+st_r). Occupancy nuke off. DD terms are diagnostics.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    GROSS_LIMIT,
    LEVER_UP,
    MAXDD_CAP,
    OCC_LONG_MIN,
    OCC_NUKE,
    OCC_PENALTY,
    OCC_SHORT_MIN,
    OCC_TRADED_MIN,
    SLIPPAGE_BPS,
    TAKER_FEE_BPS,
    TURN_LAMBDA,
    BIAS_LAMBDA,
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


def trend_corr(series: np.ndarray) -> float:
    """np.corrcoef(series, np.arange(len(series)))[1, 0]. Constant → 0."""
    eq = np.asarray(series, dtype=np.float64).reshape(-1)
    if eq.size < 3 or np.std(eq) < 1e-12:
        return 0.0
    t = np.arange(eq.size, dtype=np.float64)
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
    # v1c: unit-gross if LEVER_UP else no dust lever-up.
    gross = np.sum(np.abs(p), axis=-1, keepdims=True)
    if LEVER_UP:
        w = np.zeros_like(p)
        np.divide(p, gross, out=w, where=gross > 1e-8)
        w *= gross_limit
    else:
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
    turn_lambda: float = TURN_LAMBDA,
    bias_lambda: float = BIAS_LAMBDA,
) -> dict[str, float]:
    """v1e: loss = −corr(wealth, t) · (1 + last cumret). Lower is better."""
    port, w = portfolio_returns(positions, asset_ret, mask=mask)
    equity = np.cumprod(1.0 + port)
    maxdd, ddur, _ = max_dd_path(port)
    corr_w = trend_corr(equity)
    equity_end = float(equity[-1]) if equity.size else 1.0
    core = corr_w * equity_end
    trend_r = trend_corr(port)
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
        "trend": float(corr_w),
        "trend_equity": float(corr_w),
        "trend_returns": float(trend_r),
        "equity_end": float(equity_end),
        "cumret_last": float(equity_end - 1.0),
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
