"""Build meta-model features from Kronos outputs + market context."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    # Kronos generative features
    "kronos_p_up",
    "kronos_mean_r",
    "kronos_abs_mean_r",
    "kronos_edge",
    "kronos_disagree",  # |p_up - 0.5| * |mean_r|  (confidence x magnitude)
    # Market context
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_10",
    "vol_20",
    "vol_ratio",  # vol_10 / vol_20
    "mom_20",
    "mom_60",
    "sma_dist_20",
    "sma_dist_60",
    "drawdown_60",
    "range_10",
    "rsi_14",
    "trend_slope_20",
    "ret_skew_20",
]

# Optional column filled when supervised head is available.
OPTIONAL_FEATURE_COLS = ["sup_p_up"]


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = float(gain.iloc[-1] / (float(loss.iloc[-1]) + 1e-12))
    return 100.0 - (100.0 / (1.0 + rs))


def _market_row(ohlcv: pd.DataFrame, asof_ts: pd.Timestamp) -> dict[str, float]:
    """Market features using only bars at/before asof (no leakage)."""
    ts = pd.Timestamp(asof_ts)
    if ts.tzinfo is None and ohlcv["timestamps"].dt.tz is not None:
        ts = ts.tz_localize(ohlcv["timestamps"].dt.tz)
    hist = ohlcv[ohlcv["timestamps"] <= ts].copy()
    if len(hist) < 60:
        raise ValueError(f"Need >=60 bars of history before {asof_ts}, got {len(hist)}")
    close = hist["close"].astype(float)
    high = hist["high"].astype(float)
    low = hist["low"].astype(float)
    c = float(close.iloc[-1])
    rets = close.pct_change()
    vol10 = float(rets.iloc[-10:].std())
    vol20 = float(rets.iloc[-20:].std())
    sma20 = float(close.iloc[-20:].mean())
    sma60 = float(close.iloc[-60:].mean())
    roll_max = float(close.rolling(60).max().iloc[-1])
    # simple OLS slope on last 20 closes, scaled
    y = close.iloc[-20:].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0] / (c + 1e-12))
    return {
        "ret_1": float(c / close.iloc[-2] - 1.0),
        "ret_5": float(c / close.iloc[-6] - 1.0),
        "ret_10": float(c / close.iloc[-11] - 1.0),
        "ret_20": float(c / close.iloc[-21] - 1.0),
        "ret_60": float(c / close.iloc[-61] - 1.0) if len(close) >= 61 else 0.0,
        "vol_10": vol10,
        "vol_20": vol20,
        "vol_ratio": vol10 / (vol20 + 1e-12),
        "mom_20": float(close.iloc[-20:].mean() / close.iloc[-40:-20].mean() - 1.0)
        if len(close) >= 40
        else 0.0,
        "mom_60": float(close.iloc[-60:].mean() / close.iloc[-120:-60].mean() - 1.0)
        if len(close) >= 120
        else 0.0,
        "sma_dist_20": c / sma20 - 1.0,
        "sma_dist_60": c / sma60 - 1.0,
        "drawdown_60": c / roll_max - 1.0,
        "range_10": float((high.iloc[-10:].max() - low.iloc[-10:].min()) / c),
        "rsi_14": _rsi(close, 14),
        "trend_slope_20": slope,
        "ret_skew_20": float(rets.iloc[-20:].skew()),
    }


def steps_to_frame(
    steps: list[dict],
    ohlcv: pd.DataFrame,
    *,
    supervised_p_up: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Merge Kronos step dicts with market features and labels."""
    rows = []
    for s in steps:
        mkt = _market_row(ohlcv, s["asof"])
        p_up = float(s["p_up"])
        mean_r = float(s["mean_return"])
        asof = pd.Timestamp(s["asof"])
        row = {
            "asof": asof,
            "kronos_p_up": p_up,
            "kronos_mean_r": mean_r,
            "kronos_abs_mean_r": abs(mean_r),
            "kronos_edge": (p_up - 0.5) * np.sign(mean_r) * abs(mean_r),
            "kronos_disagree": abs(p_up - 0.5) * abs(mean_r),
            "realized_return": float(s["realized_return"]),
            "raw_signal": s["signal"],
            "y_up": 1 if float(s["realized_return"]) > 0 else 0,
            **mkt,
        }
        if supervised_p_up is not None:
            key = str(asof)
            # tolerate timezone string mismatches
            row["sup_p_up"] = float(
                supervised_p_up.get(key)
                or supervised_p_up.get(str(pd.Timestamp(s["asof"])))
                or 0.5
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)


def active_feature_cols(frame: pd.DataFrame) -> list[str]:
    cols = list(FEATURE_COLS)
    for c in OPTIONAL_FEATURE_COLS:
        if c in frame.columns:
            cols.append(c)
    return cols
