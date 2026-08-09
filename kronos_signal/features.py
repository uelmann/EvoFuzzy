"""Build meta-model features from Kronos outputs + market context."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "kronos_p_up",
    "kronos_mean_r",
    "kronos_abs_mean_r",
    "kronos_edge",  # (p_up - 0.5) * sign(mean_r) * |mean_r|
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "mom_20",
    "drawdown_60",
    "range_10",
]


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
    roll_max = close.rolling(60).max().iloc[-1]
    return {
        "ret_1": float(c / close.iloc[-2] - 1.0),
        "ret_5": float(c / close.iloc[-6] - 1.0),
        "ret_10": float(c / close.iloc[-11] - 1.0),
        "ret_20": float(c / close.iloc[-21] - 1.0),
        "vol_10": float(rets.iloc[-10:].std()),
        "vol_20": float(rets.iloc[-20:].std()),
        "mom_20": float(close.iloc[-20:].mean() / close.iloc[-40:-20].mean() - 1.0)
        if len(close) >= 40
        else 0.0,
        "drawdown_60": float(c / float(roll_max) - 1.0),
        "range_10": float(
            (high.iloc[-10:].max() - low.iloc[-10:].min()) / c
        ),
    }


def steps_to_frame(steps: list[dict], ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Merge Kronos step dicts with market features and labels."""
    rows = []
    for s in steps:
        mkt = _market_row(ohlcv, s["asof"])
        p_up = float(s["p_up"])
        mean_r = float(s["mean_return"])
        rows.append(
            {
                "asof": pd.Timestamp(s["asof"]),
                "kronos_p_up": p_up,
                "kronos_mean_r": mean_r,
                "kronos_abs_mean_r": abs(mean_r),
                "kronos_edge": (p_up - 0.5) * np.sign(mean_r) * abs(mean_r),
                "realized_return": float(s["realized_return"]),
                "raw_signal": s["signal"],
                "y_up": 1 if float(s["realized_return"]) > 0 else 0,
                **mkt,
            }
        )
    return pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)
