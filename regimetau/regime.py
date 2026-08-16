"""Causal CS-correlation regime. No learning, one frozen config."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimetau.constants import (
    BTC_SYM,
    CS_CORR_MIN_NAMES,
    CS_CORR_MIN_OBS,
    CS_CORR_WINDOW,
    PIT_N,
    REGIME_BASE,
    REGIME_HIGH,
    REGIME_LOW,
    WARMUP_OBS,
)


def _utc_idx(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert("UTC").normalize()


def _median_pairwise_corr(block: pd.DataFrame) -> float:
    if block.shape[1] < CS_CORR_MIN_NAMES or block.shape[0] < CS_CORR_MIN_OBS:
        return float("nan")
    c = block.corr()
    vals = c.values[np.triu_indices_from(c.values, 1)]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return float("nan")
    return float(np.median(vals))


def cs_corr_topn(
    panel: pd.DataFrame,
    pit: pd.DataFrame,
    *,
    window: int = CS_CORR_WINDOW,
    exclude: str = BTC_SYM,
) -> pd.Series:
    """Median pairwise corr of `window` daily log-returns in the PIT set at t."""
    p = panel.copy()
    p["date"] = _utc_idx(p["date"])
    close = p.pivot(index="date", columns="symbol", values="close").sort_index()
    rets = np.log(close / close.shift(1))
    uni = pit.copy()
    uni["date"] = _utc_idx(uni["date"])
    if "rank" in uni.columns:
        uni = uni[uni["rank"] <= PIT_N]
    by_date = {
        pd.Timestamp(dt): set(g["symbol"].astype(str)) - {exclude}
        for dt, g in uni.groupby("date")
    }
    rows = []
    dates = list(rets.index)
    pos = {d: i for i, d in enumerate(dates)}
    w = int(window)
    for dt, names in sorted(by_date.items()):
        i = pos.get(dt)
        if i is None or i < w:
            continue
        cols = [s for s in names if s in rets.columns]
        if len(cols) < CS_CORR_MIN_NAMES:
            rows.append((dt, float("nan")))
            continue
        block = rets.iloc[i - w + 1 : i + 1][cols]
        rows.append((dt, _median_pairwise_corr(block)))
    s = pd.Series({d: v for d, v in rows}, dtype=float).sort_index()
    s.index = _utc_idx(s.index)
    print(
        f"[HB] cs_corr n={len(s)} finite={int(np.isfinite(s).sum())} "
        f"median={float(s.median()) if s.notna().any() else float('nan'):.4f}",
        flush=True,
    )
    return s


def regime_labels(cs_corr: pd.Series, warmup: int = WARMUP_OBS) -> pd.DataFrame:
    """HIGH if corr_t > expanding median of past corr; LOW otherwise; BASE in warmup."""
    s = cs_corr.copy().sort_index()
    s.index = _utc_idx(s.index)
    past_med = s.shift(1).expanding(min_periods=int(warmup)).median()
    n_past = s.shift(1).expanding().count()
    high = (s > past_med) & np.isfinite(past_med) & np.isfinite(s)
    low = (~high) & np.isfinite(past_med) & np.isfinite(s)
    lab = np.full(len(s), REGIME_BASE, dtype=int)
    lab[high.to_numpy()] = REGIME_HIGH
    lab[low.to_numpy()] = REGIME_LOW
    out = pd.DataFrame(
        {
            "cs_corr": s,
            "expanding_median": past_med,
            "n_past": n_past,
            "regime": lab,
        },
        index=s.index,
    )
    n = max(len(out), 1)
    print(
        f"[HB] regime HIGH={float((out['regime'] == REGIME_HIGH).mean()):.3f} "
        f"LOW={float((out['regime'] == REGIME_LOW).mean()):.3f} "
        f"BASE={float((out['regime'] == REGIME_BASE).mean()):.3f} n={n}",
        flush=True,
    )
    return out
