"""Fixed Stage-T regime gate. No learning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import REGIME_BREADTH, REGIME_OFF_HYSTERESIS


def _utc_idx(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert("UTC").normalize()


def ew_top50_btc_ratio(panel: pd.DataFrame, pit50: pd.DataFrame, btc_id: int) -> pd.Series:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = p.pivot(index="date", columns="id", values="close").sort_index()
    close.index = _utc_idx(close.index)
    pit = pit50.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit["id"] = pit["id"].astype(int)
    rows = []
    for dt, ids in pit.groupby("date")["id"]:
        dt = pd.Timestamp(dt).tz_convert("UTC").normalize()
        if dt not in close.index or btc_id not in close.columns:
            continue
        b = float(close.loc[dt, btc_id])
        if not np.isfinite(b) or b <= 0:
            continue
        cols = [int(i) for i in ids if int(i) != int(btc_id) and int(i) in close.columns]
        if not cols:
            continue
        px = close.loc[dt, cols].astype(float)
        px = px[np.isfinite(px) & (px > 0)]
        if px.empty:
            continue
        rows.append((dt, float((px / b).mean())))
    s = pd.Series({d: v for d, v in rows}).sort_index()
    s.index = _utc_idx(s.index)
    return s


def breadth_top100(panel: pd.DataFrame, pit100: pd.DataFrame) -> pd.Series:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    p = p.sort_values(["id", "date"])
    p["sma50"] = p.groupby("id", sort=False)["close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    p["above"] = p["close"] > p["sma50"]
    pit = pit100.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit["id"] = pit["id"].astype(int)
    m = p.merge(pit[["date", "id"]], on=["date", "id"], how="inner")
    br = m.groupby("date")["above"].mean().sort_index()
    br.index = _utc_idx(br.index)
    return br


def regime_on_off(
    ratio: pd.Series,
    breadth: pd.Series,
    *,
    breadth_thr: float = REGIME_BREADTH,
    off_hyst: int = REGIME_OFF_HYSTERESIS,
) -> pd.DataFrame:
    """ON when ratio > 90d SMA AND breadth > thr. OFF after `off_hyst` consecutive failed days."""
    ratio = ratio.sort_index()
    sma = ratio.rolling(90, min_periods=30).mean()
    cond_a = ratio > sma
    br = breadth.reindex(ratio.index)
    cond_b = br > float(breadth_thr)
    raw = (cond_a.fillna(False) & cond_b.fillna(False)).astype(bool)
    on = []
    fail = 0
    state = False
    for v in raw.tolist():
        if v:
            state = True
            fail = 0
        else:
            if state:
                fail += 1
                if fail >= int(off_hyst):
                    state = False
                    fail = 0
            else:
                fail = 0
        on.append(state)
    out = pd.DataFrame(
        {
            "ratio": ratio,
            "ratio_sma90": sma,
            "breadth": br,
            "raw_on": raw.astype(int),
            "gate_on": np.asarray(on, dtype=int),
        },
        index=ratio.index,
    )
    print(
        f"[HB] regime ON frac={float(out['gate_on'].mean()):.3f} n={len(out)} "
        f"raw={float(out['raw_on'].mean()):.3f}",
        flush=True,
    )
    return out
