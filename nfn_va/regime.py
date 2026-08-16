"""Date-level FiLM market vector. Not added to the 33 name features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfn_va.constants import REGIME_EW_DAYS, REGIME_RET_DAYS


def _utc_idx(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert("UTC").normalize()


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC").normalize()
    return t.tz_convert("UTC").normalize()


def regime_frame(
    panel: pd.DataFrame,
    pit50: pd.DataFrame,
    pit100: pd.DataFrame,
    btc_id: int,
) -> pd.DataFrame:
    """m_t = [EW top-50/BTC 20d return, CS disp of 14d returns, breadth>0]."""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    p["id"] = p["id"].astype(int)
    close = p.pivot_table(index="date", columns="id", values="close", aggfunc="last").sort_index()
    close.index = _utc_idx(close.index)

    pit50 = pit50.copy()
    pit50["date"] = pd.to_datetime(pit50["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit50["id"] = pit50["id"].astype(int)
    pit100 = pit100.copy()
    pit100["date"] = pd.to_datetime(pit100["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit100["id"] = pit100["id"].astype(int)

    mem50 = { _as_utc(d): [int(i) for i in v] for d, v in pit50.groupby("date")["id"] }
    mem100 = { _as_utc(d): [int(i) for i in v] for d, v in pit100.groupby("date")["id"] }

    ratio_rows = []
    for dt, ids in mem50.items():
        if dt not in close.index or int(btc_id) not in close.columns:
            continue
        b = float(close.at[dt, int(btc_id)])
        if not np.isfinite(b) or b <= 0:
            continue
        cols = [i for i in ids if i != int(btc_id) and i in close.columns]
        if not cols:
            continue
        px = close.loc[dt, cols].astype(float)
        px = px[np.isfinite(px) & (px > 0)]
        if px.empty:
            continue
        ratio_rows.append((dt, float((px / b).mean())))
    ratio = pd.Series({d: v for d, v in ratio_rows}).sort_index()
    ratio.index = _utc_idx(ratio.index)
    ew20 = ratio.pct_change(int(REGIME_EW_DAYS), fill_method=None)

    ret = close.pct_change(int(REGIME_RET_DAYS), fill_method=None)
    disp_rows, br_rows = [], []
    for dt, ids in mem100.items():
        if dt not in ret.index:
            continue
        cols = [i for i in ids if i != int(btc_id) and i in ret.columns]
        if len(cols) < 8:
            continue
        row = ret.loc[dt, cols].astype(float)
        row = row[np.isfinite(row)]
        if len(row) < 8:
            continue
        disp_rows.append((dt, float(row.std(ddof=0))))
        br_rows.append((dt, float((row > 0).mean())))
    disp = pd.Series({d: v for d, v in disp_rows}).sort_index()
    breadth = pd.Series({d: v for d, v in br_rows}).sort_index()
    disp.index = _utc_idx(disp.index)
    breadth.index = _utc_idx(breadth.index)

    idx = close.index
    out = pd.DataFrame(index=idx)
    out["m_ew50_btc_20d"] = ew20.reindex(idx)
    out["m_cs_disp_14"] = disp.reindex(idx)
    out["m_breadth_pos"] = breadth.reindex(idx)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().fillna(0.0)
    # own-z so FiLM inputs are O(1)
    for c in out.columns:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        if np.isfinite(sd) and sd > 1e-12:
            out[c] = (out[c] - mu) / sd
        else:
            out[c] = 0.0
    out = out.clip(-5.0, 5.0)
    out = out.reset_index().rename(columns={"index": "date"})
    print(f"[p7d] regime dates={len(out)}", flush=True)
    return out
