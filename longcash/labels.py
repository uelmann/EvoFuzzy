"""USD forward-return labels for LONG-CASH (not BTC-residual)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_usd_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
    winsorize_pct: tuple[float, float] = (1.0, 99.0),
) -> pd.DataFrame:
    """
    y_usd = log(close[t+h] / close[t]), winsorized 1/99 per date.
    y_up  = 1{close[t+h]/close[t] - 1 > 0}.
    y_simple = close[t+h]/close[t] - 1 (unwinsorized; diagnostics).
    """
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    h = int(horizon)
    fwd_log = np.log(close.shift(-h) / close)
    fwd_simple = close.shift(-h) / close - 1.0
    out = feat.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["symbol", "date"])

    y_log = []
    y_s = []
    for sym, g in out.groupby("symbol", sort=False):
        if sym not in fwd_log.columns:
            y_log.append(pd.Series(np.nan, index=g.index))
            y_s.append(pd.Series(np.nan, index=g.index))
            continue
        idx = pd.DatetimeIndex(g["date"])
        y_log.append(pd.Series(fwd_log[sym].reindex(idx).to_numpy(), index=g.index))
        y_s.append(pd.Series(fwd_simple[sym].reindex(idx).to_numpy(), index=g.index))
    out[f"y_usd_h{h}"] = pd.concat(y_log).sort_index()
    out[f"y_simple_h{h}"] = pd.concat(y_s).sort_index()

    lo_p, hi_p = winsorize_pct

    def _win(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 5:
            return s
        lo = np.nanpercentile(s.to_numpy(), lo_p)
        hi = np.nanpercentile(s.to_numpy(), hi_p)
        return s.clip(lo, hi)

    out[f"y_usd_h{h}"] = out.groupby("date", sort=False)[f"y_usd_h{h}"].transform(_win)
    ys = out[f"y_simple_h{h}"].to_numpy(dtype=float)
    yup = np.where(np.isfinite(ys), (ys > 0.0).astype(float), np.nan)
    out[f"y_up_h{h}"] = yup
    return out
