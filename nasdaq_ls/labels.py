"""Residualized h-day labels vs the stock market proxy (A0 recipe)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    market_close: pd.Series,
    horizons: list[int],
    winsorize_pct: tuple[float, float] = (1.0, 99.0),
) -> pd.DataFrame:
    from nasdaq_ls.prices import close_wide

    close = close_wide(panel)
    mkt = pd.to_numeric(market_close, errors="coerce").sort_index()
    mkt.index = pd.to_datetime(mkt.index, utc=True).normalize()
    mkt = mkt.reindex(close.index).ffill()
    out = feat.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
    out = out.sort_values(["symbol", "date"])

    if "beta_btc_60_raw" in out.columns:
        beta_col = "beta_btc_60_raw"
    elif "beta_btc_60" in out.columns:
        beta_col = "beta_btc_60"
    else:
        raise ValueError("beta_btc_60_raw required")

    for h in horizons:
        fwd = np.log(close.shift(-h) / close)
        mkt_fwd = np.log(mkt.shift(-h) / mkt)
        y = []
        for sym, g in out.groupby("symbol", sort=False):
            if sym not in fwd.columns:
                y.append(pd.Series(np.nan, index=g.index))
                continue
            fr = fwd[sym].reindex(g["date"]).values
            bf = mkt_fwd.reindex(g["date"]).values
            beta = g[beta_col].values
            y.append(pd.Series(fr - beta * bf, index=g.index))
        out[f"y_h{h}"] = pd.concat(y).sort_index()
        lo_p, hi_p = winsorize_pct

        def _win(s: pd.Series) -> pd.Series:
            if s.notna().sum() < 5:
                return s
            lo = np.nanpercentile(s.values, lo_p)
            hi = np.nanpercentile(s.values, hi_p)
            return s.clip(lo, hi)

        out[f"y_h{h}"] = out.groupby("date", sort=False)[f"y_h{h}"].transform(_win)
        simple = close.shift(-h) / close - 1.0
        s = []
        for sym, g in out.groupby("symbol", sort=False):
            if sym not in simple.columns:
                s.append(pd.Series(np.nan, index=g.index))
                continue
            s.append(pd.Series(simple[sym].reindex(g["date"]).values, index=g.index))
        out[f"y_simple_h{h}"] = pd.concat(s).sort_index()
    return out
