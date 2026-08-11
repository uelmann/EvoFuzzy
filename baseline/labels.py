"""Residualized forward-return labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    horizons: list[int],
    winsorize_pct: tuple[float, float] = (1.0, 99.0),
) -> pd.DataFrame:
    """
    y_{i,t} = r_{i,t→t+h} − beta_btc_60_{i,t} · r_{BTC,t→t+h}
    beta uses time-t estimate (no lookahead). Forward returns use future closes
    only inside the label (not as features).
    """
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    btc = close["BTCUSDT"]
    out = feat.copy()
    out = out.sort_values(["symbol", "date"])

    # attach beta if missing
    if "beta_btc_60" not in out.columns:
        raise ValueError("beta_btc_60 required on features")

    for h in horizons:
        # forward log return close[t+h]/close[t]
        fwd = np.log(close.shift(-h) / close)
        btc_fwd = np.log(btc.shift(-h) / btc)
        y = []
        for sym, g in out.groupby("symbol", sort=False):
            dates = g["date"].values
            if sym not in fwd.columns:
                y.append(pd.Series(np.nan, index=g.index))
                continue
            fr = fwd[sym].reindex(g["date"]).values
            bf = btc_fwd.reindex(g["date"]).values
            beta = g["beta_btc_60"].values
            resid = fr - beta * bf
            y.append(pd.Series(resid, index=g.index))
        out[f"y_h{h}"] = pd.concat(y).sort_index()

        # winsorize per date at 1/99 — NOT z-score
        lo_p, hi_p = winsorize_pct

        def _win(s: pd.Series) -> pd.Series:
            if s.notna().sum() < 5:
                return s
            lo = np.nanpercentile(s.values, lo_p)
            hi = np.nanpercentile(s.values, hi_p)
            return s.clip(lo, hi)

        out[f"y_h{h}"] = out.groupby("date", sort=False)[f"y_h{h}"].transform(_win)

    return out
