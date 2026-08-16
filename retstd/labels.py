"""Forward USDT-return / path-std ratio and top-decile binary labels.

Working-copy only. Never write back to features_labeled.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retstd.constants import HORIZON, MIN_CS, STD_EPS, TOP_PCT, YCOL_RATIO, YCOL_SIMPLE, YCOL_STD


def forward_path_std(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """At date t, sample std (ddof=1) of daily simple returns r[t+1] .. r[t+h]."""
    close = close.sort_index()
    h = int(horizon)
    r = close.pct_change(fill_method=None)
    return r.rolling(window=h, min_periods=h).std(ddof=1).shift(-h)


def add_retstd_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int = HORIZON,
    top_pct: float = TOP_PCT,
    eps: float = STD_EPS,
    min_cs: int = MIN_CS,
) -> pd.DataFrame:
    """Replace working y_h{h} with the binary top-decile of R / (STD+eps).

    Preserves the frozen residual as y_resid_h{h} when present.
    """
    h = int(horizon)
    y_bin = f"y_h{h}"
    y_resid = f"y_resid_h{h}"
    out = feat.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    if y_bin in out.columns and y_resid not in out.columns:
        out = out.rename(columns={y_bin: y_resid})

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    close = p.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    simple = close.shift(-h) / close - 1.0
    std = forward_path_std(close, h)
    ratio = simple / (std + float(eps))

    def _stack(wide: pd.DataFrame, name: str) -> pd.DataFrame:
        try:
            s = wide.stack(future_stack=True)
        except TypeError:
            s = wide.stack()
        s = s.rename(name).reset_index()
        s.columns = ["date", "symbol", name]
        s["date"] = pd.to_datetime(s["date"], utc=True)
        return s

    extra = _stack(simple, YCOL_SIMPLE)
    extra = extra.merge(_stack(std, YCOL_STD), on=["date", "symbol"], how="outer")
    extra = extra.merge(_stack(ratio, YCOL_RATIO), on=["date", "symbol"], how="outer")
    out = out.merge(extra, on=["date", "symbol"], how="left")

    q = 1.0 - float(top_pct)

    def _top(s: pd.Series) -> pd.Series:
        m = s.notna()
        if int(m.sum()) < int(min_cs):
            return pd.Series(np.nan, index=s.index)
        thr = float(np.nanquantile(s.to_numpy(dtype=float), q))
        out_s = pd.Series(np.nan, index=s.index)
        out_s.loc[m] = (s.loc[m] >= thr).astype(float)
        return out_s

    out[y_bin] = out.groupby("date", sort=False)[YCOL_RATIO].transform(_top)
    return out
