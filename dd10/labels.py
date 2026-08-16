"""Forward USDT-return / average-drawdown ratio and top-decile binary labels.

Working-copy only. Never write back to features_labeled.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dd10.constants import ADD_EPS, HORIZON, MIN_CS, TOP_PCT, YCOL_ADD, YCOL_RATIO, YCOL_SIMPLE


def forward_avg_drawdown(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """At date t, mean_{k=1..h} (1 - close[t+k] / max(close[t..t+k]))."""
    close = close.sort_index()
    h = int(horizon)
    dds: list[np.ndarray] = []
    for k in range(1, h + 1):
        peak = close.rolling(window=k + 1, min_periods=k + 1).max().shift(-k)
        fut = close.shift(-k)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = 1.0 - fut.to_numpy(dtype=float) / peak.to_numpy(dtype=float)
        dd = np.where(np.isfinite(dd), np.clip(dd, 0.0, None), np.nan)
        dds.append(dd)
    stacked = np.nanmean(np.stack(dds, axis=0), axis=0)
    return pd.DataFrame(stacked, index=close.index, columns=close.columns)


def add_dd10_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int = HORIZON,
    top_pct: float = TOP_PCT,
    eps: float = ADD_EPS,
    min_cs: int = MIN_CS,
) -> pd.DataFrame:
    """Replace working y_h{h} with the binary top-decile of R / (ADD+eps).

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
    add = forward_avg_drawdown(close, h)
    ratio = simple / (add + float(eps))

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
    extra = extra.merge(_stack(add, YCOL_ADD), on=["date", "symbol"], how="outer")
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
