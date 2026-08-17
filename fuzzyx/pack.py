"""Pack A0 features + PIT top-N into weekly (T, S, F) tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import EXEC_DV_WINDOW, FEATURE_COLS, REBALANCE_DAYS, UNIVERSE_N
from .universe import rebalance_dates


@dataclass
class PackedPanel:
    symbols: list[str]
    reb_dates: pd.DatetimeIndex
    X: np.ndarray  # (T, S, F)
    mask: np.ndarray  # (T, S) bool
    ret_h7: np.ndarray  # (T, S) simple return over next 7 sessions
    ret_1: np.ndarray  # (T, S) next-session simple return (for daily MTM eval)


def _as_utc(s) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(s, utc=True))
    return idx.tz_convert("UTC")


def pack_weekly(
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    n: int = UNIVERSE_N,
    every: int = REBALANCE_DAYS,
) -> PackedPanel:
    feat = feat.copy()
    feat["date"] = _as_utc(feat["date"])
    feat["symbol"] = feat["symbol"].astype(str)
    uni = universe.copy()
    uni["date"] = _as_utc(uni["date"])
    uni["symbol"] = uni["symbol"].astype(str)
    uni = uni[uni["rank"] <= int(n)].copy()

    dates = pd.DatetimeIndex(sorted(feat["date"].unique()))
    reb = pd.DatetimeIndex(rebalance_dates(dates, every=every))
    reb = reb.intersection(dates)
    symbols = sorted(set(uni["symbol"].unique()) | {"BTCUSDT"})
    sym_ix = {s: i for i, s in enumerate(symbols)}
    s_n = len(symbols)
    t_n = len(reb)
    f_n = len(FEATURE_COLS)

    close = feat.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.reindex(columns=symbols)
    # session shift: next 1 and next 7 available dates
    ret1 = close.shift(-1) / close - 1.0
    ret7 = close.shift(-7) / close - 1.0

    X = np.zeros((t_n, s_n, f_n), dtype=np.float32)
    mask = np.zeros((t_n, s_n), dtype=bool)
    rh7 = np.zeros((t_n, s_n), dtype=np.float32)
    r1 = np.zeros((t_n, s_n), dtype=np.float32)

    feat_i = feat.set_index(["date", "symbol"])
    for ti, dt in enumerate(reb):
        day_u = uni[uni["date"] == dt]
        if day_u.empty:
            continue
        for _, row in day_u.iterrows():
            sym = str(row["symbol"])
            if sym not in sym_ix:
                continue
            j = sym_ix[sym]
            try:
                fr = feat_i.loc[(dt, sym)]
            except KeyError:
                continue
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[-1]
            vals = fr.reindex(FEATURE_COLS).to_numpy(dtype=np.float64)
            if not np.isfinite(vals).any():
                continue
            X[ti, j] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            h7 = ret7.loc[dt, sym] if dt in ret7.index and sym in ret7.columns else np.nan
            d1 = ret1.loc[dt, sym] if dt in ret1.index and sym in ret1.columns else np.nan
            if not np.isfinite(h7):
                continue
            rh7[ti, j] = float(np.clip(h7, -0.8, 2.0))
            r1[ti, j] = float(np.clip(d1, -0.5, 1.0)) if np.isfinite(d1) else 0.0
            mask[ti, j] = True
    return PackedPanel(
        symbols=symbols,
        reb_dates=pd.DatetimeIndex(reb),
        X=X,
        mask=mask,
        ret_h7=rh7,
        ret_1=r1,
    )


def slice_packed(p: PackedPanel, start, end) -> PackedPanel:
    m = (p.reb_dates >= pd.Timestamp(start)) & (p.reb_dates <= pd.Timestamp(end))
    return PackedPanel(
        symbols=p.symbols,
        reb_dates=p.reb_dates[m],
        X=p.X[m],
        mask=p.mask[m],
        ret_h7=p.ret_h7[m],
        ret_1=p.ret_1[m],
    )
