"""Causal OHLCV wide arrays for the formula miner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphamine.constants import BTC_SYMBOL, FIELDS


def _utc_index(idx) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
    return out


class MarketArrays:
    def __init__(
        self,
        dates: pd.DatetimeIndex,
        symbols: list[str],
        fields: dict[str, np.ndarray],
    ):
        self.dates = _utc_index(dates).tz_convert("UTC").normalize()
        self.symbols = list(symbols)
        self.fields = fields
        self.date_to_i = {pd.Timestamp(d): i for i, d in enumerate(self.dates)}
        self.sym_to_j = {s: j for j, s in enumerate(self.symbols)}
        self.shape = (len(self.dates), len(self.symbols))

    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(name)
        return self.fields[name]


def build_arrays(panel: pd.DataFrame, symbols: list[str] | None = None) -> MarketArrays:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    if symbols is not None:
        keep = set(symbols)
        p = p[p["symbol"].isin(keep)]
    p = p.sort_values(["date", "symbol"])
    dates = pd.DatetimeIndex(sorted(p["date"].unique()))
    syms = sorted(p["symbol"].unique())
    if not syms:
        raise RuntimeError("empty panel for MarketArrays")

    def _wide(col: str) -> np.ndarray:
        w = p.pivot(index="date", columns="symbol", values=col)
        w = w.reindex(index=dates, columns=syms)
        return w.to_numpy(dtype=float)

    close = _wide("close")
    volume = _wide("volume")
    dollar = _wide("dollar_volume") if "dollar_volume" in p.columns else _wide("quote_volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(np.abs(volume) > 1e-12, dollar / volume, np.nan)
        prev = np.roll(close, 1, axis=0)
        prev[0, :] = np.nan
        ret = np.log(close / prev)
    fields = {
        "open": _wide("open"),
        "high": _wide("high"),
        "low": _wide("low"),
        "close": close,
        "volume": volume,
        "dollar_volume": dollar,
        "vwap": vwap,
        "ret": ret,
    }
    missing = [k for k in FIELDS if k not in fields]
    if missing:
        raise RuntimeError(f"missing fields {missing}")
    return MarketArrays(dates, syms, fields)


def _map_date_i(dates, arr: MarketArrays) -> pd.Series:
    d = pd.to_datetime(dates, utc=True)
    if getattr(d, "dt", None) is not None:
        d = d.dt.tz_convert("UTC").dt.normalize()
    else:
        d = pd.Series(pd.DatetimeIndex(d).tz_convert("UTC").normalize())
    return d.map(arr.date_to_i)


def y_matrix(arr: MarketArrays, feat: pd.DataFrame, ycol: str) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=float)
    d = _map_date_i(feat["date"], arr)
    s = feat["symbol"].map(arr.sym_to_j)
    y = pd.to_numeric(feat[ycol], errors="coerce")
    ok = d.notna() & s.notna() & y.notna()
    if not bool(ok.any()):
        return out
    out[d[ok].astype(int).to_numpy(), s[ok].astype(int).to_numpy()] = y[ok].to_numpy(dtype=float)
    return out


def period_mask(
    arr: MarketArrays,
    feat: pd.DataFrame,
    start,
    end,
    exclude_btc: bool = True,
) -> np.ndarray:
    mask = np.zeros(arr.shape, dtype=bool)
    df = feat[["date", "symbol"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    if lo.tzinfo is None:
        lo = lo.tz_localize("UTC")
    else:
        lo = lo.tz_convert("UTC")
    if hi.tzinfo is None:
        hi = hi.tz_localize("UTC")
    else:
        hi = hi.tz_convert("UTC")
    lo = lo.normalize()
    hi = hi.normalize()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]
    if exclude_btc:
        df = df[df["symbol"] != BTC_SYMBOL]
    di = _map_date_i(df["date"], arr)
    sj = df["symbol"].map(arr.sym_to_j)
    ok = di.notna() & sj.notna()
    if bool(ok.any()):
        mask[di[ok].astype(int).to_numpy(), sj[ok].astype(int).to_numpy()] = True
    return mask


def attach_matrix(
    feat: pd.DataFrame,
    arr: MarketArrays,
    mat: np.ndarray,
    col: str,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Gather (T,N) values onto feat rows and CS-zscore per date."""
    out = feat
    if col in out.columns:
        out = out.drop(columns=[col])
    raw = np.full(len(feat), np.nan, dtype=float)
    di = _map_date_i(feat["date"], arr)
    sj = feat["symbol"].map(arr.sym_to_j)
    ok = di.notna() & sj.notna()
    if bool(ok.any()):
        raw[ok.to_numpy()] = mat[di[ok].astype(int).to_numpy(), sj[ok].astype(int).to_numpy()]
    out = out.copy()
    out[col] = raw

    def _z(s: pd.Series) -> pd.Series:
        v = s.to_numpy(dtype=float)
        m = np.isfinite(v)
        if int(m.sum()) < 3:
            return pd.Series(np.zeros(len(s)), index=s.index)
        mu = float(np.mean(v[m]))
        sd = float(np.std(v[m], ddof=0))
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        z = (v - mu) / sd
        z = np.clip(z, -clip, clip)
        z[~m] = np.nan
        return pd.Series(z, index=s.index)

    out[col] = out.groupby("date", sort=False)[col].transform(_z)
    return out
