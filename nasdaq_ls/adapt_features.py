"""Equity-clock features for NASDAQ-ADAPT-1 (not A0's 7/14/28/90 crypto windows)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline.features import _log_ret, _rolling_ols_beta_resid, _yang_zhang


ADAPT_FEATURE_COLS = [
    "ret_21",
    "ret_63",
    "ret_126",
    "ret_252",
    "mom_126_skip21",
    "mom_252_skip21",
    "rev_5",
    "rev_21",
    "close_sma63",
    "close_sma126",
    "close_sma252",
    "yz_vol_21",
    "yz_vol_63",
    "yz_vol_126",
    "vol_ratio_21_126",
    "vol_of_vol_63",
    "beta_mkt_63",
    "beta_mkt_126",
    "idio_vol_63",
    "corr_mkt_63",
    "dist_high_252",
    "dist_low_252",
    "range_pos_63",
    "amihud_21",
    "dv_z_63",
    "dv_trend_63",
]


def features_for_symbol_adapt(df: pd.DataFrame, mkt_close: pd.Series) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    d = d.set_index("date")
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    dv = d["dollar_volume"]
    r1 = np.log(c / c.shift(1))

    out = pd.DataFrame(index=d.index)
    out["ret_21"] = _log_ret(c, 21)
    out["ret_63"] = _log_ret(c, 63)
    out["ret_126"] = _log_ret(c, 126)
    out["ret_252"] = _log_ret(c, 252)
    out["mom_126_skip21"] = np.log(c.shift(21) / c.shift(126))
    out["mom_252_skip21"] = np.log(c.shift(21) / c.shift(252))
    out["rev_5"] = -_log_ret(c, 5)
    out["rev_21"] = -_log_ret(c, 21)

    sma63 = c.rolling(63, min_periods=30).mean()
    sma126 = c.rolling(126, min_periods=60).mean()
    sma252 = c.rolling(252, min_periods=120).mean()
    out["close_sma63"] = c / sma63 - 1.0
    out["close_sma126"] = c / sma126 - 1.0
    out["close_sma252"] = c / sma252 - 1.0

    out["yz_vol_21"] = _yang_zhang(o, h, l, c, 21)
    out["yz_vol_63"] = _yang_zhang(o, h, l, c, 63)
    out["yz_vol_126"] = _yang_zhang(o, h, l, c, 126)
    out["vol_ratio_21_126"] = out["yz_vol_21"] / out["yz_vol_126"].replace(0, np.nan)
    out["vol_of_vol_63"] = out["yz_vol_21"].rolling(63, min_periods=20).std()

    mkt = mkt_close.reindex(d.index)
    mkt_r = np.log(mkt / mkt.shift(1))
    beta63, idio63 = _rolling_ols_beta_resid(r1, mkt_r, 63)
    beta126, _ = _rolling_ols_beta_resid(r1, mkt_r, 126)
    out["beta_mkt_63"] = beta63
    out["beta_mkt_126"] = beta126
    out["idio_vol_63"] = idio63
    out["corr_mkt_63"] = r1.rolling(63, min_periods=20).corr(mkt_r)

    hh = h.rolling(252, min_periods=120).max()
    ll = l.rolling(252, min_periods=120).min()
    out["dist_high_252"] = c / hh - 1.0
    out["dist_low_252"] = c / ll - 1.0
    hh63 = h.rolling(63, min_periods=30).max()
    ll63 = l.rolling(63, min_periods=30).min()
    out["range_pos_63"] = (c - ll63) / (hh63 - ll63).replace(0, np.nan)

    out["amihud_21"] = (r1.abs() / dv.replace(0, np.nan)).rolling(21, min_periods=8).mean()
    dv_mean = dv.rolling(63, min_periods=20).mean()
    dv_std = dv.rolling(63, min_periods=20).std()
    out["dv_z_63"] = (dv - dv_mean) / dv_std.replace(0, np.nan)
    out["dv_trend_63"] = dv.rolling(21, min_periods=8).mean() / dv_mean.replace(0, np.nan) - 1.0

    out["symbol"] = df["symbol"].iloc[0]
    out["close"] = c
    out["dollar_volume"] = dv
    out["yz_vol_63_raw"] = out["yz_vol_63"]
    out["yz_vol_30_raw"] = out["yz_vol_63"]  # sizing fallback used by _attach_aux
    out["yz_vol_30"] = out["yz_vol_63"]
    out["mom_252_skip21_raw"] = out["mom_252_skip21"]
    out["beta_mkt_63_raw"] = out["beta_mkt_63"]
    return out.reset_index()


def apply_cs_zscore_cols(feat: pd.DataFrame, cols: list[str], clip: float = 5.0) -> pd.DataFrame:
    out = feat.copy()
    for col in cols:
        if col not in out.columns:
            continue

        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return ((s - mu) / sd).clip(-clip, clip)

        out[col] = out.groupby("date", sort=False)[col].transform(_z)
    return out


def build_adapt_feature_panel(
    panel: pd.DataFrame,
    market_close: pd.Series,
    clip: float = 5.0,
    zscore: bool = True,
) -> pd.DataFrame:
    mkt = pd.to_numeric(market_close, errors="coerce").sort_index()
    mkt.index = pd.to_datetime(mkt.index, utc=True).normalize()
    mkt = mkt[~mkt.index.duplicated(keep="last")].ffill()
    parts = []
    n_sym = panel["symbol"].nunique()
    for i, (sym, g) in enumerate(panel.groupby("symbol", sort=False), start=1):
        if len(g) < 80:
            continue
        gg = g.copy()
        gg["date"] = pd.to_datetime(gg["date"], utc=True).dt.normalize()
        if "adj_close" in gg.columns:
            gg["close"] = pd.to_numeric(gg["adj_close"], errors="coerce")
        parts.append(features_for_symbol_adapt(gg, mkt))
        if i % 25 == 0 or i == n_sym:
            print(f"[adapt features] {i}/{n_sym} symbols", flush=True)
    if not parts:
        raise RuntimeError("no ADAPT-1 features")
    feat = pd.concat(parts, ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
    if zscore:
        feat = apply_cs_zscore_cols(feat, ADAPT_FEATURE_COLS, clip=clip)
    return feat
