"""Price/volume daily features with cross-sectional z-score (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "ret_7",
    "ret_14",
    "ret_28",
    "ret_56",
    "ret_90",
    "mom_28_skip7",
    "mom_90_skip14",
    "rev_1",
    "rev_3",
    "close_sma20",
    "close_sma50",
    "close_sma100",
    "sma20_sma50",
    "ema12_ema26",
    "yz_vol_14",
    "yz_vol_30",
    "yz_vol_60",
    "pk_vol_14",
    "vol_ratio",
    "vol_of_vol_30",
    "max_ret_14",
    "min_ret_14",
    "dist_high_90",
    "dist_low_90",
    "range_pos_28",
    "skew_28",
    "skew_60",
    "beta_btc_60",
    "idio_vol_60",
    "corr_btc_28",
    "amihud_14",
    "dv_z_30",
    "dv_trend",
]


def _log_ret(close: pd.Series, w: int) -> pd.Series:
    return np.log(close / close.shift(w))


def _parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    rs = np.log(high / low) ** 2
    return np.sqrt(rs.rolling(window, min_periods=max(3, window // 3)).mean() / (4 * np.log(2)))


def _yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Yang-Zhang volatility estimator (annualization omitted; daily units)."""
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_co = np.log(close / open_)
    log_oc = np.log(open_ / close.shift(1))
    log_cc = np.log(close / close.shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    # Rogers-Satchell part
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    open_var = log_oc.rolling(window, min_periods=max(5, window // 3)).var()
    close_var = log_cc.rolling(window, min_periods=max(5, window // 3)).var()
    rs_var = rs.rolling(window, min_periods=max(5, window // 3)).mean()
    yz = open_var + k * close_var + (1 - k) * rs_var
    return np.sqrt(yz.clip(lower=0.0))


def _rolling_ols_beta_resid(
    y: pd.Series, x: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """Rolling beta via cov/var; idio vol ≈ rolling std of y − βx (causal)."""
    min_p = max(20, window // 3)
    cov = y.rolling(window, min_periods=min_p).cov(x)
    var = x.rolling(window, min_periods=min_p).var()
    beta = cov / var.replace(0, np.nan)
    resid = y - beta * x
    idio = resid.rolling(window, min_periods=min_p).std()
    return beta, idio


def features_for_symbol(df: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    """Compute raw (pre-zscore) features for one symbol; index = date."""
    d = df.sort_values("date").copy()
    d = d.set_index("date")
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    dv = d["dollar_volume"]
    r1 = np.log(c / c.shift(1))

    out = pd.DataFrame(index=d.index)
    out["ret_7"] = _log_ret(c, 7)
    out["ret_14"] = _log_ret(c, 14)
    out["ret_28"] = _log_ret(c, 28)
    out["ret_56"] = _log_ret(c, 56)
    out["ret_90"] = _log_ret(c, 90)
    out["mom_28_skip7"] = np.log(c.shift(7) / c.shift(28))
    out["mom_90_skip14"] = np.log(c.shift(14) / c.shift(90))
    out["rev_1"] = -r1
    out["rev_3"] = -_log_ret(c, 3)

    sma20 = c.rolling(20, min_periods=10).mean()
    sma50 = c.rolling(50, min_periods=20).mean()
    sma100 = c.rolling(100, min_periods=40).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["close_sma20"] = c / sma20 - 1.0
    out["close_sma50"] = c / sma50 - 1.0
    out["close_sma100"] = c / sma100 - 1.0
    out["sma20_sma50"] = sma20 / sma50 - 1.0
    out["ema12_ema26"] = ema12 / ema26 - 1.0

    out["yz_vol_14"] = _yang_zhang(o, h, l, c, 14)
    out["yz_vol_30"] = _yang_zhang(o, h, l, c, 30)
    out["yz_vol_60"] = _yang_zhang(o, h, l, c, 60)
    out["pk_vol_14"] = _parkinson(h, l, 14)
    out["vol_ratio"] = out["yz_vol_14"] / out["yz_vol_60"].replace(0, np.nan)
    out["vol_of_vol_30"] = out["yz_vol_14"].rolling(30, min_periods=10).std()

    out["max_ret_14"] = r1.rolling(14, min_periods=5).max()
    out["min_ret_14"] = r1.rolling(14, min_periods=5).min()
    hh90 = h.rolling(90, min_periods=30).max()
    ll90 = l.rolling(90, min_periods=30).min()
    out["dist_high_90"] = c / hh90 - 1.0
    out["dist_low_90"] = c / ll90 - 1.0
    hh28 = h.rolling(28, min_periods=10).max()
    ll28 = l.rolling(28, min_periods=10).min()
    out["range_pos_28"] = (c - ll28) / (hh28 - ll28).replace(0, np.nan)

    out["skew_28"] = r1.rolling(28, min_periods=10).skew()
    out["skew_60"] = r1.rolling(60, min_periods=20).skew()

    btc = btc_close.reindex(d.index)
    btc_r = np.log(btc / btc.shift(1))
    beta, idio = _rolling_ols_beta_resid(r1, btc_r, 60)
    out["beta_btc_60"] = beta
    out["idio_vol_60"] = idio
    out["corr_btc_28"] = r1.rolling(28, min_periods=10).corr(btc_r)

    out["amihud_14"] = (r1.abs() / dv.replace(0, np.nan)).rolling(14, min_periods=5).mean()
    dv_mean = dv.rolling(30, min_periods=10).mean()
    dv_std = dv.rolling(30, min_periods=10).std()
    out["dv_z_30"] = (dv - dv_mean) / dv_std.replace(0, np.nan)
    out["dv_trend"] = dv.rolling(7, min_periods=3).mean() / dv.rolling(30, min_periods=10).mean() - 1.0

    out["symbol"] = df["symbol"].iloc[0]
    out["close"] = c
    out["dollar_volume"] = dv
    out["yz_vol_30_raw"] = out["yz_vol_30"]  # keep for sizing before CS z
    out["beta_btc_60_raw"] = out["beta_btc_60"]  # keep for labels / hedge
    return out.reset_index()


def apply_cs_zscore(feat: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """Cross-sectional z-score per date; preserves *_raw columns."""
    out = feat.copy()
    for col in FEATURE_COLS:
        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return ((s - mu) / sd).clip(-clip, clip)

        out[col] = out.groupby("date", sort=False)[col].transform(_z)
    return out


def build_feature_panel(
    panel: pd.DataFrame,
    clip: float = 5.0,
    zscore: bool = True,
) -> pd.DataFrame:
    btc = panel.loc[panel["symbol"] == "BTCUSDT"].set_index("date")["close"].sort_index()
    if btc.empty:
        raise RuntimeError("BTCUSDT required in panel")
    parts = []
    n_sym = panel["symbol"].nunique()
    for i, (sym, g) in enumerate(panel.groupby("symbol", sort=False), start=1):
        if len(g) < 50:
            continue
        parts.append(features_for_symbol(g, btc))
        if i % 25 == 0 or i == n_sym:
            print(f"[features] {i}/{n_sym} symbols", flush=True)
    feat = pd.concat(parts, ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    if zscore:
        feat = apply_cs_zscore(feat, clip=clip)
    return feat
