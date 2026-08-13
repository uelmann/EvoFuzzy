"""Phase D microstructure feature block (~12 cols)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase_d.micro_data import MICRO_FEATURE_COLS


def _own_z(s: pd.Series, window: int = 30) -> pd.Series:
    mu = s.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def build_micro_features_symbol(
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
    premium: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Build raw (pre-CS-z) microstructure features for one symbol.
    All inputs filtered to symbol; dates UTC midnight.
    Liquidations unavailable → liq_* stay NaN.
    """
    # master date index from union
    pieces = []
    for df in (funding, metrics, premium, dollar_volume):
        if df is not None and not df.empty:
            pieces.append(pd.to_datetime(df["date"], utc=True))
    if not pieces:
        return pd.DataFrame(columns=["date", "symbol"] + MICRO_FEATURE_COLS)

    dates = pd.DatetimeIndex(sorted(pd.concat(pieces).unique()))
    out = pd.DataFrame({"date": dates})
    out["symbol"] = symbol

    # dollar volume trailing median
    dv = dollar_volume.copy() if dollar_volume is not None else pd.DataFrame()
    if not dv.empty:
        dv = dv.sort_values("date")
        dv["date"] = pd.to_datetime(dv["date"], utc=True)
        out = out.merge(dv[["date", "dollar_volume"]], on="date", how="left")
    else:
        out["dollar_volume"] = np.nan
    out["dv_med_30"] = out["dollar_volume"].rolling(30, min_periods=10).median()

    # funding
    if funding is not None and not funding.empty:
        f = funding.sort_values("date").copy()
        f["date"] = pd.to_datetime(f["date"], utc=True)
        out = out.merge(f[["date", "funding_rate"]], on="date", how="left")
    else:
        out["funding_rate"] = np.nan
    out["funding_now"] = out["funding_rate"]
    out["funding_z_30"] = _own_z(out["funding_now"], 30)
    out["funding_cum_7"] = out["funding_now"].rolling(7, min_periods=1).sum()
    # funding_cs_rank filled later cross-sectionally

    # premium / basis
    if premium is not None and not premium.empty:
        p = premium.sort_values("date").copy()
        p["date"] = pd.to_datetime(p["date"], utc=True)
        out = out.merge(p[["date", "premium_close"]], on="date", how="left")
    else:
        out["premium_close"] = np.nan
    out["basis_z_30"] = _own_z(out["premium_close"], 30)

    # metrics: OI, taker, L/S
    if metrics is not None and not metrics.empty:
        m = metrics.sort_values("date").copy()
        m["date"] = pd.to_datetime(m["date"], utc=True)
        cols = [
            c
            for c in [
                "sum_open_interest",
                "sum_open_interest_value",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ]
            if c in m.columns
        ]
        out = out.merge(m[["date"] + cols], on="date", how="left")
    for c in [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    oi = out["sum_open_interest"]
    out["oi_chg_1"] = oi.diff(1) / out["dv_med_30"].replace(0, np.nan)
    out["oi_chg_7"] = oi.diff(7) / out["dv_med_30"].replace(0, np.nan)
    # prefer notional OI if available
    oi_val = out["sum_open_interest_value"]
    out["oi_turnover"] = oi_val / out["dv_med_30"].replace(0, np.nan)

    out["taker_imb_z"] = _own_z(out["sum_taker_long_short_vol_ratio"], 30)
    out["ls_ratio_z"] = _own_z(out["count_long_short_ratio"], 30)

    out["liq_imb_1"] = np.nan
    out["liq_imb_7"] = np.nan

    cols = ["date", "symbol"] + [c for c in MICRO_FEATURE_COLS if c != "funding_cs_rank"]
    # funding_cs_rank added at panel level
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].copy()


def build_micro_feature_panel(
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
    premium: pd.DataFrame,
    panel_dv: pd.DataFrame,
    symbols: list[str],
    clip: float = 5.0,
) -> pd.DataFrame:
    """Build full panel with CS z-score (except funding_cs_rank which is rank then z)."""
    parts = []
    fund = funding.copy() if funding is not None else pd.DataFrame()
    met = metrics.copy() if metrics is not None else pd.DataFrame()
    prem = premium.copy() if premium is not None else pd.DataFrame()
    dv = panel_dv.copy() if panel_dv is not None else pd.DataFrame()
    for col_df in (fund, met, prem, dv):
        if not col_df.empty and "date" in col_df.columns:
            col_df["date"] = pd.to_datetime(col_df["date"], utc=True)

    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        f_s = fund[fund["symbol"] == sym] if not fund.empty else fund
        m_s = met[met["symbol"] == sym] if not met.empty else met
        p_s = prem[prem["symbol"] == sym] if not prem.empty else prem
        d_s = dv[dv["symbol"] == sym] if not dv.empty else dv
        parts.append(build_micro_features_symbol(f_s, m_s, p_s, d_s, sym))
        if i % 50 == 0 or i == n:
            print(f"[microfeat] {i}/{n} symbols", flush=True)

    if not parts:
        return pd.DataFrame(columns=["date", "symbol"] + MICRO_FEATURE_COLS)
    raw = pd.concat(parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True)

    # funding_cs_rank: cross-sectional rank of funding_now in [0,1]
    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(method="average", pct=True)

    raw["funding_cs_rank"] = raw.groupby("date")["funding_now"].transform(_rank)

    # CS z-score all micro features per date; keep NaNs (do not fill 0)
    out = raw[["date", "symbol"]].copy()
    for col in MICRO_FEATURE_COLS:
        if col not in raw.columns:
            out[col] = np.nan
            continue

        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean(skipna=True)
            sd = s.std(ddof=0, skipna=True)
            if not np.isfinite(sd) or sd == 0:
                # all nan or constant → leave nan where nan, 0 where finite constant
                return s * 0.0
            return ((s - mu) / sd).clip(-clip, clip)

        out[col] = raw.groupby("date", sort=False)[col].transform(_z)
        # preserve NaN mask
        out.loc[raw[col].isna(), col] = np.nan
    return out


def micro_coverage_on_book(
    feat_micro: pd.DataFrame,
    book_keys: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.Series:
    """Per date: fraction of book symbols with any non-null micro feature."""
    cols = cols or MICRO_FEATURE_COLS
    keys = book_keys[["date", "symbol"]].copy()
    keys["date"] = pd.to_datetime(keys["date"], utc=True)
    m = feat_micro.copy()
    m["date"] = pd.to_datetime(m["date"], utc=True)
    j = keys.merge(m, on=["date", "symbol"], how="left")
    has = j[cols].notna().any(axis=1)
    j["_has"] = has
    return j.groupby("date")["_has"].mean()
