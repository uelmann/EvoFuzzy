"""Market-context features: one vector per date, own-z over 250d, clip ±5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline.features import features_for_symbol
from round_f.constants import CTX_COLS


def utc_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True)
    return out


def _utc_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
    return s


def _map_date(dates: pd.Series, ser: pd.Series) -> pd.Series:
    if ser is None or len(ser) == 0:
        return pd.Series(np.nan, index=dates.index)
    return pd.to_datetime(dates, utc=True).map(_utc_index(ser))


def _own_z_250(s: pd.Series, clip: float = 5.0) -> pd.Series:
    mu = s.rolling(250, min_periods=60).mean()
    sd = s.rolling(250, min_periods=60).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.clip(-clip, clip)


def residual_log_returns(panel: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Daily residual log-return: r_i − β_i,t · r_BTC,t with causal 60d beta (≤ t)."""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    p = p.sort_values(["symbol", "date"])
    p["r"] = p.groupby("symbol", sort=False)["close"].transform(lambda s: np.log(s / s.shift(1)))
    btc = p.loc[p["symbol"] == "BTCUSDT", ["date", "r"]].rename(columns={"r": "r_btc"})
    p = p.merge(btc, on="date", how="left")
    f = feat[["date", "symbol", "beta_btc_60_raw"]].copy() if "beta_btc_60_raw" in feat.columns else feat[["date", "symbol"]].copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    if "beta_btc_60_raw" not in f.columns:
        f["beta_btc_60_raw"] = np.nan
    p = p.merge(f, on=["date", "symbol"], how="left")
    # fill missing beta with rolling cov/var vs BTC (causal)
    need = p["beta_btc_60_raw"].isna() & p["r"].notna() & p["r_btc"].notna()
    if need.any():
        # Assign by index (do not merge .values dates — that strips tz and breaks UTC merges).
        p["beta_fill"] = np.nan
        need_syms = p.loc[need, "symbol"].unique()
        for _, g in p.loc[p["symbol"].isin(need_syms)].groupby("symbol", sort=False):
            gg = g.sort_values("date")
            cov = gg["r"].rolling(60, min_periods=20).cov(gg["r_btc"])
            var = gg["r_btc"].rolling(60, min_periods=20).var()
            beta = cov / var.replace(0.0, np.nan)
            p.loc[gg.index, "beta_fill"] = beta.to_numpy()
        p["beta_btc_60_raw"] = p["beta_btc_60_raw"].fillna(p["beta_fill"])
        p = p.drop(columns=["beta_fill"])
    p["resid"] = p["r"] - p["beta_btc_60_raw"].fillna(0.0) * p["r_btc"]
    return p[["date", "symbol", "close", "r", "r_btc", "beta_btc_60_raw", "resid"]]


def build_context_block(
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    pit120: pd.DataFrame,
    pit40: pd.DataFrame,
    pred_a7: pd.DataFrame,
    funding: pd.DataFrame | None,
    clip: float = 5.0,
) -> pd.DataFrame:
    print("[HB] context: residuals...", flush=True)
    resid = residual_log_returns(panel, feat)
    pit120 = utc_col(pit120)
    pit40 = utc_col(pit40)

    r120 = resid.merge(pit120[["date", "symbol"]], on=["date", "symbol"], how="inner")
    cs_disp = r120.groupby("date")["resid"].std(ddof=0).sort_index()
    ctx_disp = cs_disp.rolling(30, min_periods=10).mean()

    pa = utc_col(pred_a7)
    ctx_score_disp = pa.groupby("date")["score"].std(ddof=0).sort_index()

    btc = utc_col(panel.loc[panel["symbol"] == "BTCUSDT"].sort_values("date"))
    btc_feat = utc_col(features_for_symbol(btc, btc.set_index("date")["close"]))
    ctx_btc_vol = btc_feat.set_index("date")["yz_vol_30_raw"]
    sma100 = btc.set_index("date")["close"].rolling(100, min_periods=40).mean()
    ctx_btc_trend = btc.set_index("date")["close"] / sma100 - 1.0

    fund_agg = pd.Series(dtype=float)
    if funding is not None and not funding.empty:
        f = utc_col(funding)
        f120 = f.merge(pit120[["date", "symbol"]], on=["date", "symbol"], how="inner")
        col = "funding_rate" if "funding_rate" in f120.columns else "funding_now"
        if col in f120.columns:
            fund_agg = f120.groupby("date")[col].median().sort_index()

    # breadth: close > SMA50 on pit-120
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    p = p.sort_values(["symbol", "date"])
    p["sma50"] = p.groupby("symbol", sort=False)["close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    p["above"] = p["close"] > p["sma50"]
    b120 = p.merge(pit120[["date", "symbol"]], on=["date", "symbol"], how="inner")
    ctx_breadth = b120.groupby("date")["above"].mean().sort_index()

    print("[HB] context: top-40 mean pairwise 28d corr...", flush=True)
    wide_r = resid.pivot(index="date", columns="symbol", values="r").sort_index()
    wide_r.index = pd.DatetimeIndex(pd.to_datetime(wide_r.index, utc=True))
    dates40 = pd.DatetimeIndex(pd.to_datetime(sorted(pit40["date"].unique()), utc=True))
    corr_rows = []
    by_d = {}
    for k, v in pit40.groupby("date")["symbol"].apply(list).items():
        t = pd.Timestamp(k)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        by_d[t] = list(v)
    for dt in dates40:
        dt = pd.Timestamp(dt)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        syms = [s for s in by_d.get(dt, []) if s in wide_r.columns]
        if len(syms) < 5:
            corr_rows.append((dt, np.nan))
            continue
        loc = wide_r.index.searchsorted(dt, side="right")
        win = wide_r.iloc[max(0, loc - 28) : loc][syms]
        if len(win) < 15:
            corr_rows.append((dt, np.nan))
            continue
        c = win.corr()
        tri = c.values[np.triu_indices(len(c), k=1)]
        tri = tri[np.isfinite(tri)]
        corr_rows.append((dt, float(np.mean(tri)) if len(tri) else np.nan))
    ctx_corr = pd.Series({d: v for d, v in corr_rows}).sort_index()
    ctx_corr.index = pd.DatetimeIndex(pd.to_datetime(ctx_corr.index, utc=True))

    idx = pd.DatetimeIndex(pd.to_datetime(sorted(set(pit120["date"]).union(pit40["date"])), utc=True))
    out = pd.DataFrame({"date": idx})
    out["ctx_disp"] = _map_date(out["date"], ctx_disp)
    out["ctx_score_disp"] = _map_date(out["date"], ctx_score_disp)
    out["ctx_btc_vol"] = _map_date(out["date"], ctx_btc_vol)
    out["ctx_btc_trend"] = _map_date(out["date"], ctx_btc_trend)
    out["ctx_funding_agg"] = _map_date(out["date"], fund_agg)
    out["ctx_breadth"] = _map_date(out["date"], ctx_breadth)
    out["ctx_corr"] = _map_date(out["date"], ctx_corr)
    out = out.sort_values("date")
    for c in CTX_COLS:
        out[c] = _own_z_250(out[c], clip=clip)
    print(f"[HB] context rows={len(out)}", flush=True)
    return out


def merge_context(feat: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    f = utc_col(feat)
    c = utc_col(ctx)
    cols = ["date"] + [x for x in CTX_COLS if x in c.columns]
    return f.merge(c[cols], on="date", how="left")
