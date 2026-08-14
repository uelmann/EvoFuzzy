"""Phase 2 features: 25 pruned price survivors + context + new-data. All ≤ t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    CS_CLIP,
    CTX_COLS,
    CTX_OWN_Z_MINP,
    CTX_OWN_Z_WINDOW,
    FEATURE_COLS_V1,
    NEW_COLS,
    PRICE_COLS,
    PRICE_MOM_TREND_DIST,
    STABLE_OR_WRAP,
)


def _as_utc_idx(s: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(s, utc=True))
    return idx.tz_convert("UTC").normalize()


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
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_co = np.log(close / open_)
    log_oc = np.log(open_ / close.shift(1))
    log_cc = np.log(close / close.shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    open_var = log_oc.rolling(window, min_periods=max(5, window // 3)).var()
    close_var = log_cc.rolling(window, min_periods=max(5, window // 3)).var()
    rs_var = rs.rolling(window, min_periods=max(5, window // 3)).mean()
    yz = open_var + k * close_var + (1 - k) * rs_var
    return np.sqrt(yz.clip(lower=0.0))


def _rolling_ols_beta_resid(y: pd.Series, x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    min_p = max(20, window // 3)
    cov = y.rolling(window, min_periods=min_p).cov(x)
    var = x.rolling(window, min_periods=min_p).var()
    beta = cov / var.replace(0, np.nan)
    resid = y - beta * x
    idio = resid.rolling(window, min_periods=min_p).std()
    return beta, idio


def _own_z_250(s: pd.Series, clip: float = CS_CLIP) -> pd.Series:
    mu = s.rolling(CTX_OWN_Z_WINDOW, min_periods=CTX_OWN_Z_MINP).mean()
    sd = s.rolling(CTX_OWN_Z_WINDOW, min_periods=CTX_OWN_Z_MINP).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.clip(-clip, clip)


def btc_id_from_panel(panel: pd.DataFrame) -> int:
    btc_rows = panel[panel["slug"].astype(str).str.lower().eq("bitcoin") | panel["symbol"].str.upper().eq("BTC")]
    if btc_rows.empty:
        raise RuntimeError("BTC missing from panel")
    return int(btc_rows.groupby("id").size().sort_values(ascending=False).index[0])


def features_for_id(df: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    """Raw (pre-z) price + new-data features for one id. Momentum/trend/distance on close/BTC."""
    d = df.sort_values("date").copy()
    d = d.set_index("date")
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    dv = d["dv"] if "dv" in d.columns else d["volume"]
    mcap = d["mcap"] if "mcap" in d.columns else d.get("marketCap", pd.Series(np.nan, index=d.index))
    btc = btc_close.reindex(d.index).ffill()
    c_btc = c / btc.replace(0, np.nan)
    h_btc = h / btc.replace(0, np.nan)
    l_btc = l / btc.replace(0, np.nan)
    r1 = np.log(c / c.shift(1))
    btc_r = np.log(btc / btc.shift(1))

    out = pd.DataFrame(index=d.index)
    # momentum / trend / distance on BTC-denominated price
    out["ret_14"] = _log_ret(c_btc, 14)
    out["ret_56"] = _log_ret(c_btc, 56)
    out["ret_90"] = _log_ret(c_btc, 90)
    out["mom_90_skip14"] = np.log(c_btc.shift(14) / c_btc.shift(90))
    sma20 = c_btc.rolling(20, min_periods=10).mean()
    sma50 = c_btc.rolling(50, min_periods=20).mean()
    sma100 = c_btc.rolling(100, min_periods=40).mean()
    ema12 = c_btc.ewm(span=12, adjust=False).mean()
    ema26 = c_btc.ewm(span=26, adjust=False).mean()
    out["close_sma20"] = c_btc / sma20 - 1.0
    out["close_sma50"] = c_btc / sma50 - 1.0
    out["close_sma100"] = c_btc / sma100 - 1.0
    out["sma20_sma50"] = sma20 / sma50 - 1.0
    out["ema12_ema26"] = ema12 / ema26 - 1.0
    hh90 = h_btc.rolling(90, min_periods=30).max()
    ll90 = l_btc.rolling(90, min_periods=30).min()
    out["dist_high_90"] = c_btc / hh90 - 1.0
    out["dist_low_90"] = c_btc / ll90 - 1.0

    # vol / max / min / range / skew / beta on own USD returns
    out["yz_vol_14"] = _yang_zhang(o, h, l, c, 14)
    out["yz_vol_30"] = _yang_zhang(o, h, l, c, 30)
    out["yz_vol_60"] = _yang_zhang(o, h, l, c, 60)
    out["pk_vol_14"] = _parkinson(h, l, 14)
    out["vol_ratio"] = out["yz_vol_14"] / out["yz_vol_60"].replace(0, np.nan)
    out["vol_of_vol_30"] = out["yz_vol_14"].rolling(30, min_periods=10).std()
    out["max_ret_14"] = r1.rolling(14, min_periods=5).max()
    out["min_ret_14"] = r1.rolling(14, min_periods=5).min()
    hh28 = h.rolling(28, min_periods=10).max()
    ll28 = l.rolling(28, min_periods=10).min()
    out["range_pos_28"] = (c - ll28) / (hh28 - ll28).replace(0, np.nan)
    out["skew_60"] = r1.rolling(60, min_periods=20).skew()
    beta, idio = _rolling_ols_beta_resid(r1, btc_r, 60)
    out["beta_btc_60"] = beta
    out["idio_vol_60"] = idio
    out["corr_btc_28"] = r1.rolling(28, min_periods=10).corr(btc_r)
    out["amihud_14"] = (r1.abs() / dv.replace(0, np.nan)).rolling(14, min_periods=5).mean()

    # new-data (raw; rank/Δrank filled later)
    out["log_mcap"] = np.log(mcap.replace(0, np.nan))
    first = d.index.min()
    age_days = (d.index - first).days.astype(float)
    out["log_age"] = np.log1p(age_days)
    ath = c.expanding(min_periods=1).max()
    out["dist_ath"] = c / ath.replace(0, np.nan) - 1.0
    turn = dv / mcap.replace(0, np.nan)
    out["turnover"] = turn
    tmu = turn.rolling(30, min_periods=10).mean()
    tsd = turn.rolling(30, min_periods=10).std(ddof=0)
    out["turnover_z30"] = (turn - tmu) / tsd.replace(0, np.nan)

    out["id"] = int(df["id"].iloc[0])
    out["symbol"] = str(df["symbol"].iloc[0])
    out["close"] = c
    out["dv"] = dv
    out["mcap"] = mcap
    out["ret_7_raw"] = c / c.shift(7) - 1.0
    out["r1"] = r1
    out["yz_vol_30_raw"] = out["yz_vol_30"]
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    return out


def apply_cs_zscore(feat: pd.DataFrame, cols: list[str], clip: float = CS_CLIP) -> pd.DataFrame:
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


def build_price_new_panel(panel: pd.DataFrame, btc_id: int, keep_ids: set[int]) -> pd.DataFrame:
    btc = panel.loc[panel["id"] == btc_id].sort_values("date").set_index("date")["close"]
    btc.index = _as_utc_idx(btc.index)
    sub = panel[panel["id"].isin(keep_ids | {btc_id})].copy()
    sub["date"] = pd.to_datetime(sub["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    parts = []
    ids = sorted(int(i) for i in sub["id"].unique())
    n = len(ids)
    for i, iid in enumerate(ids, start=1):
        g = sub.loc[sub["id"] == iid]
        if len(g) < 50:
            continue
        parts.append(features_for_id(g, btc))
        if i % 50 == 0 or i == n:
            print(f"[HB] features {i}/{n} ids parts={len(parts)}", flush=True)
    if not parts:
        raise RuntimeError("no feature rows")
    feat = pd.concat(parts, ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    # mcap rank among all non-stable names present that day (pre CS-z)
    rank_src = panel.copy()
    rank_src["date"] = pd.to_datetime(rank_src["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    rank_src = rank_src[~rank_src["symbol"].str.upper().isin(STABLE_OR_WRAP)]
    rank_src["mcap_rank"] = rank_src.groupby("date")["mcap"].rank(ascending=False, method="first")
    rk = rank_src[["date", "id", "mcap_rank"]].drop_duplicates(["date", "id"])
    feat = feat.merge(rk, on=["date", "id"], how="left")
    feat = feat.sort_values(["id", "date"])
    feat["d_rank_30"] = feat.groupby("id")["mcap_rank"].diff(30)
    feat["d_rank_90"] = feat.groupby("id")["mcap_rank"].diff(90)
    return feat


def _map_date(dates: pd.Series, ser: pd.Series) -> pd.Series:
    if ser is None or len(ser) == 0:
        return pd.Series(np.nan, index=dates.index)
    s = ser.copy()
    s.index = _as_utc_idx(s.index)
    return pd.to_datetime(dates, utc=True).dt.tz_convert("UTC").dt.normalize().map(s)


def build_context_block(
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    pit100: pd.DataFrame,
    pit50: pd.DataFrame,
    btc_id: int,
    clip: float = CS_CLIP,
) -> pd.DataFrame:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit100 = pit100.copy()
    pit50 = pit50.copy()
    pit100["date"] = pd.to_datetime(pit100["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit50["date"] = pd.to_datetime(pit50["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit100["id"] = pit100["id"].astype(int)
    pit50["id"] = pit50["id"].astype(int)

    f = feat.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    r1 = f[["date", "id", "r1"]].dropna()
    btc_r = f.loc[f["id"] == btc_id, ["date", "r1"]].rename(columns={"r1": "r_btc"})
    r1 = r1.merge(btc_r, on="date", how="left")
    r1["excess_1"] = r1["r1"] - r1["r_btc"]

    r100 = r1.merge(pit100[["date", "id"]], on=["date", "id"], how="inner")
    cs_std = r100.groupby("date")["excess_1"].std(ddof=0).sort_index()
    ctx_disp = cs_std.rolling(30, min_periods=10).mean()

    # 30d excess log-return CS std (PIT top-100)
    close = p.pivot(index="date", columns="id", values="close").sort_index()
    if btc_id not in close.columns:
        raise RuntimeError("BTC id missing in close pivot")
    logp = np.log(close.clip(lower=1e-18))
    ex30 = logp.diff(30).sub(logp[btc_id].diff(30), axis=0)
    members100 = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit100.groupby("date")["id"]
    }
    xs, ys = [], []
    for dt, ids in members100.items():
        if dt not in ex30.index:
            continue
        cols = [i for i in ids if i in ex30.columns]
        row = ex30.loc[dt, cols].astype(float)
        row = row[np.isfinite(row)]
        xs.append(dt)
        ys.append(float(row.std(ddof=0)) if len(row) >= 5 else np.nan)
    ctx_excess_disp = pd.Series(ys, index=pd.DatetimeIndex(xs)).sort_index()

    btc_feat = f.loc[f["id"] == btc_id].set_index("date").sort_index()
    ctx_btc_vol = btc_feat["yz_vol_30_raw"]
    btc_c = close[btc_id]
    sma100 = btc_c.rolling(100, min_periods=40).mean()
    ctx_btc_trend = btc_c / sma100 - 1.0

    p = p.sort_values(["id", "date"])
    p["sma50"] = p.groupby("id", sort=False)["close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    p["above"] = p["close"] > p["sma50"]
    b100 = p.merge(pit100[["date", "id"]], on=["date", "id"], how="inner")
    ctx_breadth = b100.groupby("date")["above"].mean().sort_index()

    print("[HB] context: PIT top-50 mean pairwise 28d corr...", flush=True)
    wide_r = f.pivot(index="date", columns="id", values="r1").sort_index()
    wide_r.index = _as_utc_idx(wide_r.index)
    members50 = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit50.groupby("date")["id"]
    }
    dates50 = pd.DatetimeIndex(sorted(members50)).tz_convert("UTC").normalize()
    corr_rows = []
    for i, dt in enumerate(dates50):
        ids = [x for x in members50.get(dt, []) if x in wide_r.columns]
        if len(ids) < 5:
            corr_rows.append((dt, np.nan))
            continue
        loc = wide_r.index.searchsorted(dt, side="right")
        win = wide_r.iloc[max(0, loc - 28) : loc][ids]
        if len(win) < 15:
            corr_rows.append((dt, np.nan))
            continue
        cmat = win.corr()
        tri = cmat.values[np.triu_indices(len(cmat), k=1)]
        tri = tri[np.isfinite(tri)]
        corr_rows.append((dt, float(np.mean(tri)) if len(tri) else np.nan))
        if i % 400 == 0:
            print(f"[HB] ctx_corr {i}/{len(dates50)}", flush=True)
    ctx_corr = pd.Series({d: v for d, v in corr_rows}).sort_index()

    # EW top-50 / BTC ratio vs 90d SMA
    ratio_rows = []
    for dt, ids in members50.items():
        if dt not in close.index:
            continue
        cols = [i for i in ids if i in close.columns and i != btc_id]
        if not cols or not np.isfinite(close.loc[dt, btc_id]):
            continue
        px = close.loc[dt, cols].astype(float)
        px = px[np.isfinite(px) & (px > 0)]
        b = float(close.loc[dt, btc_id])
        if b <= 0 or px.empty:
            continue
        ratio_rows.append((dt, float((px / b).mean())))
    ratio = pd.Series({d: v for d, v in ratio_rows}).sort_index()
    sma90 = ratio.rolling(90, min_periods=30).mean()
    ctx_alt_btc_trend = ratio / sma90 - 1.0

    idx = pd.DatetimeIndex(
        sorted(set(pit100["date"]).union(set(pit50["date"])))
    ).tz_convert("UTC").normalize()
    out = pd.DataFrame({"date": idx})
    out["ctx_disp"] = _map_date(out["date"], ctx_disp)
    out["ctx_excess_disp"] = _map_date(out["date"], ctx_excess_disp)
    out["ctx_btc_vol"] = _map_date(out["date"], ctx_btc_vol)
    out["ctx_btc_trend"] = _map_date(out["date"], ctx_btc_trend)
    out["ctx_breadth"] = _map_date(out["date"], ctx_breadth)
    out["ctx_corr"] = _map_date(out["date"], ctx_corr)
    out["ctx_alt_btc_trend"] = _map_date(out["date"], ctx_alt_btc_trend)
    out = out.sort_values("date")
    for c in CTX_COLS:
        out[c] = _own_z_250(out[c].astype(float), clip=clip)
    print(f"[HB] context rows={len(out)}", flush=True)
    return out


def assemble_feature_table(
    panel: pd.DataFrame,
    pit100: pd.DataFrame,
    pit50: pd.DataFrame,
    btc_id: int,
    clip: float = CS_CLIP,
) -> pd.DataFrame:
    pit100 = pit100.copy()
    pit100["date"] = pd.to_datetime(pit100["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit100["id"] = pit100["id"].astype(int)
    keep = set(int(x) for x in pit100["id"].unique())
    print(f"[HB] price/new features on n_ids={len(keep)+1} (pit100 union + BTC)", flush=True)
    feat = build_price_new_panel(panel, btc_id, keep)
    ctx = build_context_block(panel, feat, pit100, pit50, btc_id, clip=clip)
    feat = feat.merge(ctx, on="date", how="left")
    # restrict CS-z to PIT top-100 membership (training universe)
    key = feat.merge(pit100[["date", "id"]], on=["date", "id"], how="inner")
    zcols = [c for c in (PRICE_COLS + NEW_COLS) if c in key.columns]
    key = apply_cs_zscore(key, zcols, clip=clip)
    # context already own-z'd; leave as-is
    print(f"[HB] feature table rows={len(key)} n_feat={len(FEATURE_COLS_V1)}", flush=True)
    missing = [c for c in FEATURE_COLS_V1 if c not in key.columns]
    if missing:
        raise RuntimeError(f"missing feature cols: {missing}")
    return key


def assemble_stage_s_features(
    panel: pd.DataFrame,
    pit100: pd.DataFrame,
    btc_id: int,
    clip: float = CS_CLIP,
) -> pd.DataFrame:
    """Per-coin features only. Context columns are not computed (Stage S law)."""
    from btcb.constants import CTX_COLS, STAGE_S_COLS

    pit100 = pit100.copy()
    pit100["date"] = pd.to_datetime(pit100["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit100["id"] = pit100["id"].astype(int)
    keep = set(int(x) for x in pit100["id"].unique())
    print(f"[HB] Stage-S features on n_ids={len(keep)+1} (no context)", flush=True)
    feat = build_price_new_panel(panel, btc_id, keep)
    key = feat.merge(pit100[["date", "id"]], on=["date", "id"], how="inner")
    zcols = [c for c in STAGE_S_COLS if c in key.columns]
    key = apply_cs_zscore(key, zcols, clip=clip)
    leaked = [c for c in CTX_COLS if c in key.columns]
    if leaked:
        raise RuntimeError(f"context leaked into Stage S: {leaked}")
    missing = [c for c in STAGE_S_COLS if c not in key.columns]
    if missing:
        raise RuntimeError(f"missing Stage-S cols: {missing}")
    print(f"[HB] Stage-S rows={len(key)} n_feat={len(STAGE_S_COLS)}", flush=True)
    return key


__all__ = [
    "FEATURE_COLS_V1",
    "PRICE_COLS",
    "CTX_COLS",
    "NEW_COLS",
    "PRICE_MOM_TREND_DIST",
    "assemble_feature_table",
    "btc_id_from_panel",
    "features_for_id",
]
