"""Path signatures (SIG-A depth-4 2-d, SIG-B depth-3 3-d) — CPU, no lookahead."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from baseline.features import FEATURE_COLS

WINDOW = 60
MIN_WINDOW = 40
CLIP = 5.0

SIG_A_LEN = 8  # iisignature.logsiglength(2, 4)
SIG_B_LEN = 14  # iisignature.logsiglength(3, 3)
SIG_A_COLS = [f"sig_a_{i}" for i in range(SIG_A_LEN)]
SIG_B_COLS = [f"sig_b_{i}" for i in range(SIG_B_LEN)]
SIG_COLS = SIG_A_COLS + SIG_B_COLS


def _prepare():
    import iisignature

    return iisignature.prepare(2, 4), iisignature.prepare(3, 3), iisignature


def signatures_for_symbol(
    panel_sym: pd.DataFrame,
    feat_sym: pd.DataFrame,
    btc_ret: pd.Series,
    prep_a,
    prep_b,
    iisignature,
) -> pd.DataFrame:
    """Build raw (pre-CS-z) signatures for one symbol. Dates UTC."""
    p = panel_sym.sort_values("date").copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    p["log_close"] = np.log(p["close"].astype(float).replace(0, np.nan))
    p["r"] = p["log_close"].diff()
    beta_col = "beta_btc_60_raw" if "beta_btc_60_raw" in feat_sym.columns else "beta_btc_60"
    keep_cols = ["date"] + [c for c in (beta_col,) if c in feat_sym.columns]
    fmerge = feat_sym[keep_cols].copy() if keep_cols != ["date"] else feat_sym[["date"]].copy()
    p = p.merge(fmerge, on="date", how="left")
    if beta_col not in p.columns:
        p[beta_col] = np.nan
    btc_map = btc_ret.copy()
    btc_map.index = pd.to_datetime(btc_map.index, utc=True)
    p["r_btc"] = p["date"].map(btc_map)
    p["resid"] = p["r"] - p[beta_col].astype(float) * p["r_btc"].astype(float)
    # own-symbol dollar-volume z vs trailing 30d (NOT cross-sectional z)
    if "dollar_volume" in p.columns:
        dv = p["dollar_volume"].astype(float)
    else:
        dv = p["close"].astype(float) * p["volume"].astype(float)
    mu = dv.rolling(30, min_periods=10).mean()
    sd = dv.rolling(30, min_periods=10).std()
    p["dv_z"] = (dv - mu) / sd.replace(0, np.nan)

    n = len(p)
    dates = p["date"].to_numpy()
    resid = p["resid"].to_numpy(dtype=float)
    dvz = p["dv_z"].to_numpy(dtype=float)
    out_a = np.full((n, SIG_A_LEN), np.nan, dtype=np.float64)
    out_b = np.full((n, SIG_B_LEN), np.nan, dtype=np.float64)

    for t in range(n):
        start = max(0, t - WINDOW + 1)
        sl = slice(start, t + 1)
        rr = resid[sl]
        vv = dvz[sl]
        mask = np.isfinite(rr)
        if mask.sum() < MIN_WINDOW:
            continue
        rr = rr[mask]
        vv = np.where(np.isfinite(vv[mask]), vv[mask], 0.0)
        w = len(rr)
        vol = float(np.std(rr))
        if not np.isfinite(vol) or vol < 1e-12:
            continue
        shaped = rr / vol
        cum_r = np.cumsum(shaped)
        time_c = np.linspace(0.0, 1.0, w, dtype=np.float64)
        path_a = np.column_stack([time_c, cum_r])
        cum_v = np.cumsum(vv)
        path_b = np.column_stack([time_c, cum_r, cum_v])
        try:
            la = np.asarray(iisignature.logsig(path_a, prep_a), dtype=np.float64)
            lb = np.asarray(iisignature.logsig(path_b, prep_b), dtype=np.float64)
        except Exception:
            continue
        if la.size == SIG_A_LEN:
            out_a[t] = la
        if lb.size == SIG_B_LEN:
            out_b[t] = lb

    df = pd.DataFrame({"date": dates, "symbol": p["symbol"].iloc[0]})
    for i, c in enumerate(SIG_A_COLS):
        df[c] = out_a[:, i]
    for i, c in enumerate(SIG_B_COLS):
        df[c] = out_b[:, i]
    return df


def build_signature_panel(
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    clip: float = CLIP,
) -> pd.DataFrame:
    """All symbols; CS z-score per date, clip ±5. NaN preserved (no zero-fill)."""
    prep_a, prep_b, iisig = _prepare()
    panel = panel.copy()
    feat = feat.copy()
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)

    btc = panel[panel["symbol"] == "BTCUSDT"].sort_values("date")
    btc_ret = pd.Series(
        np.log(btc["close"].astype(float) / btc["close"].astype(float).shift(1)).values,
        index=pd.to_datetime(btc["date"], utc=True),
    )

    feat_keep_cols = ["date", "symbol"]
    if "beta_btc_60_raw" in feat.columns:
        feat_keep_cols.append("beta_btc_60_raw")
    elif "beta_btc_60" in feat.columns:
        feat_keep_cols.append("beta_btc_60")
    feat_keep = feat[feat_keep_cols].copy()

    symbols = sorted(set(feat["symbol"].unique()) | set(panel["symbol"].unique()))
    parts = []
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        p_s = panel[panel["symbol"] == sym]
        f_s = feat_keep[feat_keep["symbol"] == sym]
        if p_s.empty:
            continue
        parts.append(signatures_for_symbol(p_s, f_s, btc_ret, prep_a, prep_b, iisig))
        if i % 10 == 0 or i == len(symbols):
            print(
                f"[HB] {time.strftime('%H:%M:%S')} signatures {i}/{len(symbols)} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    if not parts:
        return pd.DataFrame(columns=["date", "symbol"] + SIG_COLS)
    raw = pd.concat(parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True)

    # Restrict to feature-panel keys
    keys = feat[["date", "symbol"]].drop_duplicates()
    keys["date"] = pd.to_datetime(keys["date"], utc=True)
    raw = keys.merge(raw, on=["date", "symbol"], how="left")

    out = raw[["date", "symbol"]].copy()
    for col in SIG_COLS:
        def _z(s: pd.Series, _col=col) -> pd.Series:
            mu = s.mean(skipna=True)
            sd = s.std(ddof=0, skipna=True)
            if not np.isfinite(sd) or sd == 0:
                return s * 0.0
            return ((s - mu) / sd).clip(-clip, clip)

        out[col] = raw.groupby("date", sort=False)[col].transform(_z)
        out.loc[raw[col].isna(), col] = np.nan
    return out


def merge_signatures(feat: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    s = sig.copy()
    s["date"] = pd.to_datetime(s["date"], utc=True)
    cols = ["date", "symbol"] + [c for c in SIG_COLS if c in s.columns]
    return f.merge(s[cols], on=["date", "symbol"], how="left")


def signature_feature_cols() -> list[str]:
    return list(FEATURE_COLS) + list(SIG_COLS)
