"""Magnitude diagnostics: top-10 mean excess and decile-mean-return curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfn_va.constants import TOP_K_PICKS


def _utc(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.normalize()


def attach_excess(scores: pd.DataFrame, labeled: pd.DataFrame, score_col: str, excess_col: str) -> pd.DataFrame:
    a = scores.copy()
    a["date"] = _utc(a["date"])
    a["id"] = a["id"].astype(int)
    b = labeled[["date", "id", excess_col]].copy()
    b["date"] = _utc(b["date"])
    b["id"] = b["id"].astype(int)
    out = a.merge(b, on=["date", "id"], how="inner")
    out = out.dropna(subset=[score_col, excess_col])
    return out


def topk_mean_excess(df: pd.DataFrame, score_col: str, excess_col: str, k: int = TOP_K_PICKS) -> dict:
    if df is None or df.empty:
        return {"mean": float("nan"), "n_dates": 0, "k": int(k)}
    rows = []
    for dt, g in df.groupby("date", sort=False):
        g = g.dropna(subset=[score_col, excess_col])
        if len(g) < max(4, int(k)):
            continue
        top = g.nlargest(int(k), score_col)
        rows.append(float(top[excess_col].mean()))
    if not rows:
        return {"mean": float("nan"), "n_dates": 0, "k": int(k)}
    return {"mean": float(np.mean(rows)), "n_dates": int(len(rows)), "k": int(k), "per_date": rows}


def decile_mean_curve(df: pd.DataFrame, score_col: str, excess_col: str, n_dec: int = 10) -> list[dict]:
    if df is None or df.empty:
        return []
    acc = {d: [] for d in range(1, n_dec + 1)}
    for dt, g in df.groupby("date", sort=False):
        g = g.dropna(subset=[score_col, excess_col])
        if len(g) < n_dec * 2:
            continue
        try:
            q = pd.qcut(g[score_col].rank(method="first"), n_dec, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        g = g.assign(_dec=q)
        for d, sub in g.groupby("_dec"):
            acc[int(d)].append(float(sub[excess_col].mean()))
    out = []
    for d in range(1, n_dec + 1):
        vals = acc[d]
        out.append({"decile": int(d), "mean_excess": float(np.mean(vals)) if vals else float("nan"), "n": int(len(vals))})
    return out


def yearly_score_stability(df: pd.DataFrame, score_col: str) -> list[dict]:
    if df is None or df.empty or score_col not in df.columns:
        return []
    a = df.copy()
    a["date"] = _utc(a["date"])
    a["year"] = a["date"].dt.year
    rows = []
    for y, g in a.groupby("year"):
        s = g[score_col].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if len(s) < 20:
            continue
        # cross-sectional: per date mean/std then average
        per = []
        for dt, gg in g.groupby("date"):
            v = gg[score_col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) < 8:
                continue
            per.append((float(np.mean(v)), float(np.std(v, ddof=0)), float(np.percentile(v, 10)), float(np.percentile(v, 90))))
        if not per:
            continue
        arr = np.asarray(per)
        rows.append(
            {
                "year": int(y),
                "n_dates": int(len(per)),
                "mean": float(arr[:, 0].mean()),
                "std": float(arr[:, 1].mean()),
                "p10": float(arr[:, 2].mean()),
                "p90": float(arr[:, 3].mean()),
            }
        )
    return rows
