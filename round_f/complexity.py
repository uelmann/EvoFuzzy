"""Complexity/regime features on trailing 90d residual log-returns (min 60)."""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pandas as pd

from round_f.constants import C22_COLS, CATCH22_NAMES, CX_COLS, EXTRA_CX_COLS

_PERM3 = list(permutations(range(3)))
_PERM3_IDX = {p: i for i, p in enumerate(_PERM3)}
_LOG6 = math.log(6.0)


def hurst_rs(x: np.ndarray) -> float:
    """Single-scale R/S Hurst: H = log(R/S) / log(n). Stated in the report."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return float("nan")
    y = x - x.mean()
    z = np.cumsum(y)
    r = float(z.max() - z.min())
    s = float(y.std(ddof=1))
    if s <= 1e-18 or r <= 0:
        return float("nan")
    return float(np.log(r / s) / np.log(n))


def variance_ratio_q5(x: np.ndarray, q: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return float("nan")
    y = x - x.mean()
    v1 = float(np.var(y, ddof=1))
    if v1 <= 1e-18:
        return float("nan")
    rq = np.convolve(y, np.ones(q), mode="valid")
    vq = float(np.var(rq, ddof=1))
    return float(vq / (q * v1))


def perm_entropy_order3(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return float("nan")
    counts = np.zeros(6, dtype=float)
    for i in range(n - 2):
        w = x[i : i + 3]
        key = tuple(np.argsort(w, kind="mergesort"))
        counts[_PERM3_IDX[key]] += 1.0
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    p = counts / tot
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / _LOG6)


def ar1_halflife(x: np.ndarray, cap: float = 90.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return float("nan")
    a, b = x[:-1], x[1:]
    va = float(np.var(a, ddof=1))
    if va <= 1e-18:
        return float("nan")
    phi = float(np.cov(a, b, ddof=1)[0, 1] / va)
    if not np.isfinite(phi) or phi <= 0.0 or phi >= 0.999:
        return float(cap)
    hl = math.log(0.5) / math.log(phi)
    if not np.isfinite(hl) or hl < 0:
        return float(cap)
    return float(min(cap, hl))


def _catch22_one(x: np.ndarray) -> dict[str, float]:
    out = {c: float("nan") for c in C22_COLS}
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return out
    try:
        import pycatch22

        res = pycatch22.catch22_all(x.tolist())
        names = res.get("names") if isinstance(res, dict) else None
        vals = res.get("values") if isinstance(res, dict) else None
        if names is None:
            return out
        m = {str(n): float(v) for n, v in zip(names, vals)}
        for raw, col in zip(CATCH22_NAMES, C22_COLS):
            if raw in m and np.isfinite(m[raw]):
                out[col] = m[raw]
            elif col[4:] in m and np.isfinite(m[col[4:]]):
                out[col] = m[col[4:]]
    except Exception:
        return out
    return out


def complexity_for_series(dates: np.ndarray, resid: np.ndarray, window: int = 90, min_obs: int = 60) -> pd.DataFrame:
    n = len(resid)
    rows = []
    for i in range(n):
        lo = max(0, i - window + 1)
        sl = resid[lo : i + 1]
        sl = sl[np.isfinite(sl)]
        rec = {"date": dates[i], "hurst_90": np.nan, "vr_5": np.nan, "perm_entropy_90": np.nan, "mr_halflife_90": np.nan}
        rec.update({c: np.nan for c in C22_COLS})
        if len(sl) >= min_obs:
            rec["hurst_90"] = hurst_rs(sl)
            rec["vr_5"] = variance_ratio_q5(sl)
            rec["perm_entropy_90"] = perm_entropy_order3(sl)
            rec["mr_halflife_90"] = ar1_halflife(sl)
            rec.update(_catch22_one(sl))
        rows.append(rec)
    return pd.DataFrame(rows)


def complexity_for_symbol(resid_sym: pd.DataFrame) -> pd.DataFrame:
    g = resid_sym.sort_values("date")
    dates = pd.to_datetime(g["date"], utc=True).to_numpy()
    x = g["resid"].to_numpy(dtype=float)
    out = complexity_for_series(dates, x)
    out["symbol"] = g["symbol"].iloc[0]
    return out


def apply_cs_z_cx(df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    out = df.copy()
    for col in CX_COLS:
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


def merge_complexity(feat: pd.DataFrame, cx: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    c = cx.copy()
    c["date"] = pd.to_datetime(c["date"], utc=True)
    cols = ["date", "symbol"] + [x for x in CX_COLS if x in c.columns]
    return f.merge(c[cols], on=["date", "symbol"], how="left")
