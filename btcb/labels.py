"""Binary excess-vs-BTC labels. y=1 iff h-day forward log-return exceeds BTC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import PHASE2_HORIZONS


def add_binary_excess_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    btc_id: int,
    horizons: tuple[int, ...] = PHASE2_HORIZONS,
) -> pd.DataFrame:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = p.pivot(index="date", columns="id", values="close").sort_index()
    if btc_id not in close.columns:
        raise RuntimeError("BTC missing for labels")
    out = feat.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    logp = np.log(close.clip(lower=1e-18))
    for h in horizons:
        fwd = logp.shift(-h) - logp
        btc_fwd = fwd[btc_id]
        excess = fwd.sub(btc_fwd, axis=0)
        long = excess.stack().rename("excess").reset_index()
        long.columns = ["date", "id", f"excess_h{h}"]
        long["date"] = pd.to_datetime(long["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        long["id"] = long["id"].astype(int)
        out = out.merge(long, on=["date", "id"], how="left")
        out[f"y_h{h}"] = (out[f"excess_h{h}"] > 0).astype(float)
        out.loc[out[f"excess_h{h}"].isna(), f"y_h{h}"] = np.nan
    return out


def add_quintile_excess_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    btc_id: int,
    horizons: tuple[int, ...] = PHASE2_HORIZONS,
    q: float = 0.80,
) -> pd.DataFrame:
    """y=1 iff h-day excess is in the top quintile of that date's cross-section (feat rows)."""
    out = add_binary_excess_labels(feat, panel, btc_id, horizons=horizons)
    for h in horizons:
        ex = f"excess_h{h}"
        yq = f"y_h{h}"

        def _topq(s: pd.Series) -> pd.Series:
            m = s.notna()
            if int(m.sum()) < 10:
                return pd.Series(np.nan, index=s.index)
            thr = np.nanquantile(s.to_numpy(), float(q))
            out_s = pd.Series(np.nan, index=s.index)
            out_s[m] = (s[m] >= thr).astype(float)
            return out_s

        out[yq] = out.groupby("date", sort=False)[ex].transform(_topq)
    return out


def add_twin_quintile_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    btc_id: int,
    horizons: tuple[int, ...] = PHASE2_HORIZONS,
    q_top: float = 0.80,
    q_bot: float = 0.20,
) -> pd.DataFrame:
    """Top and bottom quintile labels on the same date's feat cross-section."""
    out = add_quintile_excess_labels(feat, panel, btc_id, horizons=horizons, q=q_top)
    for h in horizons:
        ex = f"excess_h{h}"
        yb = f"y_bot_h{h}"

        def _botq(s: pd.Series) -> pd.Series:
            m = s.notna()
            if int(m.sum()) < 10:
                return pd.Series(np.nan, index=s.index)
            thr = np.nanquantile(s.to_numpy(), float(q_bot))
            out_s = pd.Series(np.nan, index=s.index)
            out_s[m] = (s[m] <= thr).astype(float)
            return out_s

        out[yb] = out.groupby("date", sort=False)[ex].transform(_botq)
    return out


def add_rank_grade_labels(
    feat: pd.DataFrame,
    horizon: int = 14,
    n_grades: int = 5,
) -> pd.DataFrame:
    """Integer 0..n_grades-1 within-date rank buckets of excess (higher = better)."""
    out = feat.copy()
    ex = f"excess_h{horizon}"
    ycol = f"y_rank_h{horizon}"
    if ex not in out.columns:
        raise RuntimeError(f"missing {ex} for rank-grade labels")
    n_grades = int(n_grades)

    def _grades(s: pd.Series) -> pd.Series:
        m = s.notna() & np.isfinite(s.to_numpy())
        out_s = pd.Series(np.nan, index=s.index)
        if int(m.sum()) < 10:
            return out_s
        pct = s[m].rank(method="average", pct=True)
        g = np.minimum((pct.to_numpy(dtype=float) * n_grades).astype(int), n_grades - 1)
        out_s.loc[m] = g.astype(float)
        return out_s

    out[ycol] = out.groupby("date", sort=False)[ex].transform(_grades)
    return out
