"""Bucket curves, tail spreads, tide series, and classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from baseline.evaluate import newey_west_t
from longonly.eval import _as_utc, alpha_full_and_trail
from phase_d2.metrics import window_slice
from symmetry.constants import (
    CLASSIFICATION_CRITERION,
    N_UNIVERSES_MIN,
    ROLLING_DAYS,
    SYMMETRY_RATIO_MIN,
    TOP_NW_T_MIN,
)


def middle_buckets(n_buckets: int) -> list[int]:
    if n_buckets % 2 == 1:
        return [(n_buckets + 1) // 2]
    return [n_buckets // 2, n_buckets // 2 + 1]


def assign_buckets(scores: pd.Series, n_buckets: int) -> pd.Series | None:
    s = pd.Series(scores).astype(float)
    s = s[np.isfinite(s)]
    if len(s) < n_buckets:
        return None
    try:
        b = pd.qcut(s.rank(method="first"), n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return None
    out = (b.astype(int) + 1).reindex(scores.index)
    if int(out.dropna().nunique()) < n_buckets:
        return None
    return out


def _window_mask(idx: pd.DatetimeIndex, window: str) -> np.ndarray:
    dummy = pd.Series(1.0, index=idx)
    sl = window_slice(dummy, window)
    return idx.isin(sl.index)


def daily_bucket_panel(df: pd.DataFrame, ycol: str, n_buckets: int) -> pd.DataFrame:
    """One row per (date, bucket): mean residual, hit rate, n."""
    rows: list[dict] = []
    for dt, g in df.groupby("date", sort=True):
        gg = g.dropna(subset=["score", ycol])
        if len(gg) < n_buckets:
            continue
        b = assign_buckets(gg["score"], n_buckets)
        if b is None:
            continue
        gg = gg.assign(bucket=b.values)
        yv = gg[ycol].astype(float)
        for bk, sub in gg.groupby("bucket"):
            yy = sub[ycol].astype(float)
            rows.append(
                {
                    "date": dt,
                    "bucket": int(bk),
                    "mean_y": float(yy.mean()) if len(yy) else float("nan"),
                    "hit_rate": float((yy > 0).mean()) if len(yy) else float("nan"),
                    "n": int(len(yy)),
                }
            )
        rows.append(
            {
                "date": dt,
                "bucket": 0,  # 0 = cross-sectional mean (all names that day)
                "mean_y": float(yv.mean()) if len(yv) else float("nan"),
                "hit_rate": float((yv > 0).mean()) if len(yv) else float("nan"),
                "n": int(len(yv)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "bucket", "mean_y", "hit_rate", "n"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out


def daily_spreads(panel: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    """TOP and BOTTOM spreads, plus top/mid/bot/cs means, one row per date."""
    mid = middle_buckets(n_buckets)
    top_id, bot_id = n_buckets, 1
    recs = []
    for dt, g in panel.groupby("date", sort=True):
        by = {int(r.bucket): float(r.mean_y) for _, r in g.iterrows()}
        if top_id not in by or bot_id not in by or not all(m in by for m in mid):
            continue
        if 0 not in by:
            continue
        top_m = by[top_id]
        bot_m = by[bot_id]
        mid_m = float(np.mean([by[m] for m in mid]))
        recs.append(
            {
                "date": dt,
                "top_mean": top_m,
                "mid_mean": mid_m,
                "bot_mean": bot_m,
                "cs_mean": by[0],
                "top_spread": top_m - mid_m,
                "bot_spread": mid_m - bot_m,
            }
        )
    if not recs:
        return pd.DataFrame()
    out = pd.DataFrame(recs)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values("date").set_index("date")
    return out


def summarize_spread_series(s: pd.Series, lag: int) -> dict:
    s = _as_utc(s).dropna()
    years = sorted({int(y) for y in s.index.year.unique() if y >= 2022})
    windows = ["full", "trail18m"] + [f"y{y}" for y in years]

    def _one(window: str) -> dict:
        sl = window_slice(s, window) if window != "full" else s
        sl = sl.dropna()
        mu = float(sl.mean()) if len(sl) else float("nan")
        return {
            "mean": mu,
            "nw_t": float(newey_west_t(sl.to_numpy(), lag=lag)) if len(sl) else float("nan"),
            "n": int(len(sl)),
        }

    return {w: _one(w) for w in windows}


def bucket_curve_for_window(panel: pd.DataFrame, n_buckets: int, window: str) -> dict:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    idx = pd.DatetimeIndex(p["date"].unique())
    if window != "full":
        keep = set(window_slice(pd.Series(1.0, index=idx.sort_values()), window).index)
        p = p[p["date"].isin(keep)]
    real = p[p["bucket"] > 0]
    if real.empty:
        return {"buckets": [], "spearman": float("nan"), "cs_mean": float("nan")}
    g = real.groupby("bucket").agg(mean_y=("mean_y", "mean"), hit_rate=("hit_rate", "mean"), n=("n", "sum"))
    buckets = [
        {
            "bucket": int(i),
            "mean_y": float(g.loc[i, "mean_y"]),
            "hit_rate": float(g.loc[i, "hit_rate"]),
            "n": int(g.loc[i, "n"]),
        }
        for i in range(1, n_buckets + 1)
        if i in g.index
    ]
    if len(buckets) >= 3:
        ranks = np.asarray([b["bucket"] for b in buckets], dtype=float)
        means = np.asarray([b["mean_y"] for b in buckets], dtype=float)
        rho, _ = stats.spearmanr(ranks, means)
        spear = float(rho) if np.isfinite(rho) else float("nan")
    else:
        spear = float("nan")
    cs = p[p["bucket"] == 0]["mean_y"]
    return {
        "buckets": buckets,
        "spearman": spear,
        "cs_mean": float(cs.mean()) if len(cs) else float("nan"),
        "n_days": int(p["date"].nunique()),
    }


def cell_spreads_and_label(spreads: pd.DataFrame, lag: int) -> dict:
    top = summarize_spread_series(spreads["top_spread"], lag)
    bot = summarize_spread_series(spreads["bot_spread"], lag)
    years = sorted({int(y) for y in spreads.index.year.unique() if y >= 2022})
    windows = ["full", "trail18m"] + [f"y{y}" for y in years]
    ratios = {}
    for w in windows:
        tm = top[w]["mean"]
        bm = bot[w]["mean"]
        if np.isfinite(tm) and np.isfinite(bm) and abs(bm) > 1e-15:
            ratios[w] = float(tm / bm)
        else:
            ratios[w] = float("nan")
    tfull = top["full"]
    rfull = ratios["full"]
    passed = bool(
        np.isfinite(tfull["mean"])
        and tfull["mean"] > 0
        and np.isfinite(tfull["nw_t"])
        and tfull["nw_t"] >= TOP_NW_T_MIN
        and np.isfinite(rfull)
        and rfull >= SYMMETRY_RATIO_MIN
    )
    return {
        "top": top,
        "bottom": bot,
        "ratio": ratios,
        "pass": passed,
        "need_top_positive": True,
        "need_nw_t": TOP_NW_T_MIN,
        "need_ratio": SYMMETRY_RATIO_MIN,
    }


def apply_classification(cells: dict) -> dict:
    """cells keyed (h, universe) -> cell_spreads_and_label output."""
    by_h = {7: [], 10: []}
    rows = []
    for (h, uni), blob in cells.items():
        ok = bool(blob.get("pass"))
        by_h[int(h)].append(ok)
        rows.append(
            {
                "horizon": int(h),
                "universe": uni,
                "pass": ok,
                "top_mean": blob.get("top", {}).get("full", {}).get("mean"),
                "top_nw_t": blob.get("top", {}).get("full", {}).get("nw_t"),
                "bot_mean": blob.get("bottom", {}).get("full", {}).get("mean"),
                "bot_nw_t": blob.get("bottom", {}).get("full", {}).get("nw_t"),
                "ratio": blob.get("ratio", {}).get("full"),
            }
        )
    n7 = int(sum(by_h[7]))
    n10 = int(sum(by_h[10]))
    ok = n7 >= N_UNIVERSES_MIN or n10 >= N_UNIVERSES_MIN
    return {
        "label": "SYMMETRIC" if ok else "LONG-SIDE GAP",
        "criterion": CLASSIFICATION_CRITERION,
        "n_pass_h7": n7,
        "n_pass_h10": n10,
        "need_universes": N_UNIVERSES_MIN,
        "pass": bool(ok),
        "rows": rows,
    }


def tide_tables(spreads: pd.DataFrame) -> dict:
    s = spreads.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
    top_pos = (s["top_mean"] > 0).astype(float)
    roll = top_pos.rolling(ROLLING_DAYS, min_periods=max(20, ROLLING_DAYS // 3)).mean()
    years = sorted({int(y) for y in s.index.year.unique() if y >= 2022})
    by_year = {}
    for y in years:
        sl = s[s.index.year == y]
        by_year[y] = {
            "cs_mean": float(sl["cs_mean"].mean()) if len(sl) else float("nan"),
            "top_mean": float(sl["top_mean"].mean()) if len(sl) else float("nan"),
            "bot_mean": float(sl["bot_mean"].mean()) if len(sl) else float("nan"),
            "pct_top_pos": float((sl["top_mean"] > 0).mean()) if len(sl) else float("nan"),
            "n": int(len(sl)),
        }
    return {
        "pct_top_pos_full": float(top_pos.mean()) if len(top_pos) else float("nan"),
        "pct_top_pos_trail18m": float(window_slice(top_pos, "trail18m").mean())
        if len(top_pos)
        else float("nan"),
        "by_year": by_year,
        "top_pos": top_pos,
        "roll90": roll,
        "cs_mean": s["cs_mean"],
        "top_mean": s["top_mean"],
        "bot_mean": s["bot_mean"],
    }


def ew_pit_simple(panel: pd.DataFrame, pit: pd.DataFrame, name: str = "ew") -> pd.Series:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    simple = close.pct_change()
    uni = pit.copy()
    uni["date"] = pd.to_datetime(uni["date"], utc=True)
    by = uni.groupby("date")["symbol"].apply(lambda s: list(s))
    close_idx = simple.index
    rows: dict[pd.Timestamp, float] = {}
    for dt in sorted(by.index):
        later = close_idx[close_idx > dt]
        if len(later) == 0:
            continue
        nxt = later[0]
        members = [s for s in by.loc[dt] if s in simple.columns]
        if not members:
            continue
        rows[nxt] = float(simple.loc[nxt, members].astype(float).mean())
    out = pd.Series(rows, dtype=float).sort_index()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    return out.rename(name)


def long_pick_quality(
    holdings: list[tuple],
    labeled: pd.DataFrame,
    ycol: str,
) -> dict:
    """Mean residual of held longs vs same-date CS mean on the labeled universe frame."""
    lab = labeled.copy()
    lab["date"] = pd.to_datetime(lab["date"], utc=True)
    by_date = {d: g for d, g in lab.groupby("date", sort=True)}
    recs = []
    for dt, names in holdings:
        dt = pd.Timestamp(dt)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        # holdings dates are nxt (return date); labels/scores are as-of signal date.
        # Join on the holdings date if present, else previous labeled date.
        g = by_date.get(dt)
        if g is None:
            continue
        g = g.dropna(subset=[ycol])
        if g.empty:
            continue
        cs = float(g[ycol].mean())
        held = g[g["symbol"].isin(set(names))]
        if held.empty:
            continue
        lm = float(held[ycol].mean())
        recs.append({"date": dt, "long_mean": lm, "cs_mean": cs, "excess": lm - cs, "n_long": int(len(held))})
    if not recs:
        return {
            "mean_long": float("nan"),
            "mean_cs": float("nan"),
            "mean_excess": float("nan"),
            "pct_excess_pos": float("nan"),
            "n_days": 0,
        }
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    years = sorted({int(y) for y in df.index.year.unique() if y >= 2022})
    by_year = {
        y: {
            "mean_long": float(df[df.index.year == y]["long_mean"].mean()),
            "mean_cs": float(df[df.index.year == y]["cs_mean"].mean()),
            "mean_excess": float(df[df.index.year == y]["excess"].mean()),
        }
        for y in years
    }
    return {
        "mean_long": float(df["long_mean"].mean()),
        "mean_cs": float(df["cs_mean"].mean()),
        "mean_excess": float(df["excess"].mean()),
        "pct_excess_pos": float((df["excess"] > 0).mean()),
        "n_days": int(len(df)),
        "by_year": by_year,
        "daily_excess": df["excess"],
    }


def lo_alpha_vs_benches(rets: pd.Series, benches: dict, lag: int) -> dict:
    out = {}
    for name, x in benches.items():
        out[name] = alpha_full_and_trail(rets, x, lag)
    return out
