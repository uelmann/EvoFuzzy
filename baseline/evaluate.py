"""Model-agnostic evaluation: RankIC, ICIR, NW t-stat, quintiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = np.dot(e, e) / n
    var = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma = np.dot(e[l:], e[:-l]) / n
        var += 2 * w * gamma
    se = np.sqrt(max(var, 0.0) / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


def daily_rank_ic(pred: pd.DataFrame, ycol: str, score_col: str = "score") -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []
    for dt, g in pred.groupby("date", sort=True):
        gg = g.dropna(subset=[score_col, ycol])
        if len(gg) < 5:
            continue
        corr = stats.spearmanr(gg[score_col].values, gg[ycol].values).correlation
        if corr is None or not np.isfinite(corr):
            continue
        rows.append((pd.Timestamp(dt), float(corr)))
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)), dtype=float)


def summarize_ic(ic: pd.Series, horizon: int) -> dict:
    vals = ic.values.astype(float)
    return {
        "n_days": int(len(vals)),
        "mean_ic": float(np.mean(vals)) if len(vals) else float("nan"),
        "std_ic": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
        "icir": float(np.mean(vals) / np.std(vals, ddof=1) * np.sqrt(252))
        if len(vals) > 1 and np.std(vals, ddof=1) > 0
        else float("nan"),
        "nw_tstat": newey_west_t(vals, lag=horizon),
    }


def quintile_stats(pred: pd.DataFrame, ycol: str, score_col: str = "score") -> pd.DataFrame:
    rows = []
    for dt, g in pred.groupby("date", sort=True):
        g = g.dropna(subset=[score_col, ycol])
        if len(g) < 10:
            continue
        try:
            q = pd.qcut(g[score_col], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        tmp = g.copy()
        tmp["q"] = q
        for qi, gg in tmp.groupby("q"):
            rows.append({"date": dt, "quintile": int(qi) + 1, "ret": float(gg[ycol].mean())})
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    summary = long.groupby("quintile")["ret"].mean().reset_index()
    return summary


def evaluate_predictions(
    pred: pd.DataFrame,
    horizon: int,
    universe: pd.DataFrame | None = None,
    label: str = "full",
) -> dict:
    ycol = f"y_h{horizon}"
    df = pred.copy()
    if universe is not None and not universe.empty:
        u = universe.copy()
        u["date"] = pd.to_datetime(u["date"], utc=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.merge(u[["date", "symbol"]], on=["date", "symbol"], how="inner")
    if ycol not in df.columns:
        raise ValueError(f"missing {ycol}")
    ic = daily_rank_ic(df, ycol)
    summary = summarize_ic(ic, horizon)
    summary["universe"] = label
    summary["horizon"] = horizon
    q = quintile_stats(df, ycol)
    if not q.empty:
        summary["quintile_means"] = {int(r.quintile): float(r.ret) for _, r in q.iterrows()}
        top = q.loc[q["quintile"] == q["quintile"].max(), "ret"]
        bot = q.loc[q["quintile"] == q["quintile"].min(), "ret"]
        summary["top_minus_bottom"] = float(top.mean() - bot.mean()) if len(top) and len(bot) else float("nan")
        # monotonicity: correlation of quintile index with mean ret
        summary["monotonicity"] = float(np.corrcoef(q["quintile"], q["ret"])[0, 1]) if len(q) > 2 else float("nan")
    else:
        summary["quintile_means"] = {}
        summary["top_minus_bottom"] = float("nan")
        summary["monotonicity"] = float("nan")
    summary["ic_series"] = ic
    return summary


def naive_mom28_scores(feat: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Benchmark: per-date z of ret_28 as score (already z-scored in features)."""
    out = feat[["date", "symbol", "ret_28"]].dropna().copy()
    out = out.rename(columns={"ret_28": "score"})
    out["horizon"] = horizon
    out["model_name"] = "naive_mom28"
    # attach label if present
    ycol = f"y_h{horizon}"
    if ycol in feat.columns:
        out = out.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    return out
