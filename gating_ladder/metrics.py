"""Shared metrics for the gating ladder harness."""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, summarize_ic, quintile_stats
from baseline.portfolio import _sharpe


TRAIL_DAYS = 548  # house 18m window (~365*1.5); Round F tables use ~548
ROUND_F_TOP20_H7_IC = 0.0923
ROUND_F_TOP20_H7_N_DAYS = 875
IC_TOL = 0.003


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:
        return f"UNKNOWN:{e}"


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    return obj


def maxdd_duration(equity: pd.Series) -> dict:
    eq = equity.astype(float)
    if eq.empty:
        return {"max_drawdown": float("nan"), "dd_duration_days": float("nan")}
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")
    underwater = dd < 0
    dur = 0
    best = 0
    for flag in underwater.tolist():
        dur = dur + 1 if flag else 0
        best = max(best, dur)
    return {"max_drawdown": max_dd, "dd_duration_days": int(best)}


def trail_mask(index: pd.DatetimeIndex, trail_days: int = TRAIL_DAYS) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return np.array([], dtype=bool)
    end = idx.max()
    start = end - pd.Timedelta(days=int(trail_days))
    mask = idx >= start
    return np.asarray(mask, dtype=bool)


def ic_bundle(pred: pd.DataFrame, ycol: str, horizon: int, universe: pd.DataFrame | None, label: str) -> dict:
    df = pred.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if universe is not None and not universe.empty:
        u = universe.copy()
        u["date"] = pd.to_datetime(u["date"], utc=True)
        df = df.merge(u[["date", "symbol"]], on=["date", "symbol"], how="inner")
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
    else:
        summary["quintile_means"] = {}
        summary["top_minus_bottom"] = float("nan")
    if len(ic):
        m = trail_mask(ic.index)
        ic_t = ic.iloc[m] if m.size == len(ic) else ic
        trail = summarize_ic(ic_t, horizon)
        summary["trail18m_mean_ic"] = trail.get("mean_ic")
        summary["trail18m_nw_tstat"] = trail.get("nw_tstat")
        summary["trail18m_n_days"] = trail.get("n_days")
        summary["trail18m_icir"] = trail.get("icir")
    summary["ic_dates"] = [str(d.date()) for d in ic.index[:3]] + (["..."] if len(ic) > 3 else [])
    return summary, ic


def decile_spread(
    pred: pd.DataFrame,
    ycol: str,
    panel: pd.DataFrame,
    fee_bps: float,
    slip_bps: float,
) -> dict:
    """Cross-sectional decile LS on next-bar simple-approx via label mean (gross)
    and a daily equal-weight D10-D1 book on close-to-close (net of costs).
    """
    df = pred.dropna(subset=["score", ycol]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    gross_rows = []
    for dt, g in df.groupby("date", sort=True):
        if len(g) < 10:
            continue
        try:
            q = pd.qcut(g["score"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = g.copy()
        g["q"] = q
        if g["q"].nunique() < 2:
            continue
        lo, hi = int(g["q"].min()), int(g["q"].max())
        gross_rows.append({"date": dt, "spread": float(g.loc[g["q"] == hi, ycol].mean() - g.loc[g["q"] == lo, ycol].mean())})
    gross = pd.DataFrame(gross_rows)
    gross_mean = float(gross["spread"].mean()) if len(gross) else float("nan")

    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True)
    rets = close.pct_change()
    cost_rate = (float(fee_bps) + float(slip_bps)) * 1e-4
    dates = sorted(df["date"].unique())
    prev_w = pd.Series(dtype=float)
    nets = []
    to_list = []
    for i, dt in enumerate(dates[:-1]):
        g = df[df["date"] == dt]
        if len(g) < 10:
            continue
        try:
            q = pd.qcut(g["score"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = g.copy()
        g["q"] = q
        lo, hi = int(g["q"].min()), int(g["q"].max())
        long_s = g.loc[g["q"] == hi, "symbol"]
        short_s = g.loc[g["q"] == lo, "symbol"]
        w = pd.Series(0.0, index=pd.Index(sorted(set(long_s) | set(short_s))))
        if len(long_s):
            w.loc[list(long_s)] = 0.5 / len(long_s)
        if len(short_s):
            w.loc[list(short_s)] = -0.5 / len(short_s)
        idx = w.index.union(prev_w.index)
        turnover = 0.5 * float((w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())
        nxt = dates[i + 1]
        nxt = pd.Timestamp(nxt)
        if nxt.tzinfo is None:
            nxt = nxt.tz_localize("UTC")
        gross_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in w.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    gross_r += float(wi) * float(rrow[s])
        net = gross_r - turnover * cost_rate
        nets.append(net)
        to_list.append(turnover)
        prev_w = w
    net_s = pd.Series(nets, dtype=float)
    return {
        "decile_spread_gross_mean_y": gross_mean,
        "decile_spread_net_mean": float(net_s.mean()) if len(net_s) else float("nan"),
        "decile_spread_net_sharpe": _sharpe(net_s) if len(net_s) else float("nan"),
        "decile_turnover_mean": float(np.mean(to_list)) if to_list else float("nan"),
        "n_days_net": int(len(net_s)),
    }


def slim_portfolio(res: dict) -> dict:
    drop = {
        "equity", "daily_ret", "daily_gross", "daily_hedge", "daily_cost",
        "daily_funding", "sym_contrib", "side_days", "daily_flat",
    }
    out = {k: v for k, v in res.items() if k not in drop}
    daily = res.get("daily_ret")
    if isinstance(daily, pd.Series) and len(daily):
        eq = (1.0 + daily.fillna(0.0)).cumprod()
        dd = maxdd_duration(eq)
        out["max_drawdown"] = dd["max_drawdown"]
        out["dd_duration_days"] = dd["dd_duration_days"]
        m = trail_mask(daily.index)
        trail = daily.iloc[m] if m.size == len(daily) else daily
        out["net_sharpe_trail18m"] = _sharpe(trail)
        out["n_days_trail18m"] = int(len(trail))
    return json_safe(out)
