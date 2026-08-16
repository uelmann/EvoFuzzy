"""Mechanical evaluation for LONG-CASH (viability vs cash, not vs BTC)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline.evaluate import newey_west_t
from longcash.constants import (
    ANNUALIZATION,
    AVG_GROSS_MIN,
    FULL_SHARPE_MIN,
    HORIZON,
    TRAIL_DAYS,
    TRAIL_SHARPE_MIN,
)


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) and x.std() > 0 else 0.0


def window_slice(s: pd.Series, window: str) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
    if window == "full":
        return s
    if window == "trail18m":
        if len(s.index) == 0:
            return s
        end = s.index.max()
        start = end - pd.Timedelta(days=int(TRAIL_DAYS))
        return s[(s.index >= start) & (s.index <= end)]
    if window.startswith("y"):
        y = int(window[1:])
        return s[s.index.year == y]
    raise ValueError(window)


def _as_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    return out


def cagr_maxdd(rets: pd.Series) -> tuple[float, float, float]:
    r = _as_utc(rets).fillna(0.0)
    if len(r) == 0:
        return float("nan"), float("nan"), float("nan")
    eq = (1.0 + r).cumprod()
    years = len(r) / 365.0
    cagr = float(eq.iloc[-1] ** (1.0 / max(years, 1e-6)) - 1.0) if len(r) > 1 else 0.0
    maxdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")
    total = float(eq.iloc[-1] - 1.0)
    return cagr, maxdd, total


def slim_port(res: dict) -> dict:
    drop = {
        "equity",
        "daily_ret",
        "daily_gross",
        "daily_hedge",
        "daily_cost",
        "daily_funding",
        "daily_n_pos",
        "daily_n_long",
        "daily_n_short",
        "daily_flat",
        "sym_contrib",
        "side_days",
        "daily_gross_deployed",
        "daily_btc_weight",
        "name_alpha_pnl",
    }
    return {k: v for k, v in res.items() if k not in drop}


def summarize_book(res: dict) -> dict:
    net = res.get("daily_ret")
    if net is None or not isinstance(net, pd.Series) or len(net) == 0:
        return {**slim_port(res), "error": res.get("error", "empty")}
    net = _as_utc(net).fillna(0.0)
    years = sorted({int(y) for y in net.index.year.unique() if y >= 2022})
    cagr, maxdd, total = cagr_maxdd(net)
    out = slim_port(res)
    out["n_days"] = int(len(net))
    out["net_sharpe_full"] = _sharpe(net)
    out["net_sharpe_trail18m"] = _sharpe(window_slice(net, "trail18m"))
    out["net_sharpe_by_year"] = {y: _sharpe(window_slice(net, f"y{y}")) for y in years}
    out["net_cagr"] = cagr
    out["max_drawdown"] = maxdd
    out["total_return"] = total
    out["avg_n_long"] = float(res.get("avg_n_long", float("nan")))
    out["avg_gross_deployed"] = float(res.get("avg_gross_deployed", float("nan")))
    out["pct_flat_days"] = float(res.get("pct_flat_days", float("nan")))
    out["ann_turnover"] = float(res.get("ann_turnover", float("nan")))
    out["funding_total_pnl"] = float(res.get("funding_total_pnl", float("nan")))
    out["cost_drag"] = float(res.get("cost_drag", float("nan")))
    out["gross_total_pnl"] = float(res.get("gross_total_pnl", float("nan")))
    out["net_total_pnl"] = float(res.get("net_total_pnl", float("nan")))
    out["mean_ann"] = float(net.mean() * ANNUALIZATION) if len(net) else float("nan")
    out["nw_t_vs_cash"] = newey_west_t(net.to_numpy(dtype=float), lag=HORIZON)
    trail = window_slice(net, "trail18m")
    out["nw_t_vs_cash_trail18m"] = newey_west_t(trail.to_numpy(dtype=float), lag=HORIZON) if len(trail) else float("nan")
    pnl = res.get("name_alpha_pnl") or {}
    top = sorted(pnl.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:5]
    out["top5_names"] = [{"symbol": s, "pnl": float(v)} for s, v in top]
    out["n_forced_exits"] = int(res.get("n_forced_exits", 0))
    out["forced_exit_pnl"] = float(res.get("forced_exit_pnl", 0.0))
    out["max_abs_btc_weight"] = float(res.get("max_abs_btc_weight", 0.0))
    out["btc_weight_identically_zero"] = bool(res.get("btc_weight_identically_zero", False))
    out["daily_ret"] = net
    return out


def viable(book: dict, null_verdict: str) -> dict:
    full = float(book.get("net_sharpe_full", float("nan")))
    trail = float(book.get("net_sharpe_trail18m", float("nan")))
    total = float(book.get("total_return", float("nan")))
    gross = float(book.get("avg_gross_deployed", float("nan")))
    btc0 = bool(book.get("btc_weight_identically_zero", False))
    null_ok = str(null_verdict).upper() == "GREEN"
    pass_full = bool(np.isfinite(full) and full >= FULL_SHARPE_MIN)
    pass_trail = bool(np.isfinite(trail) and trail >= TRAIL_SHARPE_MIN)
    pass_total = bool(np.isfinite(total) and total > 0)
    pass_gross = bool(np.isfinite(gross) and gross >= AVG_GROSS_MIN)
    ok = bool(pass_full and pass_trail and pass_total and pass_gross and btc0 and null_ok)
    return {
        "viable": ok,
        "verdict": "VIABLE" if ok else "NOT VIABLE",
        "pass_full": pass_full,
        "pass_trail": pass_trail,
        "pass_total": pass_total,
        "pass_gross": pass_gross,
        "pass_btc0": btc0,
        "pass_null": null_ok,
        "sharpe_full": full,
        "sharpe_trail18m": trail,
        "total_return": total,
        "avg_gross": gross,
        "null_verdict": null_verdict,
        "need_full": FULL_SHARPE_MIN,
        "need_trail": TRAIL_SHARPE_MIN,
        "need_gross": AVG_GROSS_MIN,
    }


def ew_topn_simple(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    r = close.pct_change()
    uni = universe.copy()
    uni["date"] = pd.to_datetime(uni["date"], utc=True)
    rows = []
    for dt, g in uni.groupby("date", sort=True):
        dt = pd.Timestamp(dt)
        if dt not in r.index:
            continue
        names = [s for s in g["symbol"].tolist() if s in r.columns and s != "BTCUSDT"]
        if not names:
            continue
        row = r.loc[dt, names]
        val = float(row.mean()) if row.notna().any() else 0.0
        rows.append((dt, val))
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)), dtype=float)


def btc_bh_simple(panel: pd.DataFrame) -> pd.Series:
    p = panel[panel["symbol"] == "BTCUSDT"].copy()
    if p.empty:
        return pd.Series(dtype=float)
    p["date"] = pd.to_datetime(p["date"], utc=True)
    p = p.sort_values("date")
    r = p.set_index("date")["close"].pct_change()
    return r.dropna()


def top_bucket_usd_stats(
    pred: pd.DataFrame,
    score_col: str,
    ycol: str,
    n_buckets: int = 5,
    lag: int = HORIZON,
) -> dict:
    """Per-date top-bucket mean of ycol; % days > 0 and NW-t. Informational."""
    daily = []
    for dt, g in pred.groupby("date", sort=True):
        gg = g.dropna(subset=[score_col, ycol])
        gg = gg[gg["symbol"] != "BTCUSDT"] if "symbol" in gg.columns else gg
        if len(gg) < n_buckets * 2:
            continue
        try:
            q = pd.qcut(gg[score_col], n_buckets, labels=False, duplicates="drop")
        except (ValueError, TypeError):
            continue
        qv = pd.to_numeric(pd.Series(q), errors="coerce")
        if qv.notna().sum() < n_buckets:
            continue
        top = qv.max()
        if not np.isfinite(top):
            continue
        mu = float(gg.loc[qv.to_numpy() == int(top), ycol].mean())
        if np.isfinite(mu):
            daily.append(mu)
    arr = np.asarray(daily, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return {"n_days": int(len(arr)), "pct_top_pos": float("nan"), "mean_top": float("nan"), "nw_t": float("nan")}
    return {
        "n_days": int(len(arr)),
        "pct_top_pos": float(np.mean(arr > 0.0)),
        "mean_top": float(np.mean(arr)),
        "nw_t": newey_west_t(arr, lag=lag),
    }


def null_verdict_from_cells(cells: list[dict]) -> dict:
    n_violate = sum(1 for c in cells if not c.get("bias_ok"))
    bias_pass = n_violate == 0 and bool(cells)
    skill_pass = bool(cells) and all(c.get("exceeds_p95") for c in cells)
    if not cells:
        verdict = "PARKED-NO-SKILL"
    elif not bias_pass:
        verdict = "CONTAMINATED"
    elif not skill_pass:
        verdict = "PARKED-NO-SKILL"
    else:
        verdict = "GREEN"
    return {
        "bias_pass": bool(bias_pass),
        "skill_pass": bool(skill_pass),
        "n_violate": int(n_violate),
        "n_folds": int(len(cells)),
        "verdict": verdict,
        "passed": bool(bias_pass and skill_pass),
        "cells": cells,
    }
