"""Mechanical evaluation for ALPHAMINE-LO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphamine.constants import (
    ANNUALIZATION,
    AVG_GROSS_MIN,
    BTC_SYMBOL,
    FULL_SHARPE_MIN,
    HORIZON,
    TRAIL_DAYS,
    TRAIL_SHARPE_MIN,
)
from baseline.evaluate import daily_rank_ic, newey_west_t, summarize_ic
from baseline.model import FoldSpec


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
    out["avg_n_long"] = float(res.get("avg_n_long", res.get("avg_n_positions", float("nan"))))
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


def improves(a0: dict, mine: dict, ric_a0: float, ric_mine: float, gap_a0: float, gap_mine: float, null_verdict: str) -> dict:
    btc0 = bool(a0.get("btc_weight_identically_zero")) and bool(mine.get("btc_weight_identically_zero"))
    null_ok = str(null_verdict).upper() == "GREEN"
    pass_ric = bool(np.isfinite(ric_mine) and np.isfinite(ric_a0) and ric_mine > ric_a0)
    pass_gap = bool(np.isfinite(gap_mine) and np.isfinite(gap_a0) and gap_mine > gap_a0)
    sh_m = float(mine.get("net_sharpe_full", float("nan")))
    sh_a = float(a0.get("net_sharpe_full", float("nan")))
    pass_sh = bool(np.isfinite(sh_m) and np.isfinite(sh_a) and sh_m > sh_a)
    ok = bool(pass_ric and pass_gap and pass_sh and btc0 and null_ok)
    return {
        "improves": ok,
        "verdict": "IMPROVES" if ok else "NO LIFT",
        "pass_ric": pass_ric,
        "pass_gap": pass_gap,
        "pass_sharpe": pass_sh,
        "pass_btc0": btc0,
        "pass_null": null_ok,
        "ric_a0": ric_a0,
        "ric_mine": ric_mine,
        "gap_a0": gap_a0,
        "gap_mine": gap_mine,
        "sharpe_a0": sh_a,
        "sharpe_mine": sh_m,
        "null_verdict": null_verdict,
    }


def pooled_rankic(pred: pd.DataFrame, ycol: str, score_col: str = "score") -> dict:
    g = pred.copy()
    g["date"] = pd.to_datetime(g["date"], utc=True)
    if "symbol" in g.columns:
        g = g[g["symbol"] != BTC_SYMBOL]
    ic = daily_rank_ic(g, ycol, score_col)
    return summarize_ic(ic, HORIZON)


def top_minus_universe(
    pred: pd.DataFrame,
    score_col: str,
    ycol: str,
    n_buckets: int = 5,
) -> dict:
    """Per-date (top-quintile mean − universe mean) of ycol. BTC dropped."""
    daily = []
    for dt, g in pred.groupby("date", sort=True):
        gg = g.dropna(subset=[score_col, ycol])
        if "symbol" in gg.columns:
            gg = gg[gg["symbol"] != BTC_SYMBOL]
        if len(gg) < n_buckets * 2:
            continue
        try:
            q = pd.qcut(gg[score_col], n_buckets, labels=False, duplicates="drop")
        except (ValueError, TypeError):
            continue
        qv = pd.to_numeric(pd.Series(q, index=gg.index), errors="coerce")
        if qv.notna().sum() < n_buckets:
            continue
        top = qv.max()
        if not np.isfinite(top):
            continue
        mu_top = float(gg.loc[qv.to_numpy() == int(top), ycol].mean())
        mu_uni = float(gg[ycol].mean())
        if np.isfinite(mu_top) and np.isfinite(mu_uni):
            daily.append(mu_top - mu_uni)
    arr = np.asarray(daily, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return {
            "n_days": int(len(arr)),
            "mean_gap": float("nan"),
            "pct_gap_pos": float("nan"),
            "nw_t": float("nan"),
        }
    return {
        "n_days": int(len(arr)),
        "mean_gap": float(np.mean(arr)),
        "pct_gap_pos": float(np.mean(arr > 0.0)),
        "nw_t": newey_west_t(arr, lag=HORIZON),
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
        names = [s for s in g["symbol"].tolist() if s in r.columns and s != BTC_SYMBOL]
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
    p = panel[panel["symbol"] == BTC_SYMBOL].copy()
    if p.empty:
        return pd.Series(dtype=float)
    p["date"] = pd.to_datetime(p["date"], utc=True)
    p = p.sort_values("date")
    r = p.set_index("date")["close"].pct_change()
    return r.dropna()


def last_fold_wins(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds
    out = preds.sort_values(["date", "symbol", "fold_id"])
    return out.drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)


def pick_null_folds(folds: list[FoldSpec], anchor: str = "2022-01-01") -> list[FoldSpec]:
    if not folds:
        return []

    def _utc(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    first = folds[0]
    target = _utc(anchor)
    nearest = min(folds, key=lambda f: abs((_utc(f.val_start) - target).days))
    if nearest.fold_id == first.fold_id and len(folds) > 1:
        nearest = min(folds[1:], key=lambda f: abs((_utc(f.val_start) - target).days))
    return [first, nearest]


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
