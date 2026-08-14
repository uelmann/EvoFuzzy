"""Long-only evaluation: OLS alpha, viability, attribution, benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from longonly.constants import (
    ANNUALIZATION,
    LOH_FULL_SHARPE_MIN,
    LOH_TRAIL_SHARPE_MIN,
    LOU_NW_T_MIN,
    TRAIL_DAYS,
    VIABILITY_CRITERION,
)
from phase_d2.metrics import _sharpe, window_slice
from round_f.eval import combo_from_sleeves


def _as_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    return out


def _align(s: pd.Series | None, idx: pd.DatetimeIndex, fill: float = 0.0) -> pd.Series:
    if not isinstance(s, pd.Series) or len(s) == 0:
        return pd.Series(fill, index=idx, dtype=float)
    s = _as_utc(s)
    return s.reindex(idx).fillna(fill)


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


def ols_alpha_beta_nw(y: pd.Series, x: pd.Series, lag: int) -> dict:
    """OLS y = a + b x + e. Annualized alpha = a * 365. Newey–West t on a (Bartlett, lag)."""
    y = _as_utc(y).astype(float)
    x = _as_utc(x).astype(float)
    y, x = y.align(x, join="inner")
    mask = np.isfinite(y.to_numpy()) & np.isfinite(x.to_numpy())
    yv = y.to_numpy()[mask]
    xv = x.to_numpy()[mask]
    n = int(len(yv))
    if n < 10:
        return {
            "alpha_daily": float("nan"),
            "alpha_ann": float("nan"),
            "beta": float("nan"),
            "nw_t_alpha": float("nan"),
            "n": n,
            "lag": int(lag),
        }
    X = np.column_stack([np.ones(n), xv])
    beta_hat, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    e = yv - X @ beta_hat
    u = X * e[:, None]
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    S = u.T @ u
    L = max(int(lag), 0)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gk = u[k:].T @ u[:-k]
        S = S + w * (gk + gk.T)
    V = xtx_inv @ S @ xtx_inv
    se_a = float(np.sqrt(max(V[0, 0], 0.0)))
    t_a = float(beta_hat[0] / se_a) if se_a > 0 else 0.0
    return {
        "alpha_daily": float(beta_hat[0]),
        "alpha_ann": float(beta_hat[0] * ANNUALIZATION),
        "beta": float(beta_hat[1]),
        "nw_t_alpha": t_a,
        "n": n,
        "lag": int(lag),
        "se_alpha_daily": se_a,
    }


def alpha_full_and_trail(y: pd.Series, x: pd.Series, lag: int) -> dict:
    full = ols_alpha_beta_nw(y, x, lag)
    yt = window_slice(y, "trail18m")
    xt = window_slice(x, "trail18m")
    trail = ols_alpha_beta_nw(yt, xt, lag)
    return {"full": full, "trail18m": trail}


def btc_bh_simple(panel: pd.DataFrame) -> pd.Series:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    if "BTCUSDT" not in close.columns:
        raise RuntimeError("BTCUSDT missing from panel")
    return close["BTCUSDT"].pct_change().rename("btc_bh")


def ew_top20_simple(panel: pd.DataFrame, pit20: pd.DataFrame) -> pd.Series:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    simple = close.pct_change()
    pit = pit20.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True)
    by = pit.groupby("date")["symbol"].apply(lambda s: list(s))
    dates = sorted(by.index)
    rows: dict[pd.Timestamp, float] = {}
    close_idx = simple.index
    for dt in dates:
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
    return out.rename("ew_top20")


def top_n_names(pnl: dict, n: int = 5) -> list[dict]:
    items = [(k, float(v)) for k, v in (pnl or {}).items() if k != "BTCUSDT_hedge"]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"symbol": k, "pnl": v} for k, v in items[:n]]


def loh_viable(sharpe_full: float, sharpe_trail: float) -> dict:
    pf = bool(np.isfinite(sharpe_full) and sharpe_full >= LOH_FULL_SHARPE_MIN)
    pt = bool(np.isfinite(sharpe_trail) and sharpe_trail >= LOH_TRAIL_SHARPE_MIN)
    ok = pf and pt
    return {
        "verdict": "VIABLE" if ok else "NOT VIABLE",
        "criterion": VIABILITY_CRITERION,
        "sharpe_full": float(sharpe_full) if np.isfinite(sharpe_full) else float("nan"),
        "sharpe_trail18m": float(sharpe_trail) if np.isfinite(sharpe_trail) else float("nan"),
        "need_full": LOH_FULL_SHARPE_MIN,
        "need_trail18m": LOH_TRAIL_SHARPE_MIN,
        "pass_full": pf,
        "pass_trail18m": pt,
        "pass": bool(ok),
    }


def lou_viable(alpha_full: float, nw_t: float, alpha_trail: float) -> dict:
    pa = bool(np.isfinite(alpha_full) and alpha_full > 0)
    pt = bool(np.isfinite(nw_t) and nw_t >= LOU_NW_T_MIN)
    ptr = bool(np.isfinite(alpha_trail) and alpha_trail > 0)
    ok = pa and pt and ptr
    return {
        "verdict": "VIABLE" if ok else "NOT VIABLE",
        "criterion": VIABILITY_CRITERION,
        "alpha_ann_full": float(alpha_full) if np.isfinite(alpha_full) else float("nan"),
        "nw_t_alpha_full": float(nw_t) if np.isfinite(nw_t) else float("nan"),
        "alpha_ann_trail18m": float(alpha_trail) if np.isfinite(alpha_trail) else float("nan"),
        "need_alpha_positive": True,
        "need_nw_t": LOU_NW_T_MIN,
        "need_trail_alpha_positive": True,
        "pass_alpha_full": pa,
        "pass_nw_t": pt,
        "pass_alpha_trail18m": ptr,
        "pass": bool(ok),
    }


def attribution_block(port: dict, idx: pd.DatetimeIndex | None = None) -> dict:
    """Decompose net PnL into long / short / hedge / funding / costs."""
    net = port.get("daily_ret")
    if not isinstance(net, pd.Series) or len(net) == 0:
        return {"error": "missing daily_ret"}
    net = _as_utc(net)
    use_idx = net.index if idx is None else pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
    net = net.reindex(use_idx).fillna(0.0)
    legs = {
        "long": _align(port.get("daily_long"), use_idx),
        "short": _align(port.get("daily_short"), use_idx),
        "hedge": _align(port.get("daily_hedge"), use_idx),
        "funding": _align(port.get("daily_funding"), use_idx),
        "costs": _align(port.get("daily_cost"), use_idx),
        "net": net,
    }

    def _sum_row(mask) -> dict:
        long_s = float(legs["long"][mask].sum())
        short_s = float(legs["short"][mask].sum())
        hedge_s = float(legs["hedge"][mask].sum())
        fund_s = float(legs["funding"][mask].sum())
        cost_s = float(legs["costs"][mask].sum())
        net_s = float(legs["net"][mask].sum())
        recon = long_s + short_s + hedge_s + fund_s - cost_s
        return {
            "long": long_s,
            "short": short_s,
            "hedge": hedge_s,
            "funding": fund_s,
            "costs": cost_s,
            "net": net_s,
            "recon": recon,
            "recon_gap": recon - net_s,
            "long_share_of_net": (long_s / net_s) if abs(net_s) > 1e-12 else float("nan"),
            "long_share_of_alpha": (long_s / (long_s + short_s)) if abs(long_s + short_s) > 1e-12 else float("nan"),
        }

    years = sorted({int(y) for y in use_idx.year.unique() if y >= 2022})
    by_year = {y: _sum_row(use_idx.year == y) for y in years}
    full = _sum_row(np.ones(len(use_idx), dtype=bool))
    trail_idx = window_slice(net, "trail18m").index
    trail_mask = use_idx.isin(trail_idx)
    trail = _sum_row(trail_mask)
    return {"full": full, "trail18m": trail, "by_year": by_year, "n_days": int(len(use_idx))}


def enrich_combo(p1: dict, p2: dict) -> dict:
    c = combo_from_sleeves(p1, p2)
    if "error" in c:
        return c
    idx = pd.DatetimeIndex(pd.to_datetime(c["daily_ret"].index, utc=True))
    c["daily_ret"].index = idx
    half_keys = (
        "daily_long",
        "daily_short",
        "daily_hedge",
        "daily_cost",
        "daily_funding",
        "daily_gross",
        "daily_gross_deployed",
        "daily_gross_full",
    )
    for key in half_keys:
        c[key] = 0.5 * _align(p1.get(key), idx) + 0.5 * _align(p2.get(key), idx)
    n1 = _align(p1.get("daily_n_long"), idx)
    n2 = _align(p2.get("daily_n_long"), idx)
    if n1.abs().sum() == 0:
        n1 = _align(p1.get("daily_n_pos"), idx)
    if n2.abs().sum() == 0:
        n2 = _align(p2.get("daily_n_pos"), idx)
    c["daily_n_long"] = n1 + n2
    f1 = _align(p1.get("daily_flat"), idx)
    f2 = _align(p2.get("daily_flat"), idx)
    c["daily_flat"] = ((f1 > 0.5) & (f2 > 0.5)).astype(float)
    cagr, maxdd, total = cagr_maxdd(c["daily_ret"])
    c["net_cagr"] = cagr
    c["max_drawdown"] = maxdd
    c["total_return"] = total
    c["avg_n_long"] = float(c["daily_n_long"].mean()) if len(c["daily_n_long"]) else float("nan")
    c["pct_flat_days"] = float(c["daily_flat"].mean()) if len(c["daily_flat"]) else float("nan")
    c["avg_gross_deployed"] = float(c["daily_gross_deployed"].mean()) if len(c["daily_gross_deployed"]) else float("nan")
    c["avg_gross_full"] = float(c["daily_gross_full"].mean()) if len(c["daily_gross_full"]) else float("nan")
    c["funding_total_pnl"] = float(c["daily_funding"].sum())
    c["hedge_total_pnl"] = float(c["daily_hedge"].sum())
    c["cost_drag"] = float(c["daily_cost"].sum())
    c["gross_total_pnl"] = float(c["daily_gross"].sum())
    c["net_total_pnl"] = float(c["daily_ret"].sum())
    c["ann_turnover"] = 0.5 * float(p1.get("ann_turnover", float("nan"))) + 0.5 * float(
        p2.get("ann_turnover", float("nan"))
    )
    pnl1 = dict(p1.get("name_alpha_pnl") or {})
    pnl2 = dict(p2.get("name_alpha_pnl") or {})
    names = set(pnl1) | set(pnl2)
    c["name_alpha_pnl"] = {k: 0.5 * float(pnl1.get(k, 0.0)) + 0.5 * float(pnl2.get(k, 0.0)) for k in names}
    eq = (1.0 + c["daily_ret"].fillna(0.0)).cumprod()
    c["equity"] = pd.DataFrame({"date": idx, "equity": eq.values})
    return c


def book_stats(port: dict, btc: pd.Series, lag: int, ref_combo: pd.Series | None = None) -> dict:
    rets = _as_utc(port["daily_ret"])
    years = sorted({int(y) for y in rets.index.year.unique() if y >= 2022})
    cagr, maxdd, total = cagr_maxdd(rets)
    n_long = port.get("daily_n_long")
    if not isinstance(n_long, pd.Series) or len(n_long) == 0:
        n_long = port.get("daily_n_pos")
    n_long = _align(n_long, rets.index) if isinstance(n_long, pd.Series) else pd.Series(dtype=float)
    flat = _align(port.get("daily_flat"), rets.index) if isinstance(port.get("daily_flat"), pd.Series) else pd.Series(dtype=float)
    gdep = port.get("daily_gross_deployed")
    gfull = port.get("daily_gross_full")
    fund = port.get("daily_funding")
    alpha = alpha_full_and_trail(rets, btc, lag)
    corr_ref = float("nan")
    if isinstance(ref_combo, pd.Series) and len(ref_combo):
        a, b = rets.align(_as_utc(ref_combo), join="inner")
        if len(a) > 5:
            corr_ref = float(a.corr(b))
    return {
        "n_days": int(len(rets)),
        "net_sharpe_full": _sharpe(rets),
        "net_sharpe_trail18m": _sharpe(window_slice(rets, "trail18m")),
        "net_sharpe_by_year": {y: _sharpe(window_slice(rets, f"y{y}")) for y in years},
        "net_cagr": cagr,
        "max_drawdown": maxdd,
        "total_return": total,
        "avg_n_long": float(n_long.mean()) if len(n_long) else float(port.get("avg_n_long", float("nan"))),
        "avg_gross_deployed": float(_align(gdep, rets.index).mean())
        if isinstance(gdep, pd.Series)
        else float(port.get("avg_gross_deployed", float("nan"))),
        "avg_gross_full": float(_align(gfull, rets.index).mean())
        if isinstance(gfull, pd.Series)
        else float(port.get("avg_gross_full", float("nan"))),
        "pct_flat_days": float(flat.mean()) if len(flat) else float(port.get("pct_flat_days", float("nan"))),
        "funding_total_pnl": float(_align(fund, rets.index).sum())
        if isinstance(fund, pd.Series)
        else float(port.get("funding_total_pnl", float("nan"))),
        "ann_turnover": float(port.get("ann_turnover", float("nan"))),
        "top5_names": top_n_names(port.get("name_alpha_pnl") or {}),
        "alpha": alpha,
        "corr_vs_ref_combo": corr_ref,
        "long_only": bool(port.get("long_only", False)),
        "apply_beta_hedge": bool(port.get("apply_beta_hedge", True)),
        "tau_pct": port.get("tau_pct"),
        "horizon": port.get("horizon"),
        "daily_ret": rets,
        "equity": port.get("equity"),
    }
