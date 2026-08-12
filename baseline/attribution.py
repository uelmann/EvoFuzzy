"""Attribution, concentration, median-τ, and IC diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import daily_rank_ic, summarize_ic


COLLAPSE_WINDOWS = {
    "LUNAUSDT": ("2022-05-01", "2022-05-20"),
    "LUNA2USDT": ("2022-05-01", "2022-05-20"),
    "FTTUSDT": ("2022-11-01", "2022-11-20"),
}


def symbol_attribution(res: dict, top_n: int = 10) -> dict:
    contrib = res.get("sym_contrib") or {}
    if not contrib:
        return {"top": [], "bottom": [], "total": 0.0}
    total = float(sum(contrib.values()))
    items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    def _row(sym, pnl):
        side = res.get("side_days", {}).get(sym, {})
        return {
            "symbol": sym,
            "pnl": float(pnl),
            "pct_of_total": float(pnl / total * 100) if total != 0 else 0.0,
            "long_days": int(side.get("long_days", 0)),
            "short_days": int(side.get("short_days", 0)),
            "dominant_side": "long" if side.get("long_days", 0) >= side.get("short_days", 0) else "short",
        }
    top = [_row(s, p) for s, p in items[:top_n]]
    bottom = [_row(s, p) for s, p in items[-top_n:]]
    collapse = {}
    for sym, (a, b) in COLLAPSE_WINDOWS.items():
        if sym in contrib:
            collapse[sym] = {
                "pnl": float(contrib[sym]),
                "pct_of_total": float(contrib[sym] / total * 100) if total else 0.0,
                "window": [a, b],
                **_row(sym, contrib[sym]),
            }
        else:
            # fuzzy match
            hits = [k for k in contrib if sym.replace("USDT", "") in k]
            collapse[sym] = {
                "pnl": 0.0,
                "pct_of_total": 0.0,
                "window": [a, b],
                "note": "not in contrib" if not hits else f"aliases={hits}",
                "aliases": [
                    {"symbol": h, "pnl": float(contrib[h]), "pct": float(contrib[h] / total * 100) if total else 0.0}
                    for h in hits
                ],
            }
    return {"top": top, "bottom": bottom, "total": total, "collapse": collapse}


def per_year_breakdown(res: dict) -> list[dict]:
    net = res.get("daily_ret")
    if net is None or len(net) == 0:
        return []
    gross = res.get("daily_gross", pd.Series(0.0, index=net.index))
    hedge = res.get("daily_hedge", pd.Series(0.0, index=net.index))
    cost = res.get("daily_cost", pd.Series(0.0, index=net.index))
    fund = res.get("daily_funding", pd.Series(0.0, index=net.index))
    df = pd.DataFrame(
        {
            "net": net,
            "gross": gross.reindex(net.index).fillna(0.0),
            "hedge": hedge.reindex(net.index).fillna(0.0),
            "cost": cost.reindex(net.index).fillna(0.0),
            "funding": fund.reindex(net.index).fillna(0.0),
        }
    )
    df["year"] = pd.to_datetime(df.index, utc=True).year
    rows = []
    for y, g in df.groupby("year"):
        def _sh(x):
            return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0
        rows.append(
            {
                "year": int(y),
                "n_days": int(len(g)),
                "net_sharpe": _sh(g["net"]),
                "gross_total": float(g["gross"].sum()),
                "cost_drag": float(g["cost"].sum()),
                "funding_total": float(g["funding"].sum()),
                "hedge_total": float(g["hedge"].sum()),
                "net_total": float(g["net"].sum()),
            }
        )
    return rows


def day_concentration(res: dict) -> dict:
    net = res.get("daily_ret")
    if net is None or len(net) == 0:
        return {}
    total = float(net.sum())
    ranked = net.sort_values(ascending=False)
    def _pct(k):
        if total == 0:
            return 0.0
        return float(ranked.head(k).sum() / total * 100)
    return {
        "pct_best_5_days": _pct(5),
        "pct_best_20_days": _pct(20),
        "pct_worst_5_days": float(ranked.tail(5).sum() / total * 100) if total else 0.0,
        "total_net_pnl": total,
    }


def median_tau_summary(rows: list[dict]) -> list[dict]:
    """Group by variant/horizon/lag/funding_on → median and best net Sharpe across τ."""
    if not rows:
        return []
    df = pd.DataFrame([r for r in rows if "net_sharpe" in r and "error" not in r])
    if df.empty:
        return []
    keys = ["variant", "horizon", "lag", "funding_on"]
    for k in keys:
        if k not in df.columns:
            df[k] = None
    out = []
    for gkeys, g in df.groupby(keys, dropna=False):
        # gkeys may be scalar if single key
        if not isinstance(gkeys, tuple):
            gkeys = (gkeys,)
        rec = dict(zip(keys, gkeys))
        rec["median_net_sharpe"] = float(g["net_sharpe"].median())
        rec["best_net_sharpe"] = float(g["net_sharpe"].max())
        best_row = g.loc[g["net_sharpe"].idxmax()]
        med_row = g.iloc[(g["net_sharpe"] - rec["median_net_sharpe"]).abs().argmin()]
        rec["best_tau"] = float(best_row["tau_pct"])
        rec["median_tau"] = float(med_row["tau_pct"])
        rec["median_funding_pnl"] = float(med_row.get("funding_total_pnl", 0.0) or 0.0)
        rec["best_funding_pnl"] = float(best_row.get("funding_total_pnl", 0.0) or 0.0)
        rec["n_tau"] = int(len(g))
        out.append(rec)
    return out


def ic_dispersion_diagnostic(
    pred: pd.DataFrame,
    horizon: int,
    universe: pd.DataFrame,
    n_exclude: int = 10,
) -> dict:
    """top20 RankIC mean with/without highest cross-sectional dispersion days."""
    ycol = f"y_h{horizon}"
    df = pred.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    u = universe.copy()
    u["date"] = pd.to_datetime(u["date"], utc=True)
    df = df.merge(u[["date", "symbol"]], on=["date", "symbol"], how="inner")
    ic = daily_rank_ic(df, ycol)
    # dispersion = cross-sectional std of y (or score)
    disp = df.groupby("date")[ycol].std()
    disp = disp.reindex(ic.index).dropna()
    top_disp_days = disp.nlargest(n_exclude).index
    ic_ex = ic.drop(labels=[d for d in top_disp_days if d in ic.index], errors="ignore")
    base = summarize_ic(ic, horizon)
    ex = summarize_ic(ic_ex, horizon)
    return {
        "n_days": base["n_days"],
        "mean_ic": base["mean_ic"],
        "icir": base["icir"],
        "nw_tstat": base["nw_tstat"],
        "mean_ic_excl_top_disp": ex["mean_ic"],
        "icir_excl_top_disp": ex["icir"],
        "nw_tstat_excl_top_disp": ex["nw_tstat"],
        "n_excluded": int(n_exclude),
        "advantage_disappears": bool(
            abs(ex["mean_ic"]) < 0.5 * abs(base["mean_ic"]) if np.isfinite(base["mean_ic"]) else False
        ),
        "ic_series": ic,
        "disp_series": disp,
    }
