"""Phase D.2 metric assembly, median-τ pick, and mechanical verdicts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline.attribution import median_tau_summary, per_year_breakdown
from baseline.evaluate import evaluate_predictions
from phase_d.ablation import paired_delta_ic
from phase_d2.constants import TRAIL_DAYS


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) and x.std() > 0 else 0.0


def trail_mask(idx: pd.DatetimeIndex, days: int = TRAIL_DAYS) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
    if len(idx) == 0:
        return pd.Series(dtype=bool)
    end = idx.max()
    start = end - pd.Timedelta(days=days)
    return pd.Series((idx >= start) & (idx <= end), index=idx)


def window_slice(s: pd.Series, window: str) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
    if window == "full":
        return s
    if window == "trail18m":
        m = trail_mask(s.index)
        return s.loc[m.values]
    if window.startswith("y"):
        y = int(window[1:])
        return s[s.index.year == y]
    raise ValueError(window)


def slim_port(res: dict) -> dict:
    drop = {
        "equity", "daily_ret", "daily_gross", "daily_hedge", "daily_cost",
        "daily_funding", "daily_n_pos", "daily_n_long", "daily_n_short",
        "daily_flat", "sym_contrib", "side_days", "daily_long", "daily_short",
        "daily_gross_deployed", "daily_gross_full", "name_alpha_pnl",
    }
    return {k: v for k, v in res.items() if k not in drop}


def summarize_port(res: dict, common_idx: pd.DatetimeIndex | None = None) -> dict:
    net = res.get("daily_ret")
    if net is None or not isinstance(net, pd.Series) or len(net) == 0:
        return {**slim_port(res), "error": res.get("error", "empty")}
    net = net.copy()
    net.index = pd.DatetimeIndex(pd.to_datetime(net.index, utc=True))
    used = net if common_idx is None else net.reindex(common_idx).dropna()
    years = sorted({int(y) for y in used.index.year.unique() if y >= 2022})
    out = slim_port(res)
    out["n_days"] = int(len(used))
    out["net_sharpe_full"] = _sharpe(used)
    out["net_sharpe_trail18m"] = _sharpe(window_slice(used, "trail18m"))
    out["net_sharpe_by_year"] = {y: _sharpe(window_slice(used, f"y{y}")) for y in years}
    for name, key in [
        ("gross_total_pnl", "daily_gross"),
        ("hedge_total_pnl", "daily_hedge"),
        ("cost_drag", "daily_cost"),
        ("funding_total_pnl", "daily_funding"),
        ("net_total_pnl", "daily_ret"),
    ]:
        ser = res.get(key)
        if isinstance(ser, pd.Series) and len(ser):
            ser = ser.copy()
            ser.index = pd.DatetimeIndex(pd.to_datetime(ser.index, utc=True))
            if common_idx is not None:
                ser = ser.reindex(common_idx).fillna(0.0)
            out[name] = float(ser.sum())
    out["avg_n_positions"] = float(res.get("avg_n_positions", float("nan")))
    out["avg_n_long"] = float(res.get("avg_n_long", float("nan")))
    out["avg_n_short"] = float(res.get("avg_n_short", float("nan")))
    out["pct_flat_days"] = float(res.get("pct_flat_days", float("nan")))
    out["ann_turnover"] = float(res.get("ann_turnover", float("nan")))
    out["avg_traded_rank"] = float(res.get("avg_traded_rank", float("nan")))
    out["avg_gross_deployed"] = float(res.get("avg_gross_deployed", float("nan")))
    out["avg_gross_full"] = float(res.get("avg_gross_full", float("nan")))
    out["tau_pct"] = float(res.get("tau_pct", float("nan")))
    out["tau_mode"] = res.get("tau_mode")
    out["year_rows"] = per_year_breakdown(res)
    eq = res.get("equity")
    if isinstance(eq, pd.DataFrame) and not eq.empty:
        out["equity"] = eq
    out["daily_ret"] = used
    out["name_alpha_pnl"] = dict(res.get("name_alpha_pnl") or {})
    out["sym_contrib"] = dict(res.get("sym_contrib") or {})
    series_keys = (
        "daily_n_pos", "daily_n_long", "daily_n_short", "daily_flat",
        "daily_long", "daily_short", "daily_gross", "daily_hedge",
        "daily_cost", "daily_funding", "daily_gross_deployed", "daily_gross_full",
    )
    for key in series_keys:
        ser = res.get(key)
        if isinstance(ser, pd.Series) and len(ser):
            ser = ser.copy()
            ser.index = pd.DatetimeIndex(pd.to_datetime(ser.index, utc=True))
            if common_idx is not None:
                ser = ser.reindex(common_idx)
            out[key] = ser
            if key == "daily_n_long":
                out["avg_n_long"] = float(ser.mean()) if len(ser) else float("nan")
            if key == "daily_flat":
                out["pct_flat_days"] = float(ser.mean()) if len(ser) else float("nan")
            if key == "daily_gross_deployed":
                out["avg_gross_deployed"] = float(ser.mean()) if len(ser) else float("nan")
            if key == "daily_gross_full":
                out["avg_gross_full"] = float(ser.mean()) if len(ser) else float("nan")
    return out


def pick_median_tau(runs: list[dict]) -> dict:
    """House median-τ: among τ grid, the run whose full-period net Sharpe is closest to the median."""
    rows = []
    for r in runs:
        if "error" in r and r.get("net_sharpe") is None:
            continue
        rows.append(
            {
                "variant": r.get("variant", "tranche"),
                "horizon": r.get("horizon"),
                "lag": r.get("lag", 0),
                "funding_on": r.get("funding_on", True),
                "tau_pct": r.get("tau_pct"),
                "net_sharpe": r.get("net_sharpe"),
                "funding_total_pnl": r.get("funding_total_pnl", 0.0),
            }
        )
    med = median_tau_summary(rows)
    if not med:
        return runs[0] if runs else {}
    rec = med[0]
    target = float(rec["median_tau"])
    picked = min(runs, key=lambda r: abs(float(r.get("tau_pct", 99)) - target))
    picked = dict(picked)
    picked["median_tau_meta"] = rec
    return picked


def ic_pair_on_universe(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    label: str,
) -> dict:
    ycol = f"y_h{horizon}"
    a = pred_a.copy()
    b = pred_b.copy()
    a["date"] = pd.to_datetime(a["date"], utc=True)
    b["date"] = pd.to_datetime(b["date"], utc=True)
    if ycol not in a.columns:
        a = a.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    if ycol not in b.columns:
        b = b.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    eva = evaluate_predictions(a, horizon, universe=universe, label=label)
    evb = evaluate_predictions(b, horizon, universe=universe, label=label)
    ia, ib = eva.get("ic_series", pd.Series(dtype=float)), evb.get("ic_series", pd.Series(dtype=float))
    end = max(
        pd.Timestamp(a["date"].max()) if len(a) else pd.Timestamp("1970-01-01", tz="UTC"),
        pd.Timestamp(b["date"].max()) if len(b) else pd.Timestamp("1970-01-01", tz="UTC"),
    )
    start = end - pd.Timedelta(days=TRAIL_DAYS)

    def _win(ic: pd.Series, window: str) -> pd.Series:
        if ic is None or len(ic) == 0:
            return pd.Series(dtype=float)
        ic = ic.copy()
        ic.index = pd.DatetimeIndex(pd.to_datetime(ic.index, utc=True))
        if window == "full":
            return ic
        return ic[(ic.index >= start) & (ic.index <= end)]

    tables = []
    paired = {}
    years = sorted(set(ia.index.year) | set(ib.index.year)) if len(ia) or len(ib) else []
    windows = ["full", "trail18m"] + [f"y{y}" for y in years if y >= 2022]
    for window in windows:
        if window == "full":
            sa, sb = _win(ia, "full"), _win(ib, "full")
        elif window == "trail18m":
            sa, sb = _win(ia, "trail18m"), _win(ib, "trail18m")
        else:
            y = int(window[1:])
            sa, sb = ia[ia.index.year == y] if len(ia) else ia, ib[ib.index.year == y] if len(ib) else ib
        mean_a = float(sa.mean()) if len(sa) else float("nan")
        mean_b = float(sb.mean()) if len(sb) else float("nan")
        tables.append(
            {
                "horizon": horizon,
                "universe": label,
                "window": window,
                "A_ic": mean_a,
                "B_ic": mean_b,
                "delta_ic": float(mean_b - mean_a) if np.isfinite(mean_a) and np.isfinite(mean_b) else float("nan"),
                "n_days": int(min(len(sa), len(sb))),
            }
        )
        if window in ("full", "trail18m"):
            paired[window] = paired_delta_ic(sa, sb, horizon)
    return {
        "horizon": horizon,
        "universe": label,
        "tables": tables,
        "paired_nw": paired,
        "delta_full": next((t["delta_ic"] for t in tables if t["window"] == "full"), float("nan")),
        "delta_trail18m": next((t["delta_ic"] for t in tables if t["window"] == "trail18m"), float("nan")),
    }


def apply_adoption(p_summ: dict, ic_by: dict) -> dict:
    """Mechanical verdicts from the pre-registered criterion (see addendum)."""
    universe_rows = []
    for h in (7, 10):
        p1 = p_summ.get(("P1", h)) or {}
        for cand in ("P2", "P4"):
            px = p_summ.get((cand, h)) or {}
            s18 = float(px.get("net_sharpe_trail18m", float("nan")))
            sfull = float(px.get("net_sharpe_full", float("nan")))
            b18 = float(p1.get("net_sharpe_trail18m", float("nan")))
            bfull = float(p1.get("net_sharpe_full", float("nan")))
            ok = (
                np.isfinite(s18)
                and np.isfinite(sfull)
                and np.isfinite(b18)
                and np.isfinite(bfull)
                and s18 >= b18 + 0.30
                and sfull >= bfull - 0.20
            )
            universe_rows.append(
                {
                    "candidate": cand,
                    "horizon": h,
                    "trail18m": s18,
                    "full": sfull,
                    "p1_trail18m": b18,
                    "p1_full": bfull,
                    "need_trail18m": b18 + 0.30 if np.isfinite(b18) else float("nan"),
                    "need_full": bfull - 0.20 if np.isfinite(bfull) else float("nan"),
                    "pass": bool(ok),
                }
            )
    passing = [r for r in universe_rows if r["pass"]]
    uni_verdict = "ADOPTED" if passing else "REJECTED"
    if passing:
        chosen = max(passing, key=lambda r: r["trail18m"])
        chosen_uni = "top40"
    else:
        chosen = None
        chosen_uni = "top20"

    micro_rows = []
    if chosen_uni == "top40":
        ic_label, micro_p, a_p = "top40", "P4", "P2"
    else:
        ic_label, micro_p, a_p = "top20", "P3", "P1"
    for h in (7, 10):
        ic = (ic_by.get((ic_label, h)) or {})
        d18 = float(ic.get("delta_trail18m", float("nan")))
        dfull = float(ic.get("delta_full", float("nan")))
        pm = p_summ.get((micro_p, h)) or {}
        pa = p_summ.get((a_p, h)) or {}
        ds = float(pm.get("net_sharpe_trail18m", float("nan"))) - float(pa.get("net_sharpe_trail18m", float("nan")))
        ok = (
            np.isfinite(d18)
            and np.isfinite(dfull)
            and np.isfinite(ds)
            and d18 >= 0.005
            and dfull >= 0.0
            and ds >= 0.0
        )
        micro_rows.append(
            {
                "horizon": h,
                "universe": ic_label,
                "delta_ic_trail18m": d18,
                "delta_ic_full": dfull,
                "delta_sharpe_trail18m": ds,
                "micro_run": micro_p,
                "a_run": a_p,
                "pass": bool(ok),
            }
        )
    micro_verdict = "ADOPTED" if any(r["pass"] for r in micro_rows) else "REJECTED"
    return {
        "universe_verdict": uni_verdict,
        "micro_verdict": micro_verdict,
        "chosen_universe": chosen_uni,
        "chosen_run": chosen,
        "universe_rows": universe_rows,
        "micro_rows": micro_rows,
    }


def year_attribution_2026(actual: dict, oracle: dict) -> dict:
    """Fraction of 2026 net loss due to beta error vs alpha failure (h=7 P1)."""
    def _year_net(blob: dict, year: int) -> float:
        for r in blob.get("year_rows") or []:
            if int(r.get("year", 0)) == year:
                return float(r.get("net_total", float("nan")))
        ser = blob.get("daily_ret")
        if isinstance(ser, pd.Series) and len(ser):
            y = ser[pd.DatetimeIndex(ser.index).year == year]
            return float(y.sum()) if len(y) else float("nan")
        return float("nan")

    a = _year_net(actual, 2026)
    o = _year_net(oracle, 2026)
    loss = -a if np.isfinite(a) and a < 0 else 0.0
    beta_cost = float(o - a) if np.isfinite(a) and np.isfinite(o) else float("nan")
    if loss <= 0:
        frac_beta, frac_alpha = 0.0, 0.0
        sentence = (
            f"2026 actual net={a:.4f} is not a loss; beta-estimation cost (oracle−actual)={beta_cost:.4f}."
        )
    else:
        if not np.isfinite(beta_cost):
            frac_beta, frac_alpha = float("nan"), float("nan")
            sentence = f"2026 actual net={a:.4f}; oracle unavailable."
        elif beta_cost <= 0:
            frac_beta, frac_alpha = 0.0, 1.0
            sentence = (
                f"Of the 2026 net loss ({a:.4f}), 0% is attributable to beta-estimation error "
                f"(oracle net={o:.4f} is not better) and 100% to alpha failure."
            )
        elif o >= 0:
            frac_beta, frac_alpha = 1.0, 0.0
            sentence = (
                f"Of the 2026 net loss ({a:.4f}), 100% is attributable to beta-estimation error "
                f"(oracle net={o:.4f} ≥ 0) and 0% to alpha failure."
            )
        else:
            frac_beta = min(1.0, max(0.0, beta_cost / loss))
            frac_alpha = 1.0 - frac_beta
            sentence = (
                f"Of the 2026 net loss ({a:.4f}), {frac_beta:.0%} is attributable to beta-estimation error "
                f"(oracle−actual={beta_cost:.4f}; oracle net={o:.4f}) and {frac_alpha:.0%} to alpha failure."
            )
    return {
        "actual_2026_net": a,
        "oracle_2026_net": o,
        "beta_cost": beta_cost,
        "frac_beta": frac_beta,
        "frac_alpha": frac_alpha,
        "sentence": sentence,
    }
