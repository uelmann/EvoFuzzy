"""Phase B.1 — control gates vs kr_sigma; redundancy; decay."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from baseline.portfolio import (
    _apply_hedge,
    _attach_aux,
    _funding_wide,
    _hard_threshold_state,
    _pack_metrics,
    _prepare_returns,
    _size_book,
    run_tranche_portfolio,
)

CUTOFF = pd.Timestamp("2025-08-17", tz="UTC")

ADOPTION_RULE = (
    "The kr_sigma gate is ADOPTED only if: (a) post-cutoff ΔSharpe(kr_sigma gate vs the BEST "
    "control gate, same X selection procedure) ≥ +0.10; (b) full-period Sharpe of the kr_sigma "
    "gate ≥ best control gate − 0.10; (c) redundancy diagnostics below the thresholds in §2. "
    "Otherwise verdict = REDUNDANT: adopt the best control gate instead if it beats ungated "
    "post-cutoff by ≥ +0.10, else adopt no gate."
)

CONTROL_GATES = {
    "C1_yz_vol_30": "yz_vol_30",
    "C2_idio_vol_60": "idio_vol_60",
    "C3_vol_of_vol_30": "vol_of_vol_30",
}
REF_GATE = "kr_sigma_h7"
X_GRID = [10, 20, 30]


def _resolve_gate_col(feat: pd.DataFrame, logical: str) -> str:
    """Prefer raw column when present (rank-identical to CS-z within day)."""
    if logical == "yz_vol_30" and "yz_vol_30_raw" in feat.columns:
        return "yz_vol_30_raw"
    if logical in feat.columns:
        return logical
    raise KeyError(f"gate column missing: {logical}")


def _window_mask(idx: pd.DatetimeIndex, window: str) -> np.ndarray:
    if window == "pre":
        return idx < CUTOFF
    if window == "post":
        return idx >= CUTOFF
    return np.ones(len(idx), dtype=bool)


def _metrics_from_res(res: dict) -> dict:
    """Full/pre/post Sharpe, flat%, turnover, avg positions — paired day sets."""
    if "error" in res and "daily_ret" not in res:
        return {"error": res["error"]}
    rets = res["daily_ret"]
    if not isinstance(rets, pd.Series):
        rets = pd.Series(rets)
    rets.index = pd.to_datetime(rets.index, utc=True)
    # reconstruct daily series from packed lists when available
    n = len(rets)
    eq_dates = list(rets.index)
    # flat / turnover / n_pos from equity path length — use side info if stored
    flat = res.get("_flat_series")
    to_s = res.get("_to_series")
    npos = res.get("_npos_series")
    if flat is None:
        # approximate from pct_flat only for full; build ones for windows via equity days
        flat = pd.Series(0.0, index=rets.index)
    if to_s is None:
        to_s = pd.Series(float(res.get("ann_turnover", 0.0)) / 365.0, index=rets.index)
    if npos is None:
        npos = pd.Series(float(res.get("avg_n_positions", 0.0)), index=rets.index)

    out = {"tau_pct": res.get("tau_pct"), "gate": res.get("gate_name"), "gate_col": res.get("gate_col"), "X": res.get("sigma_top_pct")}
    for w in ("full", "pre", "post"):
        m = _window_mask(rets.index, w)
        rr = rets.loc[m]
        if len(rr) < 5:
            out[f"sharpe_{w}"] = float("nan")
            out[f"n_days_{w}"] = int(len(rr))
            out[f"pct_flat_{w}"] = float("nan")
            out[f"ann_turnover_{w}"] = float("nan")
            out[f"avg_n_pos_{w}"] = float("nan")
            continue
        sh = float(rr.mean() / rr.std() * np.sqrt(365)) if rr.std() > 0 else 0.0
        out[f"sharpe_{w}"] = sh
        out[f"n_days_{w}"] = int(len(rr))
        out[f"pct_flat_{w}"] = float(flat.loc[m].mean()) if len(flat.loc[m]) else float("nan")
        out[f"ann_turnover_{w}"] = float(to_s.loc[m].mean() * 365) if len(to_s.loc[m]) else float("nan")
        out[f"avg_n_pos_{w}"] = float(npos.loc[m].mean()) if len(npos.loc[m]) else float("nan")
    out["n_blocked_entries"] = res.get("n_blocked_entries", 0)
    out["net_sharpe"] = out["sharpe_full"]
    return out


def run_tranche_with_column_gate(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    gate_values: pd.DataFrame,
    gate_col: str,
    horizon: int,
    tau_pct: float,
    top_pct: float,
    gate_name: str,
    gross_limit: float = 1.0,
    fee_bps: float = 5.0,
    slip_bps: float = 3.0,
    lag: int = 0,
    apply_funding: bool = True,
    funding: pd.DataFrame | None = None,
) -> dict:
    """
    Tranche portfolio + skip NEW entries when gate_col is in top top_pct% CS that day.
    gate_values must contain date, symbol, gate_col.
    """
    h = int(horizon)
    df = _attach_aux(preds, feat, universe)
    g = gate_values[["date", "symbol", gate_col]].copy()
    g["date"] = pd.to_datetime(g["date"], utc=True)
    g = g.rename(columns={gate_col: "_gate_val"})
    # drop any pre-existing gate/feature cols that would collide on merge
    drop_cols = [c for c in ("_gate_val", gate_col) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.merge(g, on=["date", "symbol"], how="left")

    rets = _prepare_returns(panel)
    fund_wide = _funding_wide(funding) if apply_funding else pd.DataFrame()
    dates = sorted(df["date"].unique())
    if len(dates) < h + 5 + int(lag):
        return {"error": "not enough dates", "tau_pct": tau_pct, "gate_name": gate_name}

    abs_scores = df["score"].abs().dropna().values
    tau = float(np.percentile(abs_scores, tau_pct))
    cost_rate = (fee_bps + slip_bps) * 1e-4
    tg = gross_limit / float(h)

    states: list[dict[str, int]] = [{} for _ in range(h)]
    alphas: list[pd.Series] = [pd.Series(dtype=float) for _ in range(h)]
    entry_date: dict[tuple[int, str], pd.Timestamp] = {}
    hold_days: list[int] = []
    trade_pnls: list[float] = []
    sym_pnl: dict[tuple[int, str], float] = defaultdict(float)
    sym_contrib: dict[str, float] = defaultdict(float)
    side_days: dict[str, dict] = defaultdict(lambda: {"long_days": 0, "short_days": 0})

    target_full_hist: list[pd.Series] = []
    target_alpha_hist: list[pd.Series] = []
    prev_full = pd.Series(dtype=float)
    prev_hedge = 0.0

    daily_net, daily_gross, daily_hedge, daily_cost, daily_funding = [], [], [], [], []
    to_ee, to_rs, to_hg = [], [], []
    n_pos, n_long, n_short, flat, eq_dates = [], [], [], [], []
    n_blocked = 0
    # per-day skip sets for overlap diagnostics (tranche slot that rebalances)
    daily_skips: dict[pd.Timestamp, set[str]] = {}

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt].copy()
        kslot = i % h
        prev_ak = alphas[kslot].copy()
        raw_state = _hard_threshold_state(day, tau)

        sig = day.set_index("symbol")["_gate_val"]
        finite = sig[np.isfinite(sig)]
        blocked: set[str] = set()
        if len(finite) >= 5 and top_pct > 0:
            thr = float(np.nanpercentile(finite.values, 100.0 - float(top_pct)))
            blocked = set(finite[finite >= thr].index.astype(str))

        new_state = {}
        prev_state = states[kslot]
        skipped_today: set[str] = set()
        for sym, side in raw_state.items():
            was = prev_state.get(sym, 0)
            if was == 0 and sym in blocked:
                n_blocked += 1
                skipped_today.add(sym)
                continue
            new_state[sym] = side
        daily_skips[pd.Timestamp(dt)] = skipped_today

        for sym, side in list(prev_state.items()):
            if new_state.get(sym, 0) != side:
                key = (kslot, sym)
                if key in entry_date:
                    hold_days.append(max(1, int((dt - entry_date[key]).days)))
                    trade_pnls.append(float(sym_pnl.get(key, 0.0)))
                    entry_date.pop(key, None)
                    sym_pnl.pop(key, None)
        for sym, side in new_state.items():
            if prev_state.get(sym, 0) != side:
                entry_date[(kslot, sym)] = dt
                sym_pnl[(kslot, sym)] = 0.0
        states[kslot] = new_state
        alphas[kslot] = _size_book(day, new_state, tg)

        full = pd.Series(dtype=float)
        alpha = pd.Series(dtype=float)
        for tk in range(h):
            ak = alphas[tk]
            if ak.empty:
                continue
            if tk != kslot:
                univ = set(day["symbol"])
                ak = ak[[s for s in ak.index if s in univ]]
                alphas[tk] = ak
            fk, hk = _apply_hedge(day, ak)
            alpha = alpha.add(ak, fill_value=0.0)
            full = full.add(fk, fill_value=0.0)
            for s, side in states[tk].items():
                if side > 0:
                    side_days[s]["long_days"] += 1
                elif side < 0:
                    side_days[s]["short_days"] += 1

        target_alpha_hist.append(alpha)
        target_full_hist.append(full)

        if i < lag:
            applied_alpha = pd.Series(dtype=float)
            applied_full = pd.Series(dtype=float)
            applied_hedge = 0.0
        else:
            applied_alpha = target_alpha_hist[i - lag]
            applied_full = target_full_hist[i - lag]
            applied_hedge = float(applied_full.get("BTCUSDT", 0.0) - applied_alpha.get("BTCUSDT", 0.0))

        idx = alphas[kslot].index.union(prev_ak.index)
        ee = 0.5 * float((alphas[kslot].reindex(idx).fillna(0.0) - prev_ak.reindex(idx).fillna(0.0)).abs().sum())
        rs = 0.0
        hg = 0.5 * abs(applied_hedge - prev_hedge)

        fidx = applied_full.index.union(prev_full.index)
        f = applied_full.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        turnover = 0.5 * float((f - pf).abs().sum())
        cost = turnover * cost_rate

        nxt = dates[i + 1]
        gross_r = hedge_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in applied_alpha.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    ri = float(rrow[s])
                    contrib = float(wi) * ri
                    gross_r += contrib
                    sym_contrib[s] = sym_contrib.get(s, 0.0) + contrib
            if "BTCUSDT" in rrow.index and np.isfinite(rrow["BTCUSDT"]):
                hedge_r = applied_hedge * float(rrow["BTCUSDT"])
                sym_contrib["BTCUSDT_hedge"] = sym_contrib.get("BTCUSDT_hedge", 0.0) + hedge_r

        fund_r = 0.0
        if apply_funding and not fund_wide.empty and nxt in fund_wide.index:
            row = fund_wide.loc[nxt]
            for s, wi in applied_full.items():
                if s in row.index and np.isfinite(row[s]):
                    f_i = -float(wi) * float(row[s])
                    fund_r += f_i
                    sym_contrib[s] = sym_contrib.get(s, 0.0) + f_i

        net = gross_r + hedge_r - cost + fund_r
        daily_net.append(net)
        daily_gross.append(gross_r)
        daily_hedge.append(hedge_r)
        daily_cost.append(cost)
        daily_funding.append(fund_r)
        to_ee.append(ee)
        to_rs.append(rs)
        to_hg.append(hg)
        nl = int((applied_alpha > 0).sum()) if len(applied_alpha) else 0
        ns = int((applied_alpha < 0).sum()) if len(applied_alpha) else 0
        n_long.append(nl)
        n_short.append(ns)
        n_pos.append(nl + ns)
        flat.append(1 if nl + ns == 0 else 0)
        eq_dates.append(nxt)
        prev_full, prev_hedge = applied_full, applied_hedge

    res = _pack_metrics(
        daily_net,
        daily_gross,
        daily_hedge,
        daily_cost,
        daily_funding,
        to_ee,
        to_rs,
        to_hg,
        n_pos,
        n_long,
        n_short,
        flat,
        eq_dates,
        hold_days,
        trade_pnls,
        tau_pct,
        tau,
        gate_name,
        horizon,
        lag,
        apply_funding,
        dict(sym_contrib),
        dict(side_days),
    )
    idx = pd.DatetimeIndex(eq_dates)
    res["_flat_series"] = pd.Series(flat, index=idx, dtype=float)
    res["_to_series"] = pd.Series([ee + rs + hg for ee, rs, hg in zip(to_ee, to_rs, to_hg)], index=idx)
    res["_npos_series"] = pd.Series(n_pos, index=idx, dtype=float)
    res["sigma_top_pct"] = float(top_pct)
    res["gate_name"] = gate_name
    res["gate_col"] = gate_col
    res["n_blocked_entries"] = int(n_blocked)
    res["daily_skips"] = {str(k.date()): sorted(v) for k, v in daily_skips.items()}
    return res


def attach_daily_series_ungated(res: dict) -> dict:
    """Ensure ungated result has _flat/_to/_npos series for paired window metrics."""
    if "_flat_series" in res:
        return res
    # rebuild from packed equity only — approximate flat from n_pos if absent
    rets = res["daily_ret"]
    idx = pd.DatetimeIndex(rets.index)
    # Without per-day flat from ungated path, use constant avg — caller should re-run with instrumented fn
    res["_flat_series"] = pd.Series(float(res.get("pct_flat_days", 0.0)), index=idx)
    res["_to_series"] = pd.Series(float(res.get("ann_turnover", 0.0)) / 365.0, index=idx)
    res["_npos_series"] = pd.Series(float(res.get("avg_n_positions", 0.0)), index=idx)
    res["gate_name"] = "ungated"
    res["gate_col"] = None
    res["sigma_top_pct"] = 0.0
    return res


def run_ungated_instrumented(
    preds, panel, feat, universe, funding, cfg_a0: dict, tau_pct: float, horizon: int = 7
) -> dict:
    """Re-run ungated tranche capturing daily flat/turnover/n_pos series (top_pct=0)."""
    # Use column gate with dummy col and top_pct=0 → never blocks
    dummy = preds[["date", "symbol"]].drop_duplicates().copy()
    dummy["date"] = pd.to_datetime(dummy["date"], utc=True)
    dummy["_gate_dummy"] = 0.0
    port = cfg_a0["portfolio"]
    return run_tranche_with_column_gate(
        preds,
        panel,
        feat,
        universe,
        dummy,
        gate_col="_gate_dummy",
        horizon=horizon,
        tau_pct=tau_pct,
        top_pct=0.0,
        gate_name="ungated",
        gross_limit=port["gross_limit"],
        fee_bps=port["taker_fee_bps"],
        slip_bps=port["slippage_bps"],
        lag=0,
        apply_funding=True,
        funding=funding,
    )


def mean_daily_rank_corr(a: pd.DataFrame, col_a: str, col_b: str, window: str = "full") -> dict:
    """Mean cross-sectional Spearman between col_a and col_b per date."""
    df = a[["date", "symbol", col_a, col_b]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if window == "pre":
        df = df[df["date"] < CUTOFF]
    elif window == "post":
        df = df[df["date"] >= CUTOFF]
    ics = []
    for _, g in df.groupby("date", sort=False):
        x = g[col_a].astype(float).values
        y = g[col_b].astype(float).values
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        r = stats.spearmanr(x, y)
        c = getattr(r, "correlation", None)
        if c is None:
            c = getattr(r, "statistic", np.nan)
        c = float(np.asarray(c, dtype=float).reshape(-1)[0])
        if np.isfinite(c):
            ics.append(c)
    return {
        "n_days": len(ics),
        "mean_spearman": float(np.mean(ics)) if ics else float("nan"),
        "median_spearman": float(np.median(ics)) if ics else float("nan"),
    }


def skip_overlap(
    skips_ref: dict[str, list[str]],
    skips_ctrl: dict[str, list[str]],
) -> dict:
    """Fraction of kr_sigma-skipped entries also skipped by control (same days)."""
    n_ref = 0
    n_overlap = 0
    for d, refs in skips_ref.items():
        rset = set(refs)
        if not rset:
            continue
        cset = set(skips_ctrl.get(d, []))
        n_ref += len(rset)
        n_overlap += len(rset & cset)
    return {
        "n_ref_skips": int(n_ref),
        "n_overlap": int(n_overlap),
        "overlap_frac": float(n_overlap / n_ref) if n_ref else float("nan"),
    }


def select_best_x(rows: list[dict], gate_prefix: str) -> dict | None:
    """Same X-selection procedure: maximize post-cutoff Sharpe among X in grid for that gate."""
    cands = [r for r in rows if r.get("gate") and str(r["gate"]).startswith(gate_prefix) and r.get("X", 0) > 0]
    if not cands:
        return None
    return max(cands, key=lambda r: (r.get("sharpe_post") if np.isfinite(r.get("sharpe_post", np.nan)) else -1e9))


def apply_adoption_rule(
    ungated: dict,
    kr_best: dict | None,
    ctrl_best: dict | None,
    redundancy: dict,
) -> dict:
    """
    Pre-registered adoption rule.
    X-selection: independently pick best X by post-cutoff Sharpe for kr_sigma and for each control
    family; 'best control' = max post Sharpe among C1/C2/C3 best-X rows.
    """
    red_flag = bool(redundancy.get("presumed_redundant", False))
    details = {
        "redundancy_flag": red_flag,
        "kr_best": kr_best,
        "ctrl_best": ctrl_best,
        "ungated_post": ungated.get("sharpe_post"),
    }
    if kr_best is None or ctrl_best is None:
        # fall through to REDUNDANT branch without kr
        delta_vs_ctrl = float("nan")
        a_ok = False
        b_ok = False
    else:
        delta_vs_ctrl = float(kr_best["sharpe_post"] - ctrl_best["sharpe_post"])
        a_ok = delta_vs_ctrl >= 0.10
        b_ok = float(kr_best["sharpe_full"]) >= float(ctrl_best["sharpe_full"]) - 0.10
        c_ok = not red_flag
        details.update({"a_ok": a_ok, "b_ok": b_ok, "c_ok": c_ok, "delta_vs_ctrl_post": delta_vs_ctrl})
        if a_ok and b_ok and c_ok:
            return {
                "verdict": "ADOPT_KR_SIGMA",
                "adopt": kr_best,
                "rule": ADOPTION_RULE,
                "details": details,
            }

    # REDUNDANT branch
    if ctrl_best is not None:
        d_ctrl = float(ctrl_best["sharpe_post"] - ungated.get("sharpe_post", np.nan))
        details["ctrl_vs_ungated_post"] = d_ctrl
        if np.isfinite(d_ctrl) and d_ctrl >= 0.10:
            return {
                "verdict": "REDUNDANT_ADOPT_BEST_CONTROL",
                "adopt": ctrl_best,
                "rule": ADOPTION_RULE,
                "details": details,
            }
    return {
        "verdict": "REDUNDANT_ADOPT_NO_GATE",
        "adopt": ungated,
        "rule": ADOPTION_RULE,
        "details": details,
    }


def per_year_stats(daily_ret: pd.Series) -> list[dict]:
    s = daily_ret.copy()
    s.index = pd.to_datetime(s.index, utc=True)
    rows = []
    for year, g in s.groupby(s.index.year):
        if year < 2022:
            continue
        if len(g) < 20:
            continue
        eq = (1.0 + g.fillna(0.0)).cumprod()
        years = len(g) / 365.0
        cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-6)) - 1) if len(eq) > 1 else 0.0
        sharpe = float(g.mean() / g.std() * np.sqrt(365)) if g.std() > 0 else 0.0
        maxdd = float((eq / eq.cummax() - 1.0).min())
        rows.append(
            {
                "year": int(year),
                "n_days": int(len(g)),
                "net_sharpe": sharpe,
                "cagr": cagr,
                "max_dd": maxdd,
                "total_return": float(eq.iloc[-1] - 1.0),
            }
        )
    return rows


def trailing_12m(daily_ret: pd.Series) -> dict:
    s = daily_ret.copy()
    s.index = pd.to_datetime(s.index, utc=True)
    if s.empty:
        return {"net_sharpe": float("nan"), "total_return": float("nan"), "positive_expectancy": False}
    end = s.index.max()
    start = end - pd.Timedelta(days=365)
    g = s[s.index >= start]
    if len(g) < 60:
        return {"net_sharpe": float("nan"), "total_return": float("nan"), "n_days": int(len(g)), "positive_expectancy": False}
    sharpe = float(g.mean() / g.std() * np.sqrt(365)) if g.std() > 0 else 0.0
    eq = (1.0 + g.fillna(0.0)).cumprod()
    ret = float(eq.iloc[-1] - 1.0)
    # positive expectancy ≈ positive mean daily net return (and preferably positive Sharpe)
    pos = bool(g.mean() > 0)
    return {
        "net_sharpe": sharpe,
        "total_return": ret,
        "mean_daily": float(g.mean()),
        "n_days": int(len(g)),
        "start": str(start.date()),
        "end": str(end.date()),
        "positive_expectancy": pos,
    }


def rolling_sharpe(daily_ret: pd.Series, window: int = 180) -> pd.Series:
    s = daily_ret.copy()
    s.index = pd.to_datetime(s.index, utc=True)
    mu = s.rolling(window, min_periods=max(60, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(60, window // 3)).std()
    return (mu / sd) * np.sqrt(365)
