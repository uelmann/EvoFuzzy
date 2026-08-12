"""Uncertainty-gate test on frozen A0 tranche portfolio."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from baseline.portfolio import (
    _apply_hedge,
    _attach_aux,
    _funding_wide,
    _hard_threshold_state,
    _pack_metrics,
    _prepare_returns,
    _size_book,
)

CUTOFF = pd.Timestamp("2025-08-17", tz="UTC")


def _slice_metrics(res: dict, window: str) -> dict:
    """Recompute Sharpe / turnover / flat% on equity daily series for pre/post/full."""
    if "daily_ret" not in res:
        return {"window": window, "error": "no daily_ret"}
    rets = res["daily_ret"]
    if not isinstance(rets, pd.Series):
        rets = pd.Series(rets)
    rets.index = pd.to_datetime(rets.index, utc=True)
    if window == "pre":
        rets = rets[rets.index < CUTOFF]
    elif window == "post":
        rets = rets[rets.index >= CUTOFF]
    n = len(rets)
    if n < 5:
        return {
            "window": window,
            "n_days": int(n),
            "net_sharpe": float("nan"),
            "ann_turnover": float("nan"),
            "pct_flat_days": float("nan"),
        }
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0
    # approximate flat days / turnover from packed fields if present as series — fallback to full-period scalars
    out = {
        "window": window,
        "n_days": int(n),
        "net_sharpe": sharpe,
        "net_total_pnl": float(rets.sum()),
        "ann_turnover": float(res.get("ann_turnover", float("nan"))),
        "pct_flat_days": float(res.get("pct_flat_days", float("nan"))),
    }
    return out


def run_tranche_with_sigma_gate(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    kronos_raw: pd.DataFrame,
    horizon: int,
    tau_pct: float,
    sigma_top_pct: float,
    gross_limit: float = 1.0,
    fee_bps: float = 5.0,
    slip_bps: float = 3.0,
    lag: int = 0,
    apply_funding: bool = True,
    funding: pd.DataFrame | None = None,
) -> dict:
    """
    Frozen A0 tranche logic + one rule: skip NEW entries when kr_sigma_h{horizon}
    is in the top sigma_top_pct% cross-sectionally that day.
    Existing positions may still exit via hard threshold.
    """
    h = int(horizon)
    sigma_col = f"kr_sigma_h{h}"
    df = _attach_aux(preds, feat, universe)
    k = kronos_raw.copy()
    k["date"] = pd.to_datetime(k["date"], utc=True)
    if sigma_col not in k.columns:
        # fall back to h7 if needed
        sigma_col = "kr_sigma_h7"
    df = df.merge(k[["date", "symbol", sigma_col]], on=["date", "symbol"], how="left")

    rets = _prepare_returns(panel)
    fund_wide = _funding_wide(funding) if apply_funding else pd.DataFrame()
    dates = sorted(df["date"].unique())
    if len(dates) < h + 5 + int(lag):
        return {"error": "not enough dates", "tau_pct": tau_pct, "sigma_top_pct": sigma_top_pct}

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

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt].copy()
        kslot = i % h
        prev_ak = alphas[kslot].copy()
        raw_state = _hard_threshold_state(day, tau)

        # Cross-sectional sigma gate for NEW entries only
        sig = day.set_index("symbol")[sigma_col]
        finite = sig[np.isfinite(sig)]
        blocked = set()
        if len(finite) >= 5 and sigma_top_pct > 0:
            thr = float(np.nanpercentile(finite.values, 100.0 - float(sigma_top_pct)))
            blocked = set(finite[finite >= thr].index.astype(str))

        new_state = {}
        prev_state = states[kslot]
        for sym, side in raw_state.items():
            was = prev_state.get(sym, 0)
            if was == 0 and sym in blocked:
                n_blocked += 1
                continue  # skip new entry
            new_state[sym] = side
        # allow exits: if was in prev and not in raw_state, it's already absent

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
        hedge_sum = 0.0
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
            hedge_sum += hk
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
        f"tranche_sigma_gate_{int(sigma_top_pct)}",
        horizon,
        lag,
        apply_funding,
        dict(sym_contrib),
        dict(side_days),
    )
    res["sigma_top_pct"] = float(sigma_top_pct)
    res["n_blocked_entries"] = int(n_blocked)
    res["by_window"] = {
        "full": _slice_metrics(res, "full"),
        "pre": _slice_metrics(res, "pre"),
        "post": _slice_metrics(res, "post"),
    }
    # window-specific flat% from daily flat flags
    flat_s = pd.Series(flat, index=pd.DatetimeIndex(eq_dates))
    to_s = pd.Series([ee + rs + hg for ee, rs, hg in zip(to_ee, to_rs, to_hg)], index=flat_s.index)
    for w, mask in (
        ("full", pd.Series(True, index=flat_s.index)),
        ("pre", flat_s.index < CUTOFF),
        ("post", flat_s.index >= CUTOFF),
    ):
        m = mask if isinstance(mask, pd.Series) else mask
        if int(np.sum(m)) >= 5:
            res["by_window"][w]["pct_flat_days"] = float(flat_s.loc[m].mean())
            res["by_window"][w]["ann_turnover"] = float(to_s.loc[m].mean() * 365)
    return res


def run_gate_suite(
    preds_a0: pd.DataFrame,
    panel,
    feat,
    pit20,
    kronos_raw: pd.DataFrame,
    funding,
    cfg_a0: dict,
    tau_pct: float = 60.0,
    sigma_top_pcts: list[float] | None = None,
) -> dict:
    from baseline.portfolio import run_tranche_portfolio

    sigma_top_pcts = sigma_top_pcts or [10, 20, 30]
    port = cfg_a0["portfolio"]
    baseline = run_tranche_portfolio(
        preds_a0,
        panel,
        feat,
        pit20,
        horizon=7,
        tau_pct=tau_pct,
        exit_hysteresis=port["exit_hysteresis"],
        gross_limit=port["gross_limit"],
        fee_bps=port["taker_fee_bps"],
        slip_bps=port["slippage_bps"],
        lag=0,
        apply_funding=True,
        funding=funding,
    )
    baseline["sigma_top_pct"] = 0.0
    baseline["by_window"] = {
        "full": _slice_metrics(baseline, "full"),
        "pre": _slice_metrics(baseline, "pre"),
        "post": _slice_metrics(baseline, "post"),
    }
    flat_s = pd.Series(
        [1 if x == 0 else 0 for x in []],
    )
    # recompute flat/turnover windows from packed series if available
    if "daily_ret" in baseline:
        # pct_flat from full already; leave as is for baseline windows using scalars
        for w in ("full", "pre", "post"):
            baseline["by_window"][w]["ann_turnover"] = float(baseline.get("ann_turnover", float("nan")))
            baseline["by_window"][w]["pct_flat_days"] = float(baseline.get("pct_flat_days", float("nan")))
            if w != "full":
                # sharpe already windowed; turnover/flat remain full-period reference for ungated
                pass

    rows = [
        {
            "gate": "ungated",
            "sigma_top_pct": 0,
            **{f"{w}_{k}": baseline["by_window"][w].get(k) for w in ("full", "post") for k in ("net_sharpe", "ann_turnover", "pct_flat_days")},
            "net_sharpe_full": baseline.get("net_sharpe"),
            "net_sharpe_post": baseline["by_window"]["post"].get("net_sharpe"),
            "ann_turnover": baseline.get("ann_turnover"),
            "pct_flat_days": baseline.get("pct_flat_days"),
        }
    ]
    gated_results = {"ungated": baseline}
    for x in sigma_top_pcts:
        g = run_tranche_with_sigma_gate(
            preds_a0,
            panel,
            feat,
            pit20,
            kronos_raw,
            horizon=7,
            tau_pct=tau_pct,
            sigma_top_pct=float(x),
            gross_limit=port["gross_limit"],
            fee_bps=port["taker_fee_bps"],
            slip_bps=port["slippage_bps"],
            lag=0,
            apply_funding=True,
            funding=funding,
        )
        gated_results[f"top{int(x)}"] = g
        rows.append(
            {
                "gate": f"skip_kr_sigma_top{int(x)}pct",
                "sigma_top_pct": int(x),
                "net_sharpe_full": g.get("net_sharpe"),
                "net_sharpe_post": g["by_window"]["post"].get("net_sharpe"),
                "ann_turnover": g.get("ann_turnover"),
                "pct_flat_days": g.get("pct_flat_days"),
                "ann_turnover_post": g["by_window"]["post"].get("ann_turnover"),
                "pct_flat_days_post": g["by_window"]["post"].get("pct_flat_days"),
                "n_blocked_entries": g.get("n_blocked_entries"),
                "delta_sharpe_full": float(g.get("net_sharpe", np.nan) - baseline.get("net_sharpe", np.nan)),
                "delta_sharpe_post": float(
                    g["by_window"]["post"].get("net_sharpe", np.nan)
                    - baseline["by_window"]["post"].get("net_sharpe", np.nan)
                ),
            }
        )
        print(
            f"[gate] X={x}% fullSh={g.get('net_sharpe'):.3f} postSh={g['by_window']['post'].get('net_sharpe')} "
            f"blocked={g.get('n_blocked_entries')}",
            flush=True,
        )

    # Verdict: gate "wins" if any X improves post-cutoff net Sharpe vs ungated
    best = None
    for r in rows[1:]:
        if best is None or (
            np.isfinite(r.get("delta_sharpe_post", np.nan))
            and r["delta_sharpe_post"] > best.get("delta_sharpe_post", -1e9)
        ):
            best = r
    gate_verdict = "WIN" if best and best.get("delta_sharpe_post", -1) > 0 else "NO_LIFT"
    return {
        "tau_pct": tau_pct,
        "rows": rows,
        "best": best,
        "gate_verdict": gate_verdict,
        "results": gated_results,
    }
