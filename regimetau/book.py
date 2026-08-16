"""Tranche book with date-varying fold-train τ. Imports engine helpers; does not edit them."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from baseline.portfolio import (
    _apply_hedge,
    _attach_aux,
    _cost_rate_for_rank,
    _hard_threshold_state,
    _pack_metrics,
    _prepare_returns,
    _funding_wide,
    _rank_lookup_frame,
    _rank_of,
    _size_book,
    tau_by_date_fold_train,
)
from regimetau.constants import (
    REGIME_BASE,
    REGIME_HIGH,
    REGIME_LOW,
)


def _tau_for_day(dt, regime_code: int, tau_base, tau_high, tau_low, fallback: float) -> float:
    if int(regime_code) == REGIME_HIGH:
        src = tau_high
    elif int(regime_code) == REGIME_LOW:
        src = tau_low
    else:
        src = tau_base
    v = src.get(dt)
    if v is None or not np.isfinite(v):
        v = fallback
    return float(v)


def run_regime_tau_portfolio(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    tau_pct_base: float,
    tau_pct_high: float,
    tau_pct_low: float,
    regime: pd.Series,
    folds: list | None = None,
    gross_limit: float = 1.0,
    fee_bps: float = 5.0,
    slip_bps: float = 3.0,
    lag: int = 0,
    apply_funding: bool = True,
    funding: pd.DataFrame | None = None,
    tiered_costs: bool = False,
    fee_bps_next: float = 10.0,
    slip_bps_next: float = 8.0,
    liq_cap_adv_frac: float | None = None,
    nominal_book_usd: float = 1_000_000.0,
    rank_universe: pd.DataFrame | None = None,
) -> dict:
    """Same tranche mechanics as `run_tranche_portfolio`, τ picked by regime."""
    h = int(horizon)
    df = _attach_aux(preds, feat, universe)
    rets = _prepare_returns(panel)
    fund_wide = _funding_wide(funding) if apply_funding else pd.DataFrame()
    rank_src = rank_universe if rank_universe is not None else universe
    rank_long = _rank_lookup_frame(rank_src)
    dates = sorted(df["date"].unique())
    if len(dates) < h + 5 + int(lag):
        return {"error": "not enough dates", "tau_pct": tau_pct_base, "variant": "regime-tau"}
    abs_scores = df["score"].abs().dropna().values
    if len(abs_scores) < 50:
        return {"error": "not enough scores", "tau_pct": tau_pct_base, "variant": "regime-tau"}

    tau_fallback = float(np.percentile(abs_scores, tau_pct_base))
    tau_base = tau_by_date_fold_train(df, dates, tau_pct_base, folds)
    tau_high = tau_by_date_fold_train(df, dates, tau_pct_high, folds)
    tau_low = tau_by_date_fold_train(df, dates, tau_pct_low, folds)
    for dt in dates:
        tau_base.setdefault(dt, 1e9)
        tau_high.setdefault(dt, 1e9)
        tau_low.setdefault(dt, 1e9)

    reg = regime.copy()
    reg.index = pd.DatetimeIndex(pd.to_datetime(reg.index, utc=True))
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
    daily_long, daily_short = [], []
    daily_gross_deployed, daily_gross_full = [], []
    daily_reg, daily_tau = [], []
    name_alpha_pnl: dict[str, float] = defaultdict(float)
    traded_ranks: list[float] = []
    n_forced = 0
    forced_pnl = 0.0

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt]
        rank_day = rank_long[rank_long["date"] == dt] if not rank_long.empty else day
        k = i % h
        prev_ak = alphas[k].copy()
        dt_n = pd.Timestamp(dt)
        if dt_n.tzinfo is None:
            dt_n = dt_n.tz_localize("UTC")
        else:
            dt_n = dt_n.tz_convert("UTC")
        lookup = dt_n.normalize()
        if lookup in reg.index:
            rcode = int(reg.loc[lookup])
        else:
            # last known label; BASE if none
            prior = reg[reg.index <= lookup]
            rcode = int(prior.iloc[-1]) if len(prior) else REGIME_BASE
        tau_t = _tau_for_day(dt, rcode, tau_base, tau_high, tau_low, tau_fallback)
        new_state = _hard_threshold_state(day, tau_t)

        univ = set(day["symbol"])
        for sym, side in list(states[k].items()):
            if sym not in univ and new_state.get(sym, 0) != side:
                n_forced += 1
                key = (k, sym)
                forced_pnl += float(sym_pnl.get(key, 0.0))

        for sym, side in list(states[k].items()):
            if new_state.get(sym, 0) != side:
                key = (k, sym)
                if key in entry_date:
                    hold_days.append(max(1, int((dt - entry_date[key]).days)))
                    trade_pnls.append(float(sym_pnl.get(key, 0.0)))
                    entry_date.pop(key, None)
                    sym_pnl.pop(key, None)
        for sym, side in new_state.items():
            if states[k].get(sym, 0) != side:
                entry_date[(k, sym)] = dt
                sym_pnl[(k, sym)] = 0.0
        states[k] = new_state
        alphas[k] = _size_book(
            day,
            new_state,
            tg,
            liq_cap_adv_frac=liq_cap_adv_frac,
            nominal_book_usd=nominal_book_usd,
        )

        full = pd.Series(dtype=float)
        alpha = pd.Series(dtype=float)
        hedge_sum = 0.0
        for tk in range(h):
            ak = alphas[tk]
            if ak.empty:
                continue
            if tk != k:
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

        idx = alphas[k].index.union(prev_ak.index)
        ee = 0.5 * float((alphas[k].reindex(idx).fillna(0.0) - prev_ak.reindex(idx).fillna(0.0)).abs().sum())
        rs = 0.0
        hg = 0.5 * abs(applied_hedge - prev_hedge)

        fidx = applied_full.index.union(prev_full.index)
        f = applied_full.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        dw = (f - pf).abs()
        turnover = 0.5 * float(dw.sum())
        if tiered_costs:
            cost = 0.0
            for s, mag in dw.items():
                rnk = _rank_of(day, rank_day, s)
                if s == "BTCUSDT" and not np.isfinite(rnk):
                    rnk = 1.0
                cost += 0.5 * float(mag) * _cost_rate_for_rank(
                    rnk, fee_bps, slip_bps, True, fee_bps_next, slip_bps_next
                )
        else:
            cost = turnover * cost_rate

        nxt = dates[i + 1]
        gross_r = hedge_r = long_r = short_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in applied_alpha.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    ri = float(rrow[s])
                    contrib = float(wi) * ri
                    gross_r += contrib
                    name_alpha_pnl[s] = name_alpha_pnl.get(s, 0.0) + contrib
                    if float(wi) > 0:
                        long_r += contrib
                    elif float(wi) < 0:
                        short_r += contrib
                    sym_contrib[s] = sym_contrib.get(s, 0.0) + contrib
                    key = None
                    for tk in range(h):
                        if s in states[tk]:
                            key = (tk, s)
                            break
                    if key in sym_pnl:
                        sym_pnl[key] += contrib
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
        daily_long.append(long_r)
        daily_short.append(short_r)
        daily_gross_deployed.append(float(applied_alpha.abs().sum()) if len(applied_alpha) else 0.0)
        daily_gross_full.append(float(applied_full.abs().sum()) if len(applied_full) else 0.0)
        daily_reg.append(int(rcode))
        daily_tau.append(float(tau_t))
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
        day_ranks = []
        if len(applied_alpha):
            for s, wi in applied_alpha.items():
                if abs(float(wi)) <= 1e-12:
                    continue
                rnk = _rank_of(day, rank_day, s)
                if np.isfinite(rnk):
                    day_ranks.append(float(rnk))
        if day_ranks:
            traded_ranks.append(float(np.mean(day_ranks)))
        prev_full, prev_hedge = applied_full, applied_hedge

        if i % 60 == 0:
            print(
                f"[portfolio regime-tau h={h} baseτ={tau_pct_base} "
                f"state={rcode} τ={tau_t:.5f}] day {i}/{len(dates)} L={nl} S={ns} to={turnover:.3f}",
                flush=True,
            )

    packed = _pack_metrics(
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
        tau_pct_base,
        tau_fallback,
        "regime-tau",
        horizon,
        lag,
        apply_funding,
        dict(sym_contrib),
        dict(side_days),
    )
    packed["tau_mode"] = "fold_train_regime"
    packed["tau_pct_high"] = float(tau_pct_high)
    packed["tau_pct_low"] = float(tau_pct_low)
    packed["tiered_costs"] = bool(tiered_costs)
    packed["liq_cap_adv_frac"] = liq_cap_adv_frac
    packed["nominal_book_usd"] = float(nominal_book_usd)
    packed["avg_traded_rank"] = float(np.mean(traded_ranks)) if traded_ranks else float("nan")
    packed["apply_beta_hedge"] = True
    packed["long_only"] = False
    idx = packed["daily_ret"].index
    packed["daily_long"] = pd.Series(daily_long, index=idx, dtype=float)
    packed["daily_short"] = pd.Series(daily_short, index=idx, dtype=float)
    packed["daily_gross_deployed"] = pd.Series(daily_gross_deployed, index=idx, dtype=float)
    packed["daily_gross_full"] = pd.Series(daily_gross_full, index=idx, dtype=float)
    packed["daily_regime"] = pd.Series(daily_reg, index=idx, dtype=int)
    packed["daily_tau"] = pd.Series(daily_tau, index=idx, dtype=float)
    packed["avg_gross_deployed"] = float(np.mean(daily_gross_deployed)) if daily_gross_deployed else 0.0
    packed["avg_gross_full"] = float(np.mean(daily_gross_full)) if daily_gross_full else 0.0
    packed["name_alpha_pnl"] = dict(name_alpha_pnl)
    packed["n_forced_exits"] = int(n_forced)
    packed["forced_exit_pnl"] = float(forced_pnl)
    packed["high_frac"] = float(np.mean(np.asarray(daily_reg) == REGIME_HIGH)) if daily_reg else 0.0
    packed["low_frac"] = float(np.mean(np.asarray(daily_reg) == REGIME_LOW)) if daily_reg else 0.0
    packed["base_frac"] = float(np.mean(np.asarray(daily_reg) == REGIME_BASE)) if daily_reg else 0.0
    packed["long_short_identity_gap"] = float(
        np.sum(daily_gross) - np.sum(daily_long) - np.sum(daily_short)
    )
    return packed
