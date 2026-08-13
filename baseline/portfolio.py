"""Portfolio backtest with funding accrual, execution lag, and attribution hooks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _prepare_returns(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    return np.log(close / close.shift(1))


def _funding_wide(funding: pd.DataFrame | None) -> pd.DataFrame:
    if funding is None or funding.empty:
        return pd.DataFrame()
    f = funding.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    return f.pivot(index="date", columns="symbol", values="funding_rate").sort_index()


def _attach_aux(preds: pd.DataFrame, feat: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    uni = universe.copy()
    uni["date"] = pd.to_datetime(uni["date"], utc=True)
    df = df.merge(uni[["date", "symbol"]], on=["date", "symbol"], how="inner")

    keep = ["date", "symbol", "close"]
    for c in ["yz_vol_30_raw", "yz_vol_30", "beta_btc_60_raw", "beta_btc_60"]:
        if c in feat.columns:
            keep.append(c)
    aux = feat[keep].copy()
    aux["date"] = pd.to_datetime(aux["date"], utc=True)
    df = df.merge(aux, on=["date", "symbol"], how="left")
    if "yz_vol_30_raw" not in df.columns:
        df["yz_vol_30_raw"] = df["yz_vol_30"] if "yz_vol_30" in df.columns else np.nan
    if "beta_btc_60_raw" not in df.columns:
        df["beta_btc_60_raw"] = df["beta_btc_60"] if "beta_btc_60" in df.columns else np.nan
    return df


def _inv_vol(day: pd.DataFrame, sym: str) -> float:
    row = day[day["symbol"] == sym]
    if row.empty:
        return 1.0 / 0.02
    v = float(row["yz_vol_30_raw"].iloc[0])
    if not np.isfinite(v) or v <= 0:
        v = float(row["yz_vol_30"].iloc[0]) if "yz_vol_30" in row.columns else 0.02
    if not np.isfinite(v) or v <= 0:
        v = 0.02
    return 1.0 / v


def _size_book(day: pd.DataFrame, state: dict[str, int], gross_limit: float) -> pd.Series:
    longs = [s for s, v in state.items() if v > 0]
    shorts = [s for s, v in state.items() if v < 0]
    w: dict[str, float] = {}
    if longs:
        iv = {s: _inv_vol(day, s) for s in longs}
        ssum = sum(iv.values()) or 1.0
        for s, v in iv.items():
            w[s] = 0.5 * gross_limit * v / ssum
    if shorts:
        iv = {s: _inv_vol(day, s) for s in shorts}
        ssum = sum(iv.values()) or 1.0
        for s, v in iv.items():
            w[s] = -0.5 * gross_limit * v / ssum
    return pd.Series(w, dtype=float)


def _beta_of(day: pd.DataFrame, sym: str) -> float:
    row = day[day["symbol"] == sym]
    if row.empty:
        return 0.0
    b = float(row["beta_btc_60_raw"].iloc[0])
    return b if np.isfinite(b) else 0.0


def _apply_hedge(day: pd.DataFrame, alpha: pd.Series) -> tuple[pd.Series, float]:
    if alpha.empty:
        return alpha.copy(), 0.0
    port_beta = sum(float(wi) * _beta_of(day, s) for s, wi in alpha.items())
    w_btc = -port_beta
    full = alpha.copy()
    if abs(w_btc) > 1e-12:
        full.loc["BTCUSDT"] = float(full.get("BTCUSDT", 0.0)) + w_btc
    return full, w_btc


def _update_state_hysteresis(state, day, tau, exit_thr):
    new_state = dict(state)
    for _, row in day.iterrows():
        sym = row["symbol"]
        s = float(row["score"])
        cur = new_state.get(sym, 0)
        if cur == 0:
            if s > tau:
                new_state[sym] = 1
            elif s < -tau:
                new_state[sym] = -1
        elif cur > 0 and s < exit_thr:
            new_state[sym] = 0
        elif cur < 0 and s > -exit_thr:
            new_state[sym] = 0
    univ = set(day["symbol"])
    return {k: v for k, v in new_state.items() if k in univ and v != 0}


def _hard_threshold_state(day, tau):
    st = {}
    for _, row in day.iterrows():
        s = float(row["score"])
        if s > tau:
            st[row["symbol"]] = 1
        elif s < -tau:
            st[row["symbol"]] = -1
    return st


def _sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) and x.std() > 0 else 0.0


def _funding_pnl(full: pd.Series, fund_wide: pd.DataFrame, dt: pd.Timestamp) -> float:
    """Longs pay positive funding: pnl = -Σ w_i * funding_rate_i (daily sum of 8h)."""
    if fund_wide.empty or dt not in fund_wide.index or full.empty:
        return 0.0
    row = fund_wide.loc[dt]
    pnl = 0.0
    for s, wi in full.items():
        if s in row.index and np.isfinite(row[s]):
            pnl += -float(wi) * float(row[s])
    return float(pnl)


def _pack_metrics(
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
    variant,
    horizon,
    lag: int,
    apply_funding: bool,
    sym_contrib: dict[str, float],
    side_days: dict[str, dict],
) -> dict:
    rets_s = pd.Series(daily_net, index=pd.DatetimeIndex(eq_dates))
    simple_eq = (1.0 + rets_s.fillna(0.0)).cumprod()
    n = len(rets_s)
    years = n / 365.0
    cagr = float(simple_eq.iloc[-1] ** (1 / max(years, 1e-6)) - 1) if n > 1 else 0.0
    measured_to = [ee + rs + hg for ee, rs, hg in zip(to_ee, to_rs, to_hg)]
    gross_s = pd.Series(daily_gross, index=rets_s.index)
    hedge_s = pd.Series(daily_hedge, index=rets_s.index)
    cost_s = pd.Series(daily_cost, index=rets_s.index)
    fund_s = pd.Series(daily_funding, index=rets_s.index)
    # identity check on summed simple returns
    identity_gap = float(np.sum(daily_net) - (np.sum(daily_gross) + np.sum(daily_hedge) - np.sum(daily_cost) + np.sum(daily_funding)))
    n_flat = int(np.sum(flat)) if flat else n
    return {
        "tau_pct": tau_pct,
        "tau": tau,
        "variant": variant,
        "horizon": horizon,
        "lag": lag,
        "funding_on": bool(apply_funding),
        "net_sharpe": _sharpe(rets_s),
        "gross_sharpe": _sharpe(gross_s),
        "net_cagr": cagr,
        "max_drawdown": float((simple_eq / simple_eq.cummax() - 1.0).min()) if n else 0.0,
        "total_return": float(simple_eq.iloc[-1] - 1.0) if n else 0.0,
        "avg_n_positions": float(np.mean(n_pos)) if n_pos else 0.0,
        "avg_n_long": float(np.mean(n_long)) if n_long else 0.0,
        "avg_n_short": float(np.mean(n_short)) if n_short else 0.0,
        "n_flat_days": n_flat,
        "n_days": int(n),
        "pct_flat_days": float(np.mean(flat)) if flat else 1.0,
        "ann_turnover": float(np.mean(measured_to) * 365) if measured_to else 0.0,
        # cumulative simple-return units (sum of daily contributions)
        "gross_total_pnl": float(gross_s.sum()),
        "hedge_total_pnl": float(hedge_s.sum()),
        "cost_drag": float(cost_s.sum()),
        "funding_total_pnl": float(fund_s.sum()),
        "net_total_pnl": float(rets_s.sum()),
        "identity_gap": identity_gap,
        # annualized mean return fractions (365 * daily mean)
        "gross_ann_return": float(gross_s.mean() * 365) if n else 0.0,
        "hedge_ann_return": float(hedge_s.mean() * 365) if n else 0.0,
        "cost_ann_drag": float(cost_s.mean() * 365) if n else 0.0,
        "funding_ann_return": float(fund_s.mean() * 365) if n else 0.0,
        "net_ann_return": float(rets_s.mean() * 365) if n else 0.0,
        "avg_holding_days": float(np.mean(hold_days)) if hold_days else float("nan"),
        "n_closed_trades": int(len(hold_days)),
        "trade_pnl_p10": float(np.percentile(trade_pnls, 10)) if trade_pnls else float("nan"),
        "trade_pnl_p50": float(np.percentile(trade_pnls, 50)) if trade_pnls else float("nan"),
        "trade_pnl_p90": float(np.percentile(trade_pnls, 90)) if trade_pnls else float("nan"),
        "turnover_entry_exit_ann": float(np.mean(to_ee) * 365) if to_ee else 0.0,
        "turnover_resize_ann": float(np.mean(to_rs) * 365) if to_rs else 0.0,
        "turnover_hedge_ann": float(np.mean(to_hg) * 365) if to_hg else 0.0,
        "equity": pd.DataFrame({"date": eq_dates, "equity": simple_eq.values}),
        "daily_ret": rets_s,
        "daily_gross": gross_s,
        "daily_hedge": hedge_s,
        "daily_cost": cost_s,
        "daily_funding": fund_s,
        "sym_contrib": dict(sym_contrib),
        "side_days": side_days,
    }


def run_portfolio_backtest(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    tau_pct: float,
    exit_hysteresis: float = 0.6,
    gross_limit: float = 1.0,
    fee_bps: float = 5.0,
    slip_bps: float = 3.0,
    variant: str = "daily",
    lag: int = 0,
    apply_funding: bool = False,
    funding: pd.DataFrame | None = None,
) -> dict:
    df = _attach_aux(preds, feat, universe)
    rets = _prepare_returns(panel)
    fund_wide = _funding_wide(funding) if apply_funding else pd.DataFrame()
    dates = sorted(df["date"].unique())
    if len(dates) < 10 + int(lag):
        return {"error": "not enough dates", "tau_pct": tau_pct, "variant": variant, "lag": lag}
    abs_scores = df["score"].abs().dropna().values
    if len(abs_scores) < 50:
        return {"error": "not enough scores", "tau_pct": tau_pct, "variant": variant, "lag": lag}

    tau = float(np.percentile(abs_scores, tau_pct))
    exit_thr = exit_hysteresis * tau
    cost_rate = (fee_bps + slip_bps) * 1e-4

    state: dict[str, int] = {}
    entry_date: dict[str, pd.Timestamp] = {}
    hold_days: list[int] = []
    trade_pnls: list[float] = []
    sym_pnl: dict[str, float] = defaultdict(float)
    sym_contrib: dict[str, float] = defaultdict(float)
    side_days: dict[str, dict] = defaultdict(lambda: {"long_days": 0, "short_days": 0})

    target_hist: list[pd.Series] = []
    full_hist: list[pd.Series] = []
    prev_full = pd.Series(dtype=float)
    prev_hedge = 0.0

    daily_net, daily_gross, daily_hedge, daily_cost, daily_funding = [], [], [], [], []
    to_ee, to_rs, to_hg = [], [], []
    n_pos, n_long, n_short, flat, eq_dates = [], [], [], [], []

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt]
        new_state = _update_state_hysteresis(state, day, tau, exit_thr)
        for sym, side in list(state.items()):
            if new_state.get(sym, 0) != side and sym in entry_date:
                hold_days.append(max(1, int((dt - entry_date[sym]).days)))
                trade_pnls.append(float(sym_pnl.get(sym, 0.0)))
                entry_date.pop(sym, None)
                sym_pnl.pop(sym, None)
        for sym, side in new_state.items():
            if state.get(sym, 0) != side:
                entry_date[sym] = dt
                sym_pnl[sym] = 0.0
        state = new_state
        for sym, side in state.items():
            if side > 0:
                side_days[sym]["long_days"] += 1
            elif side < 0:
                side_days[sym]["short_days"] += 1

        alpha = _size_book(day, state, gross_limit)
        full, hedge_w = _apply_hedge(day, alpha)
        target_hist.append(alpha)
        full_hist.append(full)

        # applied weights: lag days delayed
        if i < lag:
            applied_alpha = pd.Series(dtype=float)
            applied_full = pd.Series(dtype=float)
            applied_hedge = 0.0
        else:
            applied_alpha = target_hist[i - lag]
            applied_full = full_hist[i - lag]
            applied_hedge = float(applied_full.get("BTCUSDT", 0.0) - applied_alpha.get("BTCUSDT", 0.0))

        fidx = applied_full.index.union(prev_full.index)
        f = applied_full.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        turnover = 0.5 * float((f - pf).abs().sum())
        cost = turnover * cost_rate

        # turnover attribution vs previous applied
        idx = applied_alpha.index.union(prev_full.index)
        a = applied_alpha.reindex(idx).fillna(0.0)
        # approximate prev alpha from prev_full without clean split — use target hist
        prev_a = target_hist[i - lag - 1] if (i - lag - 1) >= 0 else pd.Series(dtype=float)
        pa = prev_a.reindex(idx).fillna(0.0)
        ee = rs = 0.0
        for sym in idx:
            if sym == "BTCUSDT":
                continue
            prev_side = int(np.sign(pa.get(sym, 0.0)))
            cur_side = int(np.sign(a.get(sym, 0.0)))
            dw = abs(float(a.get(sym, 0.0) - pa.get(sym, 0.0)))
            if prev_side != cur_side:
                ee += dw
            else:
                rs += dw
        hg = abs(float(applied_hedge) - float(prev_hedge))

        nxt = dates[i + 1]
        gross_r = hedge_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in applied_alpha.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    ri = float(rrow[s])
                    contrib = float(wi) * ri
                    gross_r += contrib
                    sym_pnl[s] = sym_pnl.get(s, 0.0) + contrib
                    sym_contrib[s] = sym_contrib.get(s, 0.0) + contrib
            if "BTCUSDT" in rrow.index and np.isfinite(rrow["BTCUSDT"]):
                hedge_r = float(applied_hedge) * float(rrow["BTCUSDT"])
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
        to_ee.append(0.5 * ee)
        to_rs.append(0.5 * rs)
        to_hg.append(0.5 * hg)
        nl = int((applied_alpha > 0).sum()) if len(applied_alpha) else 0
        ns = int((applied_alpha < 0).sum()) if len(applied_alpha) else 0
        n_long.append(nl)
        n_short.append(ns)
        n_pos.append(nl + ns)
        flat.append(1 if nl + ns == 0 else 0)
        eq_dates.append(nxt)
        prev_full, prev_hedge = applied_full, applied_hedge

        if i % 60 == 0:
            print(
                f"[portfolio {variant} τ={tau_pct} lag={lag} fund={apply_funding}] "
                f"day {i}/{len(dates)} npos={n_pos[-1]} net={net:.5f}",
                flush=True,
            )

    return _pack_metrics(
        daily_net, daily_gross, daily_hedge, daily_cost, daily_funding,
        to_ee, to_rs, to_hg, n_pos, n_long, n_short, flat, eq_dates, hold_days, trade_pnls,
        tau_pct, tau, variant, horizon, lag, apply_funding, dict(sym_contrib), dict(side_days),
    )


def run_tranche_portfolio(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    tau_pct: float,
    exit_hysteresis: float = 0.6,
    gross_limit: float = 1.0,
    fee_bps: float = 5.0,
    slip_bps: float = 3.0,
    lag: int = 0,
    apply_funding: bool = False,
    funding: pd.DataFrame | None = None,
    tau_mode: str = "pooled",
) -> dict:
    del exit_hysteresis
    h = int(horizon)
    df = _attach_aux(preds, feat, universe)
    rets = _prepare_returns(panel)
    fund_wide = _funding_wide(funding) if apply_funding else pd.DataFrame()
    dates = sorted(df["date"].unique())
    if len(dates) < h + 5 + int(lag):
        return {"error": "not enough dates", "tau_pct": tau_pct, "variant": "tranche", "lag": lag}
    abs_scores = df["score"].abs().dropna().values
    if len(abs_scores) < 50:
        return {"error": "not enough scores", "tau_pct": tau_pct, "variant": "tranche", "lag": lag}

    tau = float(np.percentile(abs_scores, tau_pct))
    tau_by_date: dict = {}
    if str(tau_mode) == "expanding":
        # Causal: τ at t from |score| on dates strictly before t (training-window / PIT).
        by_d = df.groupby("date")["score"].apply(lambda s: s.abs().dropna().to_numpy())
        acc: list[float] = []
        for dt in dates:
            if len(acc) >= 50:
                tau_by_date[dt] = float(np.percentile(np.asarray(acc, dtype=float), tau_pct))
            else:
                tau_by_date[dt] = tau
            extra = by_d.get(dt)
            if extra is not None and len(extra):
                acc.extend(float(x) for x in extra)
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

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt]
        k = i % h
        prev_ak = alphas[k].copy()
        tau_t = float(tau_by_date.get(dt, tau)) if tau_by_date else tau
        new_state = _hard_threshold_state(day, tau_t)

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
        alphas[k] = _size_book(day, new_state, tg)

        full = pd.Series(dtype=float)
        alpha = pd.Series(dtype=float)
        hedge_sum = 0.0
        for tk in range(h):
            ak = alphas[tk]
            if ak.empty:
                continue
            if tk != k:
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

        idx = alphas[k].index.union(prev_ak.index)
        ee = 0.5 * float((alphas[k].reindex(idx).fillna(0.0) - prev_ak.reindex(idx).fillna(0.0)).abs().sum())
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

        if i % 60 == 0:
            print(
                f"[portfolio tranche h={h} τ={tau_pct} lag={lag} fund={apply_funding}] "
                f"day {i}/{len(dates)} L={nl} S={ns} to={turnover:.3f}",
                flush=True,
            )

    packed = _pack_metrics(
        daily_net, daily_gross, daily_hedge, daily_cost, daily_funding,
        to_ee, to_rs, to_hg, n_pos, n_long, n_short, flat, eq_dates, hold_days, trade_pnls,
        tau_pct, tau, "tranche", horizon, lag, apply_funding, dict(sym_contrib), dict(side_days),
    )
    packed["tau_mode"] = str(tau_mode)
    return packed
