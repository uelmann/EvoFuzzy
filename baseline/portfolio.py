"""Portfolio backtest: daily threshold + overlapping tranche (Jegadeesh–Titman)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _prepare_returns(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    return np.log(close / close.shift(1))


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


def _update_state_hysteresis(
    state: dict[str, int],
    day: pd.DataFrame,
    tau: float,
    exit_thr: float,
) -> dict[str, int]:
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


def _hard_threshold_state(day: pd.DataFrame, tau: float) -> dict[str, int]:
    st: dict[str, int] = {}
    for _, row in day.iterrows():
        s = float(row["score"])
        if s > tau:
            st[row["symbol"]] = 1
        elif s < -tau:
            st[row["symbol"]] = -1
    return st


def _sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) and x.std() > 0 else 0.0


def _pack_metrics(
    daily_net,
    daily_gross,
    daily_hedge,
    daily_cost,
    to_ee,
    to_rs,
    to_hg,
    n_pos,
    flat,
    eq_dates,
    hold_days,
    trade_pnls,
    tau_pct,
    tau,
    variant,
    horizon,
) -> dict:
    rets_s = pd.Series(daily_net, index=pd.DatetimeIndex(eq_dates))
    simple_eq = (1.0 + rets_s.fillna(0.0)).cumprod()
    n = len(rets_s)
    years = n / 365.0
    cagr = float(simple_eq.iloc[-1] ** (1 / max(years, 1e-6)) - 1) if n > 1 else 0.0
    measured_to = [ee + rs + hg for ee, rs, hg in zip(to_ee, to_rs, to_hg)]
    return {
        "tau_pct": tau_pct,
        "tau": tau,
        "variant": variant,
        "horizon": horizon,
        "net_sharpe": _sharpe(rets_s),
        "gross_sharpe": _sharpe(pd.Series(daily_gross)),
        "net_cagr": cagr,
        "max_drawdown": float((simple_eq / simple_eq.cummax() - 1.0).min()) if n else 0.0,
        "total_return": float(simple_eq.iloc[-1] - 1.0) if n else 0.0,
        "avg_n_positions": float(np.mean(n_pos)) if n_pos else 0.0,
        "pct_flat_days": float(np.mean(flat)) if flat else 1.0,
        "ann_turnover": float(np.mean(measured_to) * 365) if measured_to else 0.0,
        "gross_total_pnl": float(np.sum(daily_gross)),
        "hedge_total_pnl": float(np.sum(daily_hedge)),
        "cost_drag": float(np.sum(daily_cost)),
        "net_total_pnl": float(np.sum(daily_net)),
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
) -> dict:
    df = _attach_aux(preds, feat, universe)
    rets = _prepare_returns(panel)
    dates = sorted(df["date"].unique())
    if len(dates) < 10:
        return {"error": "not enough dates", "tau_pct": tau_pct, "variant": variant}
    abs_scores = df["score"].abs().dropna().values
    if len(abs_scores) < 50:
        return {"error": "not enough scores", "tau_pct": tau_pct, "variant": variant}

    tau = float(np.percentile(abs_scores, tau_pct))
    exit_thr = exit_hysteresis * tau
    cost_rate = (fee_bps + slip_bps) * 1e-4

    state: dict[str, int] = {}
    entry_date: dict[str, pd.Timestamp] = {}
    hold_days: list[int] = []
    trade_pnls: list[float] = []
    sym_pnl: dict[str, float] = defaultdict(float)

    prev_alpha = pd.Series(dtype=float)
    prev_full = pd.Series(dtype=float)
    prev_hedge = 0.0

    daily_net, daily_gross, daily_hedge, daily_cost = [], [], [], []
    to_ee, to_rs, to_hg = [], [], []
    n_pos, flat, eq_dates = [], [], []

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt]
        new_state = _update_state_hysteresis(state, day, tau, exit_thr)

        for sym, side in list(state.items()):
            if new_state.get(sym, 0) != side:
                if sym in entry_date:
                    hold_days.append(max(1, int((dt - entry_date[sym]).days)))
                    trade_pnls.append(float(sym_pnl.get(sym, 0.0)))
                    entry_date.pop(sym, None)
                    sym_pnl.pop(sym, None)
        for sym, side in new_state.items():
            if state.get(sym, 0) != side:
                entry_date[sym] = dt
                sym_pnl[sym] = 0.0
        state = new_state

        alpha = _size_book(day, state, gross_limit)
        full, hedge_w = _apply_hedge(day, alpha)

        idx = alpha.index.union(prev_alpha.index)
        a = alpha.reindex(idx).fillna(0.0)
        pa = prev_alpha.reindex(idx).fillna(0.0)
        ee = rs = 0.0
        for sym in idx:
            prev_side = int(np.sign(pa[sym]))
            cur_side = int(np.sign(a[sym]))
            dw = abs(float(a[sym] - pa[sym]))
            if prev_side != cur_side:
                ee += dw
            else:
                rs += dw
        hg = abs(float(hedge_w) - float(prev_hedge))

        fidx = full.index.union(prev_full.index)
        f = full.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        turnover = 0.5 * float((f - pf).abs().sum())
        cost = turnover * cost_rate

        nxt = dates[i + 1]
        gross_r = hedge_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in alpha.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    ri = float(rrow[s])
                    gross_r += float(wi) * ri
                    sym_pnl[s] = sym_pnl.get(s, 0.0) + float(wi) * ri
            if "BTCUSDT" in rrow.index and np.isfinite(rrow["BTCUSDT"]):
                hedge_r = float(hedge_w) * float(rrow["BTCUSDT"])
        net = gross_r + hedge_r - cost

        daily_net.append(net)
        daily_gross.append(gross_r)
        daily_hedge.append(hedge_r)
        daily_cost.append(cost)
        to_ee.append(0.5 * ee)
        to_rs.append(0.5 * rs)
        to_hg.append(0.5 * hg)
        n_pos.append(len(state))
        flat.append(1 if not state else 0)
        eq_dates.append(nxt)
        prev_alpha, prev_full, prev_hedge = alpha, full, hedge_w

        if i % 30 == 0:
            print(
                f"[portfolio {variant} τ={tau_pct}] day {i}/{len(dates)} "
                f"npos={len(state)} to={turnover:.3f}",
                flush=True,
            )

    return _pack_metrics(
        daily_net, daily_gross, daily_hedge, daily_cost,
        to_ee, to_rs, to_hg, n_pos, flat, eq_dates, hold_days, trade_pnls,
        tau_pct, tau, variant, horizon,
    )


def run_tranche_portfolio(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    tau_pct: float,
    exit_hysteresis: float = 0.6,  # unused: tranche exits only at next rebalance
    gross_limit: float = 1.0,
    fee_bps: float = 5.0,
    slip_bps: float = 3.0,
) -> dict:
    """
    Split capital into h equal tranches. Tranche k rebalances on day_index % h == k.
    Each position held until that tranche's next rebalance (~h days).
    Gross ≤ 1/h per tranche; BTC beta hedge at tranche level.
    """
    del exit_hysteresis  # explicit: no daily hysteresis in tranche variant
    h = int(horizon)
    df = _attach_aux(preds, feat, universe)
    rets = _prepare_returns(panel)
    dates = sorted(df["date"].unique())
    if len(dates) < h + 5:
        return {"error": "not enough dates", "tau_pct": tau_pct, "variant": "tranche"}
    abs_scores = df["score"].abs().dropna().values
    if len(abs_scores) < 50:
        return {"error": "not enough scores", "tau_pct": tau_pct, "variant": "tranche"}

    tau = float(np.percentile(abs_scores, tau_pct))
    cost_rate = (fee_bps + slip_bps) * 1e-4
    tg = gross_limit / float(h)

    states: list[dict[str, int]] = [{} for _ in range(h)]
    alphas: list[pd.Series] = [pd.Series(dtype=float) for _ in range(h)]
    entry_date: dict[tuple[int, str], pd.Timestamp] = {}
    hold_days: list[int] = []
    trade_pnls: list[float] = []
    sym_pnl: dict[tuple[int, str], float] = defaultdict(float)

    prev_full = pd.Series(dtype=float)
    prev_hedge = 0.0
    daily_net, daily_gross, daily_hedge, daily_cost = [], [], [], []
    to_ee, to_rs, to_hg = [], [], []
    n_pos, flat, eq_dates = [], [], []

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt]
        k = i % h
        prev_ak = alphas[k].copy()
        new_state = _hard_threshold_state(day, tau)

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

        # Recompute hedges with today's betas for all tranches (weights frozen except k)
        full = pd.Series(dtype=float)
        alpha = pd.Series(dtype=float)
        hedge_sum = 0.0
        for tk in range(h):
            ak = alphas[tk]
            if ak.empty:
                continue
            # drop names no longer in today's exec universe for non-rebalanced books
            if tk != k:
                univ = set(day["symbol"])
                ak = ak[[s for s in ak.index if s in univ]]
                alphas[tk] = ak
            fk, hk = _apply_hedge(day, ak)
            alpha = alpha.add(ak, fill_value=0.0)
            full = full.add(fk, fill_value=0.0)
            hedge_sum += hk

        # turnover split
        idx = alphas[k].index.union(prev_ak.index)
        ee = 0.5 * float((alphas[k].reindex(idx).fillna(0.0) - prev_ak.reindex(idx).fillna(0.0)).abs().sum())
        rs = 0.0
        hg = 0.5 * abs(hedge_sum - prev_hedge)

        fidx = full.index.union(prev_full.index)
        f = full.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        turnover = 0.5 * float((f - pf).abs().sum())
        cost = turnover * cost_rate

        nxt = dates[i + 1]
        gross_r = hedge_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in alpha.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    ri = float(rrow[s])
                    gross_r += float(wi) * ri
                    for tk in range(h):
                        if s in alphas[tk].index:
                            sym_pnl[(tk, s)] += float(alphas[tk].get(s, 0.0)) * ri
            w_btc = float(full.get("BTCUSDT", 0.0) - alpha.get("BTCUSDT", 0.0))
            if "BTCUSDT" in rrow.index and np.isfinite(rrow["BTCUSDT"]):
                hedge_r = w_btc * float(rrow["BTCUSDT"])

        net = gross_r + hedge_r - cost
        daily_net.append(net)
        daily_gross.append(gross_r)
        daily_hedge.append(hedge_r)
        daily_cost.append(cost)
        to_ee.append(ee)
        to_rs.append(rs)
        to_hg.append(hg)
        n_active = sum(len(st) for st in states)
        n_pos.append(n_active)
        flat.append(1 if n_active == 0 else 0)
        eq_dates.append(nxt)
        prev_full, prev_hedge = full, hedge_sum

        if i % 30 == 0:
            print(
                f"[portfolio tranche h={h} τ={tau_pct}] day {i}/{len(dates)} "
                f"npos={n_active} to={turnover:.3f}",
                flush=True,
            )

    return _pack_metrics(
        daily_net, daily_gross, daily_hedge, daily_cost,
        to_ee, to_rs, to_hg, n_pos, flat, eq_dates, hold_days, trade_pnls,
        tau_pct, tau, "tranche", horizon,
    )
