"""Portfolio backtest with tau sweep, vol sizing, BTC beta hedge."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_returns(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    rets = np.log(close / close.shift(1))
    return rets


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
) -> dict:
    """
    Long if score > τ, short if score < −τ; exit when |score| < 0.6τ.
    Weights ∝ 1/yz_vol_30, gross ≤ 1. PIT top-20 only.
    BTC hedge = −Σ w_i β_i on BTCUSDT.
    Trades at close t; PnL from close-to-close next day onward (hold until exit).
    For multi-day horizon labels we still trade daily on OOS scores.
    """
    ycol = f"y_h{horizon}"
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    uni = universe.copy()
    uni["date"] = pd.to_datetime(uni["date"], utc=True)
    df = df.merge(uni[["date", "symbol"]], on=["date", "symbol"], how="inner")

    # attach vol + beta
    aux = feat[["date", "symbol", "yz_vol_30_raw", "beta_btc_60", "close"]].copy()
    aux["date"] = pd.to_datetime(aux["date"], utc=True)
    # yz_vol_30 in feat may be z-scored; prefer raw if present
    if "yz_vol_30_raw" not in feat.columns:
        aux["yz_vol_30_raw"] = feat["yz_vol_30"]
    df = df.merge(aux, on=["date", "symbol"], how="left")

    rets = _prepare_returns(panel)
    dates = sorted(df["date"].unique())
    if len(dates) < 10:
        return {"error": "not enough dates", "tau_pct": tau_pct}

    # τ from OOS |score| distribution
    abs_scores = df["score"].abs().dropna().values
    if len(abs_scores) < 50:
        return {"error": "not enough scores", "tau_pct": tau_pct}
    tau = float(np.percentile(abs_scores, tau_pct))
    exit_thr = exit_hysteresis * tau

    # state: symbol -> side (+1/-1)
    state: dict[str, int] = {}
    equity = 1.0
    eq_path = []
    daily_rets = []
    turnovers = []
    n_pos = []
    flat = []
    prev_w = pd.Series(dtype=float)

    cost_rate = (fee_bps + slip_bps) * 1e-4  # one-way

    for i, dt in enumerate(dates[:-1]):
        day = df[df["date"] == dt]
        # update positions with hysteresis
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
        # drop symbols not in today's universe
        univ_syms = set(day["symbol"])
        new_state = {k: v for k, v in new_state.items() if k in univ_syms and v != 0}
        state = new_state

        # sizing
        longs = [s for s, v in state.items() if v > 0]
        shorts = [s for s, v in state.items() if v < 0]
        w = {}
        def _inv_vol(sym):
            row = day[day["symbol"] == sym]
            if row.empty:
                return np.nan
            v = float(row["yz_vol_30_raw"].iloc[0])
            if not np.isfinite(v) or v <= 0:
                v = float(row["yz_vol_30"].iloc[0]) if "yz_vol_30" in row.columns else 0.02
            if not np.isfinite(v) or v <= 0:
                v = 0.02
            return 1.0 / v

        if longs:
            iv = {s: _inv_vol(s) for s in longs}
            ssum = sum(iv.values())
            for s, v in iv.items():
                w[s] = 0.5 * gross_limit * v / ssum
        if shorts:
            iv = {s: _inv_vol(s) for s in shorts}
            ssum = sum(iv.values())
            for s, v in iv.items():
                w[s] = -0.5 * gross_limit * v / ssum

        # beta hedge with BTC
        betas = {}
        for s in list(w):
            row = day[day["symbol"] == s]
            b = float(row["beta_btc_60"].iloc[0]) if not row.empty else 0.0
            if not np.isfinite(b):
                b = 0.0
            betas[s] = b
        port_beta = sum(w[s] * betas[s] for s in w)
        w_btc = -port_beta
        # include BTC weight separately
        w_series = pd.Series(w, dtype=float)
        if abs(w_btc) > 1e-12:
            w_series.loc["BTCUSDT"] = w_series.get("BTCUSDT", 0.0) + w_btc

        # turnover vs previous
        aligned = w_series.reindex(w_series.index.union(prev_w.index)).fillna(0.0)
        prev_a = prev_w.reindex(aligned.index).fillna(0.0)
        turnover = 0.5 * float((aligned - prev_a).abs().sum())
        cost = turnover * cost_rate

        # PnL: next day's close-to-close return
        nxt = dates[i + 1]
        if nxt not in rets.index:
            port_r = 0.0
        else:
            rrow = rets.loc[nxt]
            port_r = 0.0
            for s, wi in aligned.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    port_r += wi * float(rrow[s])
        net = port_r - cost
        equity *= np.exp(net)  # log-return approx compounding
        # use simple for path: equity *= (1+net) with net as simple approx from log
        # actually port_r is sum of log rets weighted — approximate simple
        daily_rets.append(net)
        eq_path.append({"date": nxt, "equity": equity, "ret": net})
        turnovers.append(turnover)
        n_pos.append(len(state))
        flat.append(1 if len(state) == 0 else 0)
        prev_w = aligned

        if i % 30 == 0:
            print(
                f"[portfolio τ={tau_pct}] day {i}/{len(dates)} equity={equity:.3f} "
                f"npos={len(state)} turnover={turnover:.3f}",
                flush=True,
            )

    rets_s = pd.Series(daily_rets)
    eq = pd.DataFrame(eq_path)
    if eq.empty:
        return {"error": "empty path", "tau_pct": tau_pct}
    total = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0) if len(eq) > 1 else 0.0
    # rebuild equity from simple compounding for metrics clarity
    simple_eq = (1.0 + rets_s.fillna(0.0)).cumprod()
    n = len(rets_s)
    years = n / 365.0
    cagr = float(simple_eq.iloc[-1] ** (1 / max(years, 1e-6)) - 1) if n > 1 else 0.0
    vol = float(rets_s.std() * np.sqrt(365)) if rets_s.std() > 0 else 0.0
    sharpe = float(rets_s.mean() / rets_s.std() * np.sqrt(365)) if rets_s.std() > 0 else 0.0
    dd = float((simple_eq / simple_eq.cummax() - 1.0).min())
    return {
        "tau_pct": tau_pct,
        "tau": tau,
        "net_sharpe": sharpe,
        "net_cagr": cagr,
        "max_drawdown": dd,
        "total_return": float(simple_eq.iloc[-1] - 1.0),
        "avg_n_positions": float(np.mean(n_pos)) if n_pos else 0.0,
        "pct_flat_days": float(np.mean(flat)) if flat else 1.0,
        "ann_turnover": float(np.mean(turnovers) * 365) if turnovers else 0.0,
        "equity": pd.DataFrame({"date": eq["date"], "equity": simple_eq.values}),
        "daily_ret": rets_s,
    }
