"""Always-in top-K alt-long tranche book. Never holds BTC. Residual is cash."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from alphamine.constants import (
    BTC_SYMBOL,
    FEE_BPS_NEXT,
    FEE_BPS_TOP,
    GROSS_LIMIT,
    LAG,
    LIQ_CAP_ADV_FRAC,
    NOMINAL_BOOK_USD,
    SLIP_BPS_NEXT,
    SLIP_BPS_TOP,
    TOP_K,
)
from baseline.portfolio import (
    _attach_aux,
    _cost_rate_for_rank,
    _dv_med_of,
    _funding_wide,
    _inv_vol,
    _pack_metrics,
    _rank_lookup_frame,
    _rank_of,
    _utc_ts,
)


def _simple_returns(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    return close.pct_change()


def _size_longs(
    day: pd.DataFrame,
    names: list[str],
    gross: float,
    liq_cap_adv_frac: float | None,
    nominal_book_usd: float,
) -> pd.Series:
    if not names or gross <= 0:
        return pd.Series(dtype=float)
    iv = {s: _inv_vol(day, s) for s in names}
    ssum = sum(iv.values()) or 1.0
    w = {s: float(gross) * iv[s] / ssum for s in names}
    if liq_cap_adv_frac is not None and float(liq_cap_adv_frac) > 0 and float(nominal_book_usd) > 0:
        cap_frac = float(liq_cap_adv_frac)
        book = float(nominal_book_usd)
        for s in list(w):
            dv = _dv_med_of(day, s)
            if not np.isfinite(dv) or dv <= 0:
                continue
            cap = cap_frac * dv / book
            if w[s] > cap:
                w[s] = float(cap)
    return pd.Series(w, dtype=float)


def _pick_names(day: pd.DataFrame, top_k: int) -> list[str]:
    if day.empty:
        return []
    g = day[day["symbol"] != BTC_SYMBOL].copy()
    if g.empty or "score" not in g.columns:
        return []
    g["_sc"] = pd.to_numeric(g["score"], errors="coerce")
    g = g[np.isfinite(g["_sc"])]
    if g.empty:
        return []
    g = g.sort_values("_sc", ascending=False)
    names = [s for s in g["symbol"].tolist() if s != BTC_SYMBOL][: int(top_k)]
    return names


def run_long_only_topk(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    funding: pd.DataFrame | None = None,
    apply_funding: bool = True,
    top_k: int = TOP_K,
    variant: str = "alphamine_lo",
) -> dict:
    h = int(horizon)
    df = _attach_aux(preds, feat, universe)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["symbol"] != BTC_SYMBOL]
    if "score" not in df.columns:
        return {"error": "preds missing score"}
    rets = _simple_returns(panel)
    rets.index = pd.DatetimeIndex(pd.to_datetime(rets.index, utc=True))
    fund_wide = _funding_wide(funding) if apply_funding else pd.DataFrame()
    rank_long = _rank_lookup_frame(universe)
    dates = sorted(df["date"].unique(), key=lambda x: _utc_ts(x))
    if len(dates) < h + 5 + int(LAG):
        return {"error": "not enough dates", "variant": variant, "horizon": h}

    by_date = {pd.Timestamp(dt): g for dt, g in df.groupby("date", sort=False)}
    rank_by_date = {}
    if not rank_long.empty:
        rank_long = rank_long.copy()
        rank_long["date"] = pd.to_datetime(rank_long["date"], utc=True)
        rank_by_date = {pd.Timestamp(dt): g for dt, g in rank_long.groupby("date", sort=False)}

    tg = float(GROSS_LIMIT) / float(h)
    alphas: list[pd.Series] = [pd.Series(dtype=float) for _ in range(h)]
    target_alpha_hist: list[pd.Series] = []
    prev_full = pd.Series(dtype=float)
    daily_net, daily_gross, daily_hedge, daily_cost, daily_funding = [], [], [], [], []
    to_ee, to_rs, to_hg = [], [], []
    n_pos, n_long, n_short, flat, eq_dates = [], [], [], [], []
    daily_gross_deployed = []
    name_alpha_pnl: dict[str, float] = defaultdict(float)
    side_days: dict[str, dict] = defaultdict(lambda: {"long_days": 0, "short_days": 0})
    hold_days: list[int] = []
    trade_pnls: list[float] = []
    n_forced = 0
    forced_pnl = 0.0
    max_abs_btc = 0.0
    daily_btc_w: list[float] = []

    for i, dt in enumerate(dates[:-1]):
        dt = pd.Timestamp(dt)
        day = by_date.get(dt, pd.DataFrame())
        rank_day = rank_by_date.get(dt, day)
        k = i % h
        prev_ak = alphas[k].copy()
        names = _pick_names(day, top_k) if not day.empty else []
        alphas[k] = (
            _size_longs(day, names, tg, LIQ_CAP_ADV_FRAC, NOMINAL_BOOK_USD)
            if (not day.empty and names)
            else pd.Series(dtype=float)
        )

        alpha = pd.Series(dtype=float)
        univ = set(day["symbol"]) if not day.empty else set()
        for tk in range(h):
            ak = alphas[tk]
            if ak.empty:
                continue
            if univ:
                ak = ak[[s for s in ak.index if s in univ and s != BTC_SYMBOL]]
                alphas[tk] = ak
            alpha = alpha.add(ak, fill_value=0.0)
            for s in ak.index:
                side_days[s]["long_days"] += 1

        if BTC_SYMBOL in alpha.index:
            max_abs_btc = max(max_abs_btc, float(abs(alpha.get(BTC_SYMBOL, 0.0))))
            alpha = alpha.drop(labels=[BTC_SYMBOL], errors="ignore")
        target_alpha_hist.append(alpha)

        if i < LAG:
            applied_alpha = pd.Series(dtype=float)
        else:
            applied_alpha = target_alpha_hist[i - LAG]
            applied_alpha = applied_alpha.drop(labels=[BTC_SYMBOL], errors="ignore")

        btc_w = float(applied_alpha.get(BTC_SYMBOL, 0.0)) if BTC_SYMBOL in applied_alpha.index else 0.0
        max_abs_btc = max(max_abs_btc, abs(btc_w))
        daily_btc_w.append(btc_w)

        idx = alphas[k].index.union(prev_ak.index)
        ee = 0.5 * float((alphas[k].reindex(idx).fillna(0.0) - prev_ak.reindex(idx).fillna(0.0)).abs().sum())

        fidx = applied_alpha.index.union(prev_full.index)
        f = applied_alpha.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        dw = (f - pf).abs()
        cost = 0.0
        for s, mag in dw.items():
            if s == BTC_SYMBOL:
                continue
            rnk = _rank_of(day, rank_day, s) if not day.empty else float("nan")
            cost += 0.5 * float(mag) * _cost_rate_for_rank(
                rnk, FEE_BPS_TOP, SLIP_BPS_TOP, True, FEE_BPS_NEXT, SLIP_BPS_NEXT
            )

        nxt = pd.Timestamp(dates[i + 1])
        if nxt.tzinfo is None:
            nxt = nxt.tz_localize("UTC")
        else:
            nxt = nxt.tz_convert("UTC")
        gross_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in applied_alpha.items():
                if s == BTC_SYMBOL:
                    continue
                if s in rrow.index and np.isfinite(rrow[s]):
                    contrib = float(wi) * float(rrow[s])
                    gross_r += contrib
                    name_alpha_pnl[s] = name_alpha_pnl.get(s, 0.0) + contrib
                else:
                    n_forced += 1
                    forced_pnl += 0.0
        elif len(applied_alpha):
            n_forced += int((applied_alpha.abs() > 1e-12).sum())

        fund_r = 0.0
        if apply_funding and not fund_wide.empty and nxt in fund_wide.index:
            row = fund_wide.loc[nxt]
            for s, wi in applied_alpha.items():
                if s == BTC_SYMBOL:
                    continue
                if s in row.index and np.isfinite(row[s]):
                    fund_r += -float(wi) * float(row[s])

        net = gross_r - cost + fund_r
        daily_net.append(net)
        daily_gross.append(gross_r)
        daily_hedge.append(0.0)
        daily_cost.append(cost)
        daily_funding.append(fund_r)
        to_ee.append(ee)
        to_rs.append(0.0)
        to_hg.append(0.0)
        nl = int((applied_alpha > 1e-12).sum()) if len(applied_alpha) else 0
        n_long.append(nl)
        n_short.append(0)
        n_pos.append(nl)
        flat.append(1 if nl == 0 else 0)
        eq_dates.append(nxt)
        daily_gross_deployed.append(float(applied_alpha.clip(lower=0).sum()) if len(applied_alpha) else 0.0)
        prev_full = applied_alpha
        if i % 60 == 0:
            print(
                f"[{variant} h={h}] day {i}/{len(dates)} L={nl} "
                f"gross={daily_gross_deployed[-1]:.3f} net={net:.5f}",
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
        tau_pct=0.0,
        tau=0.0,
        variant=variant,
        horizon=horizon,
        lag=LAG,
        apply_funding=apply_funding,
        sym_contrib=dict(name_alpha_pnl),
        side_days=dict(side_days),
    )
    packed["avg_gross_deployed"] = float(np.mean(daily_gross_deployed)) if daily_gross_deployed else 0.0
    packed["daily_gross_deployed"] = pd.Series(daily_gross_deployed, index=packed["daily_ret"].index, dtype=float)
    packed["name_alpha_pnl"] = dict(name_alpha_pnl)
    packed["n_forced_exits"] = int(n_forced)
    packed["forced_exit_pnl"] = float(forced_pnl)
    packed["max_abs_btc_weight"] = float(max_abs_btc)
    packed["btc_weight_identically_zero"] = bool(max_abs_btc < 1e-12)
    packed["daily_btc_weight"] = (
        pd.Series(daily_btc_w, index=packed["daily_ret"].index, dtype=float) if daily_btc_w else pd.Series(dtype=float)
    )
    packed["long_only"] = True
    packed["apply_beta_hedge"] = False
    packed["top_k"] = int(top_k)
    return packed
