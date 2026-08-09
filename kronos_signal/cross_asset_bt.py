"""Long top-K / short worst-K cross-asset backtest (CSI300-style ranking)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .panel_data import point_in_time_universe


ScoreFn = Callable[[pd.Timestamp, list[str], dict[str, pd.DataFrame]], pd.Series]


@dataclass
class CrossAssetConfig:
    universe_n: int = 30
    long_n: int = 3
    short_n: int = 3
    lookback: int = 90
    pred_len: int = 10  # hold / rebalance horizon (like Kronos A-share demo)
    min_history_days: int = 90
    cost_bps: float = 10.0  # one-way, applied on weight turnover
    start: str | None = "2021-01-01"
    end: str | None = None


def roc_score_fn(window: int = 30) -> ScoreFn:
    """Notebook-style ROC baseline: higher past return → higher rank score."""

    def _score(asof: pd.Timestamp, symbols: list[str], panels: dict[str, pd.DataFrame]) -> pd.Series:
        close = panels["close"]
        hist = close.loc[:asof, symbols]
        if len(hist) < window + 1:
            return pd.Series(dtype=float)
        ret = hist.iloc[-1] / hist.iloc[-1 - window] - 1.0
        return ret.dropna()

    return _score


def precomputed_score_fn(score_df: pd.DataFrame) -> ScoreFn:
    """Wrap a date×symbol score matrix (e.g. Kronos FT mean/last) as a ScoreFn."""
    df = score_df.copy()
    df.index = pd.to_datetime(df.index, utc=True)

    def _score(asof: pd.Timestamp, symbols: list[str], panels: dict[str, pd.DataFrame]) -> pd.Series:
        asof = pd.Timestamp(asof)
        if asof.tzinfo is None:
            asof = asof.tz_localize("UTC")
        else:
            asof = asof.tz_convert("UTC")
        if asof in df.index:
            row = df.loc[asof]
        else:
            prev = df.index[df.index <= asof]
            if len(prev) == 0:
                return pd.Series(dtype=float)
            row = df.loc[prev[-1]]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row.reindex(symbols).dropna().astype(float)

    return _score


def summarize_long_short(result: dict) -> dict:
    """JSON-friendly metrics (drop heavy frames)."""
    return {
        "n_rebalances": result["n_rebalances"],
        "total_return": result["total_return"],
        "max_drawdown": result["max_drawdown"],
        "ann_vol": result["ann_vol"],
        "sharpe": result["sharpe"],
        "btc_total_return": result["btc_total_return"],
        "turnover_mean": result["turnover_mean"],
        "n_days": int(len(result["daily_return"])),
    }


def run_long_short_backtest(
    panels: dict[str, pd.DataFrame],
    score_fn: ScoreFn,
    cfg: CrossAssetConfig | None = None,
) -> dict:
    """
    Dollar-neutral equal-weight: +1/long_n on top scores, -1/short_n on worst.

    Rebalances every `pred_len` calendar trading days present in the panel.
    Portfolio daily return = sum_i w_i * r_i(t) minus turnover costs on rebalance days.
    """
    cfg = cfg or CrossAssetConfig()
    close = panels["close"]
    mcap = panels["marketCap"]
    rets = close.pct_change()

    dates = close.index.sort_values()
    if cfg.start:
        dates = dates[dates >= pd.Timestamp(cfg.start, tz="UTC")]
    if cfg.end:
        dates = dates[dates <= pd.Timestamp(cfg.end, tz="UTC")]
    dates = list(dates)

    if len(dates) < cfg.lookback + cfg.pred_len + 5:
        raise ValueError("Not enough dates for cross-asset backtest")

    # Rebalance on a non-overlapping grid (like their hold horizon).
    rebalance_dates = dates[cfg.lookback :: cfg.pred_len]

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    picks_rows: list[dict] = []

    for rb in rebalance_dates:
        univ = point_in_time_universe(
            mcap,
            rb,
            top_n=cfg.universe_n,
            min_history_days=cfg.min_history_days,
            close=close,
        )
        if len(univ) < cfg.long_n + cfg.short_n:
            continue
        scores = score_fn(rb, univ, panels)
        scores = scores.reindex(univ).dropna()
        if len(scores) < cfg.long_n + cfg.short_n:
            continue
        ranked = scores.sort_values(ascending=False)
        longs = list(ranked.index[: cfg.long_n])
        shorts = list(ranked.index[-cfg.short_n :])
        # Hold from next bar after signal through pred_len bars (execution delay=1).
        loc = dates.index(rb)
        hold_dates = dates[loc + 1 : loc + 1 + cfg.pred_len]
        if not hold_dates:
            continue
        w = {s: 0.0 for s in univ}
        for s in longs:
            w[s] = 1.0 / cfg.long_n
        for s in shorts:
            w[s] = -1.0 / cfg.short_n
        for d in hold_dates:
            for s, val in w.items():
                if s in weights.columns:
                    weights.loc[d, s] = val
        picks_rows.append(
            {
                "asof": rb,
                "longs": ",".join(longs),
                "shorts": ",".join(shorts),
                "n_universe": len(univ),
            }
        )

    # Align returns: weight on day t earns close-to-close return of day t.
    port = (weights * rets).sum(axis=1)
    # Transaction costs on weight changes (L1 turnover / 2 * one-way bps is common;
    # here: cost = turnover * cost_bps * 1e-4 where turnover = 0.5 * sum|Δw|).
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover = 0.5 * dw
    costs = turnover * (cfg.cost_bps * 1e-4)
    net = port - costs

    active = net.loc[dates[0] : dates[-1]].dropna()
    equity = (1.0 + active.fillna(0.0)).cumprod()

    # Benchmarks on same calendar
    btc = close["BTC"] if "BTC" in close.columns else None
    if btc is not None:
        btc_r = btc.reindex(active.index).pct_change().fillna(0.0)
        btc_eq = (1.0 + btc_r).cumprod()
        btc_total = float(btc_eq.iloc[-1] / btc_eq.iloc[0] - 1.0) if len(btc_eq) > 1 else 0.0
    else:
        btc_eq = None
        btc_total = None

    # Equal-weight buy&hold of PIT top-N is expensive to do exactly; use static
    # last-day top-N EW as a rough reference only if needed — skip for now.

    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    vol = float(active.std() * np.sqrt(365)) if active.std() == active.std() else 0.0
    sharpe = (
        float(active.mean() / active.std() * np.sqrt(365))
        if active.std() and active.std() > 0
        else 0.0
    )

    return {
        "config": cfg.__dict__,
        "n_rebalances": len(picks_rows),
        "total_return": total,
        "max_drawdown": dd,
        "ann_vol": vol,
        "sharpe": sharpe,
        "btc_total_return": btc_total,
        "equity": equity,
        "btc_equity": btc_eq,
        "daily_return": active,
        "weights": weights,
        "picks": pd.DataFrame(picks_rows),
        "turnover_mean": float(turnover.loc[active.index].mean()) if len(active) else 0.0,
    }
