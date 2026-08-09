"""Long-only TopkDropout backtest — mirrors Kronos/finetune/qlib_test strategy knobs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .official_config import OfficialConfig


def load_full_panel(cfg: OfficialConfig) -> dict[str, pd.DataFrame]:
    path = Path(cfg.dataset_path) / "full_panel.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def panels_from_full(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    closes, mcaps = {}, {}
    for sym, df in data.items():
        closes[sym] = df["close"]
        if "marketCap" in df.columns:
            mcaps[sym] = df["marketCap"]
    close = pd.DataFrame(closes).sort_index()
    mcap = pd.DataFrame(mcaps).sort_index() if mcaps else close * 0.0
    return {"close": close, "marketCap": mcap}


def topk_dropout_backtest(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    cfg: OfficialConfig,
    universe_mcap: pd.DataFrame | None = None,
) -> dict:
    """
    scores: date × symbol prediction scores (higher = more bullish).
    Implements a simplified TopkDropout:
      - restrict to point-in-time top `universe_n` by mcap when provided
      - hold `topk` names, each day drop up to `n_drop` worst held if replacements exist
      - enforce hold_thresh in trading days
      - delay execution by 1 day; equal-weight long-only
    """
    topk = cfg.backtest_n_symbol_hold
    n_drop = cfg.backtest_n_symbol_drop
    hold_thresh = cfg.backtest_hold_thresh
    start, end = cfg.backtest_time_range
    dates = close.index[(close.index >= pd.Timestamp(start, tz="UTC")) & (close.index <= pd.Timestamp(end, tz="UTC"))]
    dates = list(dates)

    holdings: dict[str, int] = {}  # symbol -> days held
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    rets = close.pct_change()

    for i, d in enumerate(dates[:-1]):
        # Signal day d → positions on d+1 (delay_execution)
        exec_day = dates[i + 1]
        univ = list(close.columns)
        if universe_mcap is not None and d in universe_mcap.index:
            row = universe_mcap.loc[d].dropna()
            univ = list(row.nlargest(cfg.universe_n).index)

        day_scores = scores.loc[d].reindex(univ).dropna() if d in scores.index else pd.Series(dtype=float)
        if day_scores.empty:
            continue

        # Drop worst held (if held long enough)
        droppable = [s for s, age in holdings.items() if age >= hold_thresh and s in day_scores.index]
        droppable_sorted = sorted(droppable, key=lambda s: day_scores.get(s, -np.inf))
        to_drop = droppable_sorted[:n_drop]

        kept = [s for s in holdings if s not in to_drop]
        # Fill up to topk with best not held
        candidates = [s for s in day_scores.sort_values(ascending=False).index if s not in kept]
        need = max(topk - len(kept), 0)
        add = candidates[:need]
        new_hold = kept + add
        # If still over topk (shouldn't), trim worst
        if len(new_hold) > topk:
            new_hold = sorted(new_hold, key=lambda s: day_scores.get(s, -np.inf), reverse=True)[:topk]

        new_holdings: dict[str, int] = {}
        for s in new_hold:
            new_holdings[s] = (holdings.get(s, 0) + 1) if s in holdings else 1
        holdings = new_holdings

        if not holdings:
            continue
        w = 1.0 / len(holdings)
        for s in holdings:
            if s in weights.columns:
                weights.loc[exec_day, s] = w

    port = (weights * rets).sum(axis=1)
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    # Approximate one-way costs on turnover (open+close style)
    cost_rate = 0.5 * (cfg.open_cost + cfg.close_cost)
    net = port - 0.5 * dw * cost_rate
    active = net.loc[dates[0] : dates[-1]].fillna(0.0)
    equity = (1.0 + active).cumprod()

    btc_total = None
    btc_eq = None
    if "BTC" in close.columns:
        btc_r = close["BTC"].reindex(active.index).pct_change().fillna(0.0)
        btc_eq = (1.0 + btc_r).cumprod()
        btc_total = float(btc_eq.iloc[-1] / btc_eq.iloc[0] - 1.0) if len(btc_eq) > 1 else 0.0

    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    sharpe = (
        float(active.mean() / active.std() * np.sqrt(365))
        if active.std() and active.std() > 0
        else 0.0
    )
    return {
        "total_return": total,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "btc_total_return": btc_total,
        "equity": equity,
        "btc_equity": btc_eq,
        "daily_return": active,
        "weights": weights,
        "n_days": len(active),
    }


def roc_scores(close: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return close / close.shift(window) - 1.0
