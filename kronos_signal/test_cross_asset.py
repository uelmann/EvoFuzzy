"""Unit tests for panel + long/short backtest (ROC, no GPU)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kronos_signal.cross_asset_bt import CrossAssetConfig, roc_score_fn, run_long_short_backtest
from kronos_signal.panel_data import (
    load_historical_long,
    point_in_time_universe,
    to_wide_panels,
)

CSV = Path(__file__).resolve().parent / "data" / "historical_data.csv"


def test_panel_and_universe():
    assert CSV.exists(), f"missing {CSV}"
    long_df = load_historical_long(CSV)
    panels = to_wide_panels(long_df)
    assert "BTC" in panels["close"].columns
    asof = panels["close"].index[-1]
    univ = point_in_time_universe(
        panels["marketCap"], asof, top_n=30, min_history_days=30, close=panels["close"]
    )
    assert len(univ) == 30
    assert "BTC" in univ
    assert "USDT" not in univ


def test_roc_long_short_smoke():
    panels = to_wide_panels(load_historical_long(CSV))
    cfg = CrossAssetConfig(
        universe_n=30,
        long_n=3,
        short_n=3,
        pred_len=10,
        lookback=90,
        min_history_days=60,
        start="2023-01-01",
        cost_bps=10.0,
    )
    out = run_long_short_backtest(panels, roc_score_fn(20), cfg)
    assert out["n_rebalances"] > 5
    assert abs(out["total_return"]) < 50  # sanity, not a performance claim
    # weights should be roughly dollar-neutral on rebalance hold days
    w = out["weights"].loc[out["daily_return"].index]
    # sample a day with positions
    active_days = w.index[(w.abs().sum(axis=1) > 0)]
    assert len(active_days) > 0
    sample = w.loc[active_days[len(active_days) // 2]]
    assert abs(sample.sum()) < 1e-6
    assert abs(sample[sample > 0].sum() - 1.0) < 1e-6
    assert abs(sample[sample < 0].sum() + 1.0) < 1e-6


if __name__ == "__main__":
    test_panel_and_universe()
    test_roc_long_short_smoke()
    print("ok")
