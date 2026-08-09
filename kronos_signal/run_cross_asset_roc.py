"""Run ROC baseline cross-asset long/short backtest (no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cross_asset_bt import CrossAssetConfig, roc_score_fn, run_long_short_backtest
from .panel_data import DEFAULT_CSV, load_historical_long, to_wide_panels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--universe-n", type=int, default=30)
    p.add_argument("--long-n", type=int, default=3)
    p.add_argument("--short-n", type=int, default=3)
    p.add_argument("--pred-len", type=int, default=10)
    p.add_argument("--roc-window", type=int, default=30)
    p.add_argument("--start", type=str, default="2021-01-01")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "last_cross_asset_roc.json",
    )
    args = p.parse_args()

    long_df = load_historical_long(args.csv)
    panels = to_wide_panels(long_df)
    cfg = CrossAssetConfig(
        universe_n=args.universe_n,
        long_n=args.long_n,
        short_n=args.short_n,
        pred_len=args.pred_len,
        start=args.start,
        cost_bps=args.cost_bps,
    )
    result = run_long_short_backtest(panels, roc_score_fn(args.roc_window), cfg)
    summary = {
        "mode": "roc_baseline",
        "roc_window": args.roc_window,
        "config": result["config"],
        "n_rebalances": result["n_rebalances"],
        "total_return": result["total_return"],
        "max_drawdown": result["max_drawdown"],
        "ann_vol": result["ann_vol"],
        "sharpe": result["sharpe"],
        "btc_total_return": result["btc_total_return"],
        "turnover_mean": result["turnover_mean"],
        "picks_tail": result["picks"].tail(5).to_dict(orient="records"),
    }
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
