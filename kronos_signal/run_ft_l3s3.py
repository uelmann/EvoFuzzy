"""Kronos FT scores → top-3 long / worst-3 short (90/10 windows)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from .cross_asset_bt import (
    CrossAssetConfig,
    precomputed_score_fn,
    roc_score_fn,
    run_long_short_backtest,
    summarize_long_short,
)
from .official_topk_bt import panels_from_full
from .panel_data import DEFAULT_CSV, load_historical_long, to_wide_panels


def _load_panels(panel_pkl: Path | None, csv: Path) -> dict:
    if panel_pkl is not None and panel_pkl.exists():
        with open(panel_pkl, "rb") as f:
            return panels_from_full(pickle.load(f))
    return to_wide_panels(load_historical_long(csv))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores",
        type=Path,
        default=Path(__file__).resolve().parent / "official_runs" / "ft_prediction_scores.pkl",
    )
    p.add_argument("--panel-pkl", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--universe-n", type=int, default=30)
    p.add_argument("--long-n", type=int, default=3)
    p.add_argument("--short-n", type=int, default=3)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--pred-len", type=int, default=10, help="Hold / rebalance horizon in days")
    p.add_argument("--roc-window", type=int, default=10)
    p.add_argument("--start", type=str, default="2024-07-01")
    p.add_argument("--end", type=str, default="2026-08-08")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument(
        "--signals",
        type=str,
        default="mean,last",
        help="Comma-separated score keys to backtest (FT and ZS)",
    )
    p.add_argument(
        "--zs-scores",
        type=Path,
        default=None,
        help="Optional zero-shot (non-FT) score pickle for comparison",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "last_ft_l3s3.json",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=Path(__file__).resolve().parent / "ft_l3s3_equity.png",
    )
    args = p.parse_args()

    with open(args.scores, "rb") as f:
        scores_map = pickle.load(f)
    zs_map = None
    if args.zs_scores is not None:
        with open(args.zs_scores, "rb") as f:
            zs_map = pickle.load(f)
    panels = _load_panels(args.panel_pkl, args.csv)
    cfg = CrossAssetConfig(
        universe_n=args.universe_n,
        long_n=args.long_n,
        short_n=args.short_n,
        lookback=args.lookback,
        pred_len=args.pred_len,
        min_history_days=args.lookback,
        cost_bps=args.cost_bps,
        start=args.start,
        end=args.end,
    )

    results: dict[str, dict] = {}
    equities: dict[str, object] = {}
    keys = [s.strip() for s in args.signals.split(",") if s.strip()]

    def _run_named(prefix: str, smap: dict, key: str) -> None:
        if key not in smap:
            raise SystemExit(f"Missing score key {key!r} in {prefix}; have {list(smap)}")
        bt = run_long_short_backtest(panels, precomputed_score_fn(smap[key]), cfg)
        name = f"{prefix}_{key}"
        results[name] = summarize_long_short(bt)
        equities[name] = bt["equity"]
        print(
            f"[{name}] ret={bt['total_return']:+.1%} sharpe={bt['sharpe']:.2f} "
            f"mdd={bt['max_drawdown']:.1%} rebals={bt['n_rebalances']}",
            flush=True,
        )

    for key in keys:
        _run_named("kronos_ft", scores_map, key)
    if zs_map is not None:
        for key in keys:
            _run_named("kronos_zs", zs_map, key)

    roc_bt = run_long_short_backtest(panels, roc_score_fn(args.roc_window), cfg)
    results["roc_baseline"] = summarize_long_short(roc_bt)
    equities["roc_baseline"] = roc_bt["equity"]
    btc_eq = roc_bt["btc_equity"]
    print(
        f"[roc_baseline] ret={roc_bt['total_return']:+.1%} sharpe={roc_bt['sharpe']:.2f} "
        f"mdd={roc_bt['max_drawdown']:.1%}",
        flush=True,
    )

    out = {
        "recipe": "ft_vs_zs_l3s3" if zs_map is not None else "ft_l3s3",
        "note": (
            "L3/S3 dollar-neutral; FT vs pretrained zero-shot Kronos scores"
            if zs_map is not None
            else "FT scores; portfolio = long top-N / short worst-N"
        ),
        "config": cfg.__dict__,
        "roc_window": args.roc_window,
        "scores_path": str(args.scores),
        "zs_scores_path": str(args.zs_scores) if args.zs_scores else None,
        "signals": results,
        "primary": results.get("kronos_ft_last")
        or results.get("kronos_ft_mean")
        or next(iter(results.values())),
    }
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {args.out}", flush=True)

    # Equity plot — focus on last (best FT) + ZS last + ROC + BTC when comparing
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {
        "kronos_ft_last": "#116329",
        "kronos_ft_mean": "#1f6feb",
        "kronos_zs_last": "#8250df",
        "kronos_zs_mean": "#bf3989",
        "roc_baseline": "#bf8700",
    }
    plot_order = [
        "kronos_ft_last",
        "kronos_zs_last",
        "kronos_ft_mean",
        "kronos_zs_mean",
        "roc_baseline",
    ]
    for name in plot_order:
        eq = equities.get(name)
        if eq is None or len(eq) < 2:
            continue
        norm = eq / eq.iloc[0]
        ret = results[name]["total_return"]
        lw = 2.2 if name.endswith("_last") or name == "roc_baseline" else 1.4
        ls = "--" if name.startswith("kronos_zs") else "-"
        ax.plot(
            norm.index,
            norm.values,
            label=f"{name} {ret:+.0%}",
            lw=lw,
            ls=ls,
            color=colors.get(name, None),
        )
    if btc_eq is not None and len(btc_eq) > 1:
        btc_n = btc_eq / btc_eq.iloc[0]
        ax.plot(
            btc_n.index,
            btc_n.values,
            label=f"BTC B&H {roc_bt['btc_total_return']:+.0%}",
            lw=1.5,
            color="#8b949e",
        )
    ax.axhline(1.0, color="#d0d7de", lw=1)
    title_cmp = "FT vs zero-shot" if zs_map is not None else "FT"
    ax.set_title(
        f"L{cfg.long_n}/S{cfg.short_n} dollar-neutral ({title_cmp}) — "
        f"lookback={cfg.lookback} / hold={cfg.pred_len}d\n"
        f"{cfg.start} → {cfg.end}  |  PIT top-{cfg.universe_n}"
    )
    ax.set_ylabel("Equity (start=1)")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.plot}", flush=True)


if __name__ == "__main__":
    main()
