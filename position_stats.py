"""Print avg long / avg short / flat days per portfolio variant from Volume cache.

Usage:
    modal run position_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
import yaml

volume = modal.Volume.from_name("quant-baseline", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas==2.2.2", "pyarrow", "scipy", "lightgbm", "httpx", "pyyaml")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)
app = modal.App("quant-baseline-ls-stats", image=image)


@app.function(timeout=60 * 60 * 2, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def compute_ls_stats() -> list[dict]:
    import pandas as pd
    from baseline.data import load_funding_panel, load_panel
    from baseline.portfolio import run_portfolio_backtest, run_tranche_portfolio

    with open("/root/config.yaml") as f:
        cfg = yaml.safe_load(f)

    root = Path("/data/quant")
    feat = pd.read_parquet(root / "features" / "features_labeled.parquet")
    pit20 = pd.read_parquet(root / "universe" / "top20_pit.parquet")
    pit120 = pd.read_parquet(root / "universe" / "top120_pit.parquet")
    ever = sorted(set(pit120["symbol"].unique()) | {"BTCUSDT"})
    panel = load_panel(root / "raw" / "klines", ever)
    panel = panel[panel["symbol"].isin(ever)].copy()
    funding = load_funding_panel(root / "raw" / "funding", ever)

    rows = []
    for h in cfg["labels"]["horizons"]:
        canon = pd.read_parquet(root / "predictions" / f"lgbm_price_only_h{h}.parquet")
        ycol = f"y_h{h}"
        if ycol not in canon.columns:
            canon = canon.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        for funding_on in (False, True):
            for tp in cfg["portfolio"]["tau_percentiles"]:
                for variant, fn in (
                    ("daily", run_portfolio_backtest),
                    ("tranche", run_tranche_portfolio),
                ):
                    kw = dict(
                        preds=canon,
                        panel=panel,
                        feat=feat,
                        universe=pit20,
                        horizon=h,
                        tau_pct=tp,
                        exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                        gross_limit=cfg["portfolio"]["gross_limit"],
                        fee_bps=cfg["portfolio"]["taker_fee_bps"],
                        slip_bps=cfg["portfolio"]["slippage_bps"],
                        lag=0,
                        apply_funding=funding_on,
                        funding=funding,
                    )
                    if variant == "daily":
                        kw["variant"] = "daily"
                    res = fn(**kw)
                    if "error" in res:
                        continue
                    rows.append(
                        {
                            "variant": variant,
                            "horizon": h,
                            "tau_pct": tp,
                            "funding_on": funding_on,
                            "avg_n_long": res["avg_n_long"],
                            "avg_n_short": res["avg_n_short"],
                            "n_flat_days": res["n_flat_days"],
                            "n_days": res["n_days"],
                            "pct_flat_days": res["pct_flat_days"],
                            "net_sharpe": res["net_sharpe"],
                        }
                    )
                    print(
                        f"[ls] {variant} h={h} τ={tp} fund={funding_on} "
                        f"L={res['avg_n_long']:.2f} S={res['avg_n_short']:.2f} "
                        f"flat={res['n_flat_days']}/{res['n_days']} ({100*res['pct_flat_days']:.1f}%)",
                        flush=True,
                    )
    (root / "reports" / "ls_stats.json").write_text(json.dumps(rows, indent=2))
    volume.commit()
    return rows


@app.local_entrypoint()
def main():
    rows = compute_ls_stats.remote()
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports/ls_stats.json").write_text(json.dumps(rows, indent=2))
    print("\n===== AVG LONG / AVG SHORT / FLAT DAYS =====", flush=True)
    print(
        f"{'var':>8} {'h':>3} {'τ':>4} {'fund':>5} {'avgL':>7} {'avgS':>7} "
        f"{'flat_d':>7} {'n_days':>7} {'%flat':>7} {'netSh':>7}",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['variant']:>8} {r['horizon']:>3} {r['tau_pct']:>4} "
            f"{str(r['funding_on']):>5} {r['avg_n_long']:>7.2f} {r['avg_n_short']:>7.2f} "
            f"{r['n_flat_days']:>7} {r['n_days']:>7} {100*r['pct_flat_days']:>6.1f}% "
            f"{r['net_sharpe']:>7.2f}",
            flush=True,
        )
