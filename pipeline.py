"""
Phase A0 price-only baseline — Modal entrypoint.

Usage:
    modal run pipeline.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-baseline-a0"
VOLUME_NAME = "quant-baseline"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "lightgbm",
        "matplotlib",
        "httpx",
        "pyyaml",
        "scikit-learn",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

# Optional: mount local kronos artifacts for adapter if present
_local_crypto = Path("/opt/cursor/artifacts/crypto_data")
if _local_crypto.exists():
    image = image.add_local_dir(str(_local_crypto), remote_path="/kronos_import")

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 30, retries=0, volumes={"/data/quant": volume}, cpu=2, memory=4096)
def download_one_symbol(item: dict) -> dict:
    from baseline.data import download_symbol_months, month_range

    symbol = item["symbol"]
    start_month = item["start_month"]
    interval = item.get("interval", "1d")
    dest = Path("/data/quant/raw/klines")
    months = month_range(start_month)
    t0 = time.time()
    path = download_symbol_months(symbol, months, dest, interval=interval)
    volume.commit()
    return {"symbol": symbol, "path": str(path), "elapsed": time.time() - t0}


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def train_one_fold_job(payload: dict) -> dict:
    """Train a single CV fold; payload carries paths + fold spec + horizon."""
    import pandas as pd
    from baseline.model import FoldSpec, _fit_predict_fold
    from baseline.seedutil import seed_everything

    cfg = payload["cfg"]
    seed_everything(cfg["seed"] + int(payload["fold_id"]))
    df = pd.read_parquet(payload["feat_path"])
    fold = FoldSpec(
        fold_id=int(payload["fold_id"]),
        train_start=pd.Timestamp(payload["train_start"]),
        train_end=pd.Timestamp(payload["train_end"]),
        purge_end=pd.Timestamp(payload["purge_end"]),
        embargo_end=pd.Timestamp(payload["embargo_end"]),
        val_start=pd.Timestamp(payload["val_start"]),
        val_end=pd.Timestamp(payload["val_end"]),
        horizon=int(payload["horizon"]),
    )
    t0 = time.time()
    print(
        f"[fold] start h={fold.horizon} id={fold.fold_id} "
        f"val={fold.val_start.date()}→{fold.val_end.date()}",
        flush=True,
    )
    pred_df, meta = _fit_predict_fold(
        df,
        fold,
        seed=cfg["seed"],
        model_cfg=cfg["model"],
        inner_holdout_days=cfg["cv"]["inner_holdout_days"],
    )
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"preds_h{fold.horizon}_fold{fold.fold_id}.parquet"
    if not pred_df.empty:
        pred_df.to_parquet(pred_path, index=False)
    meta["pred_path"] = str(pred_path) if not pred_df.empty else None
    meta["wall_elapsed"] = time.time() - t0
    if meta["wall_elapsed"] > cfg["cv"].get("fold_warn_seconds", 1200):
        print(f"[WARN] fold {fold.fold_id} exceeded 20 min: {meta['wall_elapsed']:.0f}s", flush=True)
    (out_dir / f"meta_h{fold.horizon}_fold{fold.fold_id}.json").write_text(json.dumps(meta, indent=2))
    volume.commit()
    print(f"[fold] done h={fold.horizon} id={fold.fold_id} elapsed={meta['wall_elapsed']:.1f}s", flush=True)
    return meta


@app.function(timeout=60 * 60 * 6, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_pipeline() -> dict:
    import numpy as np
    import pandas as pd

    from baseline.data import (
        build_pit_topn,
        list_um_symbols,
        load_panel,
        select_train_universe,
        should_exclude,
    )
    from baseline.evaluate import evaluate_predictions, naive_mom28_scores
    from baseline.features import build_feature_panel
    from baseline.gates import run_all_gates
    from baseline.kronos_adapter import try_export_kronos_ft
    from baseline.labels import add_labels
    from baseline.model import make_folds
    from baseline.portfolio import run_portfolio_backtest
    from baseline.report import plot_equity_curves, plot_ic_analysis, write_report
    from baseline.seedutil import seed_everything

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    raw_dir = root / "raw" / "klines"
    feat_dir = root / "features"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in [raw_dir, feat_dir, pred_dir, uni_dir, rep_dir, chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t_pipe = time.time()
    print("[pipeline] listing symbols...", flush=True)
    symbols = list_um_symbols(cfg["data"]["quote"])
    symbols = [s for s in symbols if not should_exclude(s, cfg["data"]["exclude_bases"])]
    # always include BTC
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    print(f"[pipeline] {len(symbols)} USDT perps after filters", flush=True)

    # Download (skip cached parquet)
    todo = []
    for s in symbols:
        if not (raw_dir / f"{s}.parquet").exists():
            todo.append(
                {
                    "symbol": s,
                    "start_month": cfg["data"]["start_month"],
                    "interval": cfg["data"]["interval"],
                }
            )
    print(f"[pipeline] downloading {len(todo)} / {len(symbols)} symbols...", flush=True)
    if todo:
        # chunked map to avoid huge fanout
        results = []
        chunk = 80
        for i in range(0, len(todo), chunk):
            part = todo[i : i + chunk]
            print(f"[pipeline] download chunk {i//chunk+1}/{(len(todo)-1)//chunk+1} n={len(part)}", flush=True)
            results.extend(list(download_one_symbol.map(part)))
            volume.reload()
        print(f"[pipeline] download jobs finished: {len(results)}", flush=True)
    volume.reload()

    print("[pipeline] building panel...", flush=True)
    panel = load_panel(raw_dir, symbols)
    # keep symbols with enough history
    counts = panel.groupby("symbol").size()
    keep = counts[counts >= cfg["features"]["min_history_days"]].index.tolist()
    if "BTCUSDT" not in keep:
        raise RuntimeError("BTCUSDT missing after load")
    panel = panel[panel["symbol"].isin(keep)].copy()
    train_syms = select_train_universe(panel, n=cfg["data"]["train_universe_n"])
    panel_train = panel[panel["symbol"].isin(train_syms)].copy()
    print(
        f"[pipeline] panel rows={len(panel)} symbols={panel['symbol'].nunique()} "
        f"train_universe={len(train_syms)} span={panel['date'].min().date()}→{panel['date'].max().date()}",
        flush=True,
    )
    panel_train.to_parquet(root / "panel_train.parquet", index=False)

    print("[pipeline] PIT top-20 universe...", flush=True)
    pit = build_pit_topn(
        panel,  # full liquid history for DV ranks among all downloaded
        n=cfg["data"]["exec_universe_n"],
        window=cfg["data"]["exec_dv_window"],
    )
    # restrict ranks to train-universe symbols intersecting liquidity
    pit = pit[pit["symbol"].isin(train_syms)].copy()
    # re-rank within filtered set per date
    pit = (
        pit.sort_values(["date", "dv_med"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(cfg["data"]["exec_universe_n"])
        .copy()
    )
    pit["rank"] = pit.groupby("date").cumcount() + 1
    pit_path = uni_dir / "top20_pit.parquet"
    pit.to_parquet(pit_path, index=False)
    print(f"[pipeline] wrote {pit_path} rows={len(pit)}", flush=True)

    print("[pipeline] features...", flush=True)
    t0 = time.time()
    feat = build_feature_panel(panel_train, clip=cfg["features"]["zscore_clip"])
    print(f"[pipeline] features done in {time.time()-t0:.1f}s rows={len(feat)}", flush=True)

    print("[pipeline] labels...", flush=True)
    feat = add_labels(
        feat,
        panel_train,
        horizons=cfg["labels"]["horizons"],
        winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]),
    )
    feat_path = feat_dir / "features_labeled.parquet"
    feat.to_parquet(feat_path, index=False)
    volume.commit()

    caveats = [
        "Funding rate not available in Binance Vision kline dumps → funding PnL = 0.",
        "Execution assumed at close of signal day t; PnL uses next-day close-to-close returns.",
        "Training universe = top 120 by full-sample median dollar volume (documented); execution uses PIT top-20 only.",
        "Delisted perps included when present on data.binance.vision monthly dumps.",
        "Spot pre-listing fallback for majors not applied (perps dumps start at listing).",
        "Beta hedge implemented as additive BTCUSDT weight = −Σ w_i β_i.",
    ]

    # Kronos adapter (best effort)
    kronos_status = try_export_kronos_ft(pred_dir / "kronos_ft.parquet", horizon=10)

    ic_tables: dict[str, list] = {}
    all_gate_results = []
    portfolio_summaries = []
    best_eq = None
    naive_eq = None
    primary_ic = None
    primary_quints = {}

    for h in cfg["labels"]["horizons"]:
        print(f"[pipeline] CV train horizon={h}", flush=True)
        folds = make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        if not folds:
            print(f"[pipeline] no folds for h={h}", flush=True)
            continue
        payloads = []
        out_h = pred_dir / f"h{h}"
        out_h.mkdir(parents=True, exist_ok=True)
        for fr in folds:
            payloads.append(
                {
                    "cfg": cfg,
                    "feat_path": str(feat_path),
                    "out_dir": str(out_h),
                    "fold_id": fr.fold_id,
                    "train_start": str(fr.train_start),
                    "train_end": str(fr.train_end),
                    "purge_end": str(fr.purge_end),
                    "embargo_end": str(fr.embargo_end),
                    "val_start": str(fr.val_start),
                    "val_end": str(fr.val_end),
                    "horizon": h,
                }
            )
        metas = list(train_one_fold_job.map(payloads))
        volume.reload()
        preds = []
        for m in metas:
            if m.get("pred_path") and Path(m["pred_path"]).exists():
                preds.append(pd.read_parquet(m["pred_path"]))
        if not preds:
            print(f"[pipeline] no preds for h={h}", flush=True)
            continue
        pred_all = pd.concat(preds, ignore_index=True)
        # dedupe overlapping val windows: keep first fold prediction
        pred_all = pred_all.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
            ["date", "symbol"], keep="first"
        )
        canon = pred_all[["date", "symbol", "score", "horizon"]].copy()
        canon["model_name"] = "lgbm_price_only"
        canon_path = pred_dir / f"lgbm_price_only_h{h}.parquet"
        # attach y for eval
        ycol = f"y_h{h}"
        canon = canon.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        canon.to_parquet(canon_path, index=False)

        # naive benchmark aligned to same OOS dates
        naive = naive_mom28_scores(feat, h)
        oos_dates = set(pd.to_datetime(canon["date"], utc=True))
        naive = naive[pd.to_datetime(naive["date"], utc=True).isin(oos_dates)].copy()
        naive_path = pred_dir / f"naive_mom28_h{h}.parquet"
        naive.to_parquet(naive_path, index=False)

        rows = []
        for label, uni in [("full", None), ("top20", pit)]:
            ev = evaluate_predictions(canon, h, universe=uni, label=label)
            rows.append({k: v for k, v in ev.items() if k != "ic_series"})
            if h == cfg["labels"]["primary_horizon"] and label == "top20":
                primary_ic = ev["ic_series"]
                primary_quints = ev.get("quintile_means") or {}
        # naive IC top20
        ev_n = evaluate_predictions(naive, h, universe=pit, label="top20_naive")
        rows.append({k: v for k, v in ev_n.items() if k != "ic_series"})
        if kronos_status.get("exported") and h == 10:
            try:
                kdf = pd.read_parquet(pred_dir / "kronos_ft.parquet")
                # attach labels
                kdf = kdf.merge(feat[["date", "symbol", "y_h10"]], on=["date", "symbol"], how="inner")
                kdf = kdf[pd.to_datetime(kdf["date"], utc=True).isin(oos_dates)]
                if len(kdf):
                    ev_k = evaluate_predictions(kdf, 10, universe=pit, label="top20_kronos_ft")
                    rows.append({k: v for k, v in ev_k.items() if k != "ic_series"})
            except Exception as e:
                caveats.append(f"Kronos eval skipped: {e}")
        ic_tables[f"h={h}"] = rows

        # gates on ALL OOS preds for primary horizon (more days → quieter null IC)
        if h == cfg["labels"]["primary_horizon"]:
            sample = pred_all.copy()
            gates = run_all_gates(
                panel_train,
                feat,
                build_pit_topn,
                folds[0],
                cfg,
                sample,
            )
            all_gate_results = gates
            if not all(g.get("passed") for g in gates):
                raise RuntimeError(f"Sanity gates failed: {gates}")

            # portfolio sweep
            print("[pipeline] portfolio sweep...", flush=True)
            best = None
            for tp in cfg["portfolio"]["tau_percentiles"]:
                res = run_portfolio_backtest(
                    canon,
                    panel_train,
                    feat,
                    pit,
                    horizon=h,
                    tau_pct=tp,
                    exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                    gross_limit=cfg["portfolio"]["gross_limit"],
                    fee_bps=cfg["portfolio"]["taker_fee_bps"],
                    slip_bps=cfg["portfolio"]["slippage_bps"],
                )
                slim = {k: v for k, v in res.items() if k not in ("equity", "daily_ret")}
                portfolio_summaries.append(slim)
                print(f"[pipeline] τ={tp} -> {slim}", flush=True)
                if "net_sharpe" in res and (best is None or res["net_sharpe"] > best["net_sharpe"]):
                    best = res
            if best is not None:
                best_eq = best["equity"]
                best_eq.to_parquet(rep_dir / "best_equity.parquet", index=False)

            # naive portfolio at 70th pct as comparison
            naive_bt = run_portfolio_backtest(
                naive,
                panel_train,
                feat,
                pit,
                horizon=h,
                tau_pct=70,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
            )
            if "equity" in naive_bt:
                naive_eq = naive_bt["equity"]

    # BTC buy & hold over OOS span
    if best_eq is not None and len(best_eq):
        btc = panel_train[panel_train["symbol"] == "BTCUSDT"].set_index("date")["close"].sort_index()
        idx = pd.to_datetime(best_eq["date"], utc=True)
        btc = btc.reindex(idx).dropna()
        btc_eq = pd.DataFrame(
            {
                "date": btc.index,
                "equity": (btc / btc.iloc[0]).values,
            }
        )
    else:
        btc_eq = pd.DataFrame({"date": [], "equity": []})

    # charts + report
    if primary_ic is not None:
        plot_ic_analysis(chart_dir / "ic_analysis.png", primary_ic, primary_quints)
    if best_eq is not None and len(best_eq) and len(btc_eq):
        plot_equity_curves(chart_dir / "equity_curves.png", best_eq, naive_eq, btc_eq)

    write_report(
        rep_dir / "baseline_report.md",
        cfg,
        ic_tables,
        portfolio_summaries,
        all_gate_results,
        caveats,
        kronos_status,
    )

    summary = {
        "elapsed_sec": time.time() - t_pipe,
        "n_symbols_train": len(train_syms),
        "span": [str(panel_train["date"].min().date()), str(panel_train["date"].max().date())],
        "gates": all_gate_results,
        "ic_tables": {
            k: [{kk: vv for kk, vv in r.items() if kk != "ic_series"} for r in rows]
            for k, rows in ic_tables.items()
        },
        "portfolio": portfolio_summaries,
        "kronos_status": kronos_status,
        "caveats": caveats,
    }
    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()
    print("[pipeline] DONE", flush=True)
    print(json.dumps({k: summary[k] for k in summary if k != "ic_tables"}, indent=2, default=str), flush=True)
    return summary


@app.local_entrypoint()
def main():
    print("Launching Phase A0 baseline on Modal...", flush=True)
    summary = run_pipeline.remote()
    # copy reports/charts back
    local = Path("artifacts")
    (local / "reports").mkdir(parents=True, exist_ok=True)
    (local / "charts").mkdir(parents=True, exist_ok=True)

    # pull via a small helper function reading volume files
    data = fetch_artifacts.remote()
    for rel, content in data.get("text", {}).items():
        p = local / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for rel, raw in data.get("bin", {}).items():
        p = local / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(raw))

    print("\n===== FINAL STDOUT SUMMARY =====", flush=True)
    print("Portfolio sweep:", flush=True)
    print(
        f"{'τ':>4} {'Sharpe':>8} {'CAGR':>9} {'MaxDD':>9} {'%flat':>8} {'annTO':>8}",
        flush=True,
    )
    for r in summary.get("portfolio", []):
        if "error" in r:
            print(f"{r.get('tau_pct'):>4} ERROR", flush=True)
            continue
        print(
            f"{r['tau_pct']:>4} {r['net_sharpe']:>8.2f} {r['net_cagr']:>8.2%} "
            f"{r['max_drawdown']:>8.2%} {r['pct_flat_days']:>7.1%} {r['ann_turnover']:>8.1f}",
            flush=True,
        )
    print("\nRankIC:", flush=True)
    for hkey, rows in summary.get("ic_tables", {}).items():
        for r in rows:
            print(
                f"  {hkey} {r.get('universe')}: mean={r.get('mean_ic', float('nan')):.4f} "
                f"ICIR={r.get('icir', float('nan')):.3f} NW_t={r.get('nw_tstat', float('nan')):.2f}",
                flush=True,
            )
    print(f"\nArtifacts copied to ./artifacts/  elapsed={summary.get('elapsed_sec', float('nan')):.0f}s", flush=True)
    gates_ok = all(g.get("passed") for g in summary.get("gates", []))
    print(f"Gates: {'ALL PASS' if gates_ok else 'FAILED'}", flush=True)


@app.function(timeout=60 * 10, retries=0, volumes={"/data/quant": volume})
def fetch_artifacts() -> dict:
    root = Path("/data/quant")
    text = {}
    binary = {}
    for rel in [
        "reports/baseline_report.md",
        "reports/summary.json",
    ]:
        p = root / rel
        if p.exists():
            text[rel] = p.read_text()
    for rel in [
        "charts/equity_curves.png",
        "charts/ic_analysis.png",
    ]:
        p = root / rel
        if p.exists():
            binary[rel] = list(p.read_bytes())
    return {"text": text, "bin": binary}
