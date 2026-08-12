"""
Phase A0 price-only baseline — Modal entrypoint.

Usage:
    modal run pipeline.py
"""

from __future__ import annotations

import json
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
    log_curve = bool(payload.get("log_eval_curve", False))
    print(
        f"[fold] start h={fold.horizon} id={fold.fold_id} "
        f"val={fold.val_start.date()}→{fold.val_end.date()} log_curve={log_curve}",
        flush=True,
    )
    pred_df, meta = _fit_predict_fold(
        df,
        fold,
        seed=cfg["seed"],
        model_cfg=cfg["model"],
        inner_holdout_days=cfg["cv"]["inner_holdout_days"],
        log_eval_curve=log_curve,
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
    (out_dir / f"meta_h{fold.horizon}_fold{fold.fold_id}.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    volume.commit()
    print(
        f"[fold] done h={fold.horizon} id={fold.fold_id} "
        f"elapsed={meta['wall_elapsed']:.1f}s best_iter={meta.get('best_iteration')} "
        f"mode={meta.get('early_stop_mode')}",
        flush=True,
    )
    return meta


@app.function(timeout=60 * 60 * 6, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_pipeline() -> dict:
    import pandas as pd

    from baseline.data import (
        build_pit_topn,
        list_um_symbols,
        load_panel,
        luna_presence_report,
        should_exclude,
    )
    from baseline.evaluate import evaluate_predictions, naive_mom28_scores
    from baseline.features import apply_cs_zscore, build_feature_panel
    from baseline.gates import run_all_gates
    from baseline.kronos_adapter import try_export_kronos_ft
    from baseline.labels import add_labels
    from baseline.model import best_iteration_distribution, make_folds
    from baseline.portfolio import run_portfolio_backtest, run_tranche_portfolio
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
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    print(f"[pipeline] {len(symbols)} USDT perps after filters", flush=True)

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
        results = []
        chunk = 80
        for i in range(0, len(todo), chunk):
            part = todo[i : i + chunk]
            print(
                f"[pipeline] download chunk {i//chunk+1}/{(len(todo)-1)//chunk+1} n={len(part)}",
                flush=True,
            )
            results.extend(list(download_one_symbol.map(part)))
            volume.reload()
        print(f"[pipeline] download jobs finished: {len(results)}", flush=True)
    volume.reload()

    print("[pipeline] building panel...", flush=True)
    panel = load_panel(raw_dir, symbols)
    counts = panel.groupby("symbol").size()
    keep = counts[counts >= cfg["features"]["min_history_days"]].index.tolist()
    if "BTCUSDT" not in keep:
        raise RuntimeError("BTCUSDT missing after load")
    panel = panel[panel["symbol"].isin(keep)].copy()
    print(
        f"[pipeline] panel rows={len(panel)} symbols={panel['symbol'].nunique()} "
        f"span={panel['date'].min().date()}→{panel['date'].max().date()}",
        flush=True,
    )

    window = cfg["data"]["exec_dv_window"]
    print("[pipeline] PIT top-120 / top-20 universes...", flush=True)
    pit120 = build_pit_topn(panel, n=cfg["data"]["train_universe_n"], window=window)
    pit20 = build_pit_topn(panel, n=cfg["data"]["exec_universe_n"], window=window)
    pit120_path = uni_dir / "top120_pit.parquet"
    pit20_path = uni_dir / "top20_pit.parquet"
    pit120.to_parquet(pit120_path, index=False)
    pit20.to_parquet(pit20_path, index=False)
    print(f"[pipeline] wrote {pit120_path} rows={len(pit120)}; {pit20_path} rows={len(pit20)}", flush=True)

    luna_top20 = luna_presence_report(pit20)
    luna_top120 = luna_presence_report(pit120)
    print(f"[pipeline] LUNA top20: {luna_top20}", flush=True)
    print(f"[pipeline] LUNA top120: {luna_top120}", flush=True)

    # Feature panel: all symbols that ever appear in PIT-120 (+ BTC)
    ever120 = set(pit120["symbol"].unique()) | {"BTCUSDT"}
    panel_feat = panel[panel["symbol"].isin(ever120)].copy()
    print(
        f"[pipeline] feature symbols (ever in PIT-120)={panel_feat['symbol'].nunique()}",
        flush=True,
    )

    print("[pipeline] features (raw)...", flush=True)
    t0 = time.time()
    feat_raw = build_feature_panel(panel_feat, clip=cfg["features"]["zscore_clip"], zscore=False)
    print(f"[pipeline] raw features done in {time.time()-t0:.1f}s rows={len(feat_raw)}", flush=True)

    # Restrict to PIT-120 membership, then CS z-score within that universe
    pit120_keys = pit120[["date", "symbol"]].copy()
    pit120_keys["date"] = pd.to_datetime(pit120_keys["date"], utc=True)
    feat_raw["date"] = pd.to_datetime(feat_raw["date"], utc=True)
    feat = feat_raw.merge(pit120_keys, on=["date", "symbol"], how="inner")
    print(f"[pipeline] after PIT-120 filter rows={len(feat)}", flush=True)
    feat = apply_cs_zscore(feat, clip=cfg["features"]["zscore_clip"])

    print("[pipeline] labels...", flush=True)
    feat = add_labels(
        feat,
        panel_feat,
        horizons=cfg["labels"]["horizons"],
        winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]),
    )
    feat_path = feat_dir / "features_labeled.parquet"
    feat.to_parquet(feat_path, index=False)
    volume.commit()

    caveats = [
        "Funding rate not available in Binance Vision kline dumps → funding PnL = 0.",
        "Execution assumed at close of signal day t; PnL uses next-day close-to-close returns.",
        "Training universe = point-in-time top 120 by 30d rolling median dollar volume (data ≤ t).",
        "Execution universe = point-in-time top 20 (same mechanism).",
        "Cross-sectional feature z-scores computed within the PIT-120 membership each day.",
        "Labels/hedge use raw beta_btc_60 (not the z-scored feature column).",
        "Delisted perps included when present on data.binance.vision monthly dumps.",
        "Spot pre-listing fallback for majors not applied (perps dumps start at listing).",
        "Beta hedge implemented as additive BTCUSDT weight = −Σ w_i β_i.",
        "Tranche portfolio: h capital slices, rebalance offset k on day_index % h == k; hold ~h days.",
    ]

    kronos_status = try_export_kronos_ft(pred_dir / "kronos_ft.parquet", horizon=10)

    ic_tables: dict[str, list] = {}
    all_gate_results: list = []
    portfolio_summaries: list = []
    best_eq = None
    tranche_eq = None
    naive_eq = None
    primary_ic = None
    primary_quints: dict = {}
    best_iter_stats: dict = {"by_horizon": {}}
    sensitivity: dict = {}
    used_fixed_trees = False

    def _run_folds(h: int, model_cfg: dict, tag: str) -> tuple[pd.DataFrame, list[dict]]:
        folds = make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        if not folds:
            return pd.DataFrame(), []
        payloads = []
        out_h = pred_dir / f"{tag}_h{h}"
        out_h.mkdir(parents=True, exist_ok=True)
        diag_ids = {0, max(0, len(folds) // 2), len(folds) - 1}
        cfg_job = dict(cfg)
        cfg_job["model"] = model_cfg
        for fr in folds:
            payloads.append(
                {
                    "cfg": cfg_job,
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
                    "log_eval_curve": fr.fold_id in diag_ids,
                }
            )
        print(f"[pipeline] training {len(payloads)} folds h={h} tag={tag}", flush=True)
        metas = list(train_one_fold_job.map(payloads))
        volume.reload()
        preds = []
        for m in metas:
            if m.get("pred_path") and Path(m["pred_path"]).exists():
                preds.append(pd.read_parquet(m["pred_path"]))
        if not preds:
            return pd.DataFrame(), metas
        pred_all = pd.concat(preds, ignore_index=True)
        pred_all = pred_all.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
            ["date", "symbol"], keep="first"
        )
        return pred_all, metas

    for h in cfg["labels"]["horizons"]:
        print(f"[pipeline] CV train horizon={h}", flush=True)
        model_cfg = dict(cfg["model"])
        pred_all, metas = _run_folds(h, model_cfg, tag="rankic")
        dist = best_iteration_distribution(metas)
        dist["mode"] = "rank_ic"
        print(f"[pipeline] h={h} best_iteration dist: {dist}", flush=True)
        best_iter_stats["by_horizon"][f"h={h}_rank_ic_attempt"] = dict(dist)

        # Fallback if RankIC early-stop still degenerate
        if dist["n"] and dist["gt1_frac"] < 0.9:
            print(
                f"[pipeline] best_iteration>1 on only {dist['gt1_frac']:.0%} of folds — "
                f"falling back to fixed n_estimators=500",
                flush=True,
            )
            used_fixed_trees = True
            model_cfg = dict(cfg["model"])
            model_cfg["fixed_n_estimators"] = 500
            model_cfg["early_stop_metric"] = "none"
            # Sensitivity for {200,500,1000} on the horizon that needed fallback
            sens = {}
            for nt in cfg["model"].get("sensitivity_trees", [200, 500, 1000]):
                mcfg = dict(cfg["model"])
                mcfg["fixed_n_estimators"] = int(nt)
                mcfg["early_stop_metric"] = "none"
                p_s, m_s = _run_folds(h, mcfg, tag=f"sens{nt}")
                if p_s.empty:
                    continue
                tmp = p_s.copy()
                ycol = f"y_h{h}"
                if ycol not in tmp.columns:
                    tmp = tmp.merge(
                        feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left"
                    )
                # drop merge suffixes if present
                if f"{ycol}_x" in tmp.columns:
                    tmp[ycol] = tmp[f"{ycol}_y"].fillna(tmp[f"{ycol}_x"])
                ev = evaluate_predictions(tmp, h, universe=pit20, label=f"top20_trees{nt}")
                sens[str(nt)] = {
                    "mean_ic": ev.get("mean_ic"),
                    "icir": ev.get("icir"),
                    "nw_tstat": ev.get("nw_tstat"),
                    "best_iter_dist": best_iteration_distribution(m_s),
                }
                if int(nt) == 500:
                    pred_all, metas = p_s, m_s
                    dist = best_iteration_distribution(m_s)
                    dist["mode"] = "fixed_500"
            sensitivity[f"h={h}"] = sens
            if dist.get("mode") != "fixed_500":
                pred_all, metas = _run_folds(h, model_cfg, tag="fixed500")
                dist = best_iteration_distribution(metas)
                dist["mode"] = "fixed_500"

        best_iter_stats["by_horizon"][f"h={h}"] = dist
        if pred_all.empty:
            print(f"[pipeline] no preds for h={h}", flush=True)
            continue

        canon = pred_all[["date", "symbol", "score", "horizon"]].copy()
        canon["model_name"] = "lgbm_price_only"
        ycol = f"y_h{h}"
        canon = canon.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        canon_path = pred_dir / f"lgbm_price_only_h{h}.parquet"
        canon.to_parquet(canon_path, index=False)

        naive = naive_mom28_scores(feat, h)
        oos_dates = set(pd.to_datetime(canon["date"], utc=True))
        naive = naive[pd.to_datetime(naive["date"], utc=True).isin(oos_dates)].copy()
        naive.to_parquet(pred_dir / f"naive_mom28_h{h}.parquet", index=False)

        rows = []
        for label, uni in [("pit120", pit120), ("top20", pit20)]:
            ev = evaluate_predictions(canon, h, universe=uni, label=label)
            rows.append({k: v for k, v in ev.items() if k != "ic_series"})
            if h == cfg["labels"]["primary_horizon"] and label == "top20":
                primary_ic = ev["ic_series"]
                primary_quints = ev.get("quintile_means") or {}
        ev_n = evaluate_predictions(naive, h, universe=pit20, label="top20_naive")
        rows.append({k: v for k, v in ev_n.items() if k != "ic_series"})
        if kronos_status.get("exported") and h == 10:
            try:
                kdf = pd.read_parquet(pred_dir / "kronos_ft.parquet")
                kdf = kdf.merge(feat[["date", "symbol", "y_h10"]], on=["date", "symbol"], how="inner")
                kdf = kdf[pd.to_datetime(kdf["date"], utc=True).isin(oos_dates)]
                if len(kdf):
                    ev_k = evaluate_predictions(kdf, 10, universe=pit20, label="top20_kronos_ft")
                    rows.append({k: v for k, v in ev_k.items() if k != "ic_series"})
            except Exception as e:
                caveats.append(f"Kronos eval skipped: {e}")
        ic_tables[f"h={h}"] = rows

        # Gates on FIRST fold only (original strictness) for primary horizon
        if h == cfg["labels"]["primary_horizon"]:
            folds = make_folds(
                pd.DatetimeIndex(feat["date"].unique()),
                horizon=h,
                min_train_days=cfg["cv"]["min_train_days"],
                val_days=cfg["cv"]["val_days"],
                step_days=cfg["cv"]["step_days"],
            )
            sample = pred_all[pred_all["fold_id"] == pred_all["fold_id"].min()].copy()
            # ensure y attached
            if ycol not in sample.columns:
                sample = sample.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
            gates = run_all_gates(
                panel_feat,
                feat,
                build_pit_topn,
                folds[0],
                cfg,
                sample,
            )
            all_gate_results = gates
            if not all(g.get("passed") for g in gates):
                raise RuntimeError(f"Sanity gates failed: {gates}")

        # Portfolio sweeps for BOTH horizons, daily + tranche
        print(f"[pipeline] portfolio sweeps h={h}...", flush=True)
        best_daily = None
        best_tranche = None
        for tp in cfg["portfolio"]["tau_percentiles"]:
            res = run_portfolio_backtest(
                canon,
                panel_feat,
                feat,
                pit20,
                horizon=h,
                tau_pct=tp,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                variant="daily",
            )
            slim = {k: v for k, v in res.items() if k not in ("equity", "daily_ret")}
            portfolio_summaries.append(slim)
            print(f"[pipeline] daily h={h} τ={tp} -> sharpe={slim.get('net_sharpe')} cost={slim.get('cost_drag')}", flush=True)
            if "net_sharpe" in res and (best_daily is None or res["net_sharpe"] > best_daily["net_sharpe"]):
                best_daily = res

            tres = run_tranche_portfolio(
                canon,
                panel_feat,
                feat,
                pit20,
                horizon=h,
                tau_pct=tp,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
            )
            tslim = {k: v for k, v in tres.items() if k not in ("equity", "daily_ret")}
            portfolio_summaries.append(tslim)
            print(
                f"[pipeline] tranche h={h} τ={tp} -> sharpe={tslim.get('net_sharpe')} "
                f"to={tslim.get('ann_turnover')}",
                flush=True,
            )
            if "net_sharpe" in tres and (best_tranche is None or tres["net_sharpe"] > best_tranche["net_sharpe"]):
                best_tranche = tres

        if h == cfg["labels"]["primary_horizon"]:
            if best_daily is not None:
                best_eq = best_daily["equity"]
                best_eq.to_parquet(rep_dir / "best_equity_daily.parquet", index=False)
            if best_tranche is not None:
                tranche_eq = best_tranche["equity"]
                tranche_eq.to_parquet(rep_dir / "best_equity_tranche.parquet", index=False)
            naive_bt = run_portfolio_backtest(
                naive,
                panel_feat,
                feat,
                pit20,
                horizon=h,
                tau_pct=70,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                variant="daily_naive",
            )
            if "equity" in naive_bt:
                naive_eq = naive_bt["equity"]

    if used_fixed_trees:
        caveats.append(
            "Early-stopping on RankIC was unhealthy (best_iteration>1 on <90% of folds); "
            "fell back to fixed n_estimators=500 with tree-count sensitivity rows in the report."
        )
    else:
        caveats.append("Early-stopping metric = mean daily RankIC on inner holdout (maximize), patience 100.")

    # BTC buy & hold over OOS span
    if best_eq is not None and len(best_eq):
        btc = panel_feat[panel_feat["symbol"] == "BTCUSDT"].set_index("date")["close"].sort_index()
        idx = pd.to_datetime(best_eq["date"], utc=True)
        btc = btc.reindex(idx).dropna()
        btc_eq = pd.DataFrame({"date": btc.index, "equity": (btc / btc.iloc[0]).values})
    else:
        btc_eq = pd.DataFrame({"date": [], "equity": []})

    if primary_ic is not None:
        plot_ic_analysis(chart_dir / "ic_analysis.png", primary_ic, primary_quints)
    if best_eq is not None and len(best_eq) and len(btc_eq):
        plot_equity_curves(
            chart_dir / "equity_curves.png",
            best_eq,
            naive_eq,
            btc_eq,
            tranche_eq=tranche_eq,
        )

    write_report(
        rep_dir / "baseline_report.md",
        cfg,
        ic_tables,
        portfolio_summaries,
        all_gate_results,
        caveats,
        kronos_status,
        best_iter_stats=best_iter_stats,
        luna_report={"top20": luna_top20, "top120": luna_top120},
        sensitivity=sensitivity,
    )

    summary = {
        "elapsed_sec": time.time() - t_pipe,
        "n_symbols_ever_pit120": int(panel_feat["symbol"].nunique()),
        "span": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "gates": all_gate_results,
        "luna": {"top20": luna_top20, "top120": luna_top120},
        "best_iteration": best_iter_stats,
        "sensitivity": sensitivity,
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
    return summary


@app.local_entrypoint()
def main():
    print("Launching Phase A0 remediation pipeline on Modal...", flush=True)
    summary = run_pipeline.remote()
    local = Path("artifacts")
    (local / "reports").mkdir(parents=True, exist_ok=True)
    (local / "charts").mkdir(parents=True, exist_ok=True)

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
    print("RankIC:", flush=True)
    for hkey, rows in summary.get("ic_tables", {}).items():
        for r in rows:
            print(
                f"  {hkey} {r.get('universe')}: mean={r.get('mean_ic', float('nan')):.4f} "
                f"ICIR={r.get('icir', float('nan')):.3f} NW_t={r.get('nw_tstat', float('nan')):.2f}",
                flush=True,
            )
    print("\nPortfolio (net/gross Sharpe, cost, hedge, hold, turnover):", flush=True)
    print(
        f"{'var':>8} {'h':>3} {'τ':>4} {'netSh':>7} {'grSh':>7} {'cost':>8} {'hedge':>8} "
        f"{'hold':>6} {'annTO':>7}",
        flush=True,
    )
    for r in summary.get("portfolio", []):
        if "error" in r:
            print(f"{r.get('variant'):>8} {r.get('horizon'):>3} {r.get('tau_pct'):>4} ERROR", flush=True)
            continue
        print(
            f"{str(r.get('variant')):>8} {r.get('horizon'):>3} {r['tau_pct']:>4} "
            f"{r['net_sharpe']:>7.2f} {r.get('gross_sharpe', float('nan')):>7.2f} "
            f"{r.get('cost_drag', float('nan')):>8.3f} {r.get('hedge_total_pnl', float('nan')):>8.3f} "
            f"{r.get('avg_holding_days', float('nan')):>6.1f} {r['ann_turnover']:>7.1f}",
            flush=True,
        )
    print(f"\nbest_iteration: {json.dumps(summary.get('best_iteration'), default=str)}", flush=True)
    print(f"LUNA: {json.dumps(summary.get('luna'), default=str)}", flush=True)
    print(f"\nArtifacts → ./artifacts/  elapsed={summary.get('elapsed_sec', float('nan')):.0f}s", flush=True)
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
