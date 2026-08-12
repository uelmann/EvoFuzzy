"""
Phase A0 price-only baseline — Modal entrypoint (stress-test pass).

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
    dest = Path("/data/quant/raw/klines")
    months = month_range(item["start_month"])
    t0 = time.time()
    path = download_symbol_months(symbol, months, dest, interval=item.get("interval", "1d"))
    volume.commit()
    return {"symbol": symbol, "path": str(path), "elapsed": time.time() - t0}


@app.function(timeout=60 * 30, retries=0, volumes={"/data/quant": volume}, cpu=2, memory=4096)
def download_one_funding(item: dict) -> dict:
    from baseline.data import download_funding_symbol_months, month_range

    symbol = item["symbol"]
    dest = Path("/data/quant/raw/funding")
    months = month_range(item["start_month"])
    t0 = time.time()
    path = download_funding_symbol_months(symbol, months, dest)
    volume.commit()
    return {"symbol": symbol, "path": str(path), "elapsed": time.time() - t0}


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def train_one_fold_job(payload: dict) -> dict:
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
    pred_df, meta = _fit_predict_fold(
        df,
        fold,
        seed=cfg["seed"],
        model_cfg=cfg["model"],
        inner_holdout_days=cfg["cv"]["inner_holdout_days"],
        log_eval_curve=bool(payload.get("log_eval_curve", False)),
    )
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"preds_h{fold.horizon}_fold{fold.fold_id}.parquet"
    if not pred_df.empty:
        pred_df.to_parquet(pred_path, index=False)
    meta["pred_path"] = str(pred_path) if not pred_df.empty else None
    meta["wall_elapsed"] = time.time() - t0
    (out_dir / f"meta_h{fold.horizon}_fold{fold.fold_id}.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    volume.commit()
    print(
        f"[fold] done h={fold.horizon} id={fold.fold_id} "
        f"elapsed={meta['wall_elapsed']:.1f}s best_iter={meta.get('best_iteration')}",
        flush=True,
    )
    return meta


def _slim(res: dict) -> dict:
    drop = {
        "equity", "daily_ret", "daily_gross", "daily_hedge", "daily_cost",
        "daily_funding", "sym_contrib", "side_days",
    }
    return {k: v for k, v in res.items() if k not in drop}


@app.function(timeout=60 * 60 * 8, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_pipeline() -> dict:
    import pandas as pd

    from baseline.attribution import (
        day_concentration,
        ic_dispersion_diagnostic,
        median_tau_summary,
        per_year_breakdown,
        symbol_attribution,
    )
    from baseline.data import (
        build_pit_topn,
        funding_coverage_report,
        list_um_symbols,
        load_funding_panel,
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
    from baseline.report import (
        plot_attribution,
        plot_equity_curves,
        plot_ic_analysis,
        write_report,
    )
    from baseline.seedutil import seed_everything

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    feat_dir = root / "features"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in [raw_dir, fund_dir, feat_dir, pred_dir, uni_dir, rep_dir, chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t_pipe = time.time()
    print("[pipeline] listing symbols...", flush=True)
    symbols = list_um_symbols(cfg["data"]["quote"])
    symbols = [s for s in symbols if not should_exclude(s, cfg["data"]["exclude_bases"])]
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    print(f"[pipeline] {len(symbols)} USDT perps after filters", flush=True)

    todo = [
        {"symbol": s, "start_month": cfg["data"]["start_month"], "interval": cfg["data"]["interval"]}
        for s in symbols
        if not (raw_dir / f"{s}.parquet").exists()
    ]
    print(f"[pipeline] downloading {len(todo)} / {len(symbols)} kline symbols...", flush=True)
    if todo:
        for i in range(0, len(todo), 80):
            part = todo[i : i + 80]
            list(download_one_symbol.map(part))
            volume.reload()
    volume.reload()

    print("[pipeline] building panel...", flush=True)
    panel = load_panel(raw_dir, symbols)
    counts = panel.groupby("symbol").size()
    keep = counts[counts >= cfg["features"]["min_history_days"]].index.tolist()
    if "BTCUSDT" not in keep:
        raise RuntimeError("BTCUSDT missing after load")
    panel = panel[panel["symbol"].isin(keep)].copy()

    window = cfg["data"]["exec_dv_window"]
    pit120 = build_pit_topn(panel, n=cfg["data"]["train_universe_n"], window=window)
    pit20 = build_pit_topn(panel, n=cfg["data"]["exec_universe_n"], window=window)
    pit120.to_parquet(uni_dir / "top120_pit.parquet", index=False)
    pit20.to_parquet(uni_dir / "top20_pit.parquet", index=False)
    luna_top20 = luna_presence_report(pit20)
    luna_top120 = luna_presence_report(pit120)

    ever120 = sorted(set(pit120["symbol"].unique()) | {"BTCUSDT"})
    panel_feat = panel[panel["symbol"].isin(ever120)].copy()

    # Funding for PIT-universe symbols
    fund_todo = [
        {"symbol": s, "start_month": cfg["data"]["start_month"]}
        for s in ever120
        if not (fund_dir / f"{s}.parquet").exists()
    ]
    print(f"[pipeline] downloading funding for {len(fund_todo)} / {len(ever120)} symbols...", flush=True)
    if fund_todo:
        for i in range(0, len(fund_todo), 80):
            part = fund_todo[i : i + 80]
            print(f"[pipeline] funding chunk {i//80+1} n={len(part)}", flush=True)
            list(download_one_funding.map(part))
            volume.reload()
    volume.reload()
    funding = load_funding_panel(fund_dir, ever120)
    fund_cov = funding_coverage_report(funding, ever120)
    print(f"[pipeline] funding coverage: {fund_cov}", flush=True)

    feat_path = feat_dir / "features_labeled.parquet"
    reuse = bool(cfg["portfolio"].get("reuse_predictions", True))
    have_feat = feat_path.exists()
    have_preds = all((pred_dir / f"lgbm_price_only_h{h}.parquet").exists() for h in cfg["labels"]["horizons"])

    if reuse and have_feat and have_preds:
        print("[pipeline] reusing cached features + predictions from Volume", flush=True)
        feat = pd.read_parquet(feat_path)
        used_fixed_trees = False
        best_iter_stats = {"by_horizon": {}, "note": "reused prior run predictions"}
        sensitivity = {}
        all_gate_results = []
        # still run gates on first fold sample carved from OOS
        h0 = cfg["labels"]["primary_horizon"]
        pred0 = pd.read_parquet(pred_dir / f"lgbm_price_only_h{h0}.parquet")
        ycol = f"y_h{h0}"
        if ycol not in pred0.columns:
            pred0 = pred0.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        dates0 = sorted(pd.to_datetime(pred0["date"], utc=True).unique())
        first90 = set(dates0[:90])
        sample = pred0[pd.to_datetime(pred0["date"], utc=True).isin(first90)].copy()
        folds = make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h0,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        gates = run_all_gates(panel_feat, feat, build_pit_topn, folds[0], cfg, sample)
        all_gate_results = gates
        if not all(g.get("passed") for g in gates):
            raise RuntimeError(f"Sanity gates failed: {gates}")
    else:
        print("[pipeline] features (raw)...", flush=True)
        feat_raw = build_feature_panel(panel_feat, clip=cfg["features"]["zscore_clip"], zscore=False)
        pit120_keys = pit120[["date", "symbol"]].copy()
        pit120_keys["date"] = pd.to_datetime(pit120_keys["date"], utc=True)
        feat_raw["date"] = pd.to_datetime(feat_raw["date"], utc=True)
        feat = feat_raw.merge(pit120_keys, on=["date", "symbol"], how="inner")
        feat = apply_cs_zscore(feat, clip=cfg["features"]["zscore_clip"])
        feat = add_labels(feat, panel_feat, horizons=cfg["labels"]["horizons"], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))
        feat.to_parquet(feat_path, index=False)
        volume.commit()

        used_fixed_trees = False
        best_iter_stats = {"by_horizon": {}}
        sensitivity = {}
        all_gate_results = []

        def _run_folds(h, model_cfg, tag):
            folds = make_folds(
                pd.DatetimeIndex(feat["date"].unique()),
                horizon=h,
                min_train_days=cfg["cv"]["min_train_days"],
                val_days=cfg["cv"]["val_days"],
                step_days=cfg["cv"]["step_days"],
            )
            if not folds:
                return pd.DataFrame(), []
            out_h = pred_dir / f"{tag}_h{h}"
            out_h.mkdir(parents=True, exist_ok=True)
            cfg_job = dict(cfg)
            cfg_job["model"] = model_cfg
            diag = {0, max(0, len(folds) // 2), len(folds) - 1}
            payloads = [
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
                    "log_eval_curve": fr.fold_id in diag,
                }
                for fr in folds
            ]
            metas = list(train_one_fold_job.map(payloads))
            volume.reload()
            preds = [pd.read_parquet(m["pred_path"]) for m in metas if m.get("pred_path") and Path(m["pred_path"]).exists()]
            if not preds:
                return pd.DataFrame(), metas
            pred_all = pd.concat(preds, ignore_index=True)
            pred_all = pred_all.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(["date", "symbol"], keep="first")
            return pred_all, metas

        for h in cfg["labels"]["horizons"]:
            model_cfg = dict(cfg["model"])
            pred_all, metas = _run_folds(h, model_cfg, tag="rankic")
            dist = best_iteration_distribution(metas)
            dist["mode"] = "rank_ic"
            best_iter_stats["by_horizon"][f"h={h}_rank_ic_attempt"] = dict(dist)
            if dist["n"] and dist["gt1_frac"] < 0.9:
                used_fixed_trees = True
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
                        tmp = tmp.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
                    ev = evaluate_predictions(tmp, h, universe=pit20, label=f"top20_trees{nt}")
                    sens[str(nt)] = {"mean_ic": ev.get("mean_ic"), "icir": ev.get("icir"), "nw_tstat": ev.get("nw_tstat")}
                    if int(nt) == 500:
                        pred_all, metas = p_s, m_s
                        dist = best_iteration_distribution(m_s)
                        dist["mode"] = "fixed_500"
                sensitivity[f"h={h}"] = sens
            best_iter_stats["by_horizon"][f"h={h}"] = dist
            canon = pred_all[["date", "symbol", "score", "horizon"]].copy()
            canon["model_name"] = "lgbm_price_only"
            ycol = f"y_h{h}"
            canon = canon.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
            canon.to_parquet(pred_dir / f"lgbm_price_only_h{h}.parquet", index=False)
            if h == cfg["labels"]["primary_horizon"]:
                folds = make_folds(
                    pd.DatetimeIndex(feat["date"].unique()),
                    horizon=h,
                    min_train_days=cfg["cv"]["min_train_days"],
                    val_days=cfg["cv"]["val_days"],
                    step_days=cfg["cv"]["step_days"],
                )
                sample = pred_all[pred_all["fold_id"] == pred_all["fold_id"].min()].copy()
                if ycol not in sample.columns:
                    sample = sample.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
                gates = run_all_gates(panel_feat, feat, build_pit_topn, folds[0], cfg, sample)
                all_gate_results = gates
                if not all(g.get("passed") for g in gates):
                    raise RuntimeError(f"Sanity gates failed: {gates}")

    caveats = [
        "Funding accrued from data.binance.vision monthly fundingRate dumps (sum of 8h events per UTC day).",
        "Missing funding series → 0 for that symbol/day (pre-listing / absent dumps).",
        "lag=0: trade at close t on score_t; lag=1: trade at close t+1 on score_t (pessimistic).",
        "Headline metric = median net Sharpe across τ grid (funding on, lag 0); best-τ is reference.",
        "Training universe = PIT top-120; execution = PIT top-20.",
        "Kronos code untouched.",
    ]
    if reuse and have_feat and have_preds:
        caveats.append("Reused cached Volume features/predictions; stress tests recompute portfolio only.")
    if used_fixed_trees:
        caveats.append("h=10 used fixed n_estimators=500 fallback previously.")

    kronos_status = try_export_kronos_ft(pred_dir / "kronos_ft.parquet", horizon=10)

    ic_tables: dict[str, list] = {}
    portfolio_summaries: list = []
    primary_ic = None
    primary_quints = {}
    ic_diag = None
    attribution_blob = None
    lag_compare = []
    chart_lag0 = None
    chart_lag1 = None
    naive_eq = None

    # Full portfolio stress suite
    for h in cfg["labels"]["horizons"]:
        canon_path = pred_dir / f"lgbm_price_only_h{h}.parquet"
        if not canon_path.exists():
            print(f"[pipeline] missing preds for h={h}", flush=True)
            continue
        canon = pd.read_parquet(canon_path)
        ycol = f"y_h{h}"
        if ycol not in canon.columns:
            canon = canon.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")

        naive = naive_mom28_scores(feat, h)
        oos_dates = set(pd.to_datetime(canon["date"], utc=True))
        naive = naive[pd.to_datetime(naive["date"], utc=True).isin(oos_dates)].copy()

        rows = []
        for label, uni in [("pit120", pit120), ("top20", pit20)]:
            ev = evaluate_predictions(canon, h, universe=uni, label=label)
            rows.append({k: v for k, v in ev.items() if k != "ic_series"})
            if h == cfg["labels"]["primary_horizon"] and label == "top20":
                primary_ic = ev["ic_series"]
                primary_quints = ev.get("quintile_means") or {}
                ic_diag = ic_dispersion_diagnostic(canon, h, pit20, n_exclude=10)
        ev_n = evaluate_predictions(naive, h, universe=pit20, label="top20_naive")
        rows.append({k: v for k, v in ev_n.items() if k != "ic_series"})
        ic_tables[f"h={h}"] = rows

        print(f"[pipeline] portfolio stress h={h}...", flush=True)
        best_tranche_fund_on = None
        best_daily_fund_on = None

        for funding_on in (False, True):
            for tp in cfg["portfolio"]["tau_percentiles"]:
                # full τ sweep at lag=0 for both funding modes
                res = run_portfolio_backtest(
                    canon, panel_feat, feat, pit20, horizon=h, tau_pct=tp,
                    exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                    gross_limit=cfg["portfolio"]["gross_limit"],
                    fee_bps=cfg["portfolio"]["taker_fee_bps"],
                    slip_bps=cfg["portfolio"]["slippage_bps"],
                    variant="daily", lag=0, apply_funding=funding_on, funding=funding,
                )
                portfolio_summaries.append(_slim(res))
                if funding_on and "net_sharpe" in res:
                    if best_daily_fund_on is None or res["net_sharpe"] > best_daily_fund_on["net_sharpe"]:
                        best_daily_fund_on = res

                tres = run_tranche_portfolio(
                    canon, panel_feat, feat, pit20, horizon=h, tau_pct=tp,
                    exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                    gross_limit=cfg["portfolio"]["gross_limit"],
                    fee_bps=cfg["portfolio"]["taker_fee_bps"],
                    slip_bps=cfg["portfolio"]["slippage_bps"],
                    lag=0, apply_funding=funding_on, funding=funding,
                )
                portfolio_summaries.append(_slim(tres))
                if funding_on and "net_sharpe" in tres:
                    if best_tranche_fund_on is None or tres["net_sharpe"] > best_tranche_fund_on["net_sharpe"]:
                        best_tranche_fund_on = tres
                print(
                    f"[pipeline] h={h} fund={funding_on} τ={tp} "
                    f"dailySh={res.get('net_sharpe')} trancheSh={tres.get('net_sharpe')} "
                    f"fundPnL={tres.get('funding_total_pnl')}",
                    flush=True,
                )

        # Lag-1 for headline best-τ (funding on) + also median-τ later
        if best_daily_fund_on is not None:
            tp = best_daily_fund_on["tau_pct"]
            r1 = run_portfolio_backtest(
                canon, panel_feat, feat, pit20, horizon=h, tau_pct=tp,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                variant="daily", lag=1, apply_funding=True, funding=funding,
            )
            portfolio_summaries.append(_slim(r1))
            lag_compare.append(
                {
                    "variant": "daily",
                    "horizon": h,
                    "tau_pct": tp,
                    "sharpe_lag0": best_daily_fund_on["net_sharpe"],
                    "sharpe_lag1": r1.get("net_sharpe"),
                    "funding_lag0": best_daily_fund_on.get("funding_total_pnl"),
                }
            )
        if best_tranche_fund_on is not None:
            tp = best_tranche_fund_on["tau_pct"]
            t1 = run_tranche_portfolio(
                canon, panel_feat, feat, pit20, horizon=h, tau_pct=tp,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                lag=1, apply_funding=True, funding=funding,
            )
            portfolio_summaries.append(_slim(t1))
            lag_compare.append(
                {
                    "variant": "tranche",
                    "horizon": h,
                    "tau_pct": tp,
                    "sharpe_lag0": best_tranche_fund_on["net_sharpe"],
                    "sharpe_lag1": t1.get("net_sharpe"),
                    "funding_lag0": best_tranche_fund_on.get("funding_total_pnl"),
                }
            )
            if h == cfg["labels"]["primary_horizon"]:
                chart_lag0 = best_tranche_fund_on["equity"]
                chart_lag1 = t1.get("equity")
                attribution_blob = {
                    "symbols": symbol_attribution(best_tranche_fund_on),
                    "per_year": per_year_breakdown(best_tranche_fund_on),
                    "concentration": day_concentration(best_tranche_fund_on),
                    "meta": {
                        "tau_pct": best_tranche_fund_on["tau_pct"],
                        "net_sharpe": best_tranche_fund_on["net_sharpe"],
                        "gross_sharpe": best_tranche_fund_on.get("gross_sharpe"),
                        "identity_gap": best_tranche_fund_on.get("identity_gap"),
                        "net_gt_gross_sharpe": bool(
                            best_tranche_fund_on["net_sharpe"] > best_tranche_fund_on.get("gross_sharpe", 0)
                        ),
                    },
                }
                # also lag-1 for median τ of tranche h=7 will be filled after median summary

        if h == cfg["labels"]["primary_horizon"]:
            naive_bt = run_portfolio_backtest(
                naive, panel_feat, feat, pit20, horizon=h, tau_pct=70,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                variant="daily_naive", lag=0, apply_funding=True, funding=funding,
            )
            if "equity" in naive_bt:
                naive_eq = naive_bt["equity"]

    median_tau = median_tau_summary(portfolio_summaries)

    # Extra lag-1 runs for median-τ headline variants (funding on)
    for m in median_tau:
        if not m.get("funding_on"):
            continue
        if m.get("lag") not in (0, None):
            continue
        if m.get("variant") not in ("daily", "tranche"):
            continue
        h = int(m["horizon"])
        tp = float(m["median_tau"])
        canon = pd.read_parquet(pred_dir / f"lgbm_price_only_h{h}.parquet")
        if m["variant"] == "daily":
            r1 = run_portfolio_backtest(
                canon, panel_feat, feat, pit20, horizon=h, tau_pct=tp,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                variant="daily", lag=1, apply_funding=True, funding=funding,
            )
        else:
            r1 = run_tranche_portfolio(
                canon, panel_feat, feat, pit20, horizon=h, tau_pct=tp,
                exit_hysteresis=cfg["portfolio"]["exit_hysteresis"],
                gross_limit=cfg["portfolio"]["gross_limit"],
                fee_bps=cfg["portfolio"]["taker_fee_bps"],
                slip_bps=cfg["portfolio"]["slippage_bps"],
                lag=1, apply_funding=True, funding=funding,
            )
        m["median_sharpe_lag1"] = r1.get("net_sharpe")
        portfolio_summaries.append(_slim(r1))

    # BTC overlay
    if chart_lag0 is not None and len(chart_lag0):
        btc = panel_feat[panel_feat["symbol"] == "BTCUSDT"].set_index("date")["close"].sort_index()
        idx = pd.to_datetime(chart_lag0["date"], utc=True)
        btc = btc.reindex(idx).dropna()
        btc_eq = pd.DataFrame({"date": btc.index, "equity": (btc / btc.iloc[0]).values})
    else:
        btc_eq = pd.DataFrame({"date": [], "equity": []})

    if primary_ic is not None:
        plot_ic_analysis(chart_dir / "ic_analysis.png", primary_ic, primary_quints)
    if chart_lag0 is not None and len(chart_lag0) and len(btc_eq):
        plot_equity_curves(chart_dir / "equity_curves.png", chart_lag0, chart_lag1, naive_eq, btc_eq)
    if attribution_blob is not None:
        plot_attribution(
            chart_dir / "attribution.png",
            attribution_blob.get("per_year", []),
            attribution_blob.get("symbols", {}).get("top", []),
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
        funding_coverage=fund_cov,
        median_tau=median_tau,
        lag_compare=lag_compare,
        attribution=attribution_blob,
        ic_diag={k: v for k, v in (ic_diag or {}).items() if k not in ("ic_series", "disp_series")} if ic_diag else None,
    )

    # LUNA/FTT share for stdout
    collapse = (attribution_blob or {}).get("symbols", {}).get("collapse", {}) if attribution_blob else {}

    summary = {
        "elapsed_sec": time.time() - t_pipe,
        "n_symbols_ever_pit120": int(panel_feat["symbol"].nunique()),
        "span": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "gates": all_gate_results,
        "luna": {"top20": luna_top20, "top120": luna_top120},
        "funding_coverage": fund_cov,
        "best_iteration": best_iter_stats,
        "sensitivity": sensitivity,
        "ic_tables": {
            k: [{kk: vv for kk, vv in r.items() if kk != "ic_series"} for r in rows]
            for k, rows in ic_tables.items()
        },
        "ic_diag": {k: v for k, v in (ic_diag or {}).items() if k not in ("ic_series", "disp_series")} if ic_diag else None,
        "portfolio": portfolio_summaries,
        "median_tau": median_tau,
        "lag_compare": lag_compare,
        "attribution": {
            "per_year": (attribution_blob or {}).get("per_year"),
            "concentration": (attribution_blob or {}).get("concentration"),
            "collapse": collapse,
            "meta": (attribution_blob or {}).get("meta"),
            "top": (attribution_blob or {}).get("symbols", {}).get("top"),
            "bottom": (attribution_blob or {}).get("symbols", {}).get("bottom"),
        } if attribution_blob else None,
        "kronos_status": kronos_status,
        "caveats": caveats,
    }
    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()
    print("[pipeline] DONE", flush=True)
    return summary


@app.local_entrypoint()
def main():
    print("Launching Phase A0 stress-test pipeline on Modal...", flush=True)
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
    print(
        f"{'var':>8} {'h':>3} {'medSh0':>7} {'medSh1':>7} {'fundPnL':>8} {'LUNA%':>7} {'FTT%':>7}",
        flush=True,
    )
    collapse = (summary.get("attribution") or {}).get("collapse") or {}
    luna_pct = collapse.get("LUNAUSDT", {}).get("pct_of_total", float("nan"))
    if not isinstance(luna_pct, (int, float)):
        luna_pct = float("nan")
    ftt_pct = collapse.get("FTTUSDT", {}).get("pct_of_total", float("nan"))
    if not isinstance(ftt_pct, (int, float)):
        ftt_pct = float("nan")
    for m in summary.get("median_tau", []):
        if m.get("lag") not in (0, None):
            continue
        if not m.get("funding_on"):
            continue
        print(
            f"{str(m.get('variant')):>8} {m.get('horizon'):>3} "
            f"{m.get('median_net_sharpe', float('nan')):>7.2f} "
            f"{m.get('median_sharpe_lag1', float('nan')):>7.2f} "
            f"{m.get('median_funding_pnl', float('nan')):>8.3f} "
            f"{luna_pct:>6.2f}% {ftt_pct:>6.2f}%",
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
    print(f"\nIC diag: {json.dumps(summary.get('ic_diag'), default=str)}", flush=True)
    print(f"LUNA/FTT collapse: {json.dumps(collapse, default=str)[:800]}", flush=True)
    print(f"\nArtifacts → ./artifacts/  elapsed={summary.get('elapsed_sec', float('nan')):.0f}s", flush=True)
    gates_ok = all(g.get("passed") for g in summary.get("gates", []))
    print(f"Gates: {'ALL PASS' if gates_ok else 'FAILED'}", flush=True)


@app.function(timeout=60 * 10, retries=0, volumes={"/data/quant": volume})
def fetch_artifacts() -> dict:
    root = Path("/data/quant")
    text, binary = {}, {}
    for rel in ["reports/baseline_report.md", "reports/summary.json"]:
        p = root / rel
        if p.exists():
            text[rel] = p.read_text()
    for rel in ["charts/equity_curves.png", "charts/ic_analysis.png", "charts/attribution.png"]:
        p = root / rel
        if p.exists():
            binary[rel] = list(p.read_bytes())
    return {"text": text, "bin": binary}
