"""
Phase D — decay diagnostic + microstructure ablation (Modal, CPU, backtest-only).

NO schedules / cron / shadow / live jobs.

Usage:
    modal run phase_d_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-d-micro"
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
    .add_local_python_source("baseline", "phase_d")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 60 * 3, retries=1, volumes={"/data/quant": volume}, cpu=2, memory=4096)
def download_premium_job(item: dict) -> dict:
    from baseline.data import month_range
    from phase_d.micro_data import download_premium_symbol_months

    t0 = time.time()
    path = download_premium_symbol_months(
        item["symbol"],
        month_range(item.get("start_month", "2020-01")),
        Path(item["dest"]),
    )
    volume.commit()
    return {"symbol": item["symbol"], "path": str(path), "elapsed": time.time() - t0}


@app.function(timeout=60 * 60 * 4, retries=1, volumes={"/data/quant": volume}, cpu=2, memory=4096)
def download_metrics_job(item: dict) -> dict:
    """Download metrics only for requested dates (from feature panel)."""
    import io
    import zipfile

    import httpx
    import pandas as pd

    from baseline.data import VISION_FILE as VF

    symbol = item["symbol"]
    dest = Path(item["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{symbol}.parquet"
    dates = [str(d)[:10] for d in item.get("dates", [])]
    have = set()
    existing = pd.DataFrame()
    if out.exists():
        existing = pd.read_parquet(out)
        if not existing.empty:
            existing["date"] = pd.to_datetime(existing["date"], utc=True)
            have = set(existing["date"].dt.strftime("%Y-%m-%d"))
    todo = [d for d in dates if d not in have]
    rows = []
    t0 = time.time()
    for i, day in enumerate(todo):
        url = f"{VF}/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                df = pd.read_csv(zf.open(name))
            last = df.iloc[-1]
            rows.append(
                {
                    "date": pd.Timestamp(day, tz="UTC"),
                    "symbol": symbol,
                    "sum_open_interest": float(pd.to_numeric(last.get("sum_open_interest"), errors="coerce")),
                    "sum_open_interest_value": float(
                        pd.to_numeric(last.get("sum_open_interest_value"), errors="coerce")
                    ),
                    "count_long_short_ratio": float(
                        pd.to_numeric(last.get("count_long_short_ratio"), errors="coerce")
                    ),
                    "sum_taker_long_short_vol_ratio": float(
                        pd.to_numeric(last.get("sum_taker_long_short_vol_ratio"), errors="coerce")
                    ),
                }
            )
        except Exception:
            continue
        if (i + 1) % 250 == 0:
            print(f"[metrics] {symbol} {i+1}/{len(todo)}", flush=True)
            # periodic commit for long symbols
            if rows:
                tmp = pd.DataFrame(rows)
                all_df = pd.concat([existing, tmp], ignore_index=True) if not existing.empty else tmp
                all_df["date"] = pd.to_datetime(all_df["date"], utc=True)
                all_df = all_df.drop_duplicates(["date", "symbol"], keep="last")
                all_df.to_parquet(out, index=False)
                existing = all_df
                rows = []
                volume.commit()
    if rows or not out.exists():
        if rows:
            tmp = pd.DataFrame(rows)
            all_df = pd.concat([existing, tmp], ignore_index=True) if not existing.empty else tmp
        else:
            all_df = existing if not existing.empty else pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "sum_open_interest",
                    "sum_open_interest_value",
                    "count_long_short_ratio",
                    "sum_taker_long_short_vol_ratio",
                ]
            )
        if not all_df.empty:
            all_df["date"] = pd.to_datetime(all_df["date"], utc=True)
            all_df = all_df.drop_duplicates(["date", "symbol"], keep="last")
        all_df.to_parquet(out, index=False)
    volume.commit()
    n = len(pd.read_parquet(out)) if out.exists() else 0
    return {"symbol": symbol, "n_rows": n, "n_todo": len(todo), "elapsed": time.time() - t0}


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def train_d_fold_job(payload: dict) -> dict:
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
    model_cfg = dict(cfg["model"])
    if int(payload["horizon"]) == 10:
        model_cfg["fixed_n_estimators"] = 500
        model_cfg["early_stop_metric"] = "none"
    t0 = time.time()
    pred_df, meta = _fit_predict_fold(
        df,
        fold,
        seed=cfg["seed"],
        model_cfg=model_cfg,
        inner_holdout_days=cfg["cv"]["inner_holdout_days"],
        feature_cols=payload["feature_cols"],
        model_name="lgbm_a0_plus_micro",
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
        f"[D-fold] h={fold.horizon} id={fold.fold_id} status={meta.get('status')} "
        f"n_pred={len(pred_df)} elapsed={meta['wall_elapsed']:.1f}s "
        f"best_iter={meta.get('best_iteration')}",
        flush=True,
    )
    return meta


@app.function(timeout=60 * 60 * 10, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_phase_d() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.data import (
        build_pit_topn,
        load_funding_panel,
        load_panel,
        month_range,
    )
    from baseline.features import FEATURE_COLS
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything
    from phase_d.ablation import (
        KEEP_CRITERION,
        apply_keep_criterion,
        evaluate_ablation_horizon,
        merge_micro,
    )
    from phase_d.decay import plot_decay, run_decay_diagnostic
    from phase_d.micro_data import (
        MICRO_FEATURE_COLS,
        coverage_report,
        liquidation_availability_note,
        load_symbol_parquets,
    )
    from phase_d.micro_features import build_micro_feature_panel, micro_coverage_on_book
    from phase_d.report import plot_ablation_ic, print_stdout_summary, write_phaseD_report

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError(f"config.yaml drifted from frozen A0")
    print(f"[phaseD] frozen A0 OK sha256={calc}", flush=True)
    print("[phaseD] BACKTEST ONLY — no schedules/cron/shadow", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    prem_dir = root / "raw" / "premium"
    metrics_dir = root / "raw" / "metrics"
    phase_dir = root / "phase_d"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in [prem_dir, metrics_dir, phase_dir, rep_dir, chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)
    pit120["date"] = pd.to_datetime(pit120["date"], utc=True)
    pred_a7 = pd.read_parquet(pred_dir / "lgbm_price_only_h7.parquet")
    pred_a7["date"] = pd.to_datetime(pred_a7["date"], utc=True)
    pred_a10 = pd.read_parquet(pred_dir / "lgbm_price_only_h10.parquet")
    pred_a10["date"] = pd.to_datetime(pred_a10["date"], utc=True)

    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    print(f"[phaseD] loading panel ({len(ever)} symbols)...", flush=True)
    panel = load_panel(raw_dir, ever)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    funding = load_funding_panel(fund_dir, ever)

    # Gates
    folds7 = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=7,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    sample = pred_a7[pred_a7["date"] <= pred_a7["date"].min() + pd.Timedelta(days=90)].copy()
    if "y_h7" not in sample.columns:
        sample = sample.merge(feat[["date", "symbol", "y_h7"]], on=["date", "symbol"], how="left")
    gates = run_all_gates(panel, feat, build_pit_topn, folds7[0], cfg, sample)
    if not all(g.get("passed") for g in gates):
        raise RuntimeError(f"Sanity gates failed: {gates}")
    print("[phaseD] gates OK", flush=True)

    # --- 1. Decay diagnostic ---
    print("[phaseD] decay diagnostic...", flush=True)
    decay = run_decay_diagnostic(
        pred_a7,
        feat,
        panel,
        pit20,
        funding,
        tau_pct=60.0,
        cfg_portfolio=cfg["portfolio"],
    )
    plot_decay(decay, chart_dir / "phaseD_decay.png")
    print(f"[phaseD] DIAGNOSTIC VERDICT={decay['verdict']}", flush=True)

    # --- 2. Microstructure downloads ---
    print("[phaseD] downloading premiumIndexKlines...", flush=True)
    todo_prem = [
        {"symbol": s, "start_month": "2020-01", "dest": str(prem_dir)}
        for s in ever
        if not (prem_dir / f"{s}.parquet").exists()
    ]
    print(f"[phaseD] premium todo={len(todo_prem)}/{len(ever)}", flush=True)
    if todo_prem:
        for i in range(0, len(todo_prem), 80):
            part = todo_prem[i : i + 80]
            print(f"[phaseD] premium chunk {i//80+1} n={len(part)}", flush=True)
            list(download_premium_job.map(part))
            volume.reload()
    volume.reload()

    # metrics: only dates present in feature panel per symbol
    print("[phaseD] downloading metrics (dates ⊆ feature panel)...", flush=True)
    metrics_payloads = []
    for sym, g in feat.groupby("symbol"):
        dates = sorted(pd.to_datetime(g["date"], utc=True).dt.strftime("%Y-%m-%d").unique())
        # metrics start ~2020-09
        dates = [d for d in dates if d >= "2020-09-01"]
        outp = metrics_dir / f"{sym}.parquet"
        if outp.exists():
            try:
                ex = pd.read_parquet(outp, columns=["date"])
                have = set(pd.to_datetime(ex["date"], utc=True).dt.strftime("%Y-%m-%d"))
                if set(dates).issubset(have):
                    continue
            except Exception:
                pass
        metrics_payloads.append({"symbol": sym, "dest": str(metrics_dir), "dates": dates})
    print(f"[phaseD] metrics todo symbols={len(metrics_payloads)}", flush=True)
    if metrics_payloads:
        # waves of 40
        for i in range(0, len(metrics_payloads), 40):
            wave = metrics_payloads[i : i + 40]
            print(f"[phaseD] metrics wave {i//40+1} n={len(wave)}", flush=True)
            list(download_metrics_job.map(wave))
            volume.reload()
    volume.reload()

    liq_note = liquidation_availability_note()
    print(f"[phaseD] liquidations: {liq_note}", flush=True)

    premium = load_symbol_parquets(prem_dir, ever)
    metrics = load_symbol_parquets(metrics_dir, ever)
    # dollar volume from panel
    panel_dv = panel[["date", "symbol"]].copy()
    panel_dv["dollar_volume"] = (
        panel["close"] * panel["volume"]
        if "volume" in panel.columns
        else np.nan
    )
    # better: use close*volume
    p2 = panel.copy()
    p2["dollar_volume"] = p2["close"].astype(float) * p2["volume"].astype(float)
    panel_dv = p2[["date", "symbol", "dollar_volume"]]

    coverage = {
        "funding_rate": coverage_report(funding, ever, "funding_rate"),
        "premium_close": coverage_report(premium, ever, "premium_close"),
        "sum_open_interest": coverage_report(metrics, ever, "sum_open_interest"),
        "count_long_short_ratio": coverage_report(metrics, ever, "count_long_short_ratio"),
        "sum_taker_long_short_vol_ratio": coverage_report(metrics, ever, "sum_taker_long_short_vol_ratio"),
        "liquidationSnapshot_um": liq_note,
    }
    print(f"[phaseD] coverage={json.dumps({k: coverage[k] for k in coverage}, default=str)[:800]}", flush=True)

    # --- 3. Features ---
    micro_path = phase_dir / "micro_features.parquet"
    if micro_path.exists():
        print("[phaseD] reusing cached micro features", flush=True)
        micro = pd.read_parquet(micro_path)
    else:
        print("[phaseD] building microstructure features...", flush=True)
        micro = build_micro_feature_panel(funding, metrics, premium, panel_dv, ever, clip=cfg["features"]["zscore_clip"])
        micro.to_parquet(micro_path, index=False)
        volume.commit()
    micro["date"] = pd.to_datetime(micro["date"], utc=True)

    feat_d = merge_micro(feat, micro)
    feat_d_path = phase_dir / "features_a0_plus_micro.parquet"
    feat_d.to_parquet(feat_d_path, index=False)
    volume.commit()
    feature_cols_d = list(FEATURE_COLS) + list(MICRO_FEATURE_COLS)

    # coverage on top20 book dates
    cov_series = micro_coverage_on_book(micro, pit20)
    cov_series.to_frame("cov").to_parquet(phase_dir / "micro_coverage_top20.parquet")

    # --- 4. Ablation ---
    ablation_blobs = {}
    delta_by_h = {}
    for h, pred_a in [(7, pred_a7), (10, pred_a10)]:
        print(f"[phaseD] training Model D h={h}...", flush=True)
        folds = make_folds(
            pd.DatetimeIndex(feat_d["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        out_h = phase_dir / f"preds_d_h{h}"
        canon = phase_dir / f"lgbm_a0_plus_micro_h{h}.parquet"
        meta_path = out_h / f"fold_meta_h{h}.json"
        reuse_ok = False
        if canon.exists():
            pred_d = pd.read_parquet(canon)
            if not pred_d.empty:
                print(f"[phaseD] reusing Model D preds {canon} n={len(pred_d)}", flush=True)
                if meta_path.exists():
                    metas = json.loads(meta_path.read_text())
                else:
                    metas = []
                    for mp in sorted(out_h.glob(f"meta_h{h}_fold*.json")):
                        metas.append(json.loads(mp.read_text()))
                reuse_ok = True
            else:
                print(f"[phaseD] empty Model D canon — retraining h={h}", flush=True)
                canon.unlink(missing_ok=True)
        if not reuse_ok:
            out_h.mkdir(parents=True, exist_ok=True)
            payloads = [
                {
                    "cfg": cfg,
                    "feat_path": str(feat_d_path),
                    "out_dir": str(out_h),
                    "fold_id": fr.fold_id,
                    "train_start": str(fr.train_start),
                    "train_end": str(fr.train_end),
                    "purge_end": str(fr.purge_end),
                    "embargo_end": str(fr.embargo_end),
                    "val_start": str(fr.val_start),
                    "val_end": str(fr.val_end),
                    "horizon": h,
                    "feature_cols": feature_cols_d,
                }
                for fr in folds
            ]
            metas = list(train_d_fold_job.map(payloads))
            volume.reload()
            preds = [
                pd.read_parquet(m["pred_path"])
                for m in metas
                if m.get("pred_path") and Path(m["pred_path"]).exists()
            ]
            pred_d = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
            if not pred_d.empty:
                pred_d = pred_d.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
                    ["date", "symbol"], keep="first"
                )
                pred_d.to_parquet(canon, index=False)
            meta_path.write_text(json.dumps(metas, indent=2, default=str))
            volume.commit()

        blob = evaluate_ablation_horizon(
            pred_a,
            pred_d,
            feat,
            pit20,
            pit120,
            panel,
            funding,
            h,
            folds,
            metas,
            cfg,
            coverage_by_date=cov_series,
        )
        ablation_blobs[h] = blob
        delta_by_h[h] = blob["delta_daily_ic"]
        print(
            f"[phaseD] h={h} Δtrail18={blob['delta_top20_trail18m']} Δfull={blob['delta_top20_full']} "
            f"frac+={blob['frac_pos_folds_trail18m']}",
            flush=True,
        )

    keep = apply_keep_criterion(
        {
            h: {
                "delta_top20_trail18m": ablation_blobs[h]["delta_top20_trail18m"],
                "delta_top20_full": ablation_blobs[h]["delta_top20_full"],
                "frac_pos_folds_trail18m": ablation_blobs[h]["frac_pos_folds_trail18m"],
            }
            for h in ablation_blobs
        }
    )
    print(f"[phaseD] ABLATION VERDICT={keep['verdict']} :: {KEEP_CRITERION}", flush=True)

    plot_ablation_ic(delta_by_h, chart_dir / "phaseD_ablation.png")

    abl_serial = {}
    for h, blob in ablation_blobs.items():
        abl_serial[h] = {k: v for k, v in blob.items() if k not in ("delta_daily_ic",)}

    write_phaseD_report(
        rep_dir / "phaseD_report.md",
        frozen_hash=calc,
        decay=decay,
        coverage=coverage,
        ablation=abl_serial,
        keep=keep,
    )
    print_stdout_summary(decay, keep, ablation_blobs)

    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "decay": decay,
        "coverage": coverage,
        "keep": keep,
        "ablation": abl_serial,
        "gates": gates,
        "elapsed_sec": time.time() - t_pipe,
        "criterion": KEEP_CRITERION,
    }
    (rep_dir / "phaseD_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # also write decay_diagnostic aliases requested by prior phase naming if useful
    (rep_dir / "decay_diagnostic.json").write_text(json.dumps(decay, indent=2, default=str))
    volume.commit()
    print(f"[phaseD] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return summary


@app.local_entrypoint()
def main():
    print("[local] starting Phase D (CPU, backtest-only)...", flush=True)
    summary = run_phase_d.remote()
    print("[local] syncing artifacts...", flush=True)
    import subprocess

    art = Path("artifacts")
    (art / "reports").mkdir(parents=True, exist_ok=True)
    (art / "charts").mkdir(parents=True, exist_ok=True)
    for remote, local in [
        ("reports/phaseD_report.md", art / "reports" / "phaseD_report.md"),
        ("reports/phaseD_summary.json", art / "reports" / "phaseD_summary.json"),
        ("reports/decay_diagnostic.json", art / "reports" / "decay_diagnostic.json"),
        ("charts/phaseD_decay.png", art / "charts" / "phaseD_decay.png"),
        ("charts/phaseD_ablation.png", art / "charts" / "phaseD_ablation.png"),
    ]:
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote, str(local), "--force"],
            check=False,
        )
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("phaseD*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("phaseD*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
        ddg = art / "reports" / "decay_diagnostic.json"
        if ddg.exists():
            (opt / "reports" / "decay_diagnostic.json").write_bytes(ddg.read_bytes())
    print(
        json.dumps(
            {
                "diagnostic": summary.get("decay", {}).get("verdict"),
                "ablation": summary.get("keep", {}).get("verdict"),
                "gpu_used": summary.get("gpu_used"),
                "scheduled_jobs_created": summary.get("scheduled_jobs_created"),
            },
            indent=2,
        )
    )
    print("[local] Phase D complete.", flush=True)
