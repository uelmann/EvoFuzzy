"""
Phase E — path signatures (CPU) + tiny GRU (single A10G, budget-guarded).

BACKTEST ONLY. No schedules / cron / shadow / live jobs.

Usage:
    modal run phase_e_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-e-traj"
VOLUME_NAME = "quant-baseline"
MAX_GPU_HOURS = 10.0
DEFAULT_MAX_EPOCHS = 30
GRU_SEEDS_OFFSET = (0, 1, 2)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("g++", "python3-dev")
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
    .pip_install("iisignature")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "phase_e")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
)

gpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "pyyaml",
        "scikit-learn",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "phase_e")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


def _utc(ts):
    import pandas as pd

    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _strip_series(blob: dict) -> dict:
    import pandas as pd

    out = {}
    for k, v in blob.items():
        if k == "delta_daily_ic":
            continue
        if isinstance(v, pd.Series):
            continue
        out[k] = v
    return out


@app.function(
    image=cpu_image,
    timeout=60 * 90,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=8,
    memory=32768,
)
def train_e1_fold_job(payload: dict) -> dict:
    import pandas as pd
    from baseline.model import FoldSpec, _fit_predict_fold
    from baseline.seedutil import seed_everything

    cfg = payload["cfg"]
    seed_everything(cfg["seed"] + int(payload["fold_id"]))
    df = pd.read_parquet(payload["feat_path"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    fold = FoldSpec(
        fold_id=int(payload["fold_id"]),
        train_start=_utc(payload["train_start"]),
        train_end=_utc(payload["train_end"]),
        purge_end=_utc(payload["purge_end"]),
        embargo_end=_utc(payload["embargo_end"]),
        val_start=_utc(payload["val_start"]),
        val_end=_utc(payload["val_end"]),
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
        model_name="lgbm_a0_plus_sig",
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
        f"[HB] E1-fold h={fold.horizon} id={fold.fold_id} status={meta.get('status')} "
        f"n_pred={len(pred_df)} elapsed={meta['wall_elapsed']:.1f}s best_iter={meta.get('best_iteration')}",
        flush=True,
    )
    return meta


@app.function(
    image=gpu_image,
    timeout=60 * 60 * 10,
    retries=0,
    volumes={"/data/quant": volume},
    gpu="A10G",
    memory=32768,
)
def train_gru_all_job(payload: dict) -> dict:
    """Single A10G sequential train: calibrate → budget guard → all folds/seeds/horizons."""
    import pandas as pd
    from phase_e.seq_model import (
        calibrate_epoch_seconds,
        gru_param_count,
        project_gpu_hours,
        train_gru_fold,
    )

    cache_dir = Path(payload["cache_dir"])
    out_root = Path(payload["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    volume.reload()
    inner_h = int(payload["inner_holdout_days"])
    max_epochs = int(payload["max_epochs"])
    seeds = list(payload["seeds"])
    n_params = gru_param_count()
    print(f"[HB] GRU n_params={n_params} device-check starting", flush=True)

    from types import SimpleNamespace

    folds_by_h = {}
    for h, flist in payload["folds"].items():
        folds_by_h[int(h)] = [
            SimpleNamespace(
                fold_id=int(fr["fold_id"]),
                train_start=_utc(fr["train_start"]),
                train_end=_utc(fr["train_end"]),
                purge_end=_utc(fr["purge_end"]),
                embargo_end=_utc(fr["embargo_end"]),
                val_start=_utc(fr["val_start"]),
                val_end=_utc(fr["val_end"]),
                horizon=int(h),
            )
            for fr in flist
        ]

    # calibrate on first h=7 fold
    cal_fold = folds_by_h[7][0]
    cal = calibrate_epoch_seconds(cache_dir, cal_fold, horizon=7, seed=int(seeds[0]), inner_holdout_days=inner_h)
    print(f"[HB] GRU calibrate {cal}", flush=True)
    n_folds = max(len(v) for v in folds_by_h.values())
    n_h = len(folds_by_h)
    proj = project_gpu_hours(cal["sec_1_epoch"], n_folds, len(seeds), n_h, max_epochs)
    print(f"[HB] GRU budget project1 gpu_hours={proj['gpu_hours']:.3f} cap={MAX_GPU_HOURS}", flush=True)
    aborted = False
    if proj["gpu_hours"] > MAX_GPU_HOURS:
        max_epochs = max(1, max_epochs // 2)
        proj = project_gpu_hours(cal["sec_1_epoch"], n_folds, len(seeds), n_h, max_epochs)
        print(f"[HB] GRU halved epochs → {max_epochs} project2 gpu_hours={proj['gpu_hours']:.3f}", flush=True)
        if proj["gpu_hours"] > MAX_GPU_HOURS:
            aborted = True
            print("[GRU] HARD ABORT projected GPU-hours still > 10 after halving epochs", flush=True)

    budget = {"calibrate": cal, "projection": proj, "max_epochs": max_epochs, "aborted": aborted, "n_params": n_params}
    (out_root / "budget.json").write_text(json.dumps(budget, indent=2, default=str))
    volume.commit()
    if aborted:
        return {"status": "aborted_budget", "budget": budget, "n_params": n_params}

    all_meta = []
    for h, folds in sorted(folds_by_h.items()):
        for fr in folds:
            for seed in seeds:
                dest_dir = out_root / f"h{h}" / f"seed{seed}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                pp = dest_dir / f"fold{fr.fold_id}.parquet"
                mp = dest_dir / f"fold{fr.fold_id}_meta.json"
                if pp.exists() and pp.stat().st_size > 0:
                    meta = json.loads(mp.read_text()) if mp.exists() else {
                        "fold_id": fr.fold_id,
                        "seed": seed,
                        "horizon": int(h),
                        "status": "reuse",
                    }
                    meta["pred_path"] = str(pp)
                    all_meta.append(meta)
                    print(
                        f"[HB] GRU reuse h={h} fold={fr.fold_id} seed={seed} path={pp}",
                        flush=True,
                    )
                    continue
                pred_df, meta = train_gru_fold(
                    cache_dir,
                    fr,
                    horizon=int(h),
                    seed=int(seed),
                    inner_holdout_days=inner_h,
                    max_epochs=max_epochs,
                    pairwise_rank=False,
                )
                dest_dir = out_root / f"h{h}" / f"seed{seed}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                pp = dest_dir / f"fold{fr.fold_id}.parquet"
                if not pred_df.empty:
                    pred_df.to_parquet(pp, index=False)
                    meta["pred_path"] = str(pp)
                else:
                    meta["pred_path"] = None
                (dest_dir / f"fold{fr.fold_id}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
                all_meta.append(meta)
                volume.commit()
                print(
                    f"[HB] GRU done h={h} fold={fr.fold_id} seed={seed} status={meta.get('status')} "
                    f"n={meta.get('n_pred')} elapsed={meta.get('elapsed'):.1f}s",
                    flush=True,
                )

    # assemble per-seed and mean-seed canonical preds
    summary_paths = {}
    for h in folds_by_h:
        seed_frames = []
        for seed in seeds:
            pieces = []
            d = out_root / f"h{h}" / f"seed{seed}"
            for p in sorted(d.glob("fold*.parquet")):
                pieces.append(pd.read_parquet(p))
            if not pieces:
                continue
            sdf = pd.concat(pieces, ignore_index=True)
            sdf = sdf.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(["date", "symbol"], keep="first")
            sp = out_root / f"lgbm_seq_s_h{h}_seed{seed}.parquet"
            sdf.to_parquet(sp, index=False)
            seed_frames.append(sdf[["date", "symbol", "score"]].rename(columns={"score": f"score_s{seed}"}))
        if not seed_frames:
            continue
        merged = seed_frames[0]
        for extra in seed_frames[1:]:
            merged = merged.merge(extra, on=["date", "symbol"], how="outer")
        scols = [c for c in merged.columns if c.startswith("score_s")]
        merged["score"] = merged[scols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        ycol = f"y_h{h}"
        # attach label from first seed file if present
        first = pd.read_parquet(out_root / f"lgbm_seq_s_h{h}_seed{seeds[0]}.parquet")
        if ycol in first.columns:
            merged = merged.merge(first[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        canon = merged[["date", "symbol", "score"] + ([ycol] if ycol in merged.columns else [])]
        cp = out_root / f"lgbm_seq_s_h{h}.parquet"
        canon.to_parquet(cp, index=False)
        summary_paths[str(h)] = str(cp)
        print(f"[HB] GRU canon h={h} n={len(canon)} path={cp}", flush=True)
    volume.commit()
    return {
        "status": "ok",
        "budget": budget,
        "n_params": n_params,
        "max_epochs": max_epochs,
        "canon": summary_paths,
        "n_jobs": len(all_meta),
    }


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 12,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=16,
    memory=65536,
)
def run_phase_e() -> dict:
    import hashlib

    import pandas as pd

    from baseline.data import build_pit_topn, load_funding_panel, load_panel
    from baseline.evaluate import evaluate_predictions
    from baseline.features import FEATURE_COLS
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything
    from phase_e.evalutil import (
        SIG_CRITERION,
        S_VIABLE_CRITERION,
        aggregate_gain,
        apply_s_blend_criteria,
        apply_sig_keep,
        blend_scores,
        daily_score_spearman,
        evaluate_pair,
        plot_delta_ic,
    )
    from phase_e.report import print_stdout_summary, write_phaseE_report
    from phase_e.seq_model import build_sequence_cache
    from phase_e.signatures import SIG_COLS, build_signature_panel, merge_signatures, signature_feature_cols

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    print(f"[phaseE] frozen A0 OK sha256={calc}", flush=True)
    print("[phaseE] BACKTEST ONLY — no schedules/cron/shadow", flush=True)
    print("[phaseE] RUN ORDER: before queued D.2 universe test; both universes reported.", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    phase_dir = root / "phase_e"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in (phase_dir, rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)
    pit120["date"] = pd.to_datetime(pit120["date"], utc=True)
    pred_a7 = pd.read_parquet(pred_dir / "lgbm_price_only_h7.parquet")
    pred_a10 = pd.read_parquet(pred_dir / "lgbm_price_only_h10.parquet")
    pred_a7["date"] = pd.to_datetime(pred_a7["date"], utc=True)
    pred_a10["date"] = pd.to_datetime(pred_a10["date"], utc=True)

    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    print(f"[phaseE] loading panel ({len(ever)} symbols)...", flush=True)
    panel = load_panel(raw_dir, ever)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    funding = load_funding_panel(fund_dir, ever)

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
    print("[phaseE] gates OK", flush=True)

    # ----- Arm 1 signatures -----
    sig_path = phase_dir / "signature_features.parquet"
    if sig_path.exists() and sig_path.stat().st_size > 0:
        print("[phaseE] reusing cached signatures", flush=True)
        sig = pd.read_parquet(sig_path)
    else:
        print("[phaseE] building path signatures (iisignature)...", flush=True)
        sig = build_signature_panel(panel, feat, clip=cfg["features"]["zscore_clip"])
        sig.to_parquet(sig_path, index=False)
        volume.commit()
    sig["date"] = pd.to_datetime(sig["date"], utc=True)
    feat_e1 = merge_signatures(feat, sig)
    feat_e1_path = phase_dir / "features_a0_plus_sig.parquet"
    feat_e1.to_parquet(feat_e1_path, index=False)
    volume.commit()
    feature_cols_e1 = signature_feature_cols()
    cov = {c: float(feat_e1[c].notna().mean()) for c in SIG_COLS if c in feat_e1.columns}
    print(f"[phaseE] signature coverage mean={sum(cov.values())/max(len(cov),1):.3f} n_cols={len(cov)}", flush=True)

    sig_ablation = {}
    sig_delta_plot = {}
    sig_imp = {}
    metas_e1 = {}
    for h, pred_a in [(7, pred_a7), (10, pred_a10)]:
        print(f"[phaseE] training Model E1 h={h}...", flush=True)
        folds = make_folds(
            pd.DatetimeIndex(feat_e1["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        out_h = phase_dir / f"preds_e1_h{h}"
        canon = phase_dir / f"lgbm_a0_plus_sig_h{h}.parquet"
        meta_path = out_h / f"fold_meta_h{h}.json"
        reuse_ok = False
        if canon.exists():
            pred_e1 = pd.read_parquet(canon)
            if not pred_e1.empty:
                print(f"[phaseE] reusing E1 preds {canon} n={len(pred_e1)}", flush=True)
                metas = json.loads(meta_path.read_text()) if meta_path.exists() else []
                reuse_ok = True
            else:
                canon.unlink(missing_ok=True)
        if not reuse_ok:
            out_h.mkdir(parents=True, exist_ok=True)
            payloads = [
                {
                    "cfg": cfg,
                    "feat_path": str(feat_e1_path),
                    "out_dir": str(out_h),
                    "fold_id": fr.fold_id,
                    "train_start": str(fr.train_start),
                    "train_end": str(fr.train_end),
                    "purge_end": str(fr.purge_end),
                    "embargo_end": str(fr.embargo_end),
                    "val_start": str(fr.val_start),
                    "val_end": str(fr.val_end),
                    "horizon": h,
                    "feature_cols": feature_cols_e1,
                }
                for fr in folds
            ]
            metas = list(train_e1_fold_job.map(payloads))
            volume.reload()
            preds = [
                pd.read_parquet(m["pred_path"])
                for m in metas
                if m.get("pred_path") and Path(m["pred_path"]).exists()
            ]
            pred_e1 = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
            if not pred_e1.empty:
                pred_e1 = pred_e1.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
                    ["date", "symbol"], keep="first"
                )
                pred_e1.to_parquet(canon, index=False)
            meta_path.write_text(json.dumps(metas, indent=2, default=str))
            volume.commit()
        metas_e1[h] = metas
        blob = evaluate_pair(
            pred_a, pred_e1, feat, pit20, pit120, panel, funding, h, folds, cfg, b_label="E1"
        )
        sig_ablation[h] = blob
        sig_imp[h] = aggregate_gain(metas, SIG_COLS)
        for uni, ser in (blob.get("delta_daily_ic") or {}).items():
            sig_delta_plot[f"{uni} h={h}"] = ser
        u20 = (blob.get("by_universe") or {}).get("top20") or {}
        u120 = (blob.get("by_universe") or {}).get("pit120") or {}
        print(
            f"[phaseE] E1 h={h} top20 Δ18={u20.get('delta_trail18m')} Δfull={u20.get('delta_full')} "
            f"pit120 Δ18={u120.get('delta_trail18m')} Δfull={u120.get('delta_full')}",
            flush=True,
        )

    sig_keep = apply_sig_keep(sig_ablation)
    plot_delta_ic(sig_delta_plot, chart_dir / "phaseE_sig_ic.png", "SIG E1 − A0")
    print(f"[phaseE] SIG verdicts: { {k: v.get('verdict') for k,v in sig_keep['universes'].items()} }", flush=True)

    # ----- Arm 2 sequences + GRU -----
    seq_dir = phase_dir / "seq"
    seq_meta_path = seq_dir / "meta.json"
    if seq_meta_path.exists() and (seq_dir / "X.npy").exists() and (seq_dir / "index.parquet").exists():
        print("[phaseE] reusing sequence cache", flush=True)
        seq_meta = json.loads(seq_meta_path.read_text())
    else:
        print("[phaseE] building 60×33 sequence cache...", flush=True)
        seq_meta = build_sequence_cache(feat, seq_dir)
        volume.commit()

    gru_root = phase_dir / "gru"
    gru_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(cfg["seed"]) + int(o) for o in GRU_SEEDS_OFFSET]
    folds_payload = {}
    for h in (7, 10):
        folds = make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        folds_payload[h] = [
            {
                "fold_id": fr.fold_id,
                "train_start": str(fr.train_start),
                "train_end": str(fr.train_end),
                "purge_end": str(fr.purge_end),
                "embargo_end": str(fr.embargo_end),
                "val_start": str(fr.val_start),
                "val_end": str(fr.val_end),
            }
            for fr in folds
        ]

    canon7 = gru_root / "lgbm_seq_s_h7.parquet"
    canon10 = gru_root / "lgbm_seq_s_h10.parquet"
    if canon7.exists() and canon10.exists() and canon7.stat().st_size > 0 and canon10.stat().st_size > 0:
        print("[phaseE] reusing GRU canonical preds", flush=True)
        gru_result = json.loads((gru_root / "budget.json").read_text()) if (gru_root / "budget.json").exists() else {}
        gru_result = {
            "status": "reuse",
            "budget": gru_result,
            "n_params": gru_result.get("n_params"),
            "max_epochs": gru_result.get("max_epochs", DEFAULT_MAX_EPOCHS),
        }
    else:
        print("[phaseE] launching single-A10G GRU train (budget-guarded)...", flush=True)
        gru_result = train_gru_all_job.remote(
            {
                "cache_dir": str(seq_dir),
                "out_root": str(gru_root),
                "inner_holdout_days": cfg["cv"]["inner_holdout_days"],
                "max_epochs": DEFAULT_MAX_EPOCHS,
                "seeds": seeds,
                "folds": folds_payload,
            }
        )
        volume.reload()

    print(f"[phaseE] GRU result status={gru_result.get('status')} n_params={gru_result.get('n_params')}", flush=True)

    seq_s = {}
    seq_blend = {}
    seq_delta_plot = {}
    score_corr = {}
    seed_rows = []
    pred_s_by_h = {}
    pred_b_by_h = {}
    for h, pred_a in [(7, pred_a7), (10, pred_a10)]:
        folds = make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        sp = gru_root / f"lgbm_seq_s_h{h}.parquet"
        if not sp.exists():
            seq_s[h] = {"horizon": h, "tables": [], "error": "missing_S"}
            seq_blend[h] = {"horizon": h, "tables": [], "error": "missing_S"}
            continue
        pred_s = pd.read_parquet(sp)
        pred_s["date"] = pd.to_datetime(pred_s["date"], utc=True)
        pred_s_by_h[h] = pred_s
        blob_s = evaluate_pair(pred_a, pred_s, feat, pit20, pit120, panel, funding, h, folds, cfg, b_label="S", compute_sharpe=False)
        seq_s[h] = blob_s
        pred_b = blend_scores(pred_a, pred_s)
        pred_b_by_h[h] = pred_b
        blob_b = evaluate_pair(pred_a, pred_b, feat, pit20, pit120, panel, funding, h, folds, cfg, b_label="BLEND")
        seq_blend[h] = blob_b
        score_corr[h] = daily_score_spearman(pred_a, pred_s)
        for uni, ser in (blob_b.get("delta_daily_ic") or {}).items():
            seq_delta_plot[f"{uni} h={h} BLEND"] = ser
        for uni, ser in (blob_s.get("delta_daily_ic") or {}).items():
            seq_delta_plot[f"{uni} h={h} S"] = ser
        # per-seed
        for seed in seeds:
            ssp = gru_root / f"lgbm_seq_s_h{h}_seed{seed}.parquet"
            if not ssp.exists():
                continue
            ps = pd.read_parquet(ssp)
            ps["date"] = pd.to_datetime(ps["date"], utc=True)
            ycol = f"y_h{h}"
            if ycol not in ps.columns:
                ps = ps.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
            end = ps["date"].max()
            start = end - pd.Timedelta(days=int(365 * 1.5))
            for uni_name, uni in [("top20", pit20), ("pit120", pit120)]:
                for window, mask_df in [
                    ("full", ps),
                    ("trail18m", ps[(ps["date"] >= start) & (ps["date"] <= end)]),
                ]:
                    ev = evaluate_predictions(mask_df, h, universe=uni, label=uni_name)
                    seed_rows.append(
                        {
                            "horizon": h,
                            "seed": seed,
                            "universe": uni_name,
                            "window": window,
                            "mean_ic": ev.get("mean_ic"),
                            "n_days": ev.get("n_days"),
                        }
                    )

    seq_criteria = apply_s_blend_criteria(seq_s, seq_blend)
    plot_delta_ic(seq_delta_plot, chart_dir / "phaseE_seq_ic.png", "S/BLEND − A0")

    # seed spread
    spread = {}
    if seed_rows:
        sdf = pd.DataFrame(seed_rows)
        for (h, uni, window), g in sdf.groupby(["horizon", "universe", "window"]):
            vals = g["mean_ic"].astype(float)
            spread[f"h{h}_{uni}_{window}"] = {
                "min": float(vals.min()) if len(vals) else float("nan"),
                "max": float(vals.max()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "mean": float(vals.mean()) if len(vals) else float("nan"),
            }
    seed_var = {"rows": seed_rows, "spread": spread}

    gru_info = {
        "n_params": gru_result.get("n_params") or (gru_result.get("budget") or {}).get("n_params"),
        "max_epochs": gru_result.get("max_epochs") or (gru_result.get("budget") or {}).get("max_epochs"),
        "device": "cuda:A10G",
        "seeds": seeds,
        "status": gru_result.get("status"),
    }
    budget = gru_result.get("budget") or {}

    write_phaseE_report(
        rep_dir / "phaseE_report.md",
        frozen_hash=calc,
        sig_ablation={h: _strip_series(b) for h, b in sig_ablation.items()},
        sig_keep=sig_keep,
        sig_imp=sig_imp,
        seq_s={h: _strip_series(b) for h, b in seq_s.items()},
        seq_blend={h: _strip_series(b) for h, b in seq_blend.items()},
        seq_criteria=seq_criteria,
        score_corr=score_corr,
        seed_var=seed_var,
        gru_info=gru_info,
        budget=budget,
    )
    print_stdout_summary(sig_keep, seq_criteria, sig_ablation, seq_s, seq_blend, gru_info, seed_var)

    summary = {
        "frozen_sha256": calc,
        "gpu_used": gru_result.get("status") not in (None,),
        "gpu_type": "A10G",
        "scheduled_jobs_created": False,
        "run_order_note": "Phase E before queued D.2 universe test; both universes reported.",
        "sig_keep": sig_keep,
        "seq_criteria": seq_criteria,
        "sig_ablation": {str(h): _strip_series(b) for h, b in sig_ablation.items()},
        "seq_s": {str(h): _strip_series(b) for h, b in seq_s.items()},
        "seq_blend": {str(h): _strip_series(b) for h, b in seq_blend.items()},
        "sig_imp": {str(h): v for h, v in sig_imp.items()},
        "score_corr": {str(h): v for h, v in score_corr.items()},
        "seed_var": seed_var,
        "gru_info": gru_info,
        "budget": budget,
        "seq_meta": seq_meta,
        "signature_coverage": cov,
        "gates": gates,
        "criterion_sig": SIG_CRITERION,
        "criterion_seq": S_VIABLE_CRITERION,
        "elapsed_sec": time.time() - t_pipe,
    }
    (rep_dir / "phaseE_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()
    print(f"[phaseE] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return summary


@app.local_entrypoint()
def main():
    print("[local] starting Phase E (CPU signatures + single A10G GRU, backtest-only)...", flush=True)
    summary = run_phase_e.remote()
    print("[local] syncing artifacts to reports/ and charts/ ...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    (art / "reports").mkdir(parents=True, exist_ok=True)
    (art / "charts").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name in [
        ("reports/phaseE_report.md", "phaseE_report.md"),
        ("reports/phaseE_summary.json", "phaseE_summary.json"),
        ("charts/phaseE_sig_ic.png", "phaseE_sig_ic.png"),
        ("charts/phaseE_seq_ic.png", "phaseE_seq_ic.png"),
    ]:
        kind = "reports" if remote.startswith("reports") else "charts"
        dest_art = art / kind / name
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote, str(dest_art), "--force"],
            check=False,
        )
        if dest_art.exists():
            shutil.copy2(dest_art, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("phaseE*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("phaseE*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(
        json.dumps(
            {
                "sig_keep": (summary.get("sig_keep") or {}).get("universes"),
                "seq_criteria": {
                    u: {
                        "S": v.get("S_viable"),
                        "BLEND": v.get("BLEND_verdict"),
                    }
                    for u, v in ((summary.get("seq_criteria") or {}).get("universes") or {}).items()
                },
                "gru_n_params": (summary.get("gru_info") or {}).get("n_params"),
                "gpu_used": summary.get("gpu_used"),
                "scheduled_jobs_created": False,
            },
            indent=2,
            default=str,
        )
    )
    print("[local] Phase E complete.", flush=True)
