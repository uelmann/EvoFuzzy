"""
Phase E.1b — empirical-null GRU label-shuffle gate; resume E.1 §2–§4 iff GREEN.

Usage:
    modal run phase_e1b_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-e1b-null"
VOLUME_NAME = "quant-baseline"
MAX_GPU_HOURS = 6.0
DEFAULT_MAX_EPOCHS = 30
E1_GPU_CAP = 15.0

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

cpu_image = (
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
    .add_local_python_source("baseline", "phase_e", "phase_e1")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/phaseE1_addendum.md", remote_path="/root/phaseE1_addendum.md")
)

gpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    .pip_install("numpy", "pandas==2.2.2", "pyarrow", "scipy", "pyyaml", "scikit-learn")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "phase_e", "phase_e1")
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


def _folds_ns(payload_folds: dict):
    from types import SimpleNamespace

    out = {}
    for h, flist in payload_folds.items():
        out[int(h)] = [
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
    return out


@app.function(
    image=gpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": volume},
    gpu="A10G",
    memory=32768,
)
def gru_null_jobs(payload: dict) -> dict:
    """10 shuffle-seed replicates × folds × horizons; GRU train seed 42."""
    from phase_e.seq_model import project_gpu_hours, train_gru_fold
    from phase_e1.nullgate import GRU_TRAIN_SEED

    cache_dir = Path(payload["cache_dir"])
    out_root = Path(payload["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    volume.reload()
    inner_h = int(payload["inner_holdout_days"])
    max_epochs = int(payload.get("max_epochs", DEFAULT_MAX_EPOCHS))
    horizons = [int(h) for h in payload["horizons"]]
    folds_by_h = _folds_ns(payload["folds"])
    fold_ids = [int(x) for x in payload["fold_ids"]]
    shuffle_seeds = [int(x) for x in payload["shuffle_seeds"]]
    cap = float(payload.get("gpu_cap", MAX_GPU_HOURS))
    sec = float(payload.get("sec_per_epoch", 5.4))

    n_folds = len(fold_ids)
    n_reps = len(shuffle_seeds)
    proj = project_gpu_hours(sec, n_folds, n_reps, len(horizons), max_epochs)
    dropped_folds = False
    print(f"[HB] E1b GPU project gpu_hours={proj['gpu_hours']:.3f} cap={cap} folds={fold_ids}", flush=True)
    if proj["gpu_hours"] > cap:
        fold_ids = [9, 17]
        n_folds = 2
        dropped_folds = True
        proj = project_gpu_hours(sec, n_folds, n_reps, len(horizons), max_epochs)
        print(f"[HB] drop to folds {fold_ids}; project2 gpu_hours={proj['gpu_hours']:.3f}", flush=True)
    if proj["gpu_hours"] > cap:
        print("[E1b] HARD ABORT GPU-hours still over cap", flush=True)
        return {"status": "aborted_budget", "projection": proj, "dropped_folds": dropped_folds, "fold_ids": fold_ids}

    metas = []
    for h in horizons:
        frs = [f for f in folds_by_h[h] if int(f.fold_id) in set(fold_ids)]
        for fr in frs:
            for shuf_seed in shuffle_seeds:
                dest = out_root / f"h{h}" / f"fold{fr.fold_id}"
                dest.mkdir(parents=True, exist_ok=True)
                pp = dest / f"rep{shuf_seed}.parquet"
                mp = dest / f"rep{shuf_seed}_meta.json"
                if pp.exists() and pp.stat().st_size > 0:
                    meta = json.loads(mp.read_text()) if mp.exists() else {"status": "reuse"}
                    meta["pred_path"] = str(pp)
                    metas.append(meta)
                    print(f"[HB] reuse h={h} fold={fr.fold_id} shuf={shuf_seed}", flush=True)
                    continue
                pred_df, meta = train_gru_fold(
                    cache_dir,
                    fr,
                    horizon=int(h),
                    seed=int(GRU_TRAIN_SEED),
                    inner_holdout_days=inner_h,
                    max_epochs=max_epochs,
                    shuffle_labels=True,
                    shuffle_seed=int(shuf_seed),
                )
                if not pred_df.empty:
                    pred_df.to_parquet(pp, index=False)
                    meta["pred_path"] = str(pp)
                else:
                    meta["pred_path"] = None
                mp.write_text(json.dumps(meta, indent=2, default=str))
                metas.append(meta)
                volume.commit()
                print(
                    f"[HB] done h={h} fold={fr.fold_id} shuf={shuf_seed} "
                    f"status={meta.get('status')} n={meta.get('n_pred')}",
                    flush=True,
                )
    volume.commit()
    return {
        "status": "ok",
        "projection": proj,
        "dropped_folds": dropped_folds,
        "fold_ids": fold_ids,
        "horizons": horizons,
        "n_metas": len(metas),
    }


@app.function(
    image=gpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": volume},
    gpu="A10G",
    memory=32768,
)
def gru_extra_seed_jobs(payload: dict) -> dict:
    """Phase E.1 extra seeds 45–50; resume-from-parquet; cap 15 GPU-h."""
    from phase_e.seq_model import project_gpu_hours, train_gru_fold

    cache_dir = Path(payload["cache_dir"])
    out_root = Path(payload["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    volume.reload()
    inner_h = int(payload["inner_holdout_days"])
    max_epochs = int(payload.get("max_epochs", DEFAULT_MAX_EPOCHS))
    seeds = list(payload["seeds"])
    horizons = [int(h) for h in payload["horizons"]]
    folds_by_h = _folds_ns(payload["folds"])
    cap = float(payload.get("gpu_cap", E1_GPU_CAP))
    sec = float(payload.get("sec_per_epoch", 5.4))
    n_folds = max(len(v) for v in folds_by_h.values())
    proj = project_gpu_hours(sec, n_folds, len(seeds), len(horizons), max_epochs)
    print(f"[HB] extra-seed project gpu_hours={proj['gpu_hours']:.3f} cap={cap}", flush=True)
    dropped_h7 = False
    if proj["gpu_hours"] > cap and 7 in horizons and 10 in horizons:
        horizons = [10]
        dropped_h7 = True
        proj = project_gpu_hours(sec, n_folds, len(seeds), 1, max_epochs)
        print(f"[HB] drop h=7; project2 gpu_hours={proj['gpu_hours']:.3f}", flush=True)
    if proj["gpu_hours"] > cap:
        return {"status": "aborted_budget", "projection": proj, "dropped_h7": dropped_h7}
    metas = []
    for h in horizons:
        for fr in folds_by_h[h]:
            for seed in seeds:
                dest = out_root / f"h{h}" / f"seed{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                pp = dest / f"fold{fr.fold_id}.parquet"
                mp = dest / f"fold{fr.fold_id}_meta.json"
                if pp.exists() and pp.stat().st_size > 0:
                    meta = json.loads(mp.read_text()) if mp.exists() else {"status": "reuse"}
                    meta["pred_path"] = str(pp)
                    metas.append(meta)
                    print(f"[HB] reuse extra h={h} fold={fr.fold_id} seed={seed}", flush=True)
                    continue
                pred_df, meta = train_gru_fold(
                    cache_dir, fr, horizon=int(h), seed=int(seed),
                    inner_holdout_days=inner_h, max_epochs=max_epochs, shuffle_labels=False,
                )
                if not pred_df.empty:
                    pred_df.to_parquet(pp, index=False)
                    meta["pred_path"] = str(pp)
                mp.write_text(json.dumps(meta, indent=2, default=str))
                metas.append(meta)
                volume.commit()
                print(f"[HB] extra done h={h} fold={fr.fold_id} seed={seed} n={meta.get('n_pred')}", flush=True)
    volume.commit()
    return {"status": "ok", "projection": proj, "dropped_h7": dropped_h7, "horizons": horizons, "n_metas": len(metas)}


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=16,
    memory=65536,
)
def run_phase_e1b() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.data import build_pit_topn, load_funding_panel, load_panel
    from baseline.gates import run_all_gates
    from baseline.model import _fit_predict_fold, make_folds
    from baseline.seedutil import seed_everything
    from phase_e.seq_model import project_gpu_hours
    from phase_e1.e1b_report import write_e1b_report
    from phase_e1.nullgate import (
        E1B_GATE,
        FOLDS_FULL,
        GRU_TRAIN_SEED,
        PRIMARY_UNI,
        SHUFFLE_SEEDS,
        assemble_fold_ensemble,
        bias_skill_verdict,
        cell_stats,
        fold_mean_ic,
        plot_null,
    )
    from phase_e1.resume import OLD_SEEDS

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    print(f"[phaseE1b] frozen A0 OK sha256={calc}", flush=True)
    print("[phaseE1b] BACKTEST ONLY — verification, no live/schedule", flush=True)
    print(f"[phaseE1b] GATE (verbatim, before null): {E1B_GATE}", flush=True)
    addendum = Path("/root/phaseE1_addendum.md").read_text()
    print(f"[phaseE1b] addendum_chars={len(addendum)}", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat = pd.read_parquet(root / "features" / "features_labeled.parquet")
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    pit20 = pd.read_parquet(root / "universe" / "top20_pit.parquet")
    pit120 = pd.read_parquet(root / "universe" / "top120_pit.parquet")
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)
    pit120["date"] = pd.to_datetime(pit120["date"], utc=True)
    pred_a = {
        7: pd.read_parquet(root / "predictions" / "lgbm_price_only_h7.parquet"),
        10: pd.read_parquet(root / "predictions" / "lgbm_price_only_h10.parquet"),
    }
    for h in pred_a:
        pred_a[h]["date"] = pd.to_datetime(pred_a[h]["date"], utc=True)

    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    panel = load_panel(root / "raw" / "klines", ever)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    funding = load_funding_panel(root / "raw" / "funding", ever)

    folds = {}
    folds_payload = {}
    for h in (7, 10):
        folds[h] = make_folds(
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
            for fr in folds[h]
        ]

    sample = pred_a[7][pred_a[7]["date"] <= pred_a[7]["date"].min() + pd.Timedelta(days=90)].copy()
    if "y_h7" not in sample.columns:
        sample = sample.merge(feat[["date", "symbol", "y_h7"]], on=["date", "symbol"], how="left")
    a0_gates = run_all_gates(panel, feat, build_pit_topn, folds[7][0], cfg, sample)
    if not all(g.get("passed") for g in a0_gates):
        raise RuntimeError(f"A0 sanity gates failed: {a0_gates}")
    print("[phaseE1b] A0 sanity gates OK", flush=True)

    seq_dir = root / "phase_e" / "seq"
    gru_root = root / "phase_e" / "gru"
    e1b_dir = root / "phase_e1b"
    e1b_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    rep_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    budg_path = gru_root / "budget.json"
    sec = 5.4
    if budg_path.exists():
        try:
            sec = float(json.loads(budg_path.read_text())["calibrate"]["sec_1_epoch"])
        except Exception:
            pass
    proj0 = project_gpu_hours(sec, len(FOLDS_FULL), len(SHUFFLE_SEEDS), 2, DEFAULT_MAX_EPOCHS)
    print(f"[phaseE1b] budget guard project={proj0} cap={MAX_GPU_HOURS}", flush=True)

    gpu_handle = gru_null_jobs.spawn(
        {
            "cache_dir": str(seq_dir),
            "out_root": str(e1b_dir / "gru_null"),
            "inner_holdout_days": cfg["cv"]["inner_holdout_days"],
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "horizons": [7, 10],
            "folds": folds_payload,
            "fold_ids": FOLDS_FULL,
            "shuffle_seeds": SHUFFLE_SEEDS,
            "gpu_cap": MAX_GPU_HOURS,
            "sec_per_epoch": sec,
        }
    )

    # A0 empirical null on CPU while GPU runs
    a0_root = e1b_dir / "a0_null"
    a0_root.mkdir(parents=True, exist_ok=True)
    fold_map = {h: {int(fr.fold_id): fr for fr in folds[h]} for h in (7, 10)}
    a0_real = {}
    for h in (7, 10):
        pa = pred_a[h].copy()
        ycol = f"y_h{h}"
        if ycol not in pa.columns:
            pa = pa.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        if "score" not in pa.columns:
            pa["score"] = pa["y_pred"]
        for fid in FOLDS_FULL:
            fr = fold_map[h][fid]
            sub = pa[(pa["date"] >= fr.val_start) & (pa["date"] <= fr.val_end)]
            a0_real[(h, fid, "pit120")] = fold_mean_ic(sub, h, pit120, "pit120")
            a0_real[(h, fid, "top20")] = fold_mean_ic(sub, h, pit20, "top20")

    for h in (7, 10):
        for fid in FOLDS_FULL:
            fr = fold_map[h][fid]
            for shuf_seed in SHUFFLE_SEEDS:
                dest = a0_root / f"h{h}" / f"fold{fid}"
                dest.mkdir(parents=True, exist_ok=True)
                pp = dest / f"rep{shuf_seed}.parquet"
                if pp.exists() and pp.stat().st_size > 0:
                    print(f"[HB] A0 reuse h={h} fold={fid} shuf={shuf_seed}", flush=True)
                    continue
                print(f"[HB] A0 shuffle train h={h} fold={fid} shuf={shuf_seed}", flush=True)
                pred_df, meta = _fit_predict_fold(
                    feat,
                    fr,
                    seed=cfg["seed"],
                    model_cfg=cfg["model"],
                    inner_holdout_days=cfg["cv"]["inner_holdout_days"],
                    shuffle_labels=True,
                    shuffle_seed=int(shuf_seed),
                )
                if not pred_df.empty:
                    pred_df.to_parquet(pp, index=False)
                (dest / f"rep{shuf_seed}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
                volume.commit()

    gpu = gpu_handle.get()
    volume.reload()
    print(f"[phaseE1b] GPU null status={gpu.get('status')} folds={gpu.get('fold_ids')} dropped={gpu.get('dropped_folds')}", flush=True)
    if gpu.get("status") == "aborted_budget":
        raise RuntimeError(f"GPU budget abort: {gpu}")

    folds_used = [int(x) for x in (gpu.get("fold_ids") or FOLDS_FULL)]
    uni_map = {"pit120": pit120, "top20": pit20}

    def _score_null(root: Path, real_lookup, model: str) -> list[dict]:
        rows = []
        for h in (7, 10):
            for fid in folds_used:
                for uni_name, uni in uni_map.items():
                    ics = []
                    for shuf_seed in SHUFFLE_SEEDS:
                        pp = root / f"h{h}" / f"fold{fid}" / f"rep{shuf_seed}.parquet"
                        if not pp.exists():
                            ics.append(float("nan"))
                            continue
                        pdf = pd.read_parquet(pp)
                        pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
                        ycol = f"y_h{h}"
                        if ycol not in pdf.columns:
                            pdf = pdf.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
                        if "score" not in pdf.columns:
                            pdf["score"] = pdf["y_pred"]
                        ics.append(fold_mean_ic(pdf, h, uni, uni_name)["mean_ic"])
                    st = cell_stats(ics)
                    real = real_lookup.get((h, fid, uni_name), {})
                    rows.append(
                        {
                            "model": model,
                            "horizon": h,
                            "fold_id": fid,
                            "universe": uni_name,
                            "real_ic": real.get("mean_ic"),
                            **st,
                        }
                    )
        return rows

    # real 3-seed GRU ensemble IC from Phase E artifacts
    gru_real = {}
    for h in (7, 10):
        for fid in folds_used:
            ens = assemble_fold_ensemble(gru_root, h, fid, OLD_SEEDS)
            if ens.empty:
                continue
            ycol = f"y_h{h}"
            ens = ens.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
            for uni_name, uni in uni_map.items():
                gru_real[(h, fid, uni_name)] = fold_mean_ic(ens, h, uni, uni_name)

    gru_cells = _score_null(e1b_dir / "gru_null", gru_real, "GRU")
    a0_cells = _score_null(a0_root, a0_real, "A0")
    decision = bias_skill_verdict(gru_cells, n_folds_planned=len(folds_used) if len(folds_used) >= 4 else 2)
    e1b_verdict = decision["verdict"]
    print(f"[phaseE1b] BIAS pass={decision['bias_pass']} n_violate={decision['n_violate']}/{decision['n_cells']}", flush=True)
    print(f"[phaseE1b] SKILL {decision['skill_by_h']} pass={decision['skill_pass']}", flush=True)
    print(f"[phaseE1b] VERDICT={e1b_verdict}", flush=True)

    plot_null(gru_cells, chart_dir / "phaseE1b_null.png")

    resume_blob = None
    extra_gpu = {}
    if e1b_verdict == "GREEN":
        print("[phaseE1b] GREEN — extra seeds deferred to local entrypoint / gru_extra_seed_jobs", flush=True)
    elif e1b_verdict == "CONTAMINATED":
        print("[phaseE1b] CONTAMINATED — no further GRU work", flush=True)
    else:
        print("[phaseE1b] PARKED-NO-SKILL — stop, no adoption, no retuning", flush=True)

    write_e1b_report(
        rep_dir / "phaseE1b_report.md",
        frozen_hash=calc,
        budget=gpu.get("projection"),
        folds_used=folds_used,
        horizons=[7, 10],
        shuffle_seeds=SHUFFLE_SEEDS,
        dropped_folds=gpu.get("dropped_folds"),
        e1b_verdict=e1b_verdict,
        e1b_details=decision,
        null_table=gru_cells,
        a0_table=a0_cells,
        skill_by_h=decision.get("skill_by_h"),
        resume=resume_blob,
        gates=[{"name": "e1b_empirical_null", "passed": e1b_verdict == "GREEN"}],
    )
    summary = {
        "frozen_sha256": calc,
        "e1b_verdict": e1b_verdict,
        "decision": decision,
        "budget": gpu.get("projection"),
        "folds_used": folds_used,
        "dropped_folds": gpu.get("dropped_folds"),
        "gru_cells": gru_cells,
        "a0_cells": a0_cells,
        "resume": {k: resume_blob[k] for k in ("verdict", "details", "keep_lines", "stdout") if resume_blob and k in resume_blob},
        "folds_payload": folds_payload,
        "inner_holdout_days": cfg["cv"]["inner_holdout_days"],
        "sec_per_epoch": sec,
        "criterion": E1B_GATE,
        "scheduled_jobs_created": False,
        "elapsed_sec": time.time() - t_pipe,
    }
    (rep_dir / "phaseE1b_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()

    print("========== PHASE E.1b ==========", flush=True)
    print(
        f"BIAS: {'PASS' if decision['bias_pass'] else 'FAIL'} "
        f"n_violate={decision['n_violate']}/{decision['n_cells']}",
        flush=True,
    )
    print(
        f"SKILL: {'PASS' if decision['skill_pass'] else 'FAIL'} {decision['skill_by_h']}",
        flush=True,
    )
    print(f"VERDICT: {e1b_verdict}", flush=True)
    print(f"[phaseE1b] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return summary


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=16,
    memory=65536,
)
def run_e1_resume(payload: dict) -> dict:
    """GREEN path: extra seeds already trained; run E.1 §2–§4 and rewrite report."""
    import hashlib

    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from phase_e1.e1b_report import write_e1b_report
    from phase_e1.resume import run_sections_2_to_4

    calc = hashlib.sha256(Path("/root/config_frozen_a0.yaml").read_text().encode()).hexdigest()
    cfg = _cfg()
    root = Path(cfg["paths"]["volume_root"])
    feat = pd.read_parquet(root / "features" / "features_labeled.parquet")
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    pit20 = pd.read_parquet(root / "universe" / "top20_pit.parquet")
    pit120 = pd.read_parquet(root / "universe" / "top120_pit.parquet")
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)
    pit120["date"] = pd.to_datetime(pit120["date"], utc=True)
    pred_a = {
        7: pd.read_parquet(root / "predictions" / "lgbm_price_only_h7.parquet"),
        10: pd.read_parquet(root / "predictions" / "lgbm_price_only_h10.parquet"),
    }
    for h in pred_a:
        pred_a[h]["date"] = pd.to_datetime(pred_a[h]["date"], utc=True)
    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    panel = load_panel(root / "raw" / "klines", ever)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    funding = load_funding_panel(root / "raw" / "funding", ever)
    folds = {}
    for h in (7, 10):
        folds[h] = make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
    extra_gpu = payload.get("extra_gpu") or {}
    horizons_trained = extra_gpu.get("horizons") or [7, 10]
    resume_blob = run_sections_2_to_4(
        feat=feat,
        pit20=pit20,
        pit120=pit120,
        panel=panel,
        funding=funding,
        pred_a=pred_a,
        folds=folds,
        cfg=cfg,
        gru_root=root / "phase_e" / "gru",
        extra_gpu=extra_gpu,
        horizons_trained=horizons_trained,
        frozen_hash=calc,
        gates=[{"name": "e1b_empirical_null", "passed": True, "verdict": "GREEN"}],
        gates_ok=True,
        rep_dir=root / "reports",
        chart_dir=root / "charts",
        volume_commit=volume.commit,
    )
    e1b = json.loads((root / "reports" / "phaseE1b_summary.json").read_text())
    write_e1b_report(
        root / "reports" / "phaseE1b_report.md",
        frozen_hash=calc,
        budget=e1b.get("budget"),
        folds_used=e1b.get("folds_used"),
        horizons=[7, 10],
        shuffle_seeds=list(range(101, 111)),
        dropped_folds=e1b.get("dropped_folds"),
        e1b_verdict="GREEN",
        e1b_details=e1b.get("decision"),
        null_table=e1b.get("gru_cells"),
        a0_table=e1b.get("a0_cells"),
        skill_by_h=(e1b.get("decision") or {}).get("skill_by_h"),
        resume=resume_blob,
        gates=[{"name": "e1b_empirical_null", "passed": True}],
    )
    e1b["resume"] = {k: resume_blob[k] for k in ("verdict", "details", "keep_lines", "stdout") if k in resume_blob}
    (root / "reports" / "phaseE1b_summary.json").write_text(json.dumps(e1b, indent=2, default=str))
    volume.commit()
    return e1b


@app.local_entrypoint()
def main():
    print("[local] starting Phase E.1b empirical-null gate...", flush=True)
    summary = run_phase_e1b.remote()
    import shutil
    import subprocess

    def _pull():
        art = Path("artifacts")
        (art / "reports").mkdir(parents=True, exist_ok=True)
        (art / "charts").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        Path("charts").mkdir(exist_ok=True)
        for remote, name, kind in [
            ("reports/phaseE1b_report.md", "phaseE1b_report.md", "reports"),
            ("reports/phaseE1b_summary.json", "phaseE1b_summary.json", "reports"),
            ("charts/phaseE1b_null.png", "phaseE1b_null.png", "charts"),
            ("charts/phaseE1_seeds.png", "phaseE1_seeds.png", "charts"),
            ("charts/phaseE1_blend_equity.png", "phaseE1_blend_equity.png", "charts"),
        ]:
            dest = art / kind / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"], check=False)
            if dest.exists() and dest.is_file():
                shutil.copy2(dest, Path(kind) / name)
        opt = Path("/opt/cursor/artifacts")
        if opt.exists():
            for sub in ("reports", "charts"):
                (opt / sub).mkdir(parents=True, exist_ok=True)
            for src in (art / "reports").glob("phaseE1b*"):
                (opt / "reports" / src.name).write_bytes(src.read_bytes())
            for src in (art / "charts").glob("phaseE1b*"):
                (opt / "charts" / src.name).write_bytes(src.read_bytes())
            for src in (art / "charts").glob("phaseE1_*"):
                (opt / "charts" / src.name).write_bytes(src.read_bytes())

    _pull()
    if summary.get("e1b_verdict") == "GREEN":
        print("[local] GREEN — training extra seeds 45–50 then §2–§4", flush=True)
        extra = gru_extra_seed_jobs.remote(
            {
                "cache_dir": "/data/quant/phase_e/seq",
                "out_root": "/data/quant/phase_e/gru",
                "inner_holdout_days": summary.get("inner_holdout_days", 90),
                "max_epochs": DEFAULT_MAX_EPOCHS,
                "seeds": [45, 46, 47, 48, 49, 50],
                "horizons": [7, 10],
                "folds": summary["folds_payload"],
                "gpu_cap": E1_GPU_CAP,
                "sec_per_epoch": summary.get("sec_per_epoch", 5.4),
            }
        )
        summary = run_e1_resume.remote({"extra_gpu": extra})
        _pull()
    print(json.dumps({"e1b_verdict": summary.get("e1b_verdict"), "resume": summary.get("resume")}, indent=2, default=str))
    print("[local] Phase E.1b complete.", flush=True)
