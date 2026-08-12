"""
Phase B — Kronos frozen-feature ablation vs locked A0 (Modal, one-shot).

Usage:
    modal run phase_b_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-b-kronos"
VOLUME_NAME = "quant-baseline"
CRYPTO_VOLUME = "kronos-crypto-data"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
crypto_vol = modal.Volume.from_name(CRYPTO_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
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
        "einops==0.8.1",
        "huggingface_hub==0.33.1",
        "safetensors==0.6.2",
        "tqdm==4.67.1",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/data/quant/hf_cache"})
    .add_local_python_source("baseline", "phase_b")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("config_phase_b.yaml", remote_path="/root/config_phase_b.yaml")
)

app = modal.App(APP_NAME, image=image)


def _cfg_a0() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


def _cfg_b() -> dict:
    with open("/root/config_phase_b.yaml") as f:
        return yaml.safe_load(f)


@app.function(
    timeout=60 * 30,
    retries=1,
    volumes={"/data/quant": volume},
    gpu="A10G",
    memory=32768,
)
def calibrate_sec_per_row(payload: dict) -> dict:
    """Time a small batch to refine budget projection."""
    import pandas as pd
    from phase_b.kronos_features import KronosFeatureExtractor, extract_symbol_features

    root = Path("/data/quant")
    panel = pd.read_parquet(payload["panel_path"])
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    sym = payload["symbol"]
    n = int(payload.get("n_dates", 16))
    panel_sym = panel[panel["symbol"] == sym].sort_values("date")
    dates = list(pd.to_datetime(panel_sym["date"], utc=True).iloc[-n - 5 : -5])
    ext = KronosFeatureExtractor(
        model_id=payload["model_id"],
        tokenizer_id=payload["tokenizer_id"],
        context=int(payload["context"]),
        min_context=int(payload["min_context"]),
        n_paths=int(payload["n_paths"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        bf16=True,
    )
    t0 = time.time()
    _, st = extract_symbol_features(panel_sym, dates, ext, batch_size=int(payload.get("batch_size", 8)))
    elapsed = time.time() - t0
    n_comp = max(1, int(st.get("n_computed", len(dates))))
    sec = elapsed / n_comp
    volume.commit()
    return {"sec_per_row": sec, "elapsed": elapsed, "n": n_comp, "stats": st}


@app.function(
    timeout=60 * 60 * 3,
    retries=1,
    volumes={"/data/quant": volume},
    gpu="A10G",
    memory=32768,
)
def extract_symbol_chunk(payload: dict) -> dict:
    """Extract Kronos features for a list of symbols; cache shards on Volume."""
    import pandas as pd
    from phase_b.kronos_features import (
        KRONOS_FEATURE_COLS,
        KronosFeatureExtractor,
        extract_symbol_features,
    )

    root = Path("/data/quant")
    cache_dir = root / "kronos_features" / "shards"
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(payload["panel_path"])
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    keys = pd.read_parquet(payload["keys_path"])
    keys["date"] = pd.to_datetime(keys["date"], utc=True)

    ext = KronosFeatureExtractor(
        model_id=payload["model_id"],
        tokenizer_id=payload["tokenizer_id"],
        context=int(payload["context"]),
        min_context=int(payload["min_context"]),
        n_paths=int(payload["n_paths"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        bf16=bool(payload.get("bf16", True)),
    )

    summaries = []
    t_hb = time.time()
    for i, sym in enumerate(payload["symbols"]):
        shard = cache_dir / f"{sym}.parquet"
        sym_keys = keys[keys["symbol"] == sym]
        need_dates = set(pd.to_datetime(sym_keys["date"], utc=True))
        if shard.exists():
            existing = pd.read_parquet(shard)
            existing["date"] = pd.to_datetime(existing["date"], utc=True)
            have = set(existing["date"])
            # require feature cols present
            if set(KRONOS_FEATURE_COLS).issubset(existing.columns) and need_dates.issubset(have):
                summaries.append(
                    {
                        "symbol": sym,
                        "status": "cached",
                        "n_rows": int(len(existing)),
                        "n_computed": int(existing["kr_mu_h7"].notna().sum()),
                    }
                )
                continue
            missing = sorted(need_dates - have)
        else:
            existing = pd.DataFrame()
            missing = sorted(need_dates)

        if not missing:
            summaries.append({"symbol": sym, "status": "cached_empty_missing", "n_rows": 0})
            continue

        panel_sym = panel[panel["symbol"] == sym].copy()
        feat_df, st = extract_symbol_features(
            panel_sym, missing, ext, batch_size=int(payload.get("batch_size", 8))
        )
        if not existing.empty and not feat_df.empty:
            feat_df = pd.concat([existing, feat_df], ignore_index=True)
            feat_df = feat_df.drop_duplicates(["date", "symbol"], keep="last")
        elif existing.empty:
            pass
        else:
            feat_df = existing
        if not feat_df.empty:
            feat_df.to_parquet(shard, index=False)
        st["status"] = "computed"
        summaries.append(st)
        if time.time() - t_hb > 60:
            print(
                f"[extract] heartbeat sym={sym} i={i+1}/{len(payload['symbols'])} "
                f"computed={st.get('n_computed')} elapsed={st.get('elapsed_sec'):.1f}s",
                flush=True,
            )
            t_hb = time.time()
            volume.commit()

    volume.commit()
    return {
        "n_symbols": len(payload["symbols"]),
        "summaries": summaries,
        "worker": payload.get("worker_id"),
    }


@app.function(
    timeout=60 * 60 * 6,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=8,
    memory=65536,
)
def train_b_fold_job(payload: dict) -> dict:
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
        model_name="lgbm_a0_plus_kronos",
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
        f"[B-fold] h={fold.horizon} id={fold.fold_id} elapsed={meta['wall_elapsed']:.1f}s "
        f"best_iter={meta.get('best_iteration')}",
        flush=True,
    )
    return meta


@app.function(
    timeout=60 * 60 * 4,
    retries=0,
    volumes={"/data/quant": volume, "/data/crypto": crypto_vol},
    gpu="A10G",
    memory=32768,
)
def ft_reference_job(payload: dict) -> dict:
    import pandas as pd
    from phase_b.ft_reference import run_ft_reference_safe

    panel = pd.read_parquet(payload["panel_path"])
    feat = pd.read_parquet(payload["feat_path"])
    pit20 = pd.read_parquet(payload["pit20_path"])
    pit120 = pd.read_parquet(payload["pit120_path"])
    preds = pd.read_parquet(payload["preds_h10_path"])
    out = run_ft_reference_safe(
        panel,
        preds,
        feat,
        pit20,
        pit120,
        out_path=Path(payload["out_path"]),
        predictor_dir=payload["predictor_dir"],
        tokenizer_dir=payload["tokenizer_dir"],
        device="cuda:0",
    )
    volume.commit()
    return out


@app.function(
    timeout=60 * 60 * 12,
    retries=0,
    volumes={"/data/quant": volume, "/data/crypto": crypto_vol},
    cpu=16,
    memory=65536,
)
def run_phase_b() -> dict:
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel, build_pit_topn
    from baseline.features import FEATURE_COLS
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything
    from phase_b.ablation import (
        KILL_CRITERION,
        apply_kill_criterion,
        merge_kronos_features,
        run_ablation_for_horizon,
    )
    from phase_b.freeze import verify_frozen
    from phase_b.gate_test import run_gate_suite
    from phase_b.kronos_features import KRONOS_FEATURE_COLS, choose_budget_settings
    from phase_b.report import plot_gate_equity, plot_phaseB_ic, print_stdout_summary, write_phaseB_report

    t_pipe = time.time()
    print("[phaseB] verifying frozen A0 hash...", flush=True)
    # verify using files baked into image
    import hashlib
    from pathlib import Path as P

    frozen_text = P("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = P("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live = P("/root/config.yaml").read_text()
    live_h = hashlib.sha256(live.encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError(f"config.yaml drifted from frozen A0 live={live_h} frozen={calc}")
    print(f"[phaseB] frozen A0 OK sha256={calc}", flush=True)

    cfg = _cfg_a0()
    cfg_b = _cfg_b()
    seed_everything(cfg["seed"])

    root = Path(cfg["paths"]["volume_root"])
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    feat_dir = root / "features"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    kr_dir = root / "kronos_features"
    phase_dir = root / "phase_b"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in [kr_dir, phase_dir, rep_dir, chart_dir, kr_dir / "shards"]:
        d.mkdir(parents=True, exist_ok=True)

    feat_path = feat_dir / "features_labeled.parquet"
    if not feat_path.exists():
        raise RuntimeError("Missing features_labeled.parquet — run A0 first")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit120["date"] = pd.to_datetime(pit120["date"], utc=True)
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)

    # Panel for Kronos context
    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    print(f"[phaseB] loading panel for {len(ever)} symbols...", flush=True)
    panel = load_panel(raw_dir, ever)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel_path = phase_dir / "panel_ever.parquet"
    panel.to_parquet(panel_path, index=False)

    keys = feat[["date", "symbol"]].drop_duplicates()
    keys_path = kr_dir / "extract_keys.parquet"
    keys.to_parquet(keys_path, index=False)
    n_rows = len(keys)
    print(f"[phaseB] inference rows (symbol,date)={n_rows}", flush=True)

    kc = cfg_b["kronos"]
    # Budget guard
    budget_choice = choose_budget_settings(
        n_rows=n_rows,
        n_paths=int(kc["n_paths"]),
        context=int(kc["context"]),
        n_paths_fallback=int(kc["n_paths_fallback"]),
        context_fallback=int(kc["context_fallback"]),
        max_gpu_hours=float(kc["max_gpu_hours"]),
    )
    print(f"[phaseB] BUDGET GUARD initial steps={json.dumps(budget_choice['steps'])}", flush=True)
    if not budget_choice["ok"]:
        # calibrate then retry once
        print("[phaseB] projection over budget — calibrating sec/row on A10G...", flush=True)
        cal = calibrate_sec_per_row.remote(
            {
                "panel_path": str(panel_path),
                "symbol": "BTCUSDT",
                "n_dates": 12,
                "model_id": kc["model_id"],
                "tokenizer_id": kc["tokenizer_id"],
                "context": int(kc["context_fallback"]),
                "min_context": int(kc["min_context"]),
                "n_paths": int(kc["n_paths_fallback"]),
                "temperature": kc["temperature"],
                "top_p": kc["top_p"],
                "batch_size": kc["batch_size"],
            }
        )
        print(f"[phaseB] calibration={cal}", flush=True)
        budget_choice = choose_budget_settings(
            n_rows=n_rows,
            n_paths=int(kc["n_paths"]),
            context=int(kc["context"]),
            n_paths_fallback=int(kc["n_paths_fallback"]),
            context_fallback=int(kc["context_fallback"]),
            sec_per_row=float(cal["sec_per_row"]),
            max_gpu_hours=float(kc["max_gpu_hours"]),
        )
        print(f"[phaseB] BUDGET GUARD recalibrated={json.dumps(budget_choice)}", flush=True)
        if not budget_choice["ok"]:
            raise RuntimeError(f"Budget abort: {budget_choice['abort_reason']}")

    n_paths = int(budget_choice["n_paths"])
    context = int(budget_choice["context"])
    print(
        f"[phaseB] using n_paths={n_paths} context={context} "
        f"projected_gpu_h={budget_choice['projection']['gpu_hours']:.2f}",
        flush=True,
    )
    (kr_dir / "budget.json").write_text(json.dumps(budget_choice, indent=2))
    volume.commit()

    # Extract via symbol-chunk map
    symbols = sorted(keys["symbol"].unique())
    # Skip symbols fully cached
    cached = []
    todo_syms = []
    for sym in symbols:
        shard = kr_dir / "shards" / f"{sym}.parquet"
        if shard.exists():
            try:
                ex = pd.read_parquet(shard, columns=["date"] + list(KRONOS_FEATURE_COLS[:1]))
                need = set(pd.to_datetime(keys.loc[keys["symbol"] == sym, "date"], utc=True))
                have = set(pd.to_datetime(ex["date"], utc=True))
                if need.issubset(have):
                    cached.append(sym)
                    continue
            except Exception:
                pass
        todo_syms.append(sym)
    print(f"[phaseB] symbols cached={len(cached)} todo={len(todo_syms)}", flush=True)

    chunk_size = 8
    payloads = []
    for i in range(0, len(todo_syms), chunk_size):
        payloads.append(
            {
                "worker_id": i // chunk_size,
                "symbols": todo_syms[i : i + chunk_size],
                "panel_path": str(panel_path),
                "keys_path": str(keys_path),
                "model_id": kc["model_id"],
                "tokenizer_id": kc["tokenizer_id"],
                "context": context,
                "min_context": int(kc["min_context"]),
                "n_paths": n_paths,
                "temperature": kc["temperature"],
                "top_p": kc["top_p"],
                "batch_size": int(kc["batch_size"]),
                "bf16": bool(kc.get("bf16", True)),
            }
        )
    extract_summaries = []
    if payloads:
        print(f"[phaseB] launching {len(payloads)} GPU extract workers...", flush=True)
        # process in waves to avoid thundering herd on HF download
        for w in range(0, len(payloads), 4):
            wave = payloads[w : w + 4]
            print(f"[phaseB] extract wave {w//4+1} n_workers={len(wave)}", flush=True)
            extract_summaries.extend(list(extract_symbol_chunk.map(wave)))
            volume.reload()
    else:
        print("[phaseB] all Kronos feature shards cached", flush=True)

    # Assemble kronos feature panel
    volume.reload()
    parts = []
    n_computed = 0
    n_nan = 0
    for sym in symbols:
        shard = kr_dir / "shards" / f"{sym}.parquet"
        if not shard.exists():
            continue
        p = pd.read_parquet(shard)
        parts.append(p)
        if "kr_mu_h7" in p.columns:
            n_computed += int(p["kr_mu_h7"].notna().sum())
            n_nan += int(p["kr_mu_h7"].isna().sum())
    if not parts:
        raise RuntimeError("No Kronos feature shards produced")
    kronos = pd.concat(parts, ignore_index=True)
    kronos["date"] = pd.to_datetime(kronos["date"], utc=True)
    kronos = kronos.drop_duplicates(["date", "symbol"], keep="last")
    kronos_path = kr_dir / "kronos_features.parquet"
    kronos.to_parquet(kronos_path, index=False)
    coverage = {
        "n_keys": int(n_rows),
        "n_feature_rows": int(len(kronos)),
        "n_computed_mu": int(n_computed),
        "n_nan_mu": int(n_nan),
        "coverage_frac": float(n_computed / max(n_rows, 1)),
        "n_paths": n_paths,
        "context": context,
    }
    print(f"[phaseB] kronos coverage={coverage}", flush=True)
    (kr_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))
    volume.commit()

    # Merge + zscore Kronos into training frame
    print("[phaseB] merging Kronos features into A0 panel...", flush=True)
    feat_b = merge_kronos_features(feat, kronos, clip=cfg["features"]["zscore_clip"])
    feat_b_path = phase_dir / "features_a0_plus_kronos.parquet"
    feat_b.to_parquet(feat_b_path, index=False)
    volume.commit()

    feature_cols_b = list(FEATURE_COLS) + list(KRONOS_FEATURE_COLS)

    # Sanity gates on A0 (unchanged)
    pred_a7 = pd.read_parquet(pred_dir / "lgbm_price_only_h7.parquet")
    pred_a7["date"] = pd.to_datetime(pred_a7["date"], utc=True)
    folds7 = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=7,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    sample = pred_a7[pred_a7["date"] <= pred_a7["date"].min() + pd.Timedelta(days=90)].copy()
    ycol = "y_h7"
    if ycol not in sample.columns:
        sample = sample.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    gates = run_all_gates(panel, feat, build_pit_topn, folds7[0], cfg, sample)
    if not all(g.get("passed") for g in gates):
        raise RuntimeError(f"Sanity gates failed: {gates}")
    print(f"[phaseB] gates OK: {gates}", flush=True)

    # Train Model B
    ablation_blobs = {}
    delta_by_h = {}
    for h in cfg["labels"]["horizons"]:
        print(f"[phaseB] training Model B h={h}...", flush=True)
        folds = make_folds(
            pd.DatetimeIndex(feat_b["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        out_h = phase_dir / f"preds_b_h{h}"
        out_h.mkdir(parents=True, exist_ok=True)
        payloads = [
            {
                "cfg": cfg,
                "feat_path": str(feat_b_path),
                "out_dir": str(out_h),
                "fold_id": fr.fold_id,
                "train_start": str(fr.train_start),
                "train_end": str(fr.train_end),
                "purge_end": str(fr.purge_end),
                "embargo_end": str(fr.embargo_end),
                "val_start": str(fr.val_start),
                "val_end": str(fr.val_end),
                "horizon": h,
                "feature_cols": feature_cols_b,
            }
            for fr in folds
        ]
        metas = list(train_b_fold_job.map(payloads))
        volume.reload()
        preds = [
            pd.read_parquet(m["pred_path"])
            for m in metas
            if m.get("pred_path") and Path(m["pred_path"]).exists()
        ]
        pred_b = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        if not pred_b.empty:
            pred_b = pred_b.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
                ["date", "symbol"], keep="first"
            )
            pred_b.to_parquet(phase_dir / f"lgbm_a0_plus_kronos_h{h}.parquet", index=False)
        pred_a = pd.read_parquet(pred_dir / f"lgbm_price_only_h{h}.parquet")
        blob = run_ablation_for_horizon(pred_a, pred_b, feat, pit120, pit20, h, folds, metas)
        # drop heavy series from JSON later
        ablation_blobs[h] = blob
        delta_by_h[h] = blob["delta_daily_ic"]
        print(
            f"[phaseB] h={h} Δtop20 post={blob['delta_top20_post']} "
            f"frac+folds post={blob['frac_pos_folds_post']}",
            flush=True,
        )

    kill = apply_kill_criterion(
        {
            h: {
                "delta_top20_post": ablation_blobs[h]["delta_top20_post"],
                "frac_pos_folds_post": ablation_blobs[h]["frac_pos_folds_post"],
            }
            for h in ablation_blobs
        }
    )
    print(f"[phaseB] ABLATION VERDICT={kill['verdict']} :: {KILL_CRITERION}", flush=True)

    # Gate test
    print("[phaseB] uncertainty-gate test...", flush=True)
    funding = load_funding_panel(fund_dir, ever)
    gate_cfg = cfg_b["gate"]
    gate = run_gate_suite(
        pred_a7,
        panel,
        feat,
        pit20,
        kronos,
        funding,
        cfg,
        tau_pct=float(gate_cfg["tau_pct"]),
        sigma_top_pcts=list(gate_cfg["sigma_top_pcts"]),
    )

    # FT reference (non-blocking)
    print("[phaseB] FT contaminated reference...", flush=True)
    ft_cfg = cfg_b["ft_reference"]
    try:
        ft_ref = ft_reference_job.remote(
            {
                "panel_path": str(panel_path),
                "feat_path": str(feat_path),
                "pit20_path": str(uni_dir / "top20_pit.parquet"),
                "pit120_path": str(uni_dir / "top120_pit.parquet"),
                "preds_h10_path": str(pred_dir / "lgbm_price_only_h10.parquet"),
                "out_path": str(pred_dir / "kronos_ft_contaminated.parquet"),
                "predictor_dir": ft_cfg["predictor_dir"],
                "tokenizer_dir": ft_cfg["tokenizer_dir"],
            }
        )
    except Exception as e:
        ft_ref = {"status": "unavailable", "reason": str(e)}

    # Charts + report to volume and will be copied locally by local entry
    chart_path = chart_dir / "phaseB_ic.png"
    plot_phaseB_ic(delta_by_h, chart_path)
    if gate.get("gate_verdict") == "WIN" and gate.get("best"):
        key = f"top{int(gate['best']['sigma_top_pct'])}"
        ung = gate["results"]["ungated"].get("equity")
        g_eq = gate["results"].get(key, {}).get("equity")
        if ung is not None and g_eq is not None:
            plot_gate_equity(ung, g_eq, chart_dir / "phaseB_gate_equity.png")

    # Serialize ablation without Series
    abl_serial = {}
    for h, blob in ablation_blobs.items():
        abl_serial[h] = {
            k: v
            for k, v in blob.items()
            if k not in ("delta_daily_ic", "ic_a_full", "ic_b_full")
        }
        # fold per_fold ok; paired ok
        dd = blob.get("delta_daily_ic")
        if dd is not None and len(dd):
            abl_serial[h]["delta_daily_ic_tail"] = {
                str(i.date()): float(v) for i, v in list(dd.tail(5).items())
            }

    report_path = rep_dir / "phaseB_report.md"
    write_phaseB_report(
        report_path,
        frozen_hash=calc,
        budget=budget_choice,
        coverage=coverage,
        ablation=abl_serial,
        kill=kill,
        gate={k: v for k, v in gate.items() if k != "results"},
        ft_ref=ft_ref,
    )
    print_stdout_summary(ablation_blobs, kill, gate)

    # Persist summary JSON
    summary = {
        "frozen_sha256": calc,
        "budget": budget_choice,
        "coverage": coverage,
        "kill": kill,
        "gate_verdict": gate.get("gate_verdict"),
        "gate_best": gate.get("best"),
        "ablation": abl_serial,
        "ft_ref": ft_ref,
        "gates": gates,
        "elapsed_sec": time.time() - t_pipe,
        "criterion": KILL_CRITERION,
    }
    (rep_dir / "phaseB_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()
    print(f"[phaseB] DONE elapsed={time.time()-t_pipe:.1f}s verdict={kill['verdict']}", flush=True)
    return summary


@app.local_entrypoint()
def main():
    """Run Phase B on Modal and sync deliverables to ./artifacts."""
    print("[local] starting Phase B on Modal...", flush=True)
    summary = run_phase_b.remote()
    print("[local] syncing artifacts from volume...", flush=True)
    import subprocess

    art = Path("artifacts")
    (art / "reports").mkdir(parents=True, exist_ok=True)
    (art / "charts").mkdir(parents=True, exist_ok=True)
    for remote, local in [
        ("reports/phaseB_report.md", art / "reports" / "phaseB_report.md"),
        ("reports/phaseB_summary.json", art / "reports" / "phaseB_summary.json"),
        ("charts/phaseB_ic.png", art / "charts" / "phaseB_ic.png"),
        ("charts/phaseB_gate_equity.png", art / "charts" / "phaseB_gate_equity.png"),
    ]:
        try:
            subprocess.run(
                ["modal", "volume", "get", VOLUME_NAME, remote, str(local), "--force"],
                check=False,
            )
        except Exception as e:
            print(f"[local] sync skip {remote}: {e}", flush=True)
    # also copy to /opt/cursor/artifacts if present
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("phaseB*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("phaseB*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(json.dumps({k: summary.get(k) for k in ("kill", "gate_verdict", "coverage", "frozen_sha256")}, indent=2, default=str))
    print("[local] Phase B complete.", flush=True)
