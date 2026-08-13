"""
Phase E.1 — verify GRU/BLEND (leakage gates, extra seeds, portfolio). Backtest only.

Usage:
    modal run phase_e1_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-e1-verify"
VOLUME_NAME = "quant-baseline"
MAX_GPU_HOURS = 15.0
DEFAULT_MAX_EPOCHS = 30
NEW_SEEDS = [45, 46, 47, 48, 49, 50]
OLD_SEEDS = [42, 43, 44]
ENSEMBLES = {
    "E42_44": [42, 43, 44],
    "E45_47": [45, 46, 47],
    "E48_50": [48, 49, 50],
    "GRAND9": [42, 43, 44, 45, 46, 47, 48, 49, 50],
}

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
def gru_jobs(payload: dict) -> dict:
    """Shuffle-control and/or extra-seed training on one A10G with resume."""
    import pandas as pd
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
    shuffle = bool(payload.get("shuffle_labels", False))
    fold_ids = payload.get("fold_ids")  # None = all
    cap = float(payload.get("gpu_cap", MAX_GPU_HOURS))
    sec = float(payload.get("sec_per_epoch", 5.4))

    n_folds = 0
    for h in horizons:
        frs = folds_by_h[h]
        if fold_ids is not None:
            frs = [f for f in frs if f.fold_id in set(fold_ids)]
        n_folds = max(n_folds, len(frs))
    proj = project_gpu_hours(sec, n_folds, len(seeds), len(horizons), max_epochs)
    print(f"[HB] GPU project gpu_hours={proj['gpu_hours']:.3f} cap={cap} shuffle={shuffle}", flush=True)
    aborted = False
    dropped_h7 = False
    if proj["gpu_hours"] > cap and 7 in horizons and 10 in horizons:
        horizons = [10]
        dropped_h7 = True
        proj = project_gpu_hours(sec, n_folds, len(seeds), 1, max_epochs)
        print(f"[HB] drop h=7; project2 gpu_hours={proj['gpu_hours']:.3f}", flush=True)
    if proj["gpu_hours"] > cap:
        aborted = True
        print("[E1] HARD ABORT GPU-hours still over cap", flush=True)
        return {"status": "aborted_budget", "projection": proj, "dropped_h7": dropped_h7}

    metas = []
    for h in horizons:
        frs = folds_by_h[h]
        if fold_ids is not None:
            frs = [f for f in frs if int(f.fold_id) in set(int(x) for x in fold_ids)]
        for fr in frs:
            for seed in seeds:
                dest = out_root / f"h{h}" / f"seed{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                tag = "shuf" if shuffle else "fold"
                pp = dest / f"{tag}{fr.fold_id}.parquet"
                mp = dest / f"{tag}{fr.fold_id}_meta.json"
                if pp.exists() and pp.stat().st_size > 0:
                    meta = json.loads(mp.read_text()) if mp.exists() else {"status": "reuse"}
                    meta["pred_path"] = str(pp)
                    metas.append(meta)
                    print(f"[HB] reuse h={h} fold={fr.fold_id} seed={seed} shuffle={shuffle}", flush=True)
                    continue
                pred_df, meta = train_gru_fold(
                    cache_dir,
                    fr,
                    horizon=int(h),
                    seed=int(seed),
                    inner_holdout_days=inner_h,
                    max_epochs=max_epochs,
                    shuffle_labels=shuffle,
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
                    f"[HB] done h={h} fold={fr.fold_id} seed={seed} shuf={shuffle} "
                    f"status={meta.get('status')} n={meta.get('n_pred')} max_tr={meta.get('max_train_date')}",
                    flush=True,
                )
    volume.commit()
    return {
        "status": "ok",
        "projection": proj,
        "dropped_h7": dropped_h7,
        "aborted": aborted,
        "horizons": horizons,
        "n_metas": len(metas),
        "metas": [{k: v for k, v in m.items() if k != "history_tail"} for m in metas],
    }


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=16,
    memory=65536,
)
def run_phase_e1() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.data import build_pit_topn, load_funding_panel, load_panel
    from baseline.evaluate import evaluate_predictions
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from phase_e.evalutil import (
        apply_s_blend_criteria,
        blend_scores,
        daily_score_spearman,
        evaluate_pair,
        window_mask,
    )
    from phase_e1.gates import (
        fold_isolation_gate,
        future_perturbation_gate,
        prediction_alignment_gate,
        summarize_shuffle_ic,
    )
    from phase_e1.report import CONFIRM_CRITERION, plot_blend_equity, plot_seeds, write_report

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    print(f"[phaseE1] frozen A0 OK sha256={calc}", flush=True)
    print("[phaseE1] BACKTEST ONLY — verification, no live/schedule", flush=True)
    print(f"[phaseE1] CRITERION: {CONFIRM_CRITERION}", flush=True)

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
    print("[phaseE1] A0 sanity gates OK", flush=True)

    seq_dir = root / "phase_e" / "seq"
    gru_root = root / "phase_e" / "gru"
    e1_dir = root / "phase_e1"
    e1_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    rep_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(seq_dir / "index.parquet")
    idx["date"] = pd.to_datetime(idx["date"], utc=True)

    # ----- §1 gates (CPU) -----
    g_fut = future_perturbation_gate(feat)
    print(f"[phaseE1] GATE future_perturbation passed={g_fut['passed']}", flush=True)
    g_iso = fold_isolation_gate(
        idx, folds[7] + folds[10], gru_root, inner_holdout_days=int(cfg["cv"]["inner_holdout_days"])
    )
    print(f"[phaseE1] GATE fold_isolation passed={g_iso['passed']} warm={g_iso['warm_start']}", flush=True)

    # alignment from per-fold artifacts (seed 42)
    align_ok = True
    align_blob = {"name": "prediction_alignment", "passed": False, "by_h": {}}
    for h in (7, 10):
        pieces = []
        d = gru_root / f"h{h}" / "seed42"
        for p in sorted(d.glob("fold*.parquet")):
            pieces.append(pd.read_parquet(p))
        if not pieces:
            align_ok = False
            align_blob["by_h"][h] = {"error": "missing fold preds"}
            continue
        ps = pd.concat(pieces, ignore_index=True)
        blob = prediction_alignment_gate(ps, folds[h])
        align_blob["by_h"][h] = blob
        align_ok = align_ok and blob["passed"]
        print(f"[phaseE1] GATE alignment h={h} passed={blob['passed']} n_bad={blob.get('n_bad')}", flush=True)
    align_blob["passed"] = bool(align_ok)

    # shuffle GPU
    n_f = len(folds[7])
    mid_id = folds[7][n_f // 2].fold_id
    rec_id = folds[7][-1].fold_id
    print(f"[phaseE1] shuffle folds mid={mid_id} recent={rec_id}", flush=True)
    shuf = gru_jobs.remote(
        {
            "cache_dir": str(seq_dir),
            "out_root": str(e1_dir / "shuffle"),
            "inner_holdout_days": cfg["cv"]["inner_holdout_days"],
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "seeds": OLD_SEEDS,
            "horizons": [7, 10],
            "folds": folds_payload,
            "shuffle_labels": True,
            "fold_ids": [mid_id, rec_id],
            "gpu_cap": MAX_GPU_HOURS,
            "sec_per_epoch": 5.4,
        }
    )
    volume.reload()
    shuf_rows = []
    shuf_pass = True
    for h in (7, 10):
        per_fold = {}
        for fid in (mid_id, rec_id):
            ics = []
            for seed in OLD_SEEDS:
                pp = e1_dir / "shuffle" / f"h{h}" / f"seed{seed}" / f"shuf{fid}.parquet"
                if not pp.exists():
                    shuf_pass = False
                    ics.append(float("nan"))
                    continue
                pdf = pd.read_parquet(pp)
                st = summarize_shuffle_ic(pdf, h)
                ics.append(st["mean_ic"])
                shuf_rows.append({"horizon": h, "fold_id": fid, "seed": seed, **st})
            mean_ic = float(np.nanmean(ics)) if ics else float("nan")
            ok = np.isfinite(mean_ic) and abs(mean_ic) < 0.005
            per_fold[fid] = {"mean_ic": mean_ic, "passed": ok, "per_seed": ics}
            shuf_pass = shuf_pass and ok
            print(f"[phaseE1] GATE shuffle h={h} fold={fid} mean_ic={mean_ic:.5f} pass={ok}", flush=True)
    g_shuf = {
        "name": "gru_label_shuffle",
        "passed": bool(shuf_pass),
        "threshold": 0.005,
        "folds": [mid_id, rec_id],
        "rows": shuf_rows,
        "gpu": {k: v for k, v in (shuf or {}).items() if k != "metas"},
    }

    gates = [g_shuf, g_fut, g_iso, align_blob]
    gates_ok = all(g.get("passed") for g in gates)
    print(f"[phaseE1] GATES {'ALL PASS' if gates_ok else 'FAIL'} {[g['name']+':'+str(g['passed']) for g in gates]}", flush=True)

    def _dump(verdict, extra=None):
        extra = extra or {}
        write_report(
            rep_dir / "phaseE1_report.md",
            frozen_hash=calc,
            verdict=verdict,
            verdict_details=extra,
            gates=gates,
            gates_ok=gates_ok,
        )
        summary = {
            "frozen_sha256": calc,
            "gates": gates,
            "gates_ok": gates_ok,
            "verdict": verdict,
            "scheduled_jobs_created": False,
            "criterion": CONFIRM_CRITERION,
            **extra,
        }
        (rep_dir / "phaseE1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        volume.commit()
        return summary

    if not gates_ok:
        print("[phaseE1] STOP — gates failed, not running §2–§4", flush=True)
        s = _dump("NOT CONFIRMED")
        print("GATES: FAIL", flush=True)
        print("VERDICT: NOT CONFIRMED", flush=True)
        return s

    # ----- §2 extra seeds -----
    budg_path = gru_root / "budget.json"
    sec = 5.4
    if budg_path.exists():
        try:
            sec = float(json.loads(budg_path.read_text())["calibrate"]["sec_1_epoch"])
        except Exception:
            pass
    extra = gru_jobs.remote(
        {
            "cache_dir": str(seq_dir),
            "out_root": str(gru_root),
            "inner_holdout_days": cfg["cv"]["inner_holdout_days"],
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "seeds": NEW_SEEDS,
            "horizons": [7, 10],
            "folds": folds_payload,
            "shuffle_labels": False,
            "fold_ids": None,
            "gpu_cap": MAX_GPU_HOURS,
            "sec_per_epoch": sec,
        }
    )
    volume.reload()
    horizons_trained = extra.get("horizons") or [7, 10]
    print(f"[phaseE1] extra-seed status={extra.get('status')} horizons={horizons_trained} dropped_h7={extra.get('dropped_h7')}", flush=True)

    seed_pred: dict[tuple[int, int], pd.DataFrame] = {}

    def assemble(h: int, seeds: list[int]) -> pd.DataFrame:
        frames = []
        for seed in seeds:
            key = (int(h), int(seed))
            if key not in seed_pred:
                pieces = []
                d = gru_root / f"h{h}" / f"seed{seed}"
                for p in sorted(d.glob("fold*.parquet")):
                    pieces.append(pd.read_parquet(p))
                if not pieces:
                    seed_pred[key] = pd.DataFrame()
                else:
                    sdf = pd.concat(pieces, ignore_index=True)
                    sdf["date"] = pd.to_datetime(sdf["date"], utc=True)
                    sdf = sdf.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(["date", "symbol"], keep="first")
                    sp = gru_root / f"lgbm_seq_s_h{h}_seed{seed}.parquet"
                    sdf.to_parquet(sp, index=False)
                    seed_pred[key] = sdf
            sdf = seed_pred[key]
            if sdf.empty:
                continue
            frames.append(sdf[["date", "symbol", "score"]].rename(columns={"score": f"score_s{seed}"}))
        if not frames:
            return pd.DataFrame()
        merged = frames[0]
        for extra_f in frames[1:]:
            merged = merged.merge(extra_f, on=["date", "symbol"], how="outer")
        scols = [c for c in merged.columns if c.startswith("score_s")]
        merged["score"] = merged[scols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        ycol = f"y_h{h}"
        merged = merged.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        return merged[["date", "symbol", "score", ycol]]

    ensemble_table = []
    ensemble_keep = {}  # name -> set of (uni, h)
    nw_table = []
    corr_table = []
    year_table = []
    seed_points = []
    ens_hlines = []
    blend_by = {}  # (ens, h) -> pred
    s_by = {}

    for ens_name, seeds in ENSEMBLES.items():
        for h in horizons_trained:
            pred_s = assemble(h, seeds)
            if pred_s.empty:
                continue
            s_by[(ens_name, h)] = pred_s
            pred_b = blend_scores(pred_a[h], pred_s)
            blend_by[(ens_name, h)] = pred_b
            blob_s = evaluate_pair(
                pred_a[h], pred_s, feat, pit20, pit120, panel, funding, h, folds[h], cfg, b_label="S", compute_sharpe=False
            )
            blob_b = evaluate_pair(
                pred_a[h], pred_b, feat, pit20, pit120, panel, funding, h, folds[h], cfg, b_label="BLEND", compute_sharpe=False
            )
            keep = apply_s_blend_criteria({h: blob_s}, {h: blob_b})
            passing = set()
            for uni, u in (keep.get("universes") or {}).items():
                if u.get("BLEND_verdict") == "KEEP":
                    passing.add((uni, h))
            ensemble_keep.setdefault(ens_name, set()).update(passing)
            colors = {"E42_44": "orange", "E45_47": "green", "E48_50": "purple", "GRAND9": "black"}
            for uni in ("top20", "pit120"):
                u_s = (blob_s.get("by_universe") or {}).get(uni) or {}
                u_b = (blob_b.get("by_universe") or {}).get(uni) or {}
                keep_u = (keep.get("universes") or {}).get(uni) or {}
                keep_h = bool((keep_u.get("BLEND_details") or {}).get(f"h{h}", {}).get("passes"))
                for window in ("full", "trail18m"):
                    paired_s = ((blob_s.get("paired_nw") or {}).get(uni) or {}).get(window) or {}
                    paired_b = ((blob_b.get("paired_nw") or {}).get(uni) or {}).get(window) or {}
                    ensemble_table.append(
                        {
                            "ens": ens_name,
                            "horizon": h,
                            "universe": uni,
                            "window": window,
                            "A_ic": u_s.get("A_full" if window == "full" else "A_trail18m"),
                            "S_ic": u_s.get("S_full" if window == "full" else "S_trail18m"),
                            "BLEND_ic": u_b.get("BLEND_full" if window == "full" else "BLEND_trail18m"),
                            "delta_S": u_s.get("delta_full" if window == "full" else "delta_trail18m"),
                            "delta_BLEND": u_b.get("delta_full" if window == "full" else "delta_trail18m"),
                            "nw_t_BLEND": paired_b.get("nw_tstat"),
                            "frac_pos": (blob_b.get("fold_stats") or {}).get(uni, {}).get(
                                "trail18m" if window == "trail18m" else "full", {}
                            ).get("frac_positive"),
                            "keep_blend": keep_h,
                        }
                    )
                    ens_hlines.append(
                        {
                            "horizon": h,
                            "universe": uni,
                            "window": window,
                            "mean_ic": u_s.get("S_full" if window == "full" else "S_trail18m"),
                            "name": ens_name,
                            "color": colors.get(ens_name, "gray"),
                        }
                    )
                    if ens_name in ("E42_44", "GRAND9"):
                        nw_table.append(
                            {
                                "ens": ens_name,
                                "model": "S",
                                "horizon": h,
                                "universe": uni,
                                "window": window,
                                **paired_s,
                            }
                        )
                        nw_table.append(
                            {
                                "ens": ens_name,
                                "model": "BLEND",
                                "horizon": h,
                                "universe": uni,
                                "window": window,
                                **paired_b,
                            }
                        )
                        aa = pred_a[h].merge(pit20 if uni == "top20" else pit120, on=["date", "symbol"], how="inner")
                        ss = pred_s.merge(pit20 if uni == "top20" else pit120, on=["date", "symbol"], how="inner")
                        if window == "trail18m":
                            end = aa["date"].max()
                            m = window_mask(aa["date"], "trail18m", end=end)
                            aa = aa.loc[m]
                            ss = ss[ss["date"].isin(set(aa["date"]))]
                        corr_table.append(
                            {"ens": ens_name, "horizon": h, "universe": uni, "window": window, **daily_score_spearman(aa, ss)}
                        )
            if ens_name in ("E42_44", "GRAND9"):
                for t in blob_s.get("tables") or []:
                    if str(t["window"]).startswith("y"):
                        bt = next(
                            (x for x in (blob_b.get("tables") or []) if x["universe"] == t["universe"] and x["window"] == t["window"]),
                            {},
                        )
                        year_table.append(
                            {
                                "ens": ens_name,
                                "horizon": h,
                                "universe": t["universe"],
                                "year": t["window"][1:],
                                "A_ic": t.get("A_ic"),
                                "S_ic": t.get("S_ic", t.get("B_ic")),
                                "BLEND_ic": bt.get("BLEND_ic", bt.get("B_ic")),
                            }
                        )

    # per-seed distribution
    seed_dist = []
    for h in horizons_trained:
        for seed in OLD_SEEDS + NEW_SEEDS:
            ps = assemble(h, [seed])
            if ps.empty:
                continue
            for uni_name, uni in [("top20", pit20), ("pit120", pit120)]:
                ev_full = evaluate_predictions(ps, h, universe=uni, label=uni_name)
                end = ps["date"].max()
                m = window_mask(ps["date"], "trail18m", end=end)
                ev_18 = evaluate_predictions(ps.loc[m], h, universe=uni, label=uni_name)
                for window, ev in [("full", ev_full), ("trail18m", ev_18)]:
                    seed_points.append(
                        {
                            "horizon": h,
                            "seed": seed,
                            "universe": uni_name,
                            "window": window,
                            "mean_ic": ev.get("mean_ic"),
                        }
                    )
    for h in horizons_trained:
        for uni in ("top20", "pit120"):
            for window in ("full", "trail18m"):
                vals = [
                    r["mean_ic"]
                    for r in seed_points
                    if r["horizon"] == h and r["universe"] == uni and r["window"] == window and np.isfinite(r.get("mean_ic", np.nan))
                ]
                if not vals:
                    continue
                arr = np.asarray(vals, float)
                seed_dist.append(
                    {
                        "horizon": h,
                        "universe": uni,
                        "window": window,
                        "min": float(arr.min()),
                        "median": float(np.median(arr)),
                        "max": float(arr.max()),
                        "n": int(len(arr)),
                    }
                )

    # KEEP lines
    keep_lines = []
    passing_sets = []
    for name in ("E42_44", "E45_47", "E48_50"):
        sl = ensemble_keep.get(name, set())
        passing_sets.append(sl)
        keep_lines.append(f"{name} {'PASS' if sl else 'FAIL'} KEEP slices={sorted(list(sl)) if sl else 'NONE'}")
        print(f"[phaseE1] {name} pass/fail slices={sl}", flush=True)
    common = set.intersection(*passing_sets) if passing_sets else set()
    print(f"[phaseE1] common KEEP slices={common}", flush=True)

    def _nw(uni, h, window="trail18m"):
        for r in nw_table:
            if (
                r.get("ens") == "GRAND9"
                and r.get("model") == "BLEND"
                and r.get("universe") == uni
                and r.get("horizon") == h
                and r.get("window") == window
            ):
                return r.get("nw_tstat")
        return float("nan")

    confirmed_slice = None
    for uni, h in sorted(common):
        nwt = _nw(uni, h, "trail18m")
        if np.isfinite(nwt) and nwt >= 2.0:
            confirmed_slice = (uni, h, nwt)
            break

    # ----- acf -----
    def lag1_acf(pred, h):
        df = pred.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        if "score" not in df.columns:
            df["score"] = df["y_pred"]
        acs = []
        for _, g in df.groupby("symbol"):
            g = g.sort_values("date")
            s = g["score"].astype(float)
            if len(s) < 10:
                continue
            a = s.autocorr(lag=1)
            if np.isfinite(a):
                acs.append(float(a))
        return {"mean_acf": float(np.mean(acs)) if acs else float("nan"), "n_symbols": int(len(acs))}

    acf_table = []
    for ens_name in ("E42_44", "GRAND9"):
        for h in horizons_trained:
            acf_table.append({"ens": ens_name, "horizon": h, "model": "A0", **lag1_acf(pred_a[h], h)})
            if (ens_name, h) in s_by:
                acf_table.append({"ens": ens_name, "horizon": h, "model": "S", **lag1_acf(s_by[(ens_name, h)], h)})
            if (ens_name, h) in blend_by:
                acf_table.append({"ens": ens_name, "horizon": h, "model": "BLEND", **lag1_acf(blend_by[(ens_name, h)], h)})

    # ----- portfolio -----
    port = cfg["portfolio"]
    port_table = []
    equity_a = equity_b = None
    for ens_name in ("E42_44", "GRAND9"):
        for h in horizons_trained:
            pa, pb = pred_a[h], blend_by.get((ens_name, h))
            if pb is None:
                continue
            for tau_mode in ("pooled", "expanding"):
                print(f"[HB] portfolio ens={ens_name} h={h} tau_mode={tau_mode}", flush=True)
                ra = run_tranche_portfolio(
                    pa, panel, feat, pit20, horizon=h, tau_pct=60.0,
                    exit_hysteresis=port.get("exit_hysteresis", 0.6),
                    gross_limit=port.get("gross_limit", 1.0),
                    fee_bps=port.get("taker_fee_bps", 5.0),
                    slip_bps=port.get("slippage_bps", 3.0),
                    lag=0, apply_funding=True, funding=funding, tau_mode=tau_mode,
                )
                rb = run_tranche_portfolio(
                    pb, panel, feat, pit20, horizon=h, tau_pct=60.0,
                    exit_hysteresis=port.get("exit_hysteresis", 0.6),
                    gross_limit=port.get("gross_limit", 1.0),
                    fee_bps=port.get("taker_fee_bps", 5.0),
                    slip_bps=port.get("slippage_bps", 3.0),
                    lag=0, apply_funding=True, funding=funding, tau_mode=tau_mode,
                )
                da, db = ra.get("daily_ret"), rb.get("daily_ret")

                def _sh(x):
                    if x is None or not isinstance(x, pd.Series) or len(x) < 5 or x.std() == 0:
                        return float("nan")
                    return float(x.mean() / x.std() * np.sqrt(365))

                end = None
                if isinstance(da, pd.Series) and isinstance(db, pd.Series):
                    idxn = da.index.intersection(db.index)
                    da, db = da.loc[idxn], db.loc[idxn]
                    end = idxn.max() if len(idxn) else None
                start = end - pd.Timedelta(days=int(365 * 1.5)) if end is not None else None
                for window, mask in [
                    ("full", None),
                    ("trail18m", (da.index >= start) & (da.index <= end) if start is not None else None),
                ]:
                    xa = da if mask is None else da.loc[mask]
                    xb = db if mask is None else db.loc[mask]
                    port_table.append(
                        {
                            "ens": ens_name,
                            "horizon": h,
                            "tau_mode": tau_mode,
                            "window": window,
                            "A_sharpe": _sh(xa),
                            "B_sharpe": _sh(xb),
                            "delta_sharpe": _sh(xb) - _sh(xa) if np.isfinite(_sh(xa)) and np.isfinite(_sh(xb)) else float("nan"),
                            "A_to": ra.get("ann_turnover"),
                            "B_to": rb.get("ann_turnover"),
                            "A_npos": ra.get("avg_n_positions"),
                            "B_npos": rb.get("avg_n_positions"),
                            "A_flat": ra.get("pct_flat_days"),
                            "B_flat": rb.get("pct_flat_days"),
                        }
                    )
                if ens_name == "GRAND9" and tau_mode == "pooled" and h == (confirmed_slice[1] if confirmed_slice else horizons_trained[0]):
                    if isinstance(ra.get("equity"), pd.DataFrame):
                        ea = ra["equity"].copy()
                        ea["date"] = pd.to_datetime(ea["date"], utc=True)
                        equity_a = ea.set_index("date")["equity"]
                    if isinstance(rb.get("equity"), pd.DataFrame):
                        eb = rb["equity"].copy()
                        eb["date"] = pd.to_datetime(eb["date"], utc=True)
                        equity_b = eb.set_index("date")["equity"]

    # (iv) either tau convention on confirmed horizon (or h=7 default)
    h_iv = confirmed_slice[1] if confirmed_slice else (horizons_trained[0] if horizons_trained else 7)
    iv_ok = False
    iv_detail = []
    for tau_mode in ("pooled", "expanding"):
        full = next(
            (
                r
                for r in port_table
                if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == tau_mode and r["window"] == "full"
            ),
            None,
        )
        tr = next(
            (
                r
                for r in port_table
                if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == tau_mode and r["window"] == "trail18m"
            ),
            None,
        )
        if not full or not tr:
            continue
        ok = (
            np.isfinite(full["delta_sharpe"])
            and np.isfinite(tr["delta_sharpe"])
            and full["delta_sharpe"] >= -0.10
            and tr["delta_sharpe"] >= 0.0
        )
        iv_detail.append({"tau_mode": tau_mode, "delta_full": full["delta_sharpe"], "delta_18": tr["delta_sharpe"], "ok": ok})
        if ok:
            iv_ok = True

    ii_ok = bool(common)
    iii_ok = confirmed_slice is not None
    verdict = "CONFIRMED" if (gates_ok and ii_ok and iii_ok and iv_ok) else "NOT CONFIRMED"
    details = {
        "i_gates": gates_ok,
        "ii_common_slices": [list(x) for x in sorted(common)],
        "iii_slice": confirmed_slice,
        "iv": iv_detail,
        "iv_ok": iv_ok,
        "dropped_h7": extra.get("dropped_h7"),
        "horizons_trained": horizons_trained,
    }
    print(f"[phaseE1] VERDICT={verdict} details={details}", flush=True)

    plot_seeds(seed_points, ens_hlines, chart_dir / "phaseE1_seeds.png")
    if equity_a is not None and equity_b is not None:
        plot_blend_equity(equity_a, equity_b, chart_dir / "phaseE1_blend_equity.png")

    write_report(
        rep_dir / "phaseE1_report.md",
        frozen_hash=calc,
        verdict=verdict,
        verdict_details=details,
        gates=gates,
        gates_ok=gates_ok,
        budget=extra.get("projection"),
        horizons_trained=horizons_trained,
        ensemble_keep_lines=keep_lines + [f"common={sorted(common)}", f"iii={confirmed_slice}", f"iv_ok={iv_ok}"],
        ensemble_table=ensemble_table,
        seed_dist=seed_dist,
        nw_table=nw_table,
        corr_table=corr_table,
        year_table=year_table,
        acf_table=acf_table,
        port_table=port_table,
    )
    summary = {
        "frozen_sha256": calc,
        "gates_ok": gates_ok,
        "gates": gates,
        "verdict": verdict,
        "details": details,
        "criterion": CONFIRM_CRITERION,
        "ensemble_table": ensemble_table,
        "seed_dist": seed_dist,
        "nw_table": nw_table,
        "corr_table": corr_table,
        "year_table": year_table,
        "acf_table": acf_table,
        "port_table": port_table,
        "extra_gpu": extra,
        "scheduled_jobs_created": False,
        "elapsed_sec": time.time() - t_pipe,
    }
    (rep_dir / "phaseE1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()

    print("========== PHASE E.1 ==========", flush=True)
    print("GATES: " + " ".join(f"{g['name']}={'PASS' if g['passed'] else 'FAIL'}" for g in gates), flush=True)
    for line in keep_lines:
        print(f"ENSEMBLE: {line}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    # portfolio one-liner
    pfull = next(
        (r for r in port_table if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == "pooled" and r["window"] == "full"),
        {},
    )
    p18 = next(
        (r for r in port_table if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == "pooled" and r["window"] == "trail18m"),
        {},
    )
    print(
        f"PORTFOLIO h={h_iv} pooled ΔSharpe full={pfull.get('delta_sharpe')} trail18={p18.get('delta_sharpe')} "
        f"ΔTO={None if not pfull else (pfull.get('B_to') or 0)-(pfull.get('A_to') or 0)}",
        flush=True,
    )
    print(f"[phaseE1] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return summary


@app.local_entrypoint()
def main():
    print("[local] starting Phase E.1 verification...", flush=True)
    summary = run_phase_e1.remote()
    import shutil
    import subprocess

    art = Path("artifacts")
    (art / "reports").mkdir(parents=True, exist_ok=True)
    (art / "charts").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/phaseE1_report.md", "phaseE1_report.md", "reports"),
        ("reports/phaseE1_summary.json", "phaseE1_summary.json", "reports"),
        ("charts/phaseE1_seeds.png", "phaseE1_seeds.png", "charts"),
        ("charts/phaseE1_blend_equity.png", "phaseE1_blend_equity.png", "charts"),
    ]:
        dest = art / kind / name
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"], check=False)
        if dest.exists():
            shutil.copy2(dest, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("phaseE1*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("phaseE1*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(json.dumps({"verdict": summary.get("verdict"), "gates_ok": summary.get("gates_ok")}, indent=2))
    print("[local] Phase E.1 complete.", flush=True)
