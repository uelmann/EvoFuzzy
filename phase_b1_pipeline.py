"""
Phase B.1 — kr_sigma gate control experiment (CPU-only, Modal).

Usage:
    modal run phase_b1_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-b1-gate-control"
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


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_phase_b1() -> dict:
    import hashlib

    import pandas as pd

    from baseline.data import load_funding_panel, load_panel, build_pit_topn
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything
    from phase_b.control_gates import (
        ADOPTION_RULE,
        CONTROL_GATES,
        REF_GATE,
        X_GRID,
        _metrics_from_res,
        _resolve_gate_col,
        apply_adoption_rule,
        mean_daily_rank_corr,
        per_year_stats,
        rolling_sharpe,
        run_tranche_with_column_gate,
        run_ungated_instrumented,
        select_best_x,
        skip_overlap,
        trailing_12m,
    )
    from phase_b.phase_b1_report import (
        plot_gate_equities,
        plot_rolling_sharpe,
        print_stdout_summary,
        write_phaseB1_report,
    )

    t0 = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError(f"config.yaml drifted from frozen A0 live={live_h} frozen={calc}")
    print(f"[phaseB1] frozen A0 OK sha256={calc}", flush=True)
    print("[phaseB1] CPU-only — no GPU; reading Kronos features from Volume cache", flush=True)

    cfg = _cfg_a0()
    cfg_b = _cfg_b()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    pred_path = root / "predictions" / "lgbm_price_only_h7.parquet"
    kr_path = root / "kronos_features" / "kronos_features.parquet"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not kr_path.exists():
        raise RuntimeError(f"Missing Kronos feature cache at {kr_path} — refuse to run GPU inference")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    preds = pd.read_parquet(pred_path)
    preds["date"] = pd.to_datetime(preds["date"], utc=True)
    kronos = pd.read_parquet(kr_path, columns=["date", "symbol", "kr_sigma_h7"])
    kronos["date"] = pd.to_datetime(kronos["date"], utc=True)
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)
    pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")

    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    print(f"[phaseB1] loading panel ({len(ever)} symbols)...", flush=True)
    panel = load_panel(raw_dir, ever)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    funding = load_funding_panel(fund_dir, ever)

    # Sanity gates
    folds7 = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=7,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    sample = preds[preds["date"] <= preds["date"].min() + pd.Timedelta(days=90)].copy()
    if "y_h7" not in sample.columns:
        sample = sample.merge(feat[["date", "symbol", "y_h7"]], on=["date", "symbol"], how="left")
    gates = run_all_gates(panel, feat, build_pit_topn, folds7[0], cfg, sample)
    if not all(g.get("passed") for g in gates):
        raise RuntimeError(f"Sanity gates failed: {gates}")
    print(f"[phaseB1] gates OK", flush=True)

    tau_pct = float(cfg_b["gate"]["tau_pct"])
    port = cfg["portfolio"]
    print(f"[phaseB1] ungated baseline τ={tau_pct}...", flush=True)
    ungated_res = run_ungated_instrumented(preds, panel, feat, pit20, funding, cfg, tau_pct=tau_pct)
    ungated_row = _metrics_from_res(ungated_res)
    ungated_row["gate"] = "ungated"
    ungated_row["X"] = 0

    # Build gate value frames
    gate_frames = {}
    # controls from features
    for name, logical in CONTROL_GATES.items():
        col = _resolve_gate_col(feat, logical)
        gf = feat[["date", "symbol", col]].copy()
        gf = gf.rename(columns={col: logical})
        gate_frames[name] = (logical, gf)
    # reference from kronos cache
    gate_frames["REF_kr_sigma_h7"] = (REF_GATE, kronos.rename(columns={REF_GATE: REF_GATE}))

    results = {"ungated": ungated_res}
    rows = [ungated_row]

    for gname, (gcol, gdf) in gate_frames.items():
        for x in X_GRID:
            label = f"{gname}_X{x}"
            print(f"[phaseB1] gate {label} col={gcol}...", flush=True)
            res = run_tranche_with_column_gate(
                preds,
                panel,
                feat,
                pit20,
                gdf,
                gate_col=gcol,
                horizon=7,
                tau_pct=tau_pct,
                top_pct=float(x),
                gate_name=label,
                gross_limit=port["gross_limit"],
                fee_bps=port["taker_fee_bps"],
                slip_bps=port["slippage_bps"],
                lag=0,
                apply_funding=True,
                funding=funding,
            )
            results[label] = res
            row = _metrics_from_res(res)
            row["gate"] = label
            row["X"] = int(x)
            rows.append(row)
            print(
                f"[phaseB1] {label} Sh_full={row['sharpe_full']:.3f} "
                f"pre={row['sharpe_pre']:.3f} post={row['sharpe_post']:.3f}",
                flush=True,
            )

    # Redundancy diagnostics on PIT-20 × OOS days
    print("[phaseB1] redundancy diagnostics...", flush=True)
    import numpy as np

    oos = preds[["date", "symbol"]].drop_duplicates()
    diag = oos.merge(pit20[["date", "symbol"]], on=["date", "symbol"], how="inner")
    diag = diag.merge(kronos[["date", "symbol", REF_GATE]], on=["date", "symbol"], how="left")
    for _name, logical in CONTROL_GATES.items():
        col = _resolve_gate_col(feat, logical)
        piece = feat[["date", "symbol", col]].copy()
        if col != logical:
            piece = piece.rename(columns={col: logical})
        diag = diag.merge(piece, on=["date", "symbol"], how="left")

    rank_corr = {}
    reasons = []
    presumed = False
    for name, logical in CONTROL_GATES.items():
        full = mean_daily_rank_corr(diag, REF_GATE, logical, "full")
        post = mean_daily_rank_corr(diag, REF_GATE, logical, "post")
        rank_corr[name] = {"full": full, "post": post}
        if np.isfinite(full["mean_spearman"]) and full["mean_spearman"] > 0.8:
            presumed = True
            reasons.append(f"{name} mean_spearman_full={full['mean_spearman']:.3f}>0.8")
        if np.isfinite(post["mean_spearman"]) and post["mean_spearman"] > 0.8:
            presumed = True
            reasons.append(f"{name} mean_spearman_post={post['mean_spearman']:.3f}>0.8")

    # Skip overlap at X=20
    ref_key = "REF_kr_sigma_h7_X20"
    skip_ov = {}
    ref_skips = results[ref_key].get("daily_skips") or {}
    for name in CONTROL_GATES:
        ctrl_key = f"{name}_X20"
        ov = skip_overlap(ref_skips, results[ctrl_key].get("daily_skips") or {})
        skip_ov[name] = ov
        if np.isfinite(ov["overlap_frac"]) and ov["overlap_frac"] > 0.80:
            presumed = True
            reasons.append(f"{name} skip_overlap_x20={ov['overlap_frac']:.3f}>0.80")

    redundancy = {
        "rank_corr": rank_corr,
        "skip_overlap_x20": skip_ov,
        "presumed_redundant": presumed,
        "reasons": reasons,
        "thresholds": {"rank_corr": 0.8, "skip_overlap": 0.80},
    }
    print(f"[phaseB1] redundancy presumed={presumed} reasons={reasons}", flush=True)

    # X-selection: best X by post Sharpe within each family
    kr_best = select_best_x(rows, "REF_kr_sigma_h7")
    ctrl_cands = []
    for name in CONTROL_GATES:
        b = select_best_x(rows, name)
        if b is not None:
            ctrl_cands.append(b)
    ctrl_best = None
    if ctrl_cands:
        ctrl_best = max(
            ctrl_cands,
            key=lambda r: (r.get("sharpe_post") if np.isfinite(r.get("sharpe_post", np.nan)) else -1e9),
        )

    adoption = apply_adoption_rule(ungated_row, kr_best, ctrl_best, redundancy)
    print(f"[phaseB1] VERDICT={adoption['verdict']}", flush=True)
    print(f"[phaseB1] rule: {ADOPTION_RULE}", flush=True)

    # Decay
    decay_u = per_year_stats(ungated_res["daily_ret"])
    trail = trailing_12m(ungated_res["daily_ret"])
    adopt_res = results.get("ungated")
    winner_label = "ungated"
    if adoption["verdict"] == "ADOPT_KR_SIGMA" and kr_best is not None:
        adopt_res = results[kr_best["gate"]]
        winner_label = kr_best["gate"]
    elif adoption["verdict"] == "REDUNDANT_ADOPT_BEST_CONTROL" and ctrl_best is not None:
        adopt_res = results[ctrl_best["gate"]]
        winner_label = ctrl_best["gate"]
    decay_w = per_year_stats(adopt_res["daily_ret"])

    # Charts
    curves = {"ungated": ungated_res.get("equity")}
    if kr_best is not None:
        curves["kr_sigma"] = results[kr_best["gate"]].get("equity")
    if ctrl_best is not None:
        curves["best_control"] = results[ctrl_best["gate"]].get("equity")
    plot_gate_equities(curves, chart_dir / "phaseB1_gates.png")
    roll = rolling_sharpe(ungated_res["daily_ret"], 180)
    plot_rolling_sharpe(roll, chart_dir / "phaseB1_rolling_sharpe.png")

    write_phaseB1_report(
        rep_dir / "phaseB1_report.md",
        frozen_hash=calc,
        gate_rows=rows,
        redundancy=redundancy,
        adoption=adoption,
        decay_ungated=decay_u,
        decay_winner=decay_w,
        trailing=trail,
        winner_label=winner_label,
    )
    print_stdout_summary(rows, redundancy, adoption, trail)

    # JSON summary (drop heavy daily_skips)
    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "kronos_cache": str(kr_path),
        "adoption_rule": ADOPTION_RULE,
        "adoption": {
            "verdict": adoption["verdict"],
            "details": {
                k: v
                for k, v in (adoption.get("details") or {}).items()
                if k not in ("kr_best", "ctrl_best")
            },
            "kr_best_gate": (kr_best or {}).get("gate"),
            "ctrl_best_gate": (ctrl_best or {}).get("gate"),
            "adopt_gate": (adoption.get("adopt") or {}).get("gate"),
        },
        "gate_rows": rows,
        "redundancy": redundancy,
        "decay_ungated": decay_u,
        "decay_winner": decay_w,
        "trailing_12m": trail,
        "gates": gates,
        "elapsed_sec": time.time() - t0,
    }
    (rep_dir / "phaseB1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()
    print(f"[phaseB1] DONE elapsed={time.time()-t0:.1f}s verdict={adoption['verdict']}", flush=True)
    return summary


@app.local_entrypoint()
def main():
    print("[local] starting Phase B.1 (CPU)...", flush=True)
    summary = run_phase_b1.remote()
    print("[local] syncing artifacts...", flush=True)
    import subprocess

    art = Path("artifacts")
    (art / "reports").mkdir(parents=True, exist_ok=True)
    (art / "charts").mkdir(parents=True, exist_ok=True)
    for remote, local in [
        ("reports/phaseB1_report.md", art / "reports" / "phaseB1_report.md"),
        ("reports/phaseB1_summary.json", art / "reports" / "phaseB1_summary.json"),
        ("charts/phaseB1_gates.png", art / "charts" / "phaseB1_gates.png"),
        ("charts/phaseB1_rolling_sharpe.png", art / "charts" / "phaseB1_rolling_sharpe.png"),
    ]:
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote, str(local), "--force"],
            check=False,
        )
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("phaseB1*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("phaseB1*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(
        json.dumps(
            {
                "verdict": summary.get("adoption", {}).get("verdict"),
                "gpu_used": summary.get("gpu_used"),
                "trailing_12m": summary.get("trailing_12m"),
            },
            indent=2,
            default=str,
        )
    )
    print("[local] Phase B.1 complete.", flush=True)
