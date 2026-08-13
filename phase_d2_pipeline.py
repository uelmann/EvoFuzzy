"""
Phase D.2 — top-40 execution universe, causal τ, micro ablation, hedge decomp.

BACKTEST ONLY. CPU only. No schedules / live components.

Usage:
    modal run phase_d2_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-phase-d2-universe"
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
    .add_local_python_source("baseline", "phase_d", "phase_d2")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/phaseD2_addendum.md", remote_path="/root/phaseD2_addendum.md")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def train_d2_fold_job(payload: dict) -> dict:
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
        model_name="lgbm_a0_plus_micro10",
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
        f"[HB] D2-fold h={fold.horizon} id={fold.fold_id} status={meta.get('status')} "
        f"n_pred={len(pred_df)} elapsed={meta['wall_elapsed']:.1f}s",
        flush=True,
    )
    return meta


@app.function(timeout=60 * 60 * 10, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_phase_d2() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.attribution import per_year_breakdown
    from baseline.data import build_pit_topn, load_funding_panel, load_panel
    from baseline.features import FEATURE_COLS
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from phase_d.micro_data import MICRO_FEATURE_COLS_10
    from phase_d2.constants import (
        ADOPTION_CRITERION,
        FEE_BPS_NEXT,
        FEE_BPS_TOP,
        HONESTY_PREAMBLE,
        HORIZONS,
        LIQ_CAP_ADV_FRAC,
        NOMINAL_BOOK_USD,
        SLIP_BPS_NEXT,
        SLIP_BPS_TOP,
        TAU_PCTS,
    )
    from phase_d2.metrics import (
        apply_adoption,
        ic_pair_on_universe,
        pick_median_tau,
        slim_port,
        summarize_port,
        year_attribution_2026,
    )
    from phase_d2.report import (
        plot_hedge_bars,
        plot_universe_equity,
        print_stdout_summary,
        write_phaseD2_report,
    )

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    addendum = Path("/root/phaseD2_addendum.md").read_text()
    if HONESTY_PREAMBLE not in addendum or ADOPTION_CRITERION not in addendum:
        raise RuntimeError("Addendum missing verbatim preamble or criterion")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)
    print("[HB] BACKTEST ONLY — no schedules/cron/shadow; zero GPU", flush=True)
    print("[HB] preamble+criterion present in addendum (frozen before results)", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    phase_d_dir = root / "phase_d"
    phase_dir = root / "phase_d2"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in [uni_dir, phase_dir, rep_dir, chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    port_cfg = cfg["portfolio"]
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    print(f"[HB] feat rows={len(feat)} n_feat_a0={len(FEATURE_COLS)}", flush=True)

    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    print(f"[HB] loading panel n_sym={len(kline_syms)}...", flush=True)
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)

    window = int(cfg["data"]["exec_dv_window"])
    pit40 = build_pit_topn(panel, n=40, window=window)
    pit40.to_parquet(uni_dir / "top40_pit.parquet", index=False)
    if (uni_dir / "top20_pit.parquet").exists():
        pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
        pit20["date"] = pd.to_datetime(pit20["date"], utc=True)
    else:
        pit20 = build_pit_topn(panel, n=20, window=window)
        pit20.to_parquet(uni_dir / "top20_pit.parquet", index=False)
    if (uni_dir / "top120_pit.parquet").exists():
        pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")
        pit120["date"] = pd.to_datetime(pit120["date"], utc=True)
    else:
        pit120 = build_pit_topn(panel, n=int(cfg["data"]["train_universe_n"]), window=window)
    print(
        f"[HB] PIT universes top20={len(pit20)} top40={len(pit40)} top120={len(pit120)}",
        flush=True,
    )

    ever = sorted(set(feat["symbol"].unique()) | set(pit40["symbol"].unique()) | {"BTCUSDT"})
    funding = load_funding_panel(fund_dir, ever)

    pred_a = {}
    for h in HORIZONS:
        p = pd.read_parquet(pred_dir / f"lgbm_price_only_h{h}.parquet")
        p["date"] = pd.to_datetime(p["date"], utc=True)
        pred_a[h] = p
        print(f"[HB] A0 preds h={h} n={len(p)}", flush=True)

    folds7 = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=7,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    sample = pred_a[7][pred_a[7]["date"] <= pred_a[7]["date"].min() + pd.Timedelta(days=90)].copy()
    if "y_h7" not in sample.columns:
        sample = sample.merge(feat[["date", "symbol", "y_h7"]], on=["date", "symbol"], how="left")
    print("[HB] running gates (incl. top-40 lookahead)...", flush=True)
    gates = run_all_gates(panel, feat, build_pit_topn, folds7[0], cfg, sample)
    if not all(g.get("passed") for g in gates):
        raise RuntimeError(f"Sanity gates failed: {gates}")
    print("[HB] gates OK", flush=True)

    feat_d_path = phase_d_dir / "features_a0_plus_micro.parquet"
    if not feat_d_path.exists():
        raise RuntimeError(f"missing {feat_d_path}; Phase D micro panel required")
    feat_d = pd.read_parquet(feat_d_path)
    feat_d["date"] = pd.to_datetime(feat_d["date"], utc=True)
    feature_cols_d = list(FEATURE_COLS) + list(MICRO_FEATURE_COLS_10)
    missing = [c for c in feature_cols_d if c not in feat_d.columns]
    if missing:
        raise RuntimeError(f"micro10 cols missing from feat_d: {missing}")
    print(f"[HB] A+micro feature count={len(feature_cols_d)} (33+10)", flush=True)

    pred_m = {}
    for h in HORIZONS:
        print(f"[HB] A+micro10 training/reuse h={h}...", flush=True)
        folds = make_folds(
            pd.DatetimeIndex(feat_d["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        out_h = phase_dir / f"preds_m10_h{h}"
        canon = phase_dir / f"lgbm_a0_plus_micro10_h{h}.parquet"
        meta_path = out_h / f"fold_meta_h{h}.json"
        reuse_ok = False
        if canon.exists():
            pred_d = pd.read_parquet(canon)
            if not pred_d.empty:
                print(f"[HB] reusing micro10 preds {canon} n={len(pred_d)}", flush=True)
                reuse_ok = True
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
            metas = list(train_d2_fold_job.map(payloads))
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
        pred_d = pd.read_parquet(canon)
        pred_d["date"] = pd.to_datetime(pred_d["date"], utc=True)
        pred_m[h] = pred_d
        print(f"[HB] micro10 h={h} n={len(pred_d)}", flush=True)

    folds_by_h = {
        h: make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        for h in HORIZONS
    }

    specs = {
        "P1": dict(model="A", universe="top20", uni=pit20, pred=pred_a, tiered=False, cap=None),
        "P2": dict(model="A", universe="top40", uni=pit40, pred=pred_a, tiered=True, cap=LIQ_CAP_ADV_FRAC),
        "P3": dict(model="A+micro", universe="top20", uni=pit20, pred=pred_m, tiered=False, cap=None),
        "P4": dict(model="A+micro", universe="top40", uni=pit40, pred=pred_m, tiered=True, cap=LIQ_CAP_ADV_FRAC),
    }

    def _run(spec, h, tau_pct, tau_mode, hedge_mode="trailing"):
        run_id = spec.get("run_id", "?")
        key = (run_id, int(h), float(tau_pct), str(tau_mode), str(hedge_mode), bool(spec["tiered"]), spec["cap"])
        if key in _run_cache:
            print(f"[HB] reuse port {run_id} h={h} τ={tau_pct} mode={tau_mode} hedge={hedge_mode}", flush=True)
            return _run_cache[key]
        print(
            f"[HB] port {run_id} h={h} τ={tau_pct} mode={tau_mode} hedge={hedge_mode}",
            flush=True,
        )
        res = run_tranche_portfolio(
            spec["pred"][h],
            panel,
            feat,
            spec["uni"],
            horizon=h,
            tau_pct=float(tau_pct),
            exit_hysteresis=port_cfg.get("exit_hysteresis", 0.6),
            gross_limit=port_cfg.get("gross_limit", 1.0),
            fee_bps=FEE_BPS_TOP,
            slip_bps=SLIP_BPS_TOP,
            lag=0,
            apply_funding=True,
            funding=funding,
            tau_mode=tau_mode,
            folds=folds_by_h[h],
            tiered_costs=bool(spec["tiered"]),
            fee_bps_next=FEE_BPS_NEXT,
            slip_bps_next=SLIP_BPS_NEXT,
            liq_cap_adv_frac=spec["cap"],
            nominal_book_usd=NOMINAL_BOOK_USD,
            hedge_mode=hedge_mode,
            rank_universe=pit40,
        )
        _run_cache[key] = res
        return res

    _run_cache = {}

    # --- τ-fix isolation: A0 top-20 h=7 pooled vs fold_train ---
    tau_fix_raw = []
    for mode in ("pooled", "fold_train"):
        for tp in TAU_PCTS:
            spec = dict(specs["P1"], run_id="P1")
            res = _run(spec, 7, tp, mode)
            row = slim_port(res)
            row["tau_mode"] = mode
            tau_fix_raw.append(row)
    pooled60 = next(r for r in tau_fix_raw if r.get("tau_mode") == "pooled" and float(r.get("tau_pct")) == 60.0)
    train60 = next(r for r in tau_fix_raw if r.get("tau_mode") == "fold_train" and float(r.get("tau_pct")) == 60.0)
    tau_line = (
        f"A0 top-20 h=7 τ=60 net Sharpe pooled(full-OOS)={pooled60.get('net_sharpe'):.3f} "
        f"vs fold_train={train60.get('net_sharpe'):.3f} "
        f"(Δ={float(train60.get('net_sharpe', 0)) - float(pooled60.get('net_sharpe', 0)):+.3f})"
    )
    print(f"[HB] {tau_line}", flush=True)

    # --- core P1–P4 ---
    raw_by = {}  # (run_id, h, tau) -> full res
    for run_id, spec0 in specs.items():
        spec = dict(spec0, run_id=run_id)
        for h in HORIZONS:
            for tp in TAU_PCTS:
                raw_by[(run_id, h, tp)] = _run(spec, h, tp, "fold_train")

    picked = {}
    for run_id in specs:
        for h in HORIZONS:
            runs = [raw_by[(run_id, h, tp)] for tp in TAU_PCTS]
            picked[(run_id, h)] = pick_median_tau(runs)
            print(
                f"[HB] median-τ {run_id} h={h} τ={picked[(run_id, h)].get('tau_pct')} "
                f"sharpe={picked[(run_id, h)].get('net_sharpe')}",
                flush=True,
            )

    common = {}
    for h in HORIZONS:
        idxs = []
        for run_id in specs:
            ser = picked[(run_id, h)].get("daily_ret")
            if isinstance(ser, pd.Series) and len(ser):
                ix = pd.DatetimeIndex(pd.to_datetime(ser.index, utc=True))
                idxs.append(ix)
        common[h] = idxs[0]
        for ix in idxs[1:]:
            common[h] = common[h].intersection(ix)
        print(f"[HB] identical days h={h} n={len(common[h])}", flush=True)

    p_summ = {}
    p_table = []
    for run_id, spec0 in specs.items():
        for h in HORIZONS:
            summ = summarize_port(picked[(run_id, h)], common_idx=common[h])
            summ["run_id"] = run_id
            summ["horizon"] = h
            summ["model"] = spec0["model"]
            summ["universe"] = spec0["universe"]
            p_summ[(run_id, h)] = summ
            p_table.append({k: v for k, v in summ.items() if k not in ("equity", "daily_ret", "year_rows")})

    # IC
    ic_blobs = {}
    ic_tables = []
    ic_nw = []
    for uni_name, uni in [("top20", pit20), ("top40", pit40)]:
        for h in HORIZONS:
            blob = ic_pair_on_universe(pred_a[h], pred_m[h], feat, uni, h, uni_name)
            ic_blobs[(uni_name, h)] = blob
            ic_tables.extend(blob.get("tables") or [])
            for w, p in (blob.get("paired_nw") or {}).items():
                ic_nw.append({"horizon": h, "universe": uni_name, "window": w, **p})

    verdict = apply_adoption(p_summ, ic_blobs)
    print(f"[HB] universe={verdict['universe_verdict']} micro={verdict['micro_verdict']}", flush=True)

    # Oracle-beta on P1 h=7 at the same median τ (LOOKAHEAD BY DESIGN)
    p1_h7 = picked[("P1", 7)]
    spec_p1 = dict(specs["P1"], run_id="P1")
    oracle_res = _run(spec_p1, 7, p1_h7.get("tau_pct", 60.0), "fold_train", hedge_mode="oracle")
    oracle_summ = summarize_port(oracle_res, common_idx=common[7])
    p1_summ_h7 = p_summ[("P1", 7)]
    attr = year_attribution_2026(p1_summ_h7, oracle_summ)
    print(f"[HB] 2026 attribution: {attr['sentence']}", flush=True)

    # charts: P1 vs best-of-P2/P4 by trail18m (h=7 headline; if a passing row exists use that h)
    best = None
    best_name = "best-of-P2/P4"
    passing = verdict.get("universe_rows") or []
    cand_pass = [r for r in passing if r.get("pass")]
    if cand_pass:
        top = max(cand_pass, key=lambda r: r["trail18m"])
        best = p_summ[(top["candidate"], top["horizon"])]
        best_name = f"{top['candidate']} h={top['horizon']}"
    else:
        # still plot best trail18m among P2/P4 both h
        pool = [p_summ[(rid, h)] for rid in ("P2", "P4") for h in HORIZONS]
        best = max(pool, key=lambda r: float(r.get("net_sharpe_trail18m") or float("-inf")))
        best_name = f"{best.get('run_id')} h={best.get('horizon')} (not adopted)"

    plot_universe_equity(p_summ[("P1", 7)], best, best_name, chart_dir / "phaseD2_universe.png")
    plot_hedge_bars(p1_summ_h7, chart_dir / "phaseD2_hedge.png")

    extra = {"tau_fix_one_liner": tau_line, "nominal_book_usd": NOMINAL_BOOK_USD, "n_micro_features": 10}
    write_phaseD2_report(
        rep_dir / "phaseD2_report.md",
        frozen_hash=calc,
        gates=gates,
        tau_fix=tau_fix_raw,
        p_table=p_table,
        ic_tables=ic_tables,
        ic_nw=ic_nw,
        verdict=verdict,
        hedge_years=p1_summ_h7.get("year_rows") or [],
        oracle_years=oracle_summ.get("year_rows") or per_year_breakdown(oracle_res),
        attr_2026=attr,
        extra=extra,
    )
    print_stdout_summary(verdict, tau_line, attr["sentence"])

    def _jsonable(x):
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items() if k not in ("equity", "daily_ret")}
        if isinstance(x, list):
            return [_jsonable(v) for v in x]
        if isinstance(x, (pd.Timestamp,)):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "nominal_book_usd": NOMINAL_BOOK_USD,
        "liq_cap_adv_frac": LIQ_CAP_ADV_FRAC,
        "criterion": ADOPTION_CRITERION,
        "preamble": HONESTY_PREAMBLE,
        "gates": gates,
        "tau_fix": tau_fix_raw,
        "tau_fix_one_liner": tau_line,
        "p_table": p_table,
        "ic_tables": ic_tables,
        "ic_nw": ic_nw,
        "verdict": {
            k: v
            for k, v in verdict.items()
        },
        "hedge_years": p1_summ_h7.get("year_rows") or [],
        "oracle_years": oracle_summ.get("year_rows") or [],
        "attr_2026": attr,
        "elapsed_sec": time.time() - t_pipe,
        "n_micro_features": 10,
        "feature_cols_micro10": feature_cols_d,
    }
    (rep_dir / "phaseD2_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    volume.commit()
    print(f"[HB] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return {
        "frozen_sha256": calc,
        "gpu_used": False,
        "universe_verdict": verdict.get("universe_verdict"),
        "micro_verdict": verdict.get("micro_verdict"),
        "tau_fix_one_liner": tau_line,
        "attr_2026_sentence": attr.get("sentence"),
        "elapsed_sec": time.time() - t_pipe,
    }


@app.local_entrypoint()
def main():
    print("[local] starting Phase D.2 (CPU, backtest-only)...", flush=True)
    summary = run_phase_d2.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    (art / "reports").mkdir(parents=True, exist_ok=True)
    (art / "charts").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/phaseD2_report.md", "phaseD2_report.md", "reports"),
        ("reports/phaseD2_summary.json", "phaseD2_summary.json", "reports"),
        ("charts/phaseD2_universe.png", "phaseD2_universe.png", "charts"),
        ("charts/phaseD2_hedge.png", "phaseD2_hedge.png", "charts"),
    ]:
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"],
            check=False,
        )
        if dest.exists() and dest.is_file():
            shutil.copy2(dest, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("phaseD2*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("phaseD2*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] Phase D.2 complete.", flush=True)
