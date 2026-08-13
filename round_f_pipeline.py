"""
Round F — context / complexity / pruning ablations + two-sleeve combo.

BACKTEST ONLY. CPU only. Causal τ. Usage: modal run round_f_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-round-f"
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
        "pycatch22",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "phase_d", "phase_d2", "round_f")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/roundF_addendum.md", remote_path="/root/roundF_addendum.md")
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 120, retries=0, volumes={"/data/quant": volume}, cpu=2, memory=8192)
def complexity_symbol_job(payload: dict) -> dict:
    import pandas as pd

    from round_f.complexity import complexity_for_symbol

    resid = pd.read_parquet(payload["resid_path"])
    resid["date"] = pd.to_datetime(resid["date"], utc=True)
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = []
    for s in payload["symbols"]:
        dest = out_dir / f"{s}.parquet"
        if dest.exists():
            done.append(s)
            continue
        g = resid[resid["symbol"] == s].copy()
        if g.empty:
            continue
        out = complexity_for_symbol(g)
        out.to_parquet(dest, index=False)
        done.append(s)
        print(f"[HB] cx {s} n={len(out)}", flush=True)
    volume.commit()
    return {"n_sym": len(done), "elapsed": time.time() - t0}


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def train_f_fold_job(payload: dict) -> dict:
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
        model_name=payload["model_name"],
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
        f"[HB] {payload['model_name']} h={fold.horizon} fold={fold.fold_id} "
        f"n={len(pred_df)} elapsed={meta['wall_elapsed']:.1f}s",
        flush=True,
    )
    return meta


@app.function(timeout=60 * 60 * 10, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_round_f() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.data import build_pit_topn, load_funding_panel, load_panel
    from baseline.features import FEATURE_COLS
    from baseline.gates import run_all_gates
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from phase_d2.constants import (
        FEE_BPS_NEXT,
        FEE_BPS_TOP,
        HORIZONS,
        LIQ_CAP_ADV_FRAC,
        NOMINAL_BOOK_USD,
        SLIP_BPS_NEXT,
        SLIP_BPS_TOP,
        TAU_PCTS,
    )
    from phase_d2.metrics import pick_median_tau, slim_port, summarize_port
    from round_f.complexity import apply_cs_z_cx, merge_complexity
    from round_f.constants import (
        COMBO_CRITERION,
        CTX_COLS,
        CX_COLS,
        KEEP_CRITERION,
        N_PRUNE,
        P1_H,
        P1_TAU,
        P2_H,
        P2_TAU,
    )
    from round_f.context import build_context_block, merge_context, residual_log_returns
    from round_f.eval import (
        apply_combo_criterion,
        apply_keep,
        combo_from_sleeves,
        ic_tables_vs_a0,
        rank_a0_gains,
    )
    from round_f.report import plot_combo, plot_ic, print_stdout, write_roundF_report

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    addendum = Path("/root/roundF_addendum.md").read_text()
    if KEEP_CRITERION not in addendum or COMBO_CRITERION not in addendum:
        raise RuntimeError("Addendum missing verbatim criteria")
    ledger = Path("/root/numbers_ledger.md").read_text()
    if "1.401" not in ledger or "0.757" not in ledger:
        raise RuntimeError("Ledger missing deprecated pooled-τ footnote")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)
    print("[HB] BACKTEST ONLY; causal τ; zero GPU", flush=True)
    print("[HB] ledger+criteria present (frozen before results)", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    phase_dir = root / "round_f"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in [phase_dir, rep_dir, chart_dir, phase_dir / "cx_sym"]:
        d.mkdir(parents=True, exist_ok=True)

    port_cfg = cfg["portfolio"]
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    print(f"[HB] feat rows={len(feat)}", flush=True)

    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    window = int(cfg["data"]["exec_dv_window"])
    pit40 = pd.read_parquet(uni_dir / "top40_pit.parquet") if (uni_dir / "top40_pit.parquet").exists() else build_pit_topn(panel, n=40, window=window)
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")
    for u in (pit20, pit40, pit120):
        u["date"] = pd.to_datetime(u["date"], utc=True)
    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    funding = load_funding_panel(fund_dir, ever)

    pred_a = {}
    for h in HORIZONS:
        p = pd.read_parquet(pred_dir / f"lgbm_price_only_h{h}.parquet")
        p["date"] = pd.to_datetime(p["date"], utc=True)
        pred_a[h] = p

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
    print("[HB] gates...", flush=True)
    gates = run_all_gates(panel, feat, build_pit_topn, folds7[0], cfg, sample)
    if not all(g.get("passed") for g in gates):
        raise RuntimeError(f"gates failed: {gates}")
    print("[HB] gates OK", flush=True)

    ctx_path = phase_dir / "context.parquet"
    if ctx_path.exists():
        ctx = pd.read_parquet(ctx_path)
        print(f"[HB] reuse context n={len(ctx)}", flush=True)
    else:
        ctx = build_context_block(panel, feat, pit120, pit40, pred_a[7], funding)
        ctx.to_parquet(ctx_path, index=False)
        volume.commit()

    resid_path = phase_dir / "residuals.parquet"
    if resid_path.exists():
        print("[HB] reuse residuals", flush=True)
    else:
        resid = residual_log_returns(panel, feat)
        resid = resid[resid["symbol"].isin(set(feat["symbol"].unique()) | {"BTCUSDT"})]
        resid.to_parquet(resid_path, index=False)
        volume.commit()

    cx_path = phase_dir / "complexity.parquet"
    if cx_path.exists():
        cx = pd.read_parquet(cx_path)
        print(f"[HB] reuse complexity n={len(cx)}", flush=True)
    else:
        resid = pd.read_parquet(resid_path)
        symbols = sorted(resid["symbol"].unique())
        todo = [s for s in symbols if not (phase_dir / "cx_sym" / f"{s}.parquet").exists()]
        print(f"[HB] complexity map n_sym={len(symbols)} todo={len(todo)}", flush=True)
        if todo:
            waves = [todo[i : i + 25] for i in range(0, len(todo), 25)]
            payloads = [
                {"resid_path": str(resid_path), "out_dir": str(phase_dir / "cx_sym"), "symbols": w}
                for w in waves
            ]
            list(complexity_symbol_job.map(payloads))
            volume.reload()
        parts = []
        for s in symbols:
            p = phase_dir / "cx_sym" / f"{s}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
        cx = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        cx["date"] = pd.to_datetime(cx["date"], utc=True)
        cx = apply_cs_z_cx(cx)
        cx.to_parquet(cx_path, index=False)
        volume.commit()
        print(f"[HB] complexity n={len(cx)}", flush=True)

    feat_f = merge_context(feat, ctx)
    feat_f = merge_complexity(feat_f, cx)
    feat_f_path = phase_dir / "features_round_f.parquet"
    feat_f.to_parquet(feat_f_path, index=False)
    volume.commit()
    print(f"[HB] feat_f cols={len(feat_f.columns)} ctx_cov={feat_f[CTX_COLS[0]].notna().mean():.3f}", flush=True)

    ranked, n_metas = rank_a0_gains(root)
    if len(ranked) < N_PRUNE:
        raise RuntimeError(f"could not rank A0 gains n_metas={n_metas} ranked={len(ranked)}")
    pruned = ranked[:N_PRUNE]
    dropped = [c for c, _ in pruned]
    print(f"[HB] F4 drop {dropped} from n_metas={n_metas}", flush=True)

    models = {
        "F1": {"cols": list(FEATURE_COLS) + list(CTX_COLS), "name": "lgbm_f1_ctx"},
        "F2": {"cols": list(FEATURE_COLS) + list(CX_COLS), "name": "lgbm_f2_cx"},
        "F3": {"cols": list(FEATURE_COLS) + list(CTX_COLS) + list(CX_COLS), "name": "lgbm_f3_both"},
        "F4": {"cols": [c for c in FEATURE_COLS if c not in dropped], "name": "lgbm_f4_prune"},
    }
    missing = [c for c in CTX_COLS + CX_COLS if c not in feat_f.columns]
    if missing:
        raise RuntimeError(f"missing feature cols: {missing}")

    folds_by_h = {
        h: make_folds(
            pd.DatetimeIndex(feat_f["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        for h in HORIZONS
    }

    pred_f = {m: {} for m in models}
    for mid, spec in models.items():
        for h in HORIZONS:
            canon = phase_dir / f"{spec['name']}_h{h}.parquet"
            if canon.exists() and len(pd.read_parquet(canon)):
                pred_f[mid][h] = pd.read_parquet(canon)
                pred_f[mid][h]["date"] = pd.to_datetime(pred_f[mid][h]["date"], utc=True)
                print(f"[HB] reuse {mid} h={h} n={len(pred_f[mid][h])}", flush=True)
                continue
            out_h = phase_dir / f"preds_{mid}_h{h}"
            out_h.mkdir(parents=True, exist_ok=True)
            payloads = [
                {
                    "cfg": cfg,
                    "feat_path": str(feat_f_path),
                    "out_dir": str(out_h),
                    "fold_id": fr.fold_id,
                    "train_start": str(fr.train_start),
                    "train_end": str(fr.train_end),
                    "purge_end": str(fr.purge_end),
                    "embargo_end": str(fr.embargo_end),
                    "val_start": str(fr.val_start),
                    "val_end": str(fr.val_end),
                    "horizon": h,
                    "feature_cols": spec["cols"],
                    "model_name": spec["name"],
                }
                for fr in folds_by_h[h]
            ]
            metas = list(train_f_fold_job.map(payloads))
            volume.reload()
            preds = [
                pd.read_parquet(m["pred_path"])
                for m in metas
                if m.get("pred_path") and Path(m["pred_path"]).exists()
            ]
            pdf = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
            if not pdf.empty:
                pdf = pdf.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(["date", "symbol"], keep="first")
                pdf.to_parquet(canon, index=False)
            (out_h / f"fold_meta_h{h}.json").write_text(json.dumps(metas, indent=2, default=str))
            volume.commit()
            pred_f[mid][h] = pdf
            pred_f[mid][h]["date"] = pd.to_datetime(pred_f[mid][h]["date"], utc=True)
            print(f"[HB] trained {mid} h={h} n={len(pdf)}", flush=True)

    # IC
    uni_map = {"top20": pit20, "top40": pit40}
    ic_store = {}
    ic_tables, ic_nw = [], []
    delta_plot = {m: {} for m in models}
    for mid in models:
        for uni_name, uni in uni_map.items():
            for h in HORIZONS:
                blob = ic_tables_vs_a0(pred_a[h], pred_f[mid][h], feat, uni, h, uni_name, folds_by_h[h])
                ic_store[(mid, uni_name, h)] = blob
                for t in blob["tables"]:
                    ic_tables.append({"block": mid, **t})
                for w, p in blob["paired_nw"].items():
                    ic_nw.append(
                        {
                            "block": mid,
                            "horizon": h,
                            "universe": uni_name,
                            "window": w,
                            **p,
                            "frac_pos_trail18m": blob.get("frac_pos_trail18m") if w == "trail18m" else None,
                        }
                    )
                if uni_name == "top20":
                    delta_plot[mid][h] = blob.get("delta_daily")
                print(
                    f"[HB] IC {mid} {uni_name} h={h} d18={blob.get('delta_trail18m')} dfull={blob.get('delta_full')}",
                    flush=True,
                )

    def _port(preds, h, uni, tau_pct, tiered, cap):
        print(f"[HB] port h={h} uni={uni.shape[0] if hasattr(uni,'shape') else '?'} τ={tau_pct} tiered={tiered}", flush=True)
        return run_tranche_portfolio(
            preds,
            panel,
            feat,
            uni,
            horizon=h,
            tau_pct=float(tau_pct),
            exit_hysteresis=port_cfg.get("exit_hysteresis", 0.6),
            gross_limit=port_cfg.get("gross_limit", 1.0),
            fee_bps=FEE_BPS_TOP,
            slip_bps=SLIP_BPS_TOP,
            lag=0,
            apply_funding=True,
            funding=funding,
            tau_mode="fold_train",
            folds=folds_by_h[h],
            tiered_costs=bool(tiered),
            fee_bps_next=FEE_BPS_NEXT,
            slip_bps_next=SLIP_BPS_NEXT,
            liq_cap_adv_frac=cap,
            nominal_book_usd=NOMINAL_BOOK_USD,
            rank_universe=pit40,
        )

    books = {
        "top20": dict(h=P1_H, uni=pit20, tau_a0=P1_TAU, tiered=False, cap=None, label="P1"),
        "top40": dict(h=P2_H, uni=pit40, tau_a0=P2_TAU, tiered=True, cap=LIQ_CAP_ADV_FRAC, label="P2"),
    }

    a0_port = {}
    for uni_name, bk in books.items():
        res = _port(pred_a[bk["h"]], bk["h"], bk["uni"], bk["tau_a0"], bk["tiered"], bk["cap"])
        a0_port[uni_name] = summarize_port(res)

    f_port = {}
    port_rows = []
    port_d18 = {m: {} for m in models}
    for mid in models:
        for uni_name, bk in books.items():
            runs = []
            for tp in TAU_PCTS:
                runs.append(_port(pred_f[mid][bk["h"]], bk["h"], bk["uni"], tp, bk["tiered"], bk["cap"]))
            picked = pick_median_tau(runs)
            fs = summarize_port(picked)
            # identical days vs A0
            idx = a0_port[uni_name]["daily_ret"].index.intersection(fs["daily_ret"].index)
            a0s = summarize_port(a0_port[uni_name], common_idx=idx)
            fs2 = summarize_port(picked, common_idx=idx)
            d18 = float(fs2["net_sharpe_trail18m"] - a0s["net_sharpe_trail18m"])
            port_d18[mid][uni_name] = d18
            f_port[(mid, uni_name)] = fs2
            port_rows.append(
                {
                    "block": mid,
                    "book": f"{bk['label']} {uni_name} h={bk['h']}",
                    "tau_a0": bk["tau_a0"],
                    "tau_f": fs2.get("tau_pct"),
                    "a0_full": a0s.get("net_sharpe_full"),
                    "f_full": fs2.get("net_sharpe_full"),
                    "a0_trail18m": a0s.get("net_sharpe_trail18m"),
                    "f_trail18m": fs2.get("net_sharpe_trail18m"),
                    "delta_trail18m": d18,
                }
            )
            print(f"[HB] port Δ {mid} {uni_name} d18={d18:.3f}", flush=True)

    keep = {}
    for mid in models:
        ic_u = {(uni, h): ic_store[(mid, uni, h)] for uni in uni_map for h in HORIZONS}
        keep[mid] = apply_keep(mid, ic_u, port_d18[mid], prune=(mid == "F4"))

    combo = combo_from_sleeves(a0_port["top20"], a0_port["top40"])
    combo["p1_plot"] = a0_port["top20"]
    combo["p2_plot"] = a0_port["top40"]
    combo_v = apply_combo_criterion(combo, COMBO_CRITERION)
    print(f"[HB] combo {combo_v}", flush=True)

    plot_ic(delta_plot, chart_dir / "roundF_ic.png")
    plot_combo(combo, chart_dir / "roundF_combo.png")
    extra = {
        "n_a0_metas": n_metas,
        "a0_gain_bottom_to_top": ranked,
        "dropped": dropped,
        "hurst": "single-scale R/S",
    }
    write_roundF_report(
        rep_dir / "roundF_report.md",
        frozen_hash=calc,
        gates=gates,
        pruned=pruned,
        ic_tables=ic_tables,
        ic_nw=ic_nw,
        port_rows=port_rows,
        keep=keep,
        combo={k: v for k, v in combo.items() if k not in ("daily_ret", "p1_plot", "p2_plot")},
        combo_v=combo_v,
        extra=extra,
    )
    # restore equity for json skip
    print_stdout(keep, combo_v)

    def _jsonable(x):
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items() if k not in ("equity", "daily_ret", "delta_daily", "p1_plot", "p2_plot", "p1_equity", "p2_equity")}
        if isinstance(x, list):
            return [_jsonable(v) for v in x]
        if isinstance(x, tuple):
            return [_jsonable(v) for v in x]
        if isinstance(x, pd.Timestamp):
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
        "pruned": pruned,
        "n_a0_metas": n_metas,
        "port_rows": port_rows,
        "keep": keep,
        "combo": {k: combo[k] for k in combo if k not in ("daily_ret", "equity", "p1_plot", "p2_plot", "p1_equity", "p2_equity")},
        "combo_verdict": combo_v,
        "ic_tables": ic_tables,
        "ic_nw": ic_nw,
        "gates": gates,
        "elapsed_sec": time.time() - t_pipe,
        "tau_mode": "fold_train",
    }
    (rep_dir / "roundF_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    volume.commit()
    print(f"[HB] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return {
        "frozen_sha256": calc,
        "gpu_used": False,
        "keep": {m: {u: v["verdict"] for u, v in blob["by_universe"].items()} for m, blob in keep.items()},
        "combo": combo_v.get("verdict"),
        "elapsed_sec": time.time() - t_pipe,
    }


@app.local_entrypoint()
def main():
    print("[local] starting Round F (CPU, backtest-only)...", flush=True)
    summary = run_round_f.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/roundF_report.md", "roundF_report.md", "reports"),
        ("reports/roundF_summary.json", "roundF_summary.json", "reports"),
        ("charts/roundF_ic.png", "roundF_ic.png", "charts"),
        ("charts/roundF_combo.png", "roundF_combo.png", "charts"),
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
        for src in (art / "reports").glob("roundF*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("roundF*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] Round F complete.", flush=True)
