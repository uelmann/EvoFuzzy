"""
Round F5 — stacked P2′ sleeve (pruned + context) and COMBO′ update.

BACKTEST ONLY. CPU only. Causal τ. Usage: modal run round_f5_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-round-f5"
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
    .add_local_python_source("baseline", "phase_d", "phase_d2", "round_f", "round_f5")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/roundF5_addendum.md", remote_path="/root/roundF5_addendum.md")
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 90, retries=0, volumes={"/data/quant": volume}, cpu=8, memory=32768)
def train_c3_fold_job(payload: dict) -> dict:
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


@app.function(timeout=60 * 60 * 8, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_round_f5() -> dict:
    import hashlib
    import shutil

    import numpy as np
    import pandas as pd

    from baseline.data import build_pit_topn, load_funding_panel, load_panel
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
    from phase_d2.metrics import pick_median_tau, summarize_port
    from round_f.constants import CTX_COLS, P1_H, P1_TAU, P2_H, P2_TAU
    from round_f.context import merge_context
    from round_f.eval import combo_from_sleeves, ic_tables_vs_a0
    from round_f5.constants import (
        COMBO_PRIME_CRITERION,
        P2_PRIME_COLS,
        PRUNED_COLS,
        SLEEVE_CRITERION,
    )
    from round_f5.eval import (
        apply_combo_prime,
        apply_sleeve_rule,
        ic_block_pass,
        stability_vs_incumbent,
        year_pos_flat,
    )
    from round_f5.report import (
        append_ledger,
        plot_combo,
        plot_sleeves,
        print_stdout,
        write_roundF5_report,
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
    addendum = Path("/root/roundF5_addendum.md").read_text()
    if SLEEVE_CRITERION not in addendum or COMBO_PRIME_CRITERION not in addendum:
        raise RuntimeError("Addendum missing verbatim criteria")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)
    print("[HB] BACKTEST ONLY; causal τ; zero GPU", flush=True)
    print("[HB] ledger+criteria present (frozen before results)", flush=True)
    print(f"[HB] P2′ n_feat={len(P2_PRIME_COLS)} pruned={PRUNED_COLS}", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    rf_dir = root / "round_f"
    f5_dir = root / "round_f5"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in (f5_dir, rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    port_cfg = cfg["portfolio"]
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    print(f"[HB] feat rows={len(feat)}", flush=True)

    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    window = int(cfg["data"]["exec_dv_window"])
    pit40 = (
        pd.read_parquet(uni_dir / "top40_pit.parquet")
        if (uni_dir / "top40_pit.parquet").exists()
        else build_pit_topn(panel, n=40, window=window)
    )
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

    ctx_path = rf_dir / "context.parquet"
    feat_rf = rf_dir / "features_round_f.parquet"
    if feat_rf.exists():
        feat_f = pd.read_parquet(feat_rf)
        feat_f["date"] = pd.to_datetime(feat_f["date"], utc=True)
        print(f"[HB] reuse features_round_f n={len(feat_f)}", flush=True)
    elif ctx_path.exists():
        ctx = pd.read_parquet(ctx_path)
        feat_f = merge_context(feat, ctx)
        print(f"[HB] merge cached context onto feat n={len(feat_f)}", flush=True)
    else:
        raise RuntimeError("Round F context cache missing; will not recompute catch22/context")
    missing_ctx = [c for c in CTX_COLS if c not in feat_f.columns]
    if missing_ctx:
        raise RuntimeError(f"context cols missing from cache: {missing_ctx}")
    missing_p2 = [c for c in P2_PRIME_COLS if c not in feat_f.columns]
    if missing_p2:
        raise RuntimeError(f"P2′ cols missing: {missing_p2}")
    print(f"[HB] ctx_cov={feat_f[CTX_COLS[0]].notna().mean():.3f}", flush=True)

    feat_c3_path = f5_dir / "features_p2prime.parquet"
    feat_f.to_parquet(feat_c3_path, index=False)
    volume.commit()

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

    pred_c3 = {}
    for h in HORIZONS:
        canon = f5_dir / f"lgbm_p2prime_h{h}.parquet"
        if canon.exists() and len(pd.read_parquet(canon)):
            pred_c3[h] = pd.read_parquet(canon)
            pred_c3[h]["date"] = pd.to_datetime(pred_c3[h]["date"], utc=True)
            print(f"[HB] reuse C3 h={h} n={len(pred_c3[h])}", flush=True)
            continue
        out_h = f5_dir / f"preds_C3_h{h}"
        out_h.mkdir(parents=True, exist_ok=True)
        payloads = [
            {
                "cfg": cfg,
                "feat_path": str(feat_c3_path),
                "out_dir": str(out_h),
                "fold_id": fr.fold_id,
                "train_start": str(fr.train_start),
                "train_end": str(fr.train_end),
                "purge_end": str(fr.purge_end),
                "embargo_end": str(fr.embargo_end),
                "val_start": str(fr.val_start),
                "val_end": str(fr.val_end),
                "horizon": h,
                "feature_cols": list(P2_PRIME_COLS),
                "model_name": "lgbm_p2prime",
            }
            for fr in folds_by_h[h]
        ]
        metas = list(train_c3_fold_job.map(payloads))
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
        pred_c3[h] = pdf
        pred_c3[h]["date"] = pd.to_datetime(pred_c3[h]["date"], utc=True)
        print(f"[HB] trained C3 h={h} n={len(pdf)}", flush=True)

    def _load_rf(name: str, h: int) -> pd.DataFrame:
        p = rf_dir / f"{name}_h{h}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing Round F pred cache {p}")
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    pred_c1 = {h: _load_rf("lgbm_f1_ctx", h) for h in HORIZONS}
    pred_c2 = {h: _load_rf("lgbm_f4_prune", h) for h in HORIZONS}
    print("[HB] reused F1/F4 prediction caches", flush=True)

    ic_tables, ic_nw = [], []
    ic_store = {}
    pairs = [
        ("C3_vs_A0", pred_a, pred_c3),
        ("C3_vs_C1", pred_c1, pred_c3),
        ("C3_vs_C2", pred_c2, pred_c3),
    ]
    for pair, pa, pb in pairs:
        for h in HORIZONS:
            blob = ic_tables_vs_a0(pa[h], pb[h], feat, pit40, h, "top40", folds_by_h[h])
            ic_store[(pair, h)] = blob
            for t in blob["tables"]:
                ic_tables.append({"pair": pair, **t})
            for w, pnw in blob["paired_nw"].items():
                ic_nw.append(
                    {
                        "pair": pair,
                        "horizon": h,
                        "window": w,
                        **pnw,
                        "frac_pos_trail18m": blob.get("frac_pos_trail18m") if w == "trail18m" else None,
                    }
                )
            print(
                f"[HB] IC {pair} h={h} d18={blob.get('delta_trail18m')} dfull={blob.get('delta_full')}",
                flush=True,
            )

    c3_ic = ic_block_pass(ic_store[("C3_vs_A0", 7)], ic_store[("C3_vs_A0", 10)])
    print(f"[HB] C3 IC gate {c3_ic}", flush=True)

    def _port(preds, h, uni, tau_pct, tiered, cap):
        print(f"[HB] port h={h} τ={tau_pct} tiered={tiered}", flush=True)
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

    def _enrich(raw, common_idx=None):
        s = summarize_port(raw, common_idx=common_idx)
        s["year_pos_flat"] = year_pos_flat(s)
        return s

    p1_raw = _port(pred_a[P1_H], P1_H, pit20, P1_TAU, False, None)
    p1 = _enrich(p1_raw)
    print(f"[HB] P1 full={p1.get('net_sharpe_full')} trail={p1.get('net_sharpe_trail18m')}", flush=True)

    specs = {
        "C0": dict(preds=pred_a[P2_H], scan=False, tau=P2_TAU),
        "C1": dict(preds=pred_c1[P2_H], scan=True, tau=None),
        "C2": dict(preds=pred_c2[P2_H], scan=True, tau=None),
        "C3": dict(preds=pred_c3[P2_H], scan=True, tau=None),
    }
    raw_ports = {}
    for cid, sp in specs.items():
        if sp["scan"]:
            runs = [_port(sp["preds"], P2_H, pit40, tp, True, LIQ_CAP_ADV_FRAC) for tp in TAU_PCTS]
            raw_ports[cid] = pick_median_tau(runs)
        else:
            raw_ports[cid] = _port(sp["preds"], P2_H, pit40, sp["tau"], True, LIQ_CAP_ADV_FRAC)
        print(f"[HB] {cid} τ={raw_ports[cid].get('tau_pct')} sharpe={raw_ports[cid].get('net_sharpe')}", flush=True)

    common = raw_ports["C0"]["daily_ret"].index
    for cid in ("C1", "C2", "C3"):
        common = common.intersection(raw_ports[cid]["daily_ret"].index)
    common = pd.DatetimeIndex(pd.to_datetime(common, utc=True))
    cands = {cid: _enrich(raw_ports[cid], common_idx=common) for cid in specs}
    p1_idx = p1["daily_ret"].index.intersection(common)
    p1 = _enrich(p1_raw, common_idx=p1_idx)

    sleeve_v = apply_sleeve_rule(cands, c3_ic)
    selected_id = sleeve_v["selected"]
    print(f"[HB] sleeve {sleeve_v['verdict']} selected={selected_id}", flush=True)

    stability = stability_vs_incumbent(cands[selected_id], cands["C0"])
    print(f"[HB] stability sel_avg_n={stability.get('sel_avg_n_pos')} dpos={stability.get('sel_dpos')}", flush=True)

    combo_f = combo_from_sleeves(p1, cands["C0"])
    combo_p = combo_from_sleeves(p1, cands[selected_id])
    combo_v = apply_combo_prime(combo_p)
    print(f"[HB] COMBO′ {combo_v}", flush=True)

    plot_sleeves(cands, chart_dir / "roundF5_sleeves.png")
    plot_combo(combo_f, combo_p, p1, chart_dir / "roundF5_combo.png")

    sel = cands[selected_id]
    sel_by = sel.get("net_sharpe_by_year") or {}
    names = {"C0": "A0", "C1": "A0+context", "C2": "A0-pruned", "C3": "A0-pruned+context"}
    sel_ledger = {
        "row": f"P2-{selected_id}",
        "status": "ADOPTED P2 sleeve" if selected_id != "C0" else "ADOPTED P2 sleeve (incumbent)",
        "model": names[selected_id],
        "universe": "top-40",
        "h": 10,
        "tau": sel.get("tau_pct"),
        "full": sel.get("net_sharpe_full"),
        "trail18m": sel.get("net_sharpe_trail18m"),
        "y2022": sel_by.get(2022),
        "y2023": sel_by.get(2023),
        "y2024": sel_by.get(2024),
        "y2025": sel_by.get(2025),
        "y2026": sel_by.get(2026),
        "gross": sel.get("gross_total_pnl"),
        "cost": sel.get("cost_drag"),
        "funding": sel.get("funding_total_pnl"),
        "hedge": sel.get("hedge_total_pnl"),
        "avg_n_pos": sel.get("avg_n_positions"),
        "pct_flat": sel.get("pct_flat_days"),
        "ann_to": sel.get("ann_turnover"),
    }
    ref_is_prime = combo_v.get("pass")
    cref = combo_p if ref_is_prime else combo_f
    cby = cref.get("net_sharpe_by_year") or {}
    combo_ledger = {
        "row": "COMBO′" if ref_is_prime else "COMBO",
        "status": "ADOPTED reference",
        "model": f"50/50 P1+{selected_id}" if ref_is_prime else "50/50 P1+C0",
        "universe": "mixed",
        "h": "7+10",
        "tau": "causal",
        "full": cref.get("net_sharpe_full"),
        "trail18m": cref.get("net_sharpe_trail18m"),
        "y2022": cby.get(2022),
        "y2023": cby.get(2023),
        "y2024": cby.get(2024),
        "y2025": cby.get(2025),
        "y2026": cby.get(2026),
        "gross": float("nan"),
        "cost": float("nan"),
        "funding": float("nan"),
        "hedge": float("nan"),
        "avg_n_pos": float("nan"),
        "pct_flat": float("nan"),
        "ann_to": cref.get("ann_turnover"),
    }
    changelog = (
        f"Round F5: P2 sleeve {sleeve_v['verdict']} → {selected_id} "
        f"(full={float(sel_ledger['full']):.3f} trail={float(sel_ledger['trail18m']):.3f}); "
        f"COMBO′ {combo_v.get('verdict')} → reference {combo_v.get('reference')} "
        f"(full={float(combo_ledger['full']):.3f} trail={float(combo_ledger['trail18m']):.3f})."
    )
    ledger_src = Path("/root/numbers_ledger.md")
    ledger_vol = rep_dir / "numbers_ledger.md"
    shutil.copy2(ledger_src, ledger_vol)
    ledger_text = append_ledger(ledger_vol, sel_ledger, combo_ledger, changelog)

    extra = {
        "sel_full": sel.get("net_sharpe_full"),
        "sel_trail": sel.get("net_sharpe_trail18m"),
        "sel_by_year": sel_by,
        "sel_to": sel.get("ann_turnover"),
        "p2_prime_cols": list(P2_PRIME_COLS),
        "n_feat": len(P2_PRIME_COLS),
    }
    combo_f_rep = {k: v for k, v in combo_f.items() if k not in ("daily_ret", "p1_equity", "p2_equity")}
    combo_p_rep = {k: v for k, v in combo_p.items() if k not in ("daily_ret", "p1_equity", "p2_equity")}
    write_roundF5_report(
        rep_dir / "roundF5_report.md",
        frozen_hash=calc,
        gates=gates,
        cands=cands,
        ic_tables=ic_tables,
        ic_nw=ic_nw,
        c3_ic=c3_ic,
        sleeve_v=sleeve_v,
        stability=stability,
        combo_f=combo_f_rep,
        combo_p=combo_p_rep,
        combo_v=combo_v,
        ledger_diff="\n".join(ledger_text.splitlines()[-20:]),
        extra=extra,
    )
    print_stdout(sleeve_v, combo_v, cands)

    drop_keys = {
        "equity",
        "daily_ret",
        "daily_gross",
        "daily_hedge",
        "daily_cost",
        "daily_funding",
        "daily_n_pos",
        "daily_flat",
        "p1_equity",
        "p2_equity",
        "delta_daily",
    }

    def _jsonable(x):
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items() if k not in drop_keys}
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
        if isinstance(x, pd.Series):
            return None
        return x

    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "p2_prime_n_feat": len(P2_PRIME_COLS),
        "sleeve": sleeve_v,
        "combo": combo_v,
        "c3_ic": c3_ic,
        "cand_sharpes": {
            k: {
                "full": v.get("net_sharpe_full"),
                "trail18m": v.get("net_sharpe_trail18m"),
                "tau": v.get("tau_pct"),
            }
            for k, v in cands.items()
        },
        "stability": {
            k: v
            for k, v in stability.items()
            if k not in ("sel_dpos", "inc_dpos")
        }
        | {"sel_dpos": stability.get("sel_dpos"), "inc_dpos": stability.get("inc_dpos")},
        "ic_tables": ic_tables,
        "ic_nw": ic_nw,
        "gates": gates,
        "changelog": changelog,
        "elapsed_sec": time.time() - t_pipe,
        "tau_mode": "fold_train",
    }
    (rep_dir / "roundF5_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    volume.commit()
    print(f"[HB] DONE elapsed={time.time()-t_pipe:.1f}s", flush=True)
    return {
        "frozen_sha256": calc,
        "gpu_used": False,
        "selected_sleeve": selected_id,
        "sleeve_verdict": sleeve_v.get("verdict"),
        "combo_verdict": combo_v.get("verdict"),
        "reference": combo_v.get("reference"),
        "elapsed_sec": time.time() - t_pipe,
    }


@app.local_entrypoint()
def main():
    print("[local] starting Round F5 (CPU, backtest-only)...", flush=True)
    summary = run_round_f5.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/roundF5_report.md", "roundF5_report.md", "reports"),
        ("reports/roundF5_summary.json", "roundF5_summary.json", "reports"),
        ("reports/numbers_ledger.md", "numbers_ledger.md", "reports"),
        ("charts/roundF5_sleeves.png", "roundF5_sleeves.png", "charts"),
        ("charts/roundF5_combo.png", "roundF5_combo.png", "charts"),
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
        for src in (art / "reports").glob("roundF5*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
        for src in (art / "charts").glob("roundF5*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] Round F5 complete.", flush=True)
