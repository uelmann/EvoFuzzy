"""
Symmetry audit of frozen A0 scores.

ANALYSIS ONLY. CPU only. Frozen predictions/labels reused.
Usage: modal run symmetry_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-symmetry"
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
    .add_local_python_source("baseline", "phase_d", "phase_d2", "round_f", "longonly", "symmetry")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/symmetry_addendum.md", remote_path="/root/symmetry_addendum.md")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 60 * 3, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_symmetry() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from longonly.constants import (
        P1_COST_BPS,
        P1_H,
        P1_SLIP_BPS,
        P1_TAU,
        P2_H,
        P2_LIQ_CAP,
        P2_NOM_USD,
        P2_TAU,
    )
    from longonly.eval import btc_bh_simple, enrich_combo
    from phase_d2.constants import FEE_BPS_NEXT, FEE_BPS_TOP, SLIP_BPS_NEXT, SLIP_BPS_TOP
    from phase_d2.metrics import summarize_port
    from symmetry.constants import (
        CLASSIFICATION_CRITERION,
        FROZEN_A0_SHA256,
        HORIZONS,
        N_BUCKETS,
        PRED_H10,
        PRED_H10_SHA256,
        PRED_H7,
        PRED_H7_SHA256,
        UNIVERSES,
    )
    from symmetry.eval import (
        apply_classification,
        bucket_curve_for_window,
        cell_spreads_and_label,
        daily_bucket_panel,
        daily_spreads,
        ew_pit_simple,
        lo_alpha_vs_benches,
        long_pick_quality,
        tide_tables,
    )
    from symmetry.report import plot_buckets, plot_tide, print_stdout, write_report

    def _sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    if calc != FROZEN_A0_SHA256:
        raise RuntimeError(f"Frozen A0 SHA256 constant drift: {calc}")
    addendum = Path("/root/symmetry_addendum.md").read_text()
    if CLASSIFICATION_CRITERION not in addendum:
        raise RuntimeError("Addendum missing verbatim classification")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)
    print("[HB] ANALYSIS ONLY; scores/labels reused; zero GPU", flush=True)
    print(f"[HB] {CLASSIFICATION_CRITERION}", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    sy_dir = root / "symmetry"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in (sy_dir, rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    pred_h7 = Path(PRED_H7) if Path(PRED_H7).exists() else root / "predictions" / "lgbm_price_only_h7.parquet"
    pred_h10 = Path(PRED_H10) if Path(PRED_H10).exists() else root / "predictions" / "lgbm_price_only_h10.parquet"
    h7s, h10s = _sha256_file(pred_h7), _sha256_file(pred_h10)
    if h7s != PRED_H7_SHA256 or h10s != PRED_H10_SHA256:
        raise RuntimeError(f"Prediction hash mismatch h7={h7s} h10={h10s}")
    pred_hashes = {"h7": h7s, "h10": h10s}
    print(f"[HB] pred hashes OK h7={h7s} h10={h10s}", flush=True)

    port_cfg = cfg["portfolio"]
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    print(f"[HB] feat rows={len(feat)} labels={('y_h7' in feat.columns, 'y_h10' in feat.columns)}", flush=True)

    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit40 = pd.read_parquet(uni_dir / "top40_pit.parquet")
    pit120 = pd.read_parquet(uni_dir / "top120_pit.parquet")
    for u in (pit20, pit40, pit120):
        u["date"] = pd.to_datetime(u["date"], utc=True)
    uni_map = {"top20": pit20, "top40": pit40, "top120": pit120}

    preds = {}
    for h, path in ((7, pred_h7), (10, pred_h10)):
        p = pd.read_parquet(path)
        p["date"] = pd.to_datetime(p["date"], utc=True)
        ycol = f"y_h{h}"
        if ycol not in p.columns:
            p = p.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        preds[h] = p
        print(f"[HB] pred h={h} n={len(p)} y_cov={p[ycol].notna().mean():.3f}", flush=True)

    years = list(range(2022, 2027))
    windows = ["full", "trail18m"] + [f"y{y}" for y in years]
    curves = {}
    cells = {}
    tides = {}
    spread_oneliners = []

    for h in HORIZONS:
        ycol = f"y_h{h}"
        for uni in UNIVERSES:
            print(f"[HB] buckets h={h} {uni}", flush=True)
            u = uni_map[uni][["date", "symbol"]].drop_duplicates()
            df = preds[h].merge(u, on=["date", "symbol"], how="inner")
            nb = N_BUCKETS[uni]
            panel_b = daily_bucket_panel(df, ycol, nb)
            sp = daily_spreads(panel_b, nb)
            blob = cell_spreads_and_label(sp, lag=h)
            cells[(h, uni)] = blob
            tides[(h, uni)] = tide_tables(sp)
            for w in windows:
                curves[(h, uni, w)] = bucket_curve_for_window(panel_b, nb, w)
            t = blob["top"]["full"]
            btm = blob["bottom"]["full"]
            print(
                f"[HB] h={h} {uni} TOP={t['mean']:.4f} t={t['nw_t']:.2f} "
                f"BOT={btm['mean']:.4f} t={btm['nw_t']:.2f} ratio={blob['ratio']['full']:.3f} "
                f"{'PASS' if blob['pass'] else 'FAIL'}",
                flush=True,
            )

    classification = apply_classification(cells)
    print(f"[HB] LABEL {classification['label']} n7={classification['n_pass_h7']} n10={classification['n_pass_h10']}", flush=True)

    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    funding = load_funding_panel(fund_dir, ever)
    folds_by_h = {
        h: make_folds(
            pd.DatetimeIndex(feat["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        for h in (P1_H, P2_H)
    }

    def _port(pred_df, h, uni, tau_pct, tiered, cap, long_only, hedge, holdings=False):
        print(
            f"[HB] port h={h} τ={tau_pct} lo={long_only} hedge={hedge} holdings={holdings}",
            flush=True,
        )
        return run_tranche_portfolio(
            pred_df,
            panel,
            feat,
            uni,
            horizon=h,
            tau_pct=float(tau_pct),
            exit_hysteresis=port_cfg.get("exit_hysteresis", 0.6),
            gross_limit=port_cfg.get("gross_limit", 1.0),
            fee_bps=FEE_BPS_TOP if h != P1_H else P1_COST_BPS,
            slip_bps=SLIP_BPS_TOP if h != P1_H else P1_SLIP_BPS,
            lag=0,
            apply_funding=True,
            funding=funding,
            tau_mode="fold_train",
            folds=folds_by_h[h],
            tiered_costs=bool(tiered),
            fee_bps_next=FEE_BPS_NEXT,
            slip_bps_next=SLIP_BPS_NEXT,
            liq_cap_adv_frac=cap,
            nominal_book_usd=P2_NOM_USD,
            rank_universe=pit40,
            long_only=bool(long_only),
            apply_beta_hedge=bool(hedge),
            record_holdings=bool(holdings),
        )

    raw = {
        "LS_A": _port(preds[7], P1_H, pit20, P1_TAU, False, None, False, True, True),
        "LS_B": _port(preds[10], P2_H, pit40, P2_TAU, True, P2_LIQ_CAP, False, True, True),
        "LOH_A": _port(preds[7], P1_H, pit20, P1_TAU, False, None, True, True, False),
        "LOH_B": _port(preds[10], P2_H, pit40, P2_TAU, True, P2_LIQ_CAP, True, True, False),
        "LOU_A": _port(preds[7], P1_H, pit20, P1_TAU, False, None, True, False, False),
        "LOU_B": _port(preds[10], P2_H, pit40, P2_TAU, True, P2_LIQ_CAP, True, False, False),
    }
    common = raw["LS_A"]["daily_ret"].index
    for k in raw:
        common = common.intersection(raw[k]["daily_ret"].index)
    common = pd.DatetimeIndex(pd.to_datetime(common, utc=True))
    ports = {k: summarize_port(v, common_idx=common) for k, v in raw.items()}
    combo_loh = enrich_combo(ports["LOH_A"], ports["LOH_B"])
    combo_lou = enrich_combo(ports["LOU_A"], ports["LOU_B"])

    btc = btc_bh_simple(panel).reindex(common)
    ew20 = ew_pit_simple(panel, pit20, "ew_top20").reindex(common)
    ew40 = ew_pit_simple(panel, pit40, "ew_top40").reindex(common)
    benches = {"EW top-20": ew20, "EW top-40": ew40, "BTC B&H": btc}
    lo_alpha = {
        "COMBO-LO-H": lo_alpha_vs_benches(combo_loh["daily_ret"], benches, lag=10),
        "COMBO-LO-U": lo_alpha_vs_benches(combo_lou["daily_ret"], benches, lag=10),
    }

    lab20 = preds[7].merge(pit20[["date", "symbol"]], on=["date", "symbol"], how="inner")
    lab40 = preds[10].merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
    picks = {
        "Sleeve A (top-20 h=7)": long_pick_quality(raw["LS_A"].get("daily_long_names") or [], lab20, "y_h7"),
        "Sleeve B (top-40 h=10)": long_pick_quality(raw["LS_B"].get("daily_long_names") or [], lab40, "y_h10"),
    }

    def _conclusion() -> str:
        lab = classification["label"]
        # representative cells
        c40 = cells[(7, "top40")]
        rho = curves.get((7, "top40", "full"), {}).get("spearman")
        a_h = lo_alpha["COMBO-LO-H"]["EW top-40"]["full"]
        a_u = lo_alpha["COMBO-LO-U"]["EW top-40"]["full"]
        pa = picks["Sleeve A (top-20 h=7)"]
        pb = picks["Sleeve B (top-40 h=10)"]
        if lab == "SYMMETRIC":
            head = (
                "Yes — the model predicts outperformance, not only underperformance. "
                "The engine is labeled SYMMETRIC: on at least one horizon the TOP spread "
                "is positive and significant on at least two universes, with a symmetry "
                "ratio of at least 0.4."
            )
        else:
            head = (
                "Mostly no — discrimination lives on the losing tail. The engine is labeled "
                "LONG-SIDE GAP: it does not jointly clear a significant TOP spread and a "
                "symmetry ratio ≥ 0.4 on two universes at either horizon."
            )
        return (
            f"{head} Bucket curves at h=7 top-40 have Spearman ρ={rho:.3f} "
            f"(rank vs mean residual). TOP spread there is {c40['top']['full']['mean']:.4f} "
            f"(NW-t {c40['top']['full']['nw_t']:.2f}) vs BOTTOM {c40['bottom']['full']['mean']:.4f} "
            f"(t {c40['bottom']['full']['nw_t']:.2f}), ratio {c40['ratio']['full']:.3f}. "
            f"Against the fair EW top-40 basket, COMBO-LO-H alpha is {a_h['alpha_ann']:.3f} "
            f"(NW-t {a_h['nw_t_alpha']:.2f}) and COMBO-LO-U alpha is {a_u['alpha_ann']:.3f} "
            f"(t {a_u['nw_t_alpha']:.2f}). Reference long picks still beat the same-date CS mean "
            f"by {pa['mean_excess']:.4f} (Sleeve A) and {pb['mean_excess']:.4f} (Sleeve B) in residual "
            f"units — the model ranks alts, but the winner-side gap vs the loser-side is the "
            f"classification object. Neither finding changes the reference book."
        )

    extra = {"elapsed_sec": time.time() - t_pipe, "conclusion": _conclusion()}
    plot_buckets(curves, chart_dir / "symmetry_buckets.png")
    plot_tide(tides, chart_dir / "symmetry_tide.png")
    write_report(
        rep_dir / "symmetry_audit.md",
        frozen_hash=calc,
        pred_hashes=pred_hashes,
        classification=classification,
        cells=cells,
        curves=curves,
        tides=tides,
        lo_alpha=lo_alpha,
        picks=picks,
        extra=extra,
    )
    print_stdout(classification, cells, lo_alpha, picks, extra)

    drop_keys = {
        "equity",
        "daily_ret",
        "daily_gross",
        "daily_hedge",
        "daily_cost",
        "daily_funding",
        "daily_n_pos",
        "daily_n_long",
        "daily_n_short",
        "daily_flat",
        "daily_long",
        "daily_short",
        "daily_gross_deployed",
        "daily_gross_full",
        "daily_long_names",
        "top_pos",
        "roll90",
        "cs_mean",
        "top_mean",
        "bot_mean",
        "daily_excess",
        "p1_equity",
        "p2_equity",
        "name_alpha_pnl",
        "sym_contrib",
        "side_days",
    }

    def _jsonable(x):
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items() if k not in drop_keys}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        if isinstance(x, pd.Timestamp):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return None
        return x

    # tuple keys -> strings
    cells_j = {f"h{h}_{uni}": _jsonable(v) for (h, uni), v in cells.items()}
    curves_j = {f"h{h}_{uni}_{w}": _jsonable(v) for (h, uni, w), v in curves.items()}
    tides_j = {f"h{h}_{uni}": _jsonable(v) for (h, uni), v in tides.items()}
    summary = {
        "frozen_sha256": calc,
        "pred_hashes": pred_hashes,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "classification": _jsonable(classification),
        "cells": cells_j,
        "curves": curves_j,
        "tides": tides_j,
        "lo_alpha": _jsonable(lo_alpha),
        "picks": _jsonable(picks),
        "conclusion": extra["conclusion"],
        "elapsed_sec": extra["elapsed_sec"],
        "reference_book_unchanged": True,
    }
    (rep_dir / "symmetry_audit.json").write_text(json.dumps(summary, indent=2, default=str))
    volume.commit()
    print(f"[HB] DONE elapsed={time.time() - t_pipe:.1f}s", flush=True)
    return {
        "frozen_sha256": calc,
        "gpu_used": False,
        "label": classification.get("label"),
        "n_pass_h7": classification.get("n_pass_h7"),
        "n_pass_h10": classification.get("n_pass_h10"),
        "elapsed_sec": extra["elapsed_sec"],
    }


@app.local_entrypoint()
def main():
    print("[local] starting symmetry audit (CPU, analysis-only)...", flush=True)
    summary = run_symmetry.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/symmetry_audit.md", "symmetry_audit.md", "reports"),
        ("reports/symmetry_audit.json", "symmetry_audit.json", "reports"),
        ("charts/symmetry_buckets.png", "symmetry_buckets.png", "charts"),
        ("charts/symmetry_tide.png", "symmetry_tide.png", "charts"),
    ]:
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"], check=False)
        if dest.exists() and dest.is_file():
            shutil.copy2(dest, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("symmetry*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("symmetry*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] symmetry audit complete.", flush=True)
