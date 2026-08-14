"""
Long-only variants of the frozen COMBO system.

BACKTEST ONLY. CPU only. Frozen A0 scores and causal median-τ reused as-is.
Usage: modal run longonly_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-longonly"
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
    .add_local_python_source("baseline", "phase_d", "phase_d2", "round_f", "longonly")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/longonly_addendum.md", remote_path="/root/longonly_addendum.md")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 60 * 3, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_longonly() -> dict:
    import hashlib
    import shutil

    import numpy as np
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from longonly.constants import (
        CONFIG_FROZEN,
        FROZEN_A0_PATH,
        FROZEN_A0_SHA256,
        P1_COST_BPS,
        P1_H,
        P1_SLIP_BPS,
        P1_TAU,
        P2_H,
        P2_LIQ_CAP,
        P2_NOM_USD,
        P2_TAU,
        PANEL_PATH,
        PIT_TOP20,
        PIT_TOP40,
        PRED_H10,
        PRED_H7,
        VIABILITY_CRITERION,
    )
    from longonly.eval import (
        attribution_block,
        book_stats,
        btc_bh_simple,
        enrich_combo,
        ew_top20_simple,
        loh_viable,
        lou_viable,
    )
    from longonly.report import plot_longonly_equity, print_stdout, write_longonly_report
    from phase_d2.constants import FEE_BPS_NEXT, FEE_BPS_TOP, SLIP_BPS_NEXT, SLIP_BPS_TOP
    from phase_d2.metrics import summarize_port

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
    addendum = Path("/root/longonly_addendum.md").read_text()
    if VIABILITY_CRITERION not in addendum:
        raise RuntimeError("Addendum missing verbatim viability statements")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)
    print("[HB] BACKTEST ONLY; causal τ reused; zero GPU; no live components", flush=True)
    print("[HB] viability statements frozen before results", flush=True)
    print(f"[HB] {VIABILITY_CRITERION}", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    lo_dir = root / "longonly"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in (lo_dir, rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_path = Path(FROZEN_A0_PATH)
    if model_path.exists():
        print(f"[HB] frozen model sha256={_sha256_file(model_path)}", flush=True)

    pred_h7 = Path(PRED_H7)
    pred_h10 = Path(PRED_H10)
    if not pred_h7.exists():
        pred_h7 = root / "predictions" / "lgbm_price_only_h7.parquet"
    if not pred_h10.exists():
        pred_h10 = root / "predictions" / "lgbm_price_only_h10.parquet"
    pred_hashes = {
        "h7": _sha256_file(pred_h7),
        "h10": _sha256_file(pred_h10),
        "h7_path": str(pred_h7),
        "h10_path": str(pred_h10),
    }
    print(f"[HB] reused scores h7={pred_hashes['h7']} h10={pred_hashes['h10']}", flush=True)

    port_cfg = cfg["portfolio"]
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    print(f"[HB] feat rows={len(feat)}", flush=True)

    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit40 = pd.read_parquet(uni_dir / "top40_pit.parquet")
    for u in (pit20, pit40):
        u["date"] = pd.to_datetime(u["date"], utc=True)
    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    funding = load_funding_panel(fund_dir, ever)

    p7 = pd.read_parquet(pred_h7)
    p10 = pd.read_parquet(pred_h10)
    p7["date"] = pd.to_datetime(p7["date"], utc=True)
    p10["date"] = pd.to_datetime(p10["date"], utc=True)

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

    def _port(preds, h, uni, tau_pct, tiered, cap, long_only, hedge):
        print(
            f"[HB] port h={h} τ={tau_pct} long_only={long_only} hedge={hedge} tiered={tiered}",
            flush=True,
        )
        return run_tranche_portfolio(
            preds,
            panel,
            feat,
            uni,
            horizon=h,
            tau_pct=float(tau_pct),
            exit_hysteresis=port_cfg.get("exit_hysteresis", 0.6),
            gross_limit=port_cfg.get("gross_limit", 1.0),
            fee_bps=FEE_BPS_TOP if not (h == P1_H) else P1_COST_BPS,
            slip_bps=SLIP_BPS_TOP if not (h == P1_H) else P1_SLIP_BPS,
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
        )

    jobs = [
        ("LS_A", p7, P1_H, pit20, P1_TAU, False, None, False, True),
        ("LS_B", p10, P2_H, pit40, P2_TAU, True, P2_LIQ_CAP, False, True),
        ("LOH_A", p7, P1_H, pit20, P1_TAU, False, None, True, True),
        ("LOH_B", p10, P2_H, pit40, P2_TAU, True, P2_LIQ_CAP, True, True),
        ("LOU_A", p7, P1_H, pit20, P1_TAU, False, None, True, False),
        ("LOU_B", p10, P2_H, pit40, P2_TAU, True, P2_LIQ_CAP, True, False),
    ]
    raw = {}
    for name, preds, h, uni, tau, tiered, cap, lo, hg in jobs:
        raw[name] = _port(preds, h, uni, tau, tiered, cap, lo, hg)
        print(
            f"[HB] {name} sharpe={raw[name].get('net_sharpe')} "
            f"n_long={raw[name].get('avg_n_long')} gross={raw[name].get('avg_gross_deployed')}",
            flush=True,
        )

    common = raw["LS_A"]["daily_ret"].index
    for k in raw:
        common = common.intersection(raw[k]["daily_ret"].index)
    common = pd.DatetimeIndex(pd.to_datetime(common, utc=True))
    print(f"[HB] identical-days n={len(common)}", flush=True)

    ports = {k: summarize_port(v, common_idx=common) for k, v in raw.items()}
    for k, v in ports.items():
        v["name_alpha_pnl"] = dict(raw[k].get("name_alpha_pnl") or {})
        v["long_only"] = bool(raw[k].get("long_only", False))
        v["apply_beta_hedge"] = bool(raw[k].get("apply_beta_hedge", True))
        v["horizon"] = raw[k].get("horizon")
        v["ann_turnover"] = raw[k].get("ann_turnover")

    combo_ls = enrich_combo(ports["LS_A"], ports["LS_B"])
    combo_loh = enrich_combo(ports["LOH_A"], ports["LOH_B"])
    combo_lou = enrich_combo(ports["LOU_A"], ports["LOU_B"])
    for c, hflag, hedge in (
        (combo_ls, False, True),
        (combo_loh, True, True),
        (combo_lou, True, False),
    ):
        c["long_only"] = hflag
        c["apply_beta_hedge"] = hedge
        c["horizon"] = 10
        c["tau_pct"] = "50/50"

    btc = btc_bh_simple(panel).reindex(common).fillna(0.0)
    ew = ew_top20_simple(panel, pit20).reindex(common).fillna(0.0)

    named = {
        "LO-H Sleeve A": ports["LOH_A"],
        "LO-H Sleeve B": ports["LOH_B"],
        "COMBO-LO-H": combo_loh,
        "LO-U Sleeve A": ports["LOU_A"],
        "LO-U Sleeve B": ports["LOU_B"],
        "COMBO-LO-U": combo_lou,
        "Reference Sleeve A": ports["LS_A"],
        "Reference Sleeve B": ports["LS_B"],
        "Reference COMBO": combo_ls,
    }
    lag_of = {
        "LO-H Sleeve A": P1_H,
        "LO-H Sleeve B": P2_H,
        "COMBO-LO-H": P2_H,
        "LO-U Sleeve A": P1_H,
        "LO-U Sleeve B": P2_H,
        "COMBO-LO-U": P2_H,
        "Reference Sleeve A": P1_H,
        "Reference Sleeve B": P2_H,
        "Reference COMBO": P2_H,
    }
    ref_rets = combo_ls["daily_ret"]
    books = {
        name: book_stats(port, btc, lag_of[name], ref_combo=ref_rets)
        for name, port in named.items()
    }

    attr = {name: attribution_block(port, idx=common) for name, port in named.items()}

    loh_v = loh_viable(books["COMBO-LO-H"]["net_sharpe_full"], books["COMBO-LO-H"]["net_sharpe_trail18m"])
    lou_alpha = books["COMBO-LO-U"]["alpha"]
    lou_v = lou_viable(
        lou_alpha["full"]["alpha_ann"],
        lou_alpha["full"]["nw_t_alpha"],
        lou_alpha["trail18m"]["alpha_ann"],
    )
    loh_sleeve_v = {
        "Sleeve A": loh_viable(books["LO-H Sleeve A"]["net_sharpe_full"], books["LO-H Sleeve A"]["net_sharpe_trail18m"]),
        "Sleeve B": loh_viable(books["LO-H Sleeve B"]["net_sharpe_full"], books["LO-H Sleeve B"]["net_sharpe_trail18m"]),
    }
    lou_sleeve_v = {
        "Sleeve A": lou_viable(
            books["LO-U Sleeve A"]["alpha"]["full"]["alpha_ann"],
            books["LO-U Sleeve A"]["alpha"]["full"]["nw_t_alpha"],
            books["LO-U Sleeve A"]["alpha"]["trail18m"]["alpha_ann"],
        ),
        "Sleeve B": lou_viable(
            books["LO-U Sleeve B"]["alpha"]["full"]["alpha_ann"],
            books["LO-U Sleeve B"]["alpha"]["full"]["nw_t_alpha"],
            books["LO-U Sleeve B"]["alpha"]["trail18m"]["alpha_ann"],
        ),
    }

    corr_h = books["COMBO-LO-H"]["corr_vs_ref_combo"]
    corr_u = books["COMBO-LO-U"]["corr_vs_ref_combo"]
    corr_oneliner = (
        f"Corr vs reference COMBO: COMBO-LO-H={corr_h:.3f}, COMBO-LO-U={corr_u:.3f} "
        f"(Sleeve A LO-H={books['LO-H Sleeve A']['corr_vs_ref_combo']:.3f}, "
        f"LO-U={books['LO-U Sleeve A']['corr_vs_ref_combo']:.3f}; "
        f"Sleeve B LO-H={books['LO-H Sleeve B']['corr_vs_ref_combo']:.3f}, "
        f"LO-U={books['LO-U Sleeve B']['corr_vs_ref_combo']:.3f})."
    )
    extra = {
        "elapsed_sec": time.time() - t_pipe,
        "corr_oneliner": corr_oneliner,
        "identical_days": int(len(common)),
        "config": CONFIG_FROZEN,
        "panel_path": PANEL_PATH,
        "pit_top20": PIT_TOP20,
        "pit_top40": PIT_TOP40,
    }

    plot_longonly_equity(
        combo_loh["daily_ret"],
        combo_lou["daily_ret"],
        btc,
        combo_ls["daily_ret"],
        chart_dir / "longonly_equity.png",
    )

    write_longonly_report(
        rep_dir / "longonly_report.md",
        frozen_hash=calc,
        pred_hashes=pred_hashes,
        books=books,
        loh_v=loh_v,
        lou_v=lou_v,
        loh_sleeve_v=loh_sleeve_v,
        lou_sleeve_v=lou_sleeve_v,
        attr=attr,
        benches={"btc": btc, "ew_top20": ew},
        extra=extra,
    )
    print_stdout(loh_v, lou_v, attr, extra)

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
        "p1_equity",
        "p2_equity",
        "name_alpha_pnl",
        "sym_contrib",
        "side_days",
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
        if isinstance(x, pd.DataFrame):
            return None
        return x

    summary = {
        "frozen_sha256": calc,
        "pred_hashes": pred_hashes,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "viability_criterion": VIABILITY_CRITERION,
        "loh": loh_v,
        "lou": lou_v,
        "loh_sleeves": loh_sleeve_v,
        "lou_sleeves": lou_sleeve_v,
        "books": {k: _jsonable(v) for k, v in books.items()},
        "attribution": _jsonable(attr),
        "corr_oneliner": corr_oneliner,
        "reference_long_share_of_net": (attr.get("Reference COMBO") or {}).get("full", {}).get(
            "long_share_of_net"
        ),
        "elapsed_sec": time.time() - t_pipe,
        "tau_mode": "fold_train",
        "reference_book_unchanged": True,
        "identical_days": int(len(common)),
    }
    (rep_dir / "longonly_report.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    shutil.copy2(rep_dir / "longonly_report.md", lo_dir / "longonly_report.md")
    shutil.copy2(rep_dir / "longonly_report.json", lo_dir / "longonly_report.json")
    volume.commit()
    print(f"[HB] DONE elapsed={time.time() - t_pipe:.1f}s", flush=True)
    return {
        "frozen_sha256": calc,
        "gpu_used": False,
        "loh_verdict": loh_v.get("verdict"),
        "lou_verdict": lou_v.get("verdict"),
        "reference_long_share_of_net": summary["reference_long_share_of_net"],
        "corr_oneliner": corr_oneliner,
        "elapsed_sec": time.time() - t_pipe,
    }


@app.local_entrypoint()
def main():
    print("[local] starting long-only evaluation (CPU, backtest-only)...", flush=True)
    summary = run_longonly.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/longonly_report.md", "longonly_report.md", "reports"),
        ("reports/longonly_report.json", "longonly_report.json", "reports"),
        ("charts/longonly_equity.png", "longonly_equity.png", "charts"),
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
        for src in (art / "reports").glob("longonly*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        chart = art / "charts" / "longonly_equity.png"
        if chart.exists():
            (opt / "charts" / "longonly_equity.png").write_bytes(chart.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] long-only evaluation complete.", flush=True)
