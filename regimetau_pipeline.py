"""
REGIME-TAU — causal CS-corr overlay on frozen A0 scores.

BACKTEST ONLY. CPU only. Frozen COMBO / SPREAD-LS / LONG-TIDE untouched.
Does not edit existing modules. Usage: modal run regimetau_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-regimetau"
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
    .add_local_python_source("baseline", "phase_d", "phase_d2", "round_f", "longonly", "regimetau")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/regimetau_addendum.md", remote_path="/root/regimetau_addendum.md")
)

app = modal.App(APP_NAME, image=image)


def _cfg() -> dict:
    with open("/root/config.yaml") as f:
        return yaml.safe_load(f)


@app.function(timeout=60 * 60 * 3, retries=0, volumes={"/data/quant": volume}, cpu=16, memory=65536)
def run_regimetau() -> dict:
    import hashlib
    import shutil

    import numpy as np
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from longonly.eval import book_stats, enrich_combo
    from phase_d2.constants import FEE_BPS_NEXT, SLIP_BPS_NEXT
    from phase_d2.metrics import summarize_port
    from regimetau.book import run_regime_tau_portfolio
    from regimetau.constants import (
        CONFIG_FROZEN,
        FROZEN_A0_PATH,
        FROZEN_A0_SHA256,
        P1_COST_BPS,
        P1_H,
        P1_SLIP_BPS,
        P1_TAU_BASE,
        P1_TAU_HIGH,
        P1_TAU_LOW,
        P2_H,
        P2_LIQ_CAP,
        P2_NOM_USD,
        P2_TAU_BASE,
        P2_TAU_HIGH,
        P2_TAU_LOW,
        PANEL_PATH,
        PRED_H10,
        PRED_H7,
        VIABILITY_CRITERION,
    )
    from regimetau.eval import apply_viability, regime_slice_sharpe
    from regimetau.regime import cs_corr_topn, regime_labels
    from regimetau.report import plot_equity, print_stdout, write_report

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
    addendum = Path("/root/regimetau_addendum.md").read_text()
    if VIABILITY_CRITERION not in addendum:
        raise RuntimeError("Addendum missing verbatim viability statements")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)
    print("[HB] BACKTEST ONLY; REGIME-TAU; zero GPU; no live components", flush=True)
    print(f"[HB] {VIABILITY_CRITERION}", flush=True)

    cfg = _cfg()
    seed_everything(cfg["seed"])
    root = Path(cfg["paths"]["volume_root"])
    feat_path = root / "features" / "features_labeled.parquet"
    uni_dir = root / "universe"
    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    out_dir = root / "regimetau"
    rep_dir = root / "reports"
    chart_dir = root / "charts"
    for d in (out_dir, rep_dir, chart_dir):
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

    cs = cs_corr_topn(panel, pit40)
    reg_df = regime_labels(cs)
    regime = reg_df["regime"]

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

    def _ref(preds, h, uni, tau_pct, tiered, cap):
        print(f"[HB] ref port h={h} τ={tau_pct} tiered={tiered}", flush=True)
        return run_tranche_portfolio(
            preds,
            panel,
            feat,
            uni,
            horizon=h,
            tau_pct=float(tau_pct),
            exit_hysteresis=port_cfg.get("exit_hysteresis", 0.6),
            gross_limit=port_cfg.get("gross_limit", 1.0),
            fee_bps=P1_COST_BPS,
            slip_bps=P1_SLIP_BPS,
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
        )

    def _reg(preds, h, uni, base, high, low, tiered, cap):
        print(f"[HB] regime port h={h} base={base} high={high} low={low}", flush=True)
        return run_regime_tau_portfolio(
            preds,
            panel,
            feat,
            uni,
            horizon=h,
            tau_pct_base=float(base),
            tau_pct_high=float(high),
            tau_pct_low=float(low),
            regime=regime,
            folds=folds_by_h[h],
            gross_limit=port_cfg.get("gross_limit", 1.0),
            fee_bps=P1_COST_BPS,
            slip_bps=P1_SLIP_BPS,
            lag=0,
            apply_funding=True,
            funding=funding,
            tiered_costs=bool(tiered),
            fee_bps_next=FEE_BPS_NEXT,
            slip_bps_next=SLIP_BPS_NEXT,
            liq_cap_adv_frac=cap,
            nominal_book_usd=P2_NOM_USD,
            rank_universe=pit40,
        )

    raw = {
        "LS_A": _ref(p7, P1_H, pit20, P1_TAU_BASE, False, None),
        "LS_B": _ref(p10, P2_H, pit40, P2_TAU_BASE, True, P2_LIQ_CAP),
        "RG_A": _reg(p7, P1_H, pit20, P1_TAU_BASE, P1_TAU_HIGH, P1_TAU_LOW, False, None),
        "RG_B": _reg(p10, P2_H, pit40, P2_TAU_BASE, P2_TAU_HIGH, P2_TAU_LOW, True, P2_LIQ_CAP),
    }
    for name, res in raw.items():
        print(
            f"[HB] {name} sharpe={res.get('net_sharpe')} n_pos={res.get('avg_n_positions')} "
            f"err={res.get('error')}",
            flush=True,
        )

    common = raw["LS_A"]["daily_ret"].index
    for k in raw:
        common = common.intersection(raw[k]["daily_ret"].index)
    common = pd.DatetimeIndex(pd.to_datetime(common, utc=True))
    print(f"[HB] identical-days n={len(common)}", flush=True)

    ports = {k: summarize_port(v, common_idx=common) for k, v in raw.items()}
    for k, v in ports.items():
        v["daily_ret"] = raw[k]["daily_ret"].copy()
        v["daily_ret"].index = pd.DatetimeIndex(pd.to_datetime(v["daily_ret"].index, utc=True))
        v["daily_ret"] = v["daily_ret"].reindex(common).fillna(0.0)
        v["name_alpha_pnl"] = dict(raw[k].get("name_alpha_pnl") or {})
        v["horizon"] = raw[k].get("horizon")
        v["ann_turnover"] = raw[k].get("ann_turnover")
        for key in (
            "daily_long",
            "daily_short",
            "daily_hedge",
            "daily_cost",
            "daily_funding",
            "daily_gross",
            "daily_gross_deployed",
            "daily_gross_full",
            "daily_n_long",
            "daily_n_pos",
            "daily_flat",
            "daily_regime",
            "daily_tau",
            "equity",
        ):
            if key in raw[k]:
                v[key] = raw[k][key]

    combo_ls = enrich_combo(ports["LS_A"], ports["LS_B"])
    combo_rg = enrich_combo(ports["RG_A"], ports["RG_B"])
    combo_rg["daily_regime"] = raw["RG_A"].get("daily_regime")
    combo_ls["horizon"] = 10
    combo_rg["horizon"] = 10

    named = {
        "REGIME Sleeve A": ports["RG_A"],
        "REGIME Sleeve B": ports["RG_B"],
        "COMBO-REGIME-TAU": combo_rg,
        "Reference Sleeve A": ports["LS_A"],
        "Reference Sleeve B": ports["LS_B"],
        "Reference COMBO": combo_ls,
    }
    lag_of = {
        "REGIME Sleeve A": P1_H,
        "REGIME Sleeve B": P2_H,
        "COMBO-REGIME-TAU": P2_H,
        "Reference Sleeve A": P1_H,
        "Reference Sleeve B": P2_H,
        "Reference COMBO": P2_H,
    }
    ref_rets = combo_ls["daily_ret"]
    books = {
        name: book_stats(port, ref_rets * 0.0, lag_of[name], ref_combo=ref_rets)
        for name, port in named.items()
    }

    reg_series = raw["RG_A"].get("daily_regime")
    slices = {
        "COMBO-REGIME-TAU": regime_slice_sharpe(combo_rg["daily_ret"], reg_series)
        if isinstance(reg_series, pd.Series)
        else {},
        "Reference COMBO": regime_slice_sharpe(combo_ls["daily_ret"], reg_series)
        if isinstance(reg_series, pd.Series)
        else {},
    }
    verdict = apply_viability(
        books["COMBO-REGIME-TAU"]["net_sharpe_full"],
        books["COMBO-REGIME-TAU"]["net_sharpe_trail18m"],
        books["Reference COMBO"]["net_sharpe_full"],
        books["Reference COMBO"]["net_sharpe_trail18m"],
    )

    extra = {
        "elapsed_sec": time.time() - t_pipe,
        "identical_days": int(len(common)),
        "config": CONFIG_FROZEN,
        "panel_path": PANEL_PATH,
        "high_frac": float(raw["RG_A"].get("high_frac", float("nan"))),
        "low_frac": float(raw["RG_A"].get("low_frac", float("nan"))),
        "base_frac": float(raw["RG_A"].get("base_frac", float("nan"))),
        "n_forced_exits": int(raw["RG_A"].get("n_forced_exits", 0))
        + int(raw["RG_B"].get("n_forced_exits", 0)),
        "forced_exit_pnl": float(raw["RG_A"].get("forced_exit_pnl", 0.0))
        + float(raw["RG_B"].get("forced_exit_pnl", 0.0)),
        "cs_corr_median": float(cs.median()) if cs.notna().any() else float("nan"),
        "gpu": False,
    }

    plot_equity(
        combo_rg["daily_ret"],
        combo_ls["daily_ret"],
        chart_dir / "regimetau_equity.png",
    )
    write_report(
        rep_dir / "regimetau_report.md",
        frozen_hash=calc,
        pred_hashes=pred_hashes,
        books=books,
        verdict=verdict,
        slices=slices,
        extra=extra,
    )
    print_stdout(verdict, extra)

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
        "daily_regime",
        "daily_tau",
        "p1_equity",
        "p2_equity",
        "name_alpha_pnl",
        "sym_contrib",
        "side_days",
    }

    def _jsonable(x, drop=None):
        drop = drop or drop_keys
        if isinstance(x, dict):
            return {k: _jsonable(v, drop) for k, v in x.items() if k not in drop}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v, drop) for v in x]
        if isinstance(x, pd.Series):
            return None
        if isinstance(x, pd.DataFrame):
            return None
        if isinstance(x, (np.floating, float)):
            v = float(x)
            return None if not np.isfinite(v) else v
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if x is None:
            return None
        if isinstance(x, Path):
            return str(x)
        return x

    blob = {
        "verdict": _jsonable(verdict),
        "books": _jsonable({k: {kk: vv for kk, vv in v.items() if kk != "daily_ret"} for k, v in books.items()}),
        "slices": _jsonable(slices),
        "extra": _jsonable(extra),
        "pred_hashes": pred_hashes,
        "frozen_a0": calc,
    }
    (rep_dir / "regimetau_report.json").write_text(json.dumps(blob, indent=2))
    (out_dir / "regimetau_report.json").write_text(json.dumps(blob, indent=2))
    shutil.copyfile(rep_dir / "regimetau_report.md", out_dir / "regimetau_report.md")
    volume.commit()
    print(f"[HB] DONE elapsed={time.time() - t_pipe:.1f}s", flush=True)
    return blob


@app.local_entrypoint()
def main() -> None:
    blob = run_regimetau.remote()
    verdict = (blob or {}).get("verdict") or {}
    print(
        f"REGIME-TAU {verdict.get('label')} "
        f"full={verdict.get('reg_full')} trail={verdict.get('reg_trail')}",
        flush=True,
    )
    Path("/tmp/regimetau_summary.json").write_text(json.dumps(blob.get("verdict", {}), indent=2))
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/regimetau_report.md", "regimetau_report.md", "reports"),
        ("reports/regimetau_report.json", "regimetau_report.json", "reports"),
        ("charts/regimetau_equity.png", "regimetau_equity.png", "charts"),
    ]:
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"],
            check=False,
        )
        if dest.exists() and dest.is_file():
            shutil.copy2(dest, Path(kind) / name)
