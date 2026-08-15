"""
BTC-BEATER Phase 3 — SPREAD-LS challenger.

BACKTEST ONLY. Portfolio layer only. CPU only. Frozen COMBO untouched.
Reuses 2.c spread cache byte-identical. No retraining.
Usage: modal run --detach btcb_phase3_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p3"
VOL_Q = "quant-baseline"

quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

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
    .add_local_python_source("baseline", "btcb", "phase_d", "phase_d2", "round_f", "longonly")
    .add_local_file("reports/btcb_phase3_addendum.md", remote_path="/root/btcb_phase3_addendum.md")
    .add_local_file("universe/btcb_top50_floor.parquet", remote_path="/root/btcb_top50_floor.parquet")
    .add_local_file("universe/btcb_top100_floor.parquet", remote_path="/root/btcb_top100_floor.parquet")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
)

app = modal.App(APP_NAME, image=image)


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {
        "daily_ret",
        "btc_ret",
        "equity",
        "n_long",
        "n_short",
        "n_shortable",
        "incomplete",
        "long_gross",
        "short_gross",
        "cash",
        "realized_beta_90d",
        "id_to_sym",
        "ls_daily",
        "combo_daily",
    }
    if isinstance(x, dict):
        return {str(k): _jsonable(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, drop) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return None
    return x


@app.function(
    timeout=60 * 60 * 4,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p3() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd
    import yaml

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.portfolio import run_tranche_portfolio
    from baseline.seedutil import seed_everything
    from btcb.constants import (
        DEATH_CONVENTION,
        PHASE2_HORIZONS,
        PHASE2_PRIMARY_H,
        PHASE3_CRITERION,
        PHASE3_FUNDING_CAVEAT,
        SEED,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.phase3_report import plot_equity_dd, plot_overlap, plot_rolling_beta, write_phase3
    from btcb.spread_ls import (
        attach_beta,
        build_shortable,
        combo_overlap_stats,
        ew_basket,
        hash_pred_dir,
        load_twin_from_cache,
        mechanical_verdicts_ls,
        run_spread_ls,
        squeeze_table,
    )
    from longonly.constants import (
        FROZEN_A0_SHA256,
        P1_COST_BPS,
        P1_H,
        P1_SLIP_BPS,
        P1_TAU,
        P2_H,
        P2_LIQ_CAP,
        P2_NOM_USD,
        P2_TAU,
        PRED_H10,
        PRED_H7,
    )
    from longonly.eval import enrich_combo
    from phase_d2.constants import FEE_BPS_NEXT, FEE_BPS_TOP, SLIP_BPS_NEXT, SLIP_BPS_TOP

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase3_addendum.md").read_text()
    if PHASE3_CRITERION not in addendum or PHASE3_FUNDING_CAVEAT not in addendum or DEATH_CONVENTION not in addendum:
        raise RuntimeError("Phase 3 addendum missing verbatim criterion/caveat/convention")
    print("[HB] BTC-BEATER P3 BACKTEST ONLY; portfolio layer; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {PHASE3_CRITERION}", flush=True)
    print(f"[HB] {PHASE3_FUNDING_CAVEAT}", flush=True)
    print(f"[HB] {DEATH_CONVENTION}", flush=True)

    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file or calc != FROZEN_A0_SHA256:
        raise RuntimeError(f"Frozen A0 hash mismatch calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    print(f"[HB] frozen A0 OK sha256={calc}", flush=True)

    def commit():
        quant_vol.commit()

    panel_path = Path("/data/quant/btcb/full/panel.parquet")
    if not panel_path.exists():
        raise RuntimeError(f"missing panel {panel_path}")
    print(f"[HB] loading CMC panel {panel_path}", flush=True)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id}", flush=True)

    def _load_pit(name: str) -> pd.DataFrame:
        cands = [
            Path(f"/data/quant/btcb/universe/{name}"),
            Path(f"/data/quant/universe/{name}"),
            Path(f"/root/{name}"),
        ]
        for p in cands:
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                df["id"] = df["id"].astype(int)
                print(f"[HB] pit {name} from {p} rows={len(df)}", flush=True)
                return df
        raise RuntimeError(f"missing floored PIT {name} (must reuse 2.b, do not rebuild)")

    pit50 = _load_pit("btcb_top50_floor.parquet")
    pit100 = _load_pit("btcb_top100_floor.parquet")

    print("[HB] re-applying frozen 2.b cleaner (no new hygiene)...", flush=True)
    cleaned, _clog = clean_panel(panel, btc_id=btc_id)

    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if not feat_path.exists():
        raise RuntimeError(f"missing 2.b Stage-S features {feat_path}")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)
    print(f"[HB] reused feat_s rows={len(feat)}", flush=True)

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    if not pred_dir.exists():
        raise RuntimeError(f"missing 2.c pred cache {pred_dir}")
    pred_hash = hash_pred_dir(pred_dir)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n_files={pred_hash['n_files']}", flush=True)

    twins = {}
    for h in PHASE2_HORIZONS:
        twins[h] = load_twin_from_cache(pred_dir, h)
        print(f"[HB] twin h={h} rows={len(twins[h])} dates={twins[h]['date'].nunique()}", flush=True)

    print("[HB] loading Binance USDT-M klines (listing/delisting table)...", flush=True)
    raw_dir = Path("/data/quant/raw/klines")
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    if not kline_syms:
        raise RuntimeError(f"no Binance klines in {raw_dir}")
    kline_panel = load_panel(raw_dir, kline_syms)
    combo_panel = kline_panel.copy()
    combo_panel["date"] = pd.to_datetime(combo_panel["date"], utc=True)
    kline_panel["date"] = pd.to_datetime(kline_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    print(f"[HB] kline symbols={len(kline_syms)} rows={len(kline_panel)}", flush=True)
    shortable = build_shortable(cleaned, kline_panel, btc_id)
    ns = [len(v) for v in shortable.values()]
    print(
        f"[HB] shortable dates={len(shortable)} mean_ids={float(sum(ns)/len(ns)) if ns else 0:.1f} "
        f"max_ids={max(ns) if ns else 0}",
        flush=True,
    )

    close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    btc_simple = close[btc_id].pct_change()
    members100 = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit100.groupby("date")["id"]
    }

    books = {}
    for h in PHASE2_HORIZONS:
        for matched in (False, True):
            tag = ("bm" if matched else "dn") + f"_h{h}"
            print(f"[HB] SPREAD-LS {tag}...", flush=True)
            packed = run_spread_ls(
                cleaned,
                pit100,
                twins[h],
                feat,
                shortable,
                btc_id,
                h=int(h),
                beta_matched=matched,
            )
            if packed.get("error"):
                raise RuntimeError(f"book {tag} failed: {packed}")
            packed = attach_beta(packed, btc_simple)
            books[tag] = packed
            print(
                f"[HB] {tag} sharpe={packed.get('net_sharpe')} trail={packed.get('net_sharpe_trail18m')} "
                f"beta={packed.get('realized_beta_full')} nL={packed.get('avg_n_long')} "
                f"nS={packed.get('avg_n_short')} sh={packed.get('avg_shortable')}",
                flush=True,
            )
            commit()

    head = books["dn_h14"]
    if int(head.get("btc_in_book_hits") or 0) != 0:
        raise RuntimeError("BTC leaked into headline book")

    print("[HB] EW floored top-100 squeeze basket...", flush=True)
    all_dates = [d for d in close.index if d in members100]
    basket = ew_basket(close, members100, all_dates)
    squeeze = squeeze_table(head["daily_ret"], basket)

    print("[HB] replaying frozen COMBO (A0 scores, product untouched)...", flush=True)
    with open("/root/config.yaml") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["volume_root"])
    feat_a0 = pd.read_parquet(root / "features" / "features_labeled.parquet")
    feat_a0["date"] = pd.to_datetime(feat_a0["date"], utc=True)
    uni_dir = root / "universe"
    pit20 = pd.read_parquet(uni_dir / "top20_pit.parquet")
    pit40 = pd.read_parquet(uni_dir / "top40_pit.parquet")
    for u in (pit20, pit40):
        u["date"] = pd.to_datetime(u["date"], utc=True)
    fund_dir = root / "raw" / "funding"
    ever = sorted(set(feat_a0["symbol"].unique()) | {"BTCUSDT"})
    funding = load_funding_panel(fund_dir, ever)
    pred_h7 = Path(PRED_H7)
    pred_h10 = Path(PRED_H10)
    if not pred_h7.exists():
        pred_h7 = root / "predictions" / "lgbm_price_only_h7.parquet"
    if not pred_h10.exists():
        pred_h10 = root / "predictions" / "lgbm_price_only_h10.parquet"
    p7 = pd.read_parquet(pred_h7)
    p10 = pd.read_parquet(pred_h10)
    p7["date"] = pd.to_datetime(p7["date"], utc=True)
    p10["date"] = pd.to_datetime(p10["date"], utc=True)
    port_cfg = cfg["portfolio"]
    folds_by_h = {
        h: make_folds(
            pd.DatetimeIndex(feat_a0["date"].unique()),
            horizon=h,
            min_train_days=cfg["cv"]["min_train_days"],
            val_days=cfg["cv"]["val_days"],
            step_days=cfg["cv"]["step_days"],
        )
        for h in (P1_H, P2_H)
    }

    def _combo_sleeve(preds, h, uni, tau_pct, tiered, cap):
        return run_tranche_portfolio(
            preds,
            combo_panel,
            feat_a0,
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
            long_only=False,
            apply_beta_hedge=True,
        )

    ls_a = _combo_sleeve(p7, P1_H, pit20, P1_TAU, False, None)
    print(f"[HB] COMBO sleeve A sharpe={ls_a.get('net_sharpe')}", flush=True)
    ls_b = _combo_sleeve(p10, P2_H, pit40, P2_TAU, True, P2_LIQ_CAP)
    print(f"[HB] COMBO sleeve B sharpe={ls_b.get('net_sharpe')}", flush=True)
    combo = enrich_combo(ls_a, ls_b)
    print(f"[HB] COMBO sharpe={combo.get('net_sharpe_full')}", flush=True)
    overlap = combo_overlap_stats(head["daily_ret"], combo["daily_ret"])
    print(
        f"[HB] overlap n={overlap.get('n_days')} ls={overlap.get('ls_sharpe')} "
        f"combo={overlap.get('combo_sharpe')} corr={overlap.get('corr')}",
        flush=True,
    )

    verdicts = mechanical_verdicts_ls(head, overlap)
    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "btc_in_book_hits": int(head.get("btc_in_book_hits") or 0),
        "pred_sha256": pred_hash["sha256"],
        "pred_n_files": pred_hash["n_files"],
        "n_binance_symbols": int(len(kline_syms)),
        "combo_a0_sha256": calc,
        "combo_sleeve_a_sharpe": ls_a.get("net_sharpe"),
        "combo_sleeve_b_sharpe": ls_b.get("net_sharpe"),
        "funding_applied": False,
    }

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_phase3(
        rep_dir / "btcb_phase3_spreadls.md",
        verdicts=verdicts,
        books=books,
        overlap=overlap,
        squeeze=squeeze,
        extra=extra,
    )
    plot_equity_dd(head, chart_dir / "btcb_p3_equity.png")
    plot_rolling_beta(head, chart_dir / "btcb_p3_beta.png")
    plot_overlap(overlap, chart_dir / "btcb_p3_overlap.png")

    payload = {
        "criterion": PHASE3_CRITERION,
        "funding_caveat": PHASE3_FUNDING_CAVEAT,
        "death_convention": DEATH_CONVENTION,
        "verdicts": _jsonable(verdicts),
        "books": {k: _jsonable(v) for k, v in books.items()},
        "overlap": _jsonable(overlap),
        "squeeze": _jsonable(squeeze),
        "pred_hash": {"sha256": pred_hash["sha256"], "n_files": pred_hash["n_files"]},
        "extra": _jsonable(extra),
        "gpu_used": False,
    }
    (rep_dir / "btcb_phase3_spreadls.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "btcb_phase3_done.txt").write_text(
        json.dumps({"elapsed_sec": time.time() - t0, "gpu_used": False}, indent=2)
    )
    commit()

    viable_s = "VIABLE" if verdicts.get("viable") else "NOT VIABLE"
    sleeve_s = "SLEEVE-GRADE" if verdicts.get("sleeve_grade") else "NOT SLEEVE-GRADE"
    repl_s = "REPLACEMENT CANDIDATE" if verdicts.get("replacement_candidate") else "NOT REPLACEMENT"
    print(f"VERDICT: SPREAD-LS {viable_s}", flush=True)
    print(f"VERDICT: {sleeve_s}", flush=True)
    print(f"VERDICT: {repl_s}", flush=True)
    print(f"net Sharpe full={head.get('net_sharpe')} trail18m={head.get('net_sharpe_trail18m')}", flush=True)
    print(f"realized beta vs BTC={head.get('realized_beta_full')}", flush=True)
    print(f"corr COMBO={overlap.get('corr')}", flush=True)
    print(f"avg shortable={head.get('avg_shortable')}", flush=True)
    print(f"FUNDING=0 caveat in force. COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "viable": bool(verdicts.get("viable")),
        "sleeve_grade": bool(verdicts.get("sleeve_grade")),
        "replacement_candidate": bool(verdicts.get("replacement_candidate")),
        "net_sharpe_full": head.get("net_sharpe"),
        "net_sharpe_trail18m": head.get("net_sharpe_trail18m"),
        "realized_beta": head.get("realized_beta_full"),
        "corr_combo": overlap.get("corr"),
        "avg_shortable": head.get("avg_shortable"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] starting BTC-BEATER P3 (spawn, then wait)...", flush=True)
    fc = run_btcb_p3.spawn()
    print(f"[local] spawned {getattr(fc, 'object_id', fc)}", flush=True)
    summary = fc.get()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase3_spreadls.md", "reports"),
        ("reports/btcb_phase3_spreadls.json", "reports"),
        ("charts/btcb_p3_equity.png", "charts"),
        ("charts/btcb_p3_beta.png", "charts"),
        ("charts/btcb_p3_overlap.png", "charts"),
    ]
    for remote, kind in pulls:
        name = Path(remote).name
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOL_Q, remote, str(dest), "--force"], check=False)
        candidate = dest if dest.is_file() else dest / name
        if candidate.exists() and candidate.is_file():
            out = Path(kind) / name
            out.parent.mkdir(exist_ok=True)
            shutil.copy2(candidate, out)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("btcb_phase3*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_p3*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] BTC-BEATER P3 complete.", flush=True)
