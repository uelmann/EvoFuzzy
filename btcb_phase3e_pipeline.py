"""
BTC-BEATER Phase 3.e — pricing-gap forensics.

ANALYSIS ONLY. Reuses frozen 3.c artifacts. CPU only. Zero GPU.
Usage: modal run btcb_phase3e_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p3e"
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
    .add_local_python_source("baseline", "btcb")
    .add_local_file(
        "reports/btcb_phase3e_addendum.md",
        remote_path="/root/btcb_phase3e_addendum.md",
    )
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME, image=image)
CMC_PANEL = Path("/data/quant/btcb/full/panel.parquet")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {"ic_bn", "ic_cmc", "daily_ret", "equity", "id_to_sym"}
    if isinstance(x, dict):
        return {str(k): _jsonable(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, drop) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


@app.function(
    timeout=60 * 60,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def run_phase3e() -> dict:
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from btcb.binance_replay import (
        build_id_symbol_map,
        close_wide_from_panel,
        coverage_tables,
        funding_wide_from_panel,
        replay_long_leg,
    )
    from btcb.constants import (
        COMBO_SPREADLS_CORR,
        DEATH_CONVENTION,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_POSITION_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_HYBRID_SHARPE,
        PHASE3E_H,
        PHASE3E_OUTCOME,
        PHASE3E_REF_BN_SHARPE,
        PHASE3E_REF_CMC_SUB_SHARPE,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.phase3e import (
        by_side,
        by_tier,
        classify_stale,
        combine_bn_close,
        concentration_and_stale,
        funding_structure,
        funding_vs_repricing,
        fwd_excess_wide,
        gap_waterfall_shares,
        name_day_table,
        never_listed_contribution,
        rankic_on_prices,
        replayable_daily,
        signal_verdict,
    )
    from btcb.phase3e_report import plot_gap_waterfall, plot_rankic_bars, update_ledger_3e, write_phase3e
    from btcb.spread_ls import _hash_position_log, build_shortable, hash_pred_dir, load_twin_from_cache

    t0 = time.time()
    addendum = Path("/root/btcb_phase3e_addendum.md").read_text()
    for txt in (PHASE3E_OUTCOME, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"3.e addendum missing freeze text: {txt[:80]}")
    print("[HB] BTC-BEATER P3e FORENSICS ONLY; zero GPU; no book redesign", flush=True)
    print(f"[HB] {PHASE3E_OUTCOME}", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    panel = pd.read_parquet(CMC_PANEL)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id} rows={len(panel)}", flush=True)

    pit = None
    for p in (
        Path("/data/quant/btcb/universe/btcb_top100_floor.parquet"),
        Path("/data/quant/universe/btcb_top100_floor.parquet"),
    ):
        if p.exists():
            pit = pd.read_parquet(p)
            pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            pit["id"] = pit["id"].astype(int)
            pit = pit[pit["date"] <= end].copy()
            print(f"[HB] pit from {p} rows={len(pit)}", flush=True)
            break
    if pit is None:
        raise RuntimeError("missing floored PIT top-100")

    print("[HB] re-applying frozen 2.b cleaner (no CMC writes)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()
    cmc_close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    cmc_close.index = pd.to_datetime(cmc_close.index, utc=True).tz_convert("UTC").normalize()

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    twin = load_twin_from_cache(pred_dir, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()

    plog_path = Path("/data/quant/btcb/phase3c/position_log.parquet")
    if not plog_path.exists():
        raise RuntimeError("missing frozen 3.c position log")
    plog = pd.read_parquet(plog_path)
    pos_sha = _hash_position_log(plog)
    print(f"[HB] position log sha256={pos_sha}", flush=True)
    if pos_sha != PHASE3C_POSITION_SHA256:
        raise RuntimeError(f"position log mutated {pos_sha}")

    print("[HB] loading Binance caches (no new downloads)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    raw_dir = Path("/data/quant/raw/klines")
    fund_dir = Path("/data/quant/raw/funding")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet"))
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    fund_syms = sorted(p.stem for p in fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    spot_panel = load_panel(spot_dir, spot_syms)
    kline_panel = load_panel(raw_dir, kline_syms)
    funding = (
        load_funding_panel(fund_dir, fund_syms)
        if fund_syms
        else pd.DataFrame(columns=["date", "symbol", "funding_rate", "n_events"])
    )
    for df in (spot_panel, kline_panel):
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if not funding.empty:
        funding["date"] = pd.to_datetime(funding["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        funding["symbol"] = funding["symbol"].astype(str).str.upper()

    all_ids = sorted(set(int(i) for i in plog["id"].unique()) | {int(btc_id)} | set(int(i) for i in pit["id"].unique()))
    nonempty_spot = set(spot_panel["symbol"].unique())
    nonempty_perp = set(kline_panel["symbol"].unique())
    id_to_spot = build_id_symbol_map(all_ids, cleaned, nonempty_spot, nonempty_spot)
    id_to_perp = build_id_symbol_map(all_ids, cleaned, nonempty_perp, nonempty_perp)
    spot_wide = close_wide_from_panel(spot_panel, id_to_spot)
    perp_wide = close_wide_from_panel(kline_panel, id_to_perp)
    fund_wide = funding_wide_from_panel(funding, id_to_perp)
    shortable = build_shortable(cleaned, kline_panel, btc_id)
    id_to_sym = cleaned.sort_values("date").groupby("id")["symbol"].last().to_dict()

    print("[HB] name-day replay (same positions)...", flush=True)
    nd = name_day_table(plog, cmc_close, spot_wide, perp_wide, fund_wide, pit)
    print(f"[HB] name-days={len(nd)} replayable={int(nd['replayable'].sum())}", flush=True)
    daily = replayable_daily(nd)
    fr = funding_vs_repricing(daily)
    print(
        f"[HB] BN-only ON={fr['sharpe_bn_on']:.3f} OFF={fr['sharpe_bn_off']:.3f} "
        f"CMC={fr['sharpe_cmc_sub']:.3f} dFund={fr['d_sharpe_funding']:.3f} dRepr={fr['d_sharpe_repricing']:.3f}",
        flush=True,
    )
    if abs(float(fr["sharpe_bn_on"]) - float(PHASE3E_REF_BN_SHARPE)) > 0.03:
        print("[WARN] BN-only ON Sharpe drifted from 3.c reference", flush=True)
    if abs(float(fr["sharpe_cmc_sub"]) - float(PHASE3E_REF_CMC_SUB_SHARPE)) > 0.03:
        print("[WARN] CMC-subset Sharpe drifted from 3.c reference", flush=True)

    side = by_side(nd)
    tier = by_tier(nd)
    print("[HB] stale-price classification on replayable name-days...", flush=True)
    classified = classify_stale(nd, cmc_close, spot_wide, perp_wide)
    conc = concentration_and_stale(classified)
    print(
        f"[HB] stale share={conc.get('stale_share_of_gap')} top30_abs={conc.get('top_share_of_abs')}",
        flush=True,
    )
    wf = gap_waterfall_shares(fr, conc)

    print("[HB] RankIC on Binance h=14 excess (replayable names)...", flush=True)
    bn_close = combine_bn_close(spot_wide, perp_wide)
    cmc_ex = fwd_excess_wide(cmc_close, int(btc_id), h=PHASE3E_H)
    bn_ex = fwd_excess_wide(bn_close, int(btc_id), h=PHASE3E_H)
    rankic = rankic_on_prices(twin, pit, cmc_ex, bn_ex, h=PHASE3E_H)
    verdict = signal_verdict(rankic)
    print(
        f"[HB] RankIC BN full={verdict.get('rankic_bn_full')} CMC={verdict.get('rankic_cmc_full')} "
        f"trail BN={verdict.get('rankic_bn_trail')} CMC={verdict.get('rankic_cmc_trail')} "
        f"label={verdict.get('label')}",
        flush=True,
    )

    print("[HB] funding structure + never-listed longs...", flush=True)
    fund_s = funding_structure(nd, fund_wide, shortable)
    cov = coverage_tables(plog, spot_wide, perp_wide, id_to_spot, id_to_perp, id_to_sym)
    never_ids = {
        int(x["id"])
        for x in (cov.get("long") or {}).get("never_listed") or []
        if x.get("reason") == "never_listed"
    }
    never = never_listed_contribution(nd, never_ids)

    print("[HB] hybrid overlay from frozen log (ledger numbers)...", flush=True)
    hyb_out = replay_long_leg(plog, cmc_close, spot_wide, perp_wide, fund_wide)
    hyb = hyb_out["full"]
    print(f"[HB] hybrid Sharpe={hyb.get('net_sharpe'):.3f} (ref {PHASE3C_REF_HYBRID_SHARPE:.3f})", flush=True)

    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    if cmc_panel_sha1 != cmc_panel_sha0:
        raise RuntimeError("CMC panel mutated during 3.e")
    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "pred_sha256": pred_hash["sha256"],
        "position_sha256": pos_sha,
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_readonly_ok": True,
        "combo_corr": float(COMBO_SPREADLS_CORR),
        "hybrid_sharpe": hyb.get("net_sharpe"),
        "hybrid_trail": hyb.get("net_sharpe_trail18m"),
        "cmc_sharpe": 1.8177621190077422,
        "id_to_sym": {str(k): v for k, v in id_to_sym.items()},
    }
    for r in conc.get("top_rows") or []:
        r["symbol"] = id_to_sym.get(int(r["id"]))

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_3e(
        ledger_path,
        confirmed=bool(verdict.get("confirmed")),
        verdict=verdict,
        extra=extra,
        hybrid=hyb,
        cmc={"net_sharpe": extra["cmc_sharpe"], "net_sharpe_trail18m": 2.457536436771376, "maxdd": -0.2581255543376627},
    )

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase3e(
        rep_dir / "btcb_phase3e_forensics.md",
        verdict=verdict,
        fr=fr,
        side=side,
        tier=tier,
        conc=conc,
        wf=wf,
        rankic=rankic,
        funding=fund_s,
        never=never,
        extra=extra,
    )
    plot_gap_waterfall(wf, chart_dir / "btcb_phase3e_gap_waterfall.png")
    plot_rankic_bars(rankic, chart_dir / "btcb_phase3e_rankic.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": PHASE3E_OUTCOME,
        "verdict": verdict,
        "funding_vs_repricing": fr,
        "by_side": side,
        "by_tier": tier,
        "concentration": conc,
        "waterfall": wf,
        "rankic": rankic,
        "funding": fund_s,
        "never_listed": never,
        "extra": extra,
    }
    (rep_dir / "btcb_phase3e_forensics.json").write_text(json.dumps(_jsonable(payload), indent=2, default=str))
    quant_vol.commit()

    print(
        f"VERDICT: {verdict.get('label')} RankIC_BN={float(verdict.get('rankic_bn_full') or float('nan')):.4f} "
        f"RankIC_CMC={float(verdict.get('rankic_cmc_full') or float('nan')):.4f} "
        f"(full Δ={float(verdict.get('d_full') or float('nan')):.4f}); "
        f"trail BN={float(verdict.get('rankic_bn_trail') or float('nan')):.4f} "
        f"CMC={float(verdict.get('rankic_cmc_trail') or float('nan')):.4f} "
        f"(Δ={float(verdict.get('d_trail') or float('nan')):.4f})",
        flush=True,
    )
    print(
        f"GAP WATERFALL: funding {100*float(wf.get('pct_funding') or 0):.1f}% "
        f"stale {100*float(wf.get('pct_stale') or 0):.1f}% "
        f"diffuse {100*float(wf.get('pct_diffuse') or 0):.1f}% "
        f"(PnL gap={float(wf.get('total_pnl_gap') or 0):.4f})",
        flush=True,
    )
    print(
        f"RankIC pair: BN full={float(verdict.get('rankic_bn_full') or float('nan')):.4f} "
        f"CMC same-names={float(verdict.get('rankic_cmc_full') or float('nan')):.4f}",
        flush=True,
    )
    print(
        f"SHORT-CARRY: held {float(fund_s.get('held_bps_day') or float('nan')):.2f} bps/day "
        f"vs universe {float(fund_s.get('universe_bps_day') or float('nan')):.2f} "
        f"(delta {float(fund_s.get('delta_bps_day') or float('nan')):.2f})",
        flush=True,
    )
    print(
        f"NEVER-LISTED longs: contrib={float(never.get('pnl_cmc') or 0):.4f} "
        f"share={100*float(never.get('share') or 0):.2f}% n={never.get('n_names')}",
        flush=True,
    )
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "verdict": verdict,
        "waterfall": wf,
        "funding": {
            "held_bps_day": fund_s.get("held_bps_day"),
            "universe_bps_day": fund_s.get("universe_bps_day"),
            "delta_bps_day": fund_s.get("delta_bps_day"),
        },
        "never": never,
        "extra": {k: extra[k] for k in extra if k != "id_to_sym"},
    }


@app.local_entrypoint()
def main():
    print("[local] Phase 3.e forensics...", flush=True)
    fc = run_phase3e.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase3e_forensics.md", "reports"),
        ("reports/btcb_phase3e_forensics.json", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase3e_gap_waterfall.png", "charts"),
        ("charts/btcb_phase3e_rankic.png", "charts"),
    ]
    for remote, kind in pulls:
        name = Path(remote).name
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOL_Q, remote, str(dest), "--force"], check=False)
        candidate = dest if dest.is_file() else dest / name
        if candidate.exists() and candidate.is_file():
            Path(kind).mkdir(exist_ok=True)
            shutil.copy2(candidate, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "charts").glob("btcb_phase3e*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase3e*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] Phase 3.e complete.", flush=True)
