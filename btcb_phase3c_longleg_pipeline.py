"""
SPREAD-LS long-leg overlay — replay frozen 3.c position log.

ANALYSIS ONLY. No new signals, no engine re-run, no CMC writes.
Usage: modal run btcb_phase3c_longleg_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p3c-longleg"
VOL_Q = "quant-baseline"
quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "matplotlib",
        "httpx",
        "pyyaml",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    timeout=60 * 60,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def run_long_leg() -> dict:
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from btcb.binance_replay import (
        build_id_symbol_map,
        close_wide_from_panel,
        funding_wide_from_panel,
        replay_long_leg,
    )
    from btcb.constants import PHASE2C_PRED_SHA256, PHASE3C_REF_END
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.phase3c_report import plot_long_leg_equity
    from btcb.spread_ls import hash_pred_dir

    t0 = time.time()
    print("[HB] long-leg replay from frozen 3.c position log; no engine; no CMC writes", flush=True)

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256:
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    print(f"[HB] 2.c cache ok sha256={pred_hash['sha256']}", flush=True)

    plog_path = Path("/data/quant/btcb/phase3c/position_log.parquet")
    if not plog_path.exists():
        raise RuntimeError("missing frozen position log; run Phase 3.c first")
    plog = pd.read_parquet(plog_path)
    print(f"[HB] position log rows={len(plog)} ids={plog['id'].nunique()}", flush=True)

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    panel = pd.read_parquet("/data/quant/btcb/full/panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()

    spot_dir = Path("/data/quant/raw/spot_klines")
    raw_dir = Path("/data/quant/raw/klines")
    fund_dir = Path("/data/quant/raw/funding")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet"))
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    fund_syms = sorted(p.stem for p in fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    spot_panel = load_panel(spot_dir, spot_syms)
    kline_panel = load_panel(raw_dir, kline_syms)
    funding = load_funding_panel(fund_dir, fund_syms) if fund_syms else pd.DataFrame(
        columns=["date", "symbol", "funding_rate", "n_events"]
    )
    for df in (spot_panel, kline_panel):
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if not funding.empty:
        funding["date"] = pd.to_datetime(funding["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        funding["symbol"] = funding["symbol"].astype(str).str.upper()

    all_ids = sorted(set(int(i) for i in plog["id"].unique()))
    nonempty_spot = set(spot_panel["symbol"].unique())
    nonempty_perp = set(kline_panel["symbol"].unique())
    id_to_spot = build_id_symbol_map(all_ids, cleaned, nonempty_spot, nonempty_spot)
    id_to_perp = build_id_symbol_map(all_ids, cleaned, nonempty_perp, nonempty_perp)

    cmc_close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    cmc_close.index = pd.to_datetime(cmc_close.index, utc=True).tz_convert("UTC").normalize()
    spot_wide = close_wide_from_panel(spot_panel, id_to_spot)
    perp_wide = close_wide_from_panel(kline_panel, id_to_perp)
    fund_wide = funding_wide_from_panel(funding, id_to_perp)
    print("[HB] replaying long/short/full from log...", flush=True)
    out = replay_long_leg(plog, cmc_close, spot_wide, perp_wide, fund_wide)

    lng = out["long"]
    sh = out["short"]
    full = out["full"]
    print(
        f"[HB] LONG Sharpe={lng.get('net_sharpe'):.3f} trail={lng.get('net_sharpe_trail18m'):.3f} "
        f"MaxDD={100*float(lng.get('maxdd')):.1f}% total={100*float(lng.get('book_total')):.1f}% "
        f"avg_gross={float(lng.get('avg_long_gross')):.3f}",
        flush=True,
    )
    print(
        f"[HB] SHORT Sharpe={sh.get('net_sharpe'):.3f} trail={sh.get('net_sharpe_trail18m'):.3f} "
        f"MaxDD={100*float(sh.get('maxdd')):.1f}% fund={float(sh.get('funding_total_pnl') or 0):.4f}",
        flush=True,
    )
    print(
        f"[HB] FULL  Sharpe={full.get('net_sharpe'):.3f} trail={full.get('net_sharpe_trail18m'):.3f} "
        f"MaxDD={100*float(full.get('maxdd')):.1f}%",
        flush=True,
    )

    chart_dir = Path("/data/quant/charts")
    rep_dir = Path("/data/quant/reports")
    chart_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    plot_long_leg_equity(
        lng["equity"],
        full["equity"],
        sh["equity"],
        chart_dir / "btcb_phase3c_long_leg_equity.png",
        long_sharpe=lng.get("net_sharpe"),
        full_sharpe=full.get("net_sharpe"),
    )
    md = "\n".join(
        [
            "# SPREAD-LS long-leg overlay (Phase 3.c position log)",
            "",
            "ANALYSIS ONLY. Frozen β-matched h=14 weights, **not renormalized**. "
            "Longs priced hybrid (Binance spot where live, else CMC). No funding on longs. "
            "Not an official book. 3.c validation / suspension unchanged.",
            "",
            f"- **Long leg:** Sharpe full `{lng.get('net_sharpe'):.3f}` / trail-18m `{lng.get('net_sharpe_trail18m'):.3f}` "
            f"/ MaxDD `{100*float(lng.get('maxdd')):.1f}%` / total `{100*float(lng.get('book_total')):.1f}%` "
            f"/ avg long gross `{float(lng.get('avg_long_gross')):.3f}`",
            f"- **Short leg (overlay):** Sharpe `{sh.get('net_sharpe'):.3f}` / trail `{sh.get('net_sharpe_trail18m'):.3f}` "
            f"/ MaxDD `{100*float(sh.get('maxdd')):.1f}%` / funding `{float(sh.get('funding_total_pnl') or 0):.4f}`",
            f"- **Full hybrid (overlay):** Sharpe `{full.get('net_sharpe'):.3f}` / trail `{full.get('net_sharpe_trail18m'):.3f}` "
            f"/ MaxDD `{100*float(full.get('maxdd')):.1f}%`",
            "",
            "Chart: `charts/btcb_phase3c_long_leg_equity.png`.",
            "",
        ]
    )
    (rep_dir / "btcb_phase3c_long_leg.md").write_text(md)
    slim = {
        "long": {k: lng.get(k) for k in ("n_days", "start", "end", "net_sharpe", "net_sharpe_trail18m", "maxdd", "book_total", "avg_long_gross")},
        "short": {k: sh.get(k) for k in ("n_days", "net_sharpe", "net_sharpe_trail18m", "maxdd", "book_total", "funding_total_pnl")},
        "full": {k: full.get(k) for k in ("n_days", "net_sharpe", "net_sharpe_trail18m", "maxdd", "book_total")},
        "note": "weights not renormalized; analysis only",
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }
    (rep_dir / "btcb_phase3c_long_leg.json").write_text(json.dumps(slim, indent=2, default=str))
    quant_vol.commit()
    print(
        f"LONG-LEG Sharpe full={lng.get('net_sharpe'):.3f} / trailing={lng.get('net_sharpe_trail18m'):.3f} "
        f"MaxDD={100*float(lng.get('maxdd')):.1f}%",
        flush=True,
    )
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return slim


@app.local_entrypoint()
def main():
    print("[local] long-leg overlay...", flush=True)
    fc = run_long_leg.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase3c_long_leg.md", "reports"),
        ("reports/btcb_phase3c_long_leg.json", "reports"),
        ("charts/btcb_phase3c_long_leg_equity.png", "charts"),
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
        src = art / "charts" / "btcb_phase3c_long_leg_equity.png"
        if src.exists():
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase3c_long_leg*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] long-leg overlay complete.", flush=True)
