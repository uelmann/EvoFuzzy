"""
BTC-BEATER Phase 4.b — reconstruct crude-book equity curves from cached preds.

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. No training.
Frozen products untouched. Master only.
Usage: modal run btcb_phase4b_equity_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p4b-equity"
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
    .env({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/btcb_phase4b_twinrank.md", remote_path="/root/btcb_phase4b_twinrank.md")
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


def _load_close_long(raw_dir: Path, symbols: list[str]):
    import pandas as pd

    parts = []
    for sym in symbols:
        pq = raw_dir / f"{sym}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if df is None or df.empty or "close" not in df.columns:
            continue
        if "symbol" not in df.columns:
            df["symbol"] = sym
        parts.append(df[["date", "close", "symbol"]])
    if not parts:
        return pd.DataFrame(columns=["date", "close", "symbol"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "close"])


@app.function(
    timeout=60 * 60 * 2,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=65536,
)
def run_btcb_p4b_equity() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd

    from btcb.academic_factor import pit_members
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
        PHASE4B_H,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.oracle_ladder import _as_utc, ffill_members, formation_dates, run_periodic_long
    from btcb.phase4b import (
        assert_books_match,
        load_tagged_preds,
        merge_dir_spread,
        persist_book_daily_rets,
        twinrank_from_heads,
    )
    from btcb.phase4b_report import ensure_equity_chart_notes, plot_equity_curves
    from btcb.phase4v2 import collapse_fold_preds, cs_rank_blend, preds_to_score_at
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache

    t0 = time.time()
    print("[HB] Phase 4.b equity reconstruction; no training; nothing adopted", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    start = pd.Timestamp(PHASE3C_REF_START, tz="UTC")
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

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    twin = load_twin_from_cache(pred_dir_2c, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()

    print("[HB] loading Binance spot close (canonical books)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet")) if spot_dir.exists() else []
    all_ids = sorted(set(int(i) for i in pit["id"].unique()) | {int(btc_id)})
    id_to_spot = build_id_symbol_map(all_ids, cleaned, set(spot_syms), set(spot_syms))
    id_to_spot[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in set(spot_syms) else id_to_spot.get(int(btc_id))
    spot_needed = sorted({s for s in id_to_spot.values() if s})
    spot_long = _load_close_long(spot_dir, spot_needed)
    close = close_wide_from_panel(spot_long.rename(columns={"close": "close"}), id_to_spot)
    if int(btc_id) not in close.columns:
        raise RuntimeError("BTCUSDT spot missing from close wide")
    close = close[close.index <= end].sort_index()
    close.index = pd.DatetimeIndex([_as_utc(d) for d in close.index])
    print(f"[HB] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    p4v2_pred = Path("/data/quant/btcb/phase4v2/preds")
    pred_out = Path("/data/quant/btcb/phase4b/preds")
    rank_top = load_tagged_preds(p4v2_pred, "rank_s", PHASE4B_H)
    rank_bot = load_tagged_preds(pred_out, "rank_bot_s", PHASE4B_H)
    dir_top = load_tagged_preds(pred_out, "dir_top", PHASE4B_H)
    missing = [
        name
        for name, df in (("rank_s", rank_top), ("rank_bot_s", rank_bot), ("dir_top", dir_top))
        if df is None or df.empty
    ]
    if missing:
        raise RuntimeError(f"missing Phase 4.b pred caches (no retrain): {missing}")
    print(
        f"[HB] caches rank_top={len(rank_top)} rank_bot={len(rank_bot)} dir_top={len(dir_top)}",
        flush=True,
    )

    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
    top_c = collapse_fold_preds(rank_top, "p")
    bot_c = collapse_fold_preds(rank_bot, "p")
    twin_sig = twinrank_from_heads(top_c, bot_c, "p", "p", "twinrank")
    blend = cs_rank_blend(frozen, twin_sig, "spread", "twinrank", "spread_twinrank")
    dir_raw = merge_dir_spread(dir_top, twin)
    if dir_raw.empty:
        raise RuntimeError("DIR-spread merge empty; fold alignment failed")
    dir_df = collapse_fold_preds(dir_raw, "dir_spread")
    dir_blend = cs_rank_blend(dir_df, twin_sig, "dir_spread", "twinrank", "dir_twinrank")

    signals = {
        "frozen_spread": (frozen, "spread"),
        "twinrank": (twin_sig, "twinrank"),
        "spread_twinrank": (blend, "spread_twinrank"),
        "dir_spread": (dir_df, "dir_spread"),
        "dir_twinrank": (dir_blend, "dir_twinrank"),
    }

    dates = list(close.index)
    members = ffill_members(pit_members(pit, btc_id), dates)
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(PHASE4B_H))
    print(f"[HB] crude books formations={len(pairs14)}", flush=True)
    books = {}
    for name, (df, col) in signals.items():
        scores = preds_to_score_at(df, col, [t for t, _, _ in pairs14])
        packed = run_periodic_long(
            close,
            members,
            btc_id,
            scores,
            pairs14,
            cost_bps=float(ALT_BPS),
            label=name,
        )
        books[name] = packed
        print(
            f"[HB] book {name} CAGR={packed.get('cagr')} MaxDD={packed.get('maxdd')}",
            flush=True,
        )

    rec_path = Path("/data/quant/reports/btcb_phase4b_twinrank.json")
    recorded = {}
    if rec_path.exists():
        recorded = (json.loads(rec_path.read_text()) or {}).get("books") or {}
        assert_books_match(books, recorded)
        print("[HB] reconstructed books match recorded CAGR/MaxDD/Sharpe", flush=True)

    work = Path("/data/quant/btcb/phase4b")
    persist_book_daily_rets(books, work / "books")
    chart_dir = Path("/data/quant/charts")
    rep_dir = Path("/data/quant/reports")
    for d in (chart_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(books, chart_dir / "btcb_phase4b_equity.png")

    md_src = Path("/data/quant/reports/btcb_phase4b_twinrank.md")
    if not md_src.exists():
        md_src = Path("/root/btcb_phase4b_twinrank.md")
    md_text = md_src.read_text()
    md_text = ensure_equity_chart_notes(md_text)
    (rep_dir / "btcb_phase4b_twinrank.md").write_text(md_text)

    quant_vol.commit()
    summary = {
        name: {
            "cagr": packed.get("cagr"),
            "maxdd": packed.get("maxdd"),
            "sharpe": packed.get("sharpe"),
            "total": packed.get("total"),
            "n_formations": packed.get("n_formations"),
            "n_days": packed.get("n_days"),
            "start": packed.get("start"),
            "end": packed.get("end"),
        }
        for name, packed in books.items()
    }
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {"books": summary, "elapsed_sec": time.time() - t0, "gpu_used": False}


@app.local_entrypoint()
def main():
    print("[local] Phase 4.b equity curves from cached preds...", flush=True)
    fc = run_btcb_p4b_equity.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("charts/btcb_phase4b_equity.png", "charts"),
        ("reports/btcb_phase4b_twinrank.md", "reports"),
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
    png = Path("charts/btcb_phase4b_equity.png")
    if opt.exists() and png.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        (opt / "charts" / png.name).write_bytes(png.read_bytes())
        (opt / "screenshots" / png.name).write_bytes(png.read_bytes())
        md = Path("reports/btcb_phase4b_twinrank.md")
        if md.exists():
            (opt / "reports" / md.name).write_bytes(md.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] Phase 4.b equity complete.", flush=True)
