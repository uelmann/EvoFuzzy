"""
BTC-BEATER academic factor — unconstrained D10−D1 on CMC + implementation tax.

ANALYSIS ONLY. Frozen 2.c signals. CPU only. Zero GPU.
Usage: modal run btcb_academic_factor_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-academic-factor"
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
        "reports/btcb_academic_factor_addendum.md",
        remote_path="/root/btcb_academic_factor_addendum.md",
    )
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

    drop = drop or {
        "daily_ret",
        "equity",
        "daily",
        "id_to_sym",
        "contrib",
        "position_log",
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
    timeout=60 * 60,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def run_academic_factor_job() -> dict:
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from btcb.academic_factor import (
        factor_metrics,
        last_close_map,
        paper_alpha_verdict,
        pit_members,
        replay_cmc_book_from_log,
        run_academic_factor,
        series_corr,
        slim_factor,
        spread_wide,
        waterfall_table,
    )
    from btcb.academic_factor_report import plot_factor_equity, plot_waterfall, write_academic_factor
    from btcb.binance_replay import (
        build_id_symbol_map,
        close_wide_from_panel,
        funding_wide_from_panel,
        replay_long_leg,
    )
    from btcb.constants import (
        ACADEMIC_FACTOR_CRITERION,
        ACADEMIC_FACTOR_H,
        ACADEMIC_FACTOR_NAIVE_BPS,
        DEATH_CONVENTION,
        LS_LONG_BPS,
        LS_SHORT_FEE_BPS,
        LS_SHORT_SLIP_BPS,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_POSITION_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_HYBRID_SHARPE,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import build_floored_pit, clean_panel
    from btcb.spread_ls import _hash_position_log, build_shortable, hash_pred_dir, load_twin_from_cache

    t0 = time.time()
    addendum = Path("/root/btcb_academic_factor_addendum.md").read_text()
    for txt in (ACADEMIC_FACTOR_CRITERION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"academic-factor addendum missing freeze text: {txt[:80]}")
    print("[HB] BTC-BEATER academic factor ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {ACADEMIC_FACTOR_CRITERION}", flush=True)

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

    def _load_pit(name: str):
        for p in (
            Path(f"/data/quant/btcb/universe/{name}"),
            Path(f"/data/quant/universe/{name}"),
            Path(f"/root/{name}"),
        ):
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                df["id"] = df["id"].astype(int)
                df = df[df["date"] <= end].copy()
                print(f"[HB] pit {name} from {p} rows={len(df)}", flush=True)
                return df
        return None

    pit50 = _load_pit("btcb_top50_floor.parquet")
    pit100 = _load_pit("btcb_top100_floor.parquet")
    if pit50 is None or pit100 is None:
        raise RuntimeError("missing floored PIT top-50/100")

    print("[HB] re-applying frozen 2.b cleaner (no CMC writes)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()
    last_map = last_close_map(cleaned)
    close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n_files={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(
            f"2.c cache hash mismatch got={pred_hash['sha256']} n={pred_hash['n_files']}"
        )
    twin = load_twin_from_cache(pred_dir, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()
    swide = spread_wide(twin)
    print(f"[HB] twin h={PHASE3C_REF_H} rows={len(twin)} dates={twin['date'].nunique()}", flush=True)

    raw_dir = Path("/data/quant/raw/klines")
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    kline_panel = load_panel(raw_dir, kline_syms)
    kline_panel["date"] = pd.to_datetime(kline_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    kline_panel["symbol"] = kline_panel["symbol"].astype(str).str.upper()
    shortable = build_shortable(cleaned, kline_panel, btc_id)
    print(f"[HB] shortable dates={len(shortable)}", flush=True)

    mem50 = pit_members(pit50, btc_id)
    mem100 = pit_members(pit100, btc_id)
    print("[HB] building informational mcap PIT (in-memory, not written)...", flush=True)
    pit_m50, _ = build_floored_pit(cleaned, 50, score="mcap")
    pit_m100, _ = build_floored_pit(cleaned, 100, score="mcap")
    mem_m50 = pit_members(pit_m50, btc_id)
    mem_m100 = pit_members(pit_m100, btc_id)

    naive = float(ACADEMIC_FACTOR_NAIVE_BPS)
    real_s = float(LS_SHORT_FEE_BPS + LS_SHORT_SLIP_BPS)

    def _run(mem, h, shortable_map, lbps, sbps, label):
        print(f"[HB] RUN {label} h={h} shortable={shortable_map is not None} bps={lbps}/{sbps}", flush=True)
        packed = run_academic_factor(
            close,
            mem,
            swide,
            last_map,
            btc_id,
            h=int(h),
            shortable=shortable_map,
            long_bps=float(lbps),
            short_bps=float(sbps),
            label=label,
        )
        if packed.get("error"):
            raise RuntimeError(f"{label} failed: {packed}")
        print(
            f"[HB] {label} Sharpe={packed.get('sharpe'):.3f} trail={packed.get('sharpe_trail18m'):.3f} "
            f"NW-t={packed.get('nw_t'):.2f} n={packed.get('n_days')}",
            flush=True,
        )
        return packed

    tables_raw = {
        "daily_dv100_gross": _run(mem100, 1, None, 0.0, 0.0, "daily-dv100-gross"),
        "daily_dv100_naive": _run(mem100, 1, None, naive, naive, "daily-dv100-naive"),
        "daily_dv50_gross": _run(mem50, 1, None, 0.0, 0.0, "daily-dv50-gross"),
        "daily_dv50_naive": _run(mem50, 1, None, naive, naive, "daily-dv50-naive"),
        "jt_dv100_gross": _run(mem100, ACADEMIC_FACTOR_H, None, 0.0, 0.0, "jt-dv100-gross"),
        "jt_dv100_naive": _run(mem100, ACADEMIC_FACTOR_H, None, naive, naive, "jt-dv100-naive"),
        "jt_dv50_gross": _run(mem50, ACADEMIC_FACTOR_H, None, 0.0, 0.0, "jt-dv50-gross"),
        "jt_dv50_naive": _run(mem50, ACADEMIC_FACTOR_H, None, naive, naive, "jt-dv50-naive"),
        "jt_mcap100_gross": _run(mem_m100, ACADEMIC_FACTOR_H, None, 0.0, 0.0, "jt-mcap100-gross"),
        "jt_mcap100_naive": _run(mem_m100, ACADEMIC_FACTOR_H, None, naive, naive, "jt-mcap100-naive"),
        "jt_mcap50_gross": _run(mem_m50, ACADEMIC_FACTOR_H, None, 0.0, 0.0, "jt-mcap50-gross"),
        "jt_mcap50_naive": _run(mem_m50, ACADEMIC_FACTOR_H, None, naive, naive, "jt-mcap50-naive"),
        "jt_dv100_shortable": _run(mem100, ACADEMIC_FACTOR_H, shortable, naive, naive, "jt-dv100-shortable"),
        "jt_dv100_real": _run(
            mem100, ACADEMIC_FACTOR_H, shortable, float(LS_LONG_BPS), real_s, "jt-dv100-real"
        ),
    }

    plog_path = Path("/data/quant/btcb/phase3c/position_log.parquet")
    if not plog_path.exists():
        raise RuntimeError("missing frozen 3.c position log")
    plog = pd.read_parquet(plog_path)
    pos_sha = _hash_position_log(plog)
    print(f"[HB] position log sha256={pos_sha}", flush=True)
    if pos_sha != PHASE3C_POSITION_SHA256:
        raise RuntimeError(f"position log mutated {pos_sha}")

    print("[HB] CMC-priced implementable book from frozen log...", flush=True)
    cmc_book = replay_cmc_book_from_log(plog, close)
    cmc_metrics = factor_metrics(cmc_book)

    print("[HB] hybrid implementable book from frozen log (Binance+funding)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    fund_dir = Path("/data/quant/raw/funding")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet"))
    fund_syms = sorted(p.stem for p in fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    spot_panel = load_panel(spot_dir, spot_syms)
    funding = (
        load_funding_panel(fund_dir, fund_syms)
        if fund_syms
        else pd.DataFrame(columns=["date", "symbol", "funding_rate", "n_events"])
    )
    for df in (spot_panel,):
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
    spot_wide = close_wide_from_panel(spot_panel, id_to_spot)
    perp_wide = close_wide_from_panel(kline_panel, id_to_perp)
    fund_wide = funding_wide_from_panel(funding, id_to_perp)
    hyb_out = replay_long_leg(plog, close, spot_wide, perp_wide, fund_wide)
    hyb_net = hyb_out["full_net"]
    hyb_metrics = factor_metrics(hyb_net)
    hyb_metrics["net_sharpe"] = hyb_metrics["sharpe"]
    hyb_metrics["net_sharpe_trail18m"] = hyb_metrics["sharpe_trail18m"]
    print(
        f"[HB] HYBRID Sharpe={hyb_metrics.get('sharpe'):.3f} (ref {PHASE3C_REF_HYBRID_SHARPE:.3f}) "
        f"trail={hyb_metrics.get('sharpe_trail18m'):.3f}",
        flush=True,
    )
    if abs(float(hyb_metrics.get("sharpe")) - float(PHASE3C_REF_HYBRID_SHARPE)) > 0.02:
        print("[WARN] hybrid Sharpe drifted from 3.c reference; continuing (measurement)", flush=True)

    jt_g = tables_raw["jt_dv100_gross"]
    verdict = paper_alpha_verdict(jt_g)
    wf = waterfall_table(
        jt_g,
        tables_raw["jt_dv100_naive"],
        tables_raw["jt_dv100_shortable"],
        tables_raw["jt_dv100_real"],
        hyb_metrics,
    )
    corr = {
        "vs_cmc": series_corr(jt_g["daily_ret"], cmc_book),
        "vs_hybrid": series_corr(jt_g["daily_ret"], hyb_net),
    }
    legs = {
        "factor": jt_g,
        "long": jt_g.get("long") or {},
        "short": jt_g.get("short") or {},
        "universe": jt_g.get("universe") or {},
        "lmU": jt_g.get("lmU") or {},
        "umS": jt_g.get("umS") or {},
    }

    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    if cmc_panel_sha1 != cmc_panel_sha0:
        raise RuntimeError("CMC panel mutated during academic-factor run")
    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "pred_sha256": pred_hash["sha256"],
        "pred_n_files": pred_hash["n_files"],
        "position_sha256": pos_sha,
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_readonly_ok": True,
        "cmc_book_sharpe": cmc_metrics.get("sharpe"),
        "hybrid_sharpe": hyb_metrics.get("sharpe"),
    }

    tables_slim = {k: slim_factor(v) for k, v in tables_raw.items()}
    chart_dir = Path("/data/quant/charts")
    rep_dir = Path("/data/quant/reports")
    chart_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    write_academic_factor(
        rep_dir / "btcb_academic_factor.md",
        tables=tables_slim,
        legs=legs,
        corr=corr,
        waterfall=wf,
        verdict=verdict,
        extra=extra,
    )
    hyb_eq = (1.0 + hyb_net.fillna(0.0)).cumprod()
    plot_factor_equity(
        jt_g["equity"],
        hyb_eq,
        chart_dir / "btcb_academic_factor_equity.png",
        factor_label=f"FACTOR-JT top-100 GROSS  Sh={jt_g.get('sharpe'):.2f}",
        book_label=f"3.c hybrid book  Sh={hyb_metrics.get('sharpe'):.2f}",
    )
    plot_waterfall(
        wf["rows"],
        chart_dir / "btcb_academic_factor_waterfall.png",
        tax=wf.get("tax"),
    )
    payload = {
        "criterion": ACADEMIC_FACTOR_CRITERION,
        "verdict": verdict,
        "waterfall": wf,
        "corr": corr,
        "tables": tables_slim,
        "legs": {
            "long": slim_factor(legs["long"]),
            "short": slim_factor(legs["short"]),
            "universe": slim_factor(legs["universe"]),
            "lmU": slim_factor(legs["lmU"]),
            "umS": slim_factor(legs["umS"]),
        },
        "extra": extra,
        "cmc_book": slim_factor(cmc_metrics),
        "hybrid_book": slim_factor(hyb_metrics),
    }
    # slim_factor on factor_metrics-only dicts (no avg_n_*) is fine
    (rep_dir / "btcb_academic_factor.json").write_text(json.dumps(_jsonable(payload), indent=2, default=str))
    quant_vol.commit()

    rows = wf["rows"]

    def _s(i):
        return float(rows[i]["sharpe"])

    def _d(i):
        return float(rows[i]["delta"])

    print(
        f"PAPER-ALPHA {verdict['label']} gross_sharpe={verdict['sharpe']:.3f} "
        f"nw_t={verdict['nw_t']:.2f} (need Sharpe>=1.0 NW-t>=3.0)",
        flush=True,
    )
    print(
        f"WATERFALL paper={_s(0):.3f} → naive={_s(1):.3f} (Δ={_d(1):+.3f}) "
        f"→ shortability={_s(2):.3f} (Δ={_d(2):+.3f}) "
        f"→ real_costs={_s(3):.3f} (Δ={_d(3):+.3f}) "
        f"→ hybrid={_s(4):.3f} (Δ={_d(4):+.3f}); TAX={wf['tax']:.3f}",
        flush=True,
    )
    print(
        f"LEG long-minus-uni Sharpe={float(legs['lmU'].get('sharpe')):.3f} "
        f"ann_mean={float(legs['lmU'].get('ann_mean')):.3f}; "
        f"uni-minus-short Sharpe={float(legs['umS'].get('sharpe')):.3f} "
        f"ann_mean={float(legs['umS'].get('ann_mean')):.3f}",
        flush=True,
    )
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "verdict": verdict,
        "waterfall": wf,
        "corr": corr,
        "extra": extra,
        "legs": {
            "lmU": {k: legs["lmU"].get(k) for k in ("sharpe", "ann_mean", "nw_t")},
            "umS": {k: legs["umS"].get(k) for k in ("sharpe", "ann_mean", "nw_t")},
        },
    }


@app.local_entrypoint()
def main():
    print("[local] academic factor overlay...", flush=True)
    fc = run_academic_factor_job.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_academic_factor.md", "reports"),
        ("reports/btcb_academic_factor.json", "reports"),
        ("charts/btcb_academic_factor_equity.png", "charts"),
        ("charts/btcb_academic_factor_waterfall.png", "charts"),
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
        for src in (art / "charts").glob("btcb_academic_factor*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_academic_factor*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] academic factor complete.", flush=True)
