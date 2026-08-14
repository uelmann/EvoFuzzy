"""
BTC-BEATER Phase 0 (data audit) + Phase 1 (naive benchmark).

BACKTEST ONLY. CPU only. Frozen COMBO untouched.
Usage: modal run btcb_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p0p1"
VOL_Q = "quant-baseline"
VOL_K = "kronos-crypto-data"

quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)
kronos_vol = modal.Volume.from_name(VOL_K, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "matplotlib",
        "httpx",
        "pyyaml",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("reports/btcb_addendum.md", remote_path="/root/btcb_addendum.md")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    timeout=60 * 60 * 3,
    retries=0,
    volumes={"/data/quant": quant_vol, "/data/kronos": kronos_vol},
    cpu=8,
    memory=32768,
)
def run_btcb() -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    from baseline.data import load_panel
    from btcb.audit import (
        binance_agreement,
        find_named,
        load_cmc_panel,
        per_coin_span,
        quality_flags,
        scan_usable_start,
        schema_report,
        draw_sample,
        year_end_top,
        _terminal,
    )
    from btcb.benchmark import naive_rotation
    from btcb.constants import (
        AGREE_N,
        LIQUID_AGREE,
        PHASE0_GATE,
        PHASE1_LABEL,
        PIT_DV_WINDOW,
        PIT_NS,
        SAMPLE_TOPN,
        SAMPLE_YEARS,
        SEED,
    )
    from btcb.report import plot_benchmark, write_phase0, write_phase1
    from btcb.universe import build_pit_topn, trailing_rank_frame

    t0 = time.time()
    addendum = Path("/root/btcb_addendum.md").read_text()
    if PHASE0_GATE not in addendum or PHASE1_LABEL not in addendum:
        raise RuntimeError("Addendum missing verbatim gate/label")
    print("[HB] BTC-BEATER P0+P1 BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {PHASE0_GATE}", flush=True)
    print(f"[HB] {PHASE1_LABEL}", flush=True)
    np.random.seed(SEED)

    src_full = Path("/data/kronos/historical_data_full.csv")
    src_small = Path("/data/kronos/historical_data.csv")
    src = src_full if src_full.exists() else src_small
    if not src.exists():
        raise RuntimeError("CMC historical csv missing on kronos-crypto-data volume")
    print(f"[HB] source={src} bytes={src.stat().st_size}", flush=True)

    cache = Path("/data/quant/btcb/cmc_panel.parquet")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print("[HB] reuse cached cmc_panel.parquet", flush=True)
        panel = pd.read_parquet(cache)
        panel["date"] = pd.to_datetime(panel["date"], utc=True)
    else:
        print("[HB] parsing CMC csv...", flush=True)
        panel = load_cmc_panel(src)
        panel.to_parquet(cache, index=False)
        quant_vol.commit()
        print(f"[HB] cached panel n={len(panel)}", flush=True)

    schema = schema_report(panel)
    print(f"[HB] schema rows={schema['n_rows']} ids={schema['n_ids']} {schema['date_min']}→{schema['date_max']}", flush=True)
    span = per_coin_span(panel)
    data_end = pd.Timestamp(panel["date"].max())
    if data_end.tzinfo is None:
        data_end = data_end.tz_localize("UTC")
    span["terminal"] = [_terminal(r["last"], data_end) for _, r in span.iterrows()]
    print(f"[HB] coins={len(span)} survivors={(span.terminal=='SURVIVOR').sum()} ended={(span.terminal=='ENDED').sum()}", flush=True)

    graveyard = find_named(span, panel, data_end)
    n_named_present = sum(1 for r in graveyard if r.get("present"))
    print(f"[HB] named graveyard present {n_named_present}/{len(graveyard)}", flush=True)

    sample_df = draw_sample(panel, span)
    print(f"[HB] sample n={len(sample_df)}", flush=True)
    sample_recs = []
    for _, r in sample_df.iterrows():
        sample_recs.append(
            {
                "id": int(r["id"]),
                "symbol": r["symbol"],
                "name": r["name"],
                "slug": r["slug"],
                "first": str(pd.Timestamp(r["first"]).date()),
                "last": str(pd.Timestamp(r["last"]).date()),
                "n": int(r["n"]),
                "gap_frac": float(r["gap_frac"]) if np.isfinite(r["gap_frac"]) else float("nan"),
                "terminal": r["terminal"],
                "in_years": r.get("in_years"),
            }
        )

    print("[HB] trailing PIT score...", flush=True)
    score, mcap_w, method = trailing_rank_frame(panel)
    print(f"[HB] PIT method={method} dates={len(score)} names={score.shape[1]}", flush=True)

    uni_dir = Path("/data/quant/btcb/universe")
    uni_dir.mkdir(parents=True, exist_ok=True)
    pit_files = {}
    for n in PIT_NS:
        print(f"[HB] building PIT top-{n}", flush=True)
        pit = build_pit_topn(score, n)
        dest = uni_dir / f"btcb_top{n}_pit.parquet"
        pit.to_parquet(dest, index=False)
        pit_files[n] = str(dest)
        print(f"[HB] PIT top-{n} rows={len(pit)} dates={pit['date'].nunique() if len(pit) else 0}", flush=True)

    n50 = int((score.notna().sum(axis=1) >= 50).sum()) if score.size else 0
    gate = scan_usable_start(sample_df, score, data_end)
    print(f"[HB] GATE {gate['verdict']}", flush=True)

    quality = quality_flags(panel, span)
    print(f"[HB] quality redenom={quality['n_redenom_suspects']} gap_p50={quality['gap_p50']:.3f}", flush=True)

    # Binance agreement
    raw_dir = Path("/data/quant/raw/klines")
    ksyms = sorted(p.stem for p in raw_dir.glob("*.parquet")) if raw_dir.exists() else []
    bases = {s[:-4] for s in ksyms if s.endswith("USDT")}
    overlap = [s for s in LIQUID_AGREE if s in set(panel["symbol"].str.upper()) and s in bases][:AGREE_N]
    print(f"[HB] agreement overlap={overlap}", flush=True)
    if overlap and ksyms:
        bnc = load_panel(raw_dir, [f"{s}USDT" for s in overlap])
        agree = binance_agreement(panel, bnc, overlap)
    else:
        agree = {"rows": [], "n_compared": 0, "median_corr": float("nan"), "suspect": True}
    print(f"[HB] median corr={agree.get('median_corr')} suspect={agree.get('suspect')}", flush=True)

    extra0 = {
        "source_path": str(src),
        "pit_method": method,
        "dv_window": PIT_DV_WINDOW,
        "pit_n50": n50,
        "pit_ndates": int(len(score)),
        "data_end": str(data_end.date()),
        "n_survivor": int((span["terminal"] == "SURVIVOR").sum()),
        "n_ended": int((span["terminal"] == "ENDED").sum()),
        "year_end_top200_n": {
            str(y): int(len(year_end_top(panel, y, SAMPLE_TOPN))) for y in SAMPLE_YEARS
        },
        "elapsed_sec": time.time() - t0,
        "source_sha256": hashlib.sha256(src.read_bytes() if src.stat().st_size < 50_000_000 else str(src.stat().st_size).encode()).hexdigest()
        if src.stat().st_size < 50_000_000
        else f"size={src.stat().st_size}",
    }
    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    repo_uni = Path("/data/quant/btcb/universe")
    for d in (rep_dir, chart_dir, repo_uni):
        d.mkdir(parents=True, exist_ok=True)

    write_phase0(
        rep_dir / "btcb_phase0_audit.md",
        schema=schema,
        graveyard=graveyard,
        sample=sample_recs,
        quality=quality,
        agree=agree,
        gate=gate,
        extra=extra0,
    )

    naive = control = None
    skipped = bool(gate.get("blocked") or not gate.get("usable_from"))
    extra1 = {"elapsed_sec": time.time() - t0, "skipped": skipped}
    if not skipped:
        print(f"[HB] Phase 1 window from {gate['usable_from']}", flush=True)
        pit50 = pd.read_parquet(uni_dir / "btcb_top50_pit.parquet")
        start = gate["usable_from"]
        naive = naive_rotation(panel, pit50, start, degenerate_btc=False)
        control = naive_rotation(panel, pit50, start, degenerate_btc=True)
        extra1["elapsed_sec"] = time.time() - t0
        plot_benchmark(naive, gate.get("usable_from"), chart_dir / "btcb_benchmark.png")
        print(
            f"[HB] naive rel_sharpe={naive.get('rel_sharpe')} book={naive.get('book_total')} "
            f"btc={naive.get('btc_total')} live={naive.get('live_benchmark')}",
            flush=True,
        )
        print(
            f"[HB] control book={control.get('book_total')} btc={control.get('btc_total')} "
            f"rel_sharpe={control.get('rel_sharpe')}",
            flush=True,
        )
    else:
        print("[HB] Phase 1 skipped (BLOCKED)", flush=True)

    write_phase1(
        rep_dir / "btcb_phase1_benchmark.md",
        naive=naive or {},
        control=control or {},
        gate=gate,
        extra=extra1,
    )

    drop = {"daily_ret", "btc_ret", "equity", "equity_btc", "rel_equity", "w_btc", "scan"}

    def _jsonable(x):
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items() if k not in drop}
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

    p0j = {
        "schema": schema,
        "graveyard": graveyard,
        "sample": sample_recs,
        "quality": quality,
        "agree": agree,
        "gate": _jsonable(gate),
        "extra": extra0,
        "pit_files": pit_files,
        "pit_method": method,
        "gpu_used": False,
    }
    p1j = {
        "skipped": skipped,
        "naive": _jsonable(naive or {}),
        "control": _jsonable(control or {}),
        "label": PHASE1_LABEL,
        "gate_verdict": gate.get("verdict"),
        "gpu_used": False,
    }
    if naive and isinstance(naive.get("equity"), pd.Series):
        eq = naive["equity"]
        eqb = naive["equity_btc"]
        rel = naive["rel_equity"]
        wbtc = naive["w_btc"]
        p1j["rel_equity_line"] = [
            {
                "date": str(pd.Timestamp(d).date()),
                "book": float(eq.loc[d]),
                "btc": float(eqb.loc[d]) if d in eqb.index else float("nan"),
                "rel": float(rel.loc[d]) if d in rel.index else float("nan"),
                "w_btc": float(wbtc.loc[d]) if d in wbtc.index else float("nan"),
            }
            for d in eq.index
        ]
    (rep_dir / "btcb_phase0_audit.json").write_text(json.dumps(p0j, indent=2, default=str))
    (rep_dir / "btcb_phase1_benchmark.json").write_text(json.dumps(p1j, indent=2, default=str))

    # copy PIT into a path we'll volume-get as universe/
    quant_vol.commit()

    n_named_miss = sum(1 for r in graveyard if not r.get("present"))
    print(f"USABLE-WINDOW: {gate.get('verdict')}", flush=True)
    print(
        f"Graveyard: named present {n_named_present}/{len(set(g['query'] for g in graveyard))} unique queries; "
        f"top-200 sample {len(sample_recs)}/{30} present "
        f"(ended={sum(1 for r in sample_recs if r['terminal']=='ENDED')} "
        f"survivor={sum(1 for r in sample_recs if r['terminal']=='SURVIVOR')}).",
        flush=True,
    )
    if naive:
        print(
            f"Benchmark relative-line Sharpe={naive.get('rel_sharpe'):.3f} "
            f"CAGR={naive.get('rel_cagr'):.3f}; "
            f"book total={naive.get('book_total'):.3f} vs BTC={naive.get('btc_total'):.3f}; "
            f"label={'LIVE BENCHMARK' if naive.get('live_benchmark') else 'NOT A LIVE BENCHMARK'}",
            flush=True,
        )
    else:
        print("Benchmark: SKIPPED", flush=True)
    print("COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s", flush=True)
    return {
        "verdict": gate.get("verdict"),
        "usable_from": gate.get("usable_from"),
        "blocked": gate.get("blocked"),
        "sample_n": len(sample_recs),
        "named_present": n_named_present,
        "rel_sharpe": None if not naive else naive.get("rel_sharpe"),
        "live": None if not naive else naive.get("live_benchmark"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] starting BTC-BEATER P0+P1...", flush=True)
    summary = run_btcb.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    Path("universe").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase0_audit.md", "reports"),
        ("reports/btcb_phase0_audit.json", "reports"),
        ("reports/btcb_phase1_benchmark.md", "reports"),
        ("reports/btcb_phase1_benchmark.json", "reports"),
        ("charts/btcb_benchmark.png", "charts"),
        ("btcb/universe/btcb_top50_pit.parquet", "universe"),
        ("btcb/universe/btcb_top100_pit.parquet", "universe"),
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
        for src in (art / "reports").glob("btcb*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] BTC-BEATER P0+P1 complete.", flush=True)
