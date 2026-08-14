"""
BTC-BEATER Phase 0.c — full CMC re-download (active+inactive), re-audit, honest-window benchmark.

DATA + ANALYSIS ONLY. CPU only. Frozen COMBO untouched.
Usage: modal run btcb_phase0c_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p0c"
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
        "requests",
        "pyyaml",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("reports/btcb_phase0c_addendum.md", remote_path="/root/btcb_phase0c_addendum.md")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    timeout=60 * 60 * 6,
    retries=0,
    volumes={"/data/quant": quant_vol, "/data/kronos": kronos_vol},
    cpu=8,
    memory=32768,
)
def run_btcb_0c() -> dict:
    import numpy as np
    import pandas as pd

    from btcb.audit import find_named_0c, per_coin_span, quality_flags, schema_report, _terminal
    from btcb.benchmark import naive_rotation_v3
    from btcb.cmc_client import CmcPublic
    from btcb.constants import (
        DEATH_CONVENTION,
        DOWNLOAD_SLEEP_S,
        ENDED_BEFORE_YEAR,
        PHASE0C_GATE,
        PHASE1_LABEL,
        PIT_NS,
        PIT_DV_WINDOW,
    )
    from btcb.coverage import scan_usable_from_snapshots
    from btcb.download import (
        assemble_panel,
        build_target_ids,
        credit_guard,
        download_ohlcv,
        load_current_828,
        seed_from_existing_panel,
    )
    from btcb.report import plot_benchmark_v3, plot_coverage, write_phase0c, write_phase1_v3
    from btcb.universe import build_pit_topn_ids, trailing_rank_frame_by_id

    t0 = time.time()
    addendum = Path("/root/btcb_phase0c_addendum.md").read_text()
    if PHASE0C_GATE not in addendum or DEATH_CONVENTION not in addendum or PHASE1_LABEL not in addendum:
        raise RuntimeError("Phase 0.c addendum missing verbatim gate/convention/label")
    print("[HB] BTC-BEATER P0c DATA+ANALYSIS ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {PHASE0C_GATE}", flush=True)
    print(f"[HB] {DEATH_CONVENTION}", flush=True)
    print(f"[HB] {PHASE1_LABEL}", flush=True)
    print("[HB] Old-archive / 2018-circular benchmarks discarded unread.", flush=True)

    root = Path("/data/quant/btcb/full")
    snap_dir = root / "snapshots"
    ohlcv_dir = root / "ohlcv"
    for d in (root, snap_dir, ohlcv_dir):
        d.mkdir(parents=True, exist_ok=True)

    def commit():
        quant_vol.commit()

    api = CmcPublic(sleep_s=DOWNLOAD_SLEEP_S)
    print("[HB] fetching full map (active+inactive+untracked)...", flush=True)
    cmap = api.fetch_full_map()
    map_path = root / "cmc_map.parquet"
    cmap.to_parquet(map_path, index=False)
    n_active = int((cmap["listing_status"] == "active").sum())
    n_inactive = int((cmap["listing_status"] == "inactive").sum())
    n_untracked = int((cmap["listing_status"] == "untracked").sum())
    print(f"[HB] map n={len(cmap)} active={n_active} inactive={n_inactive} untracked={n_untracked}", flush=True)

    current_path = Path("/data/kronos/universe_meta.csv")
    current_ids = load_current_828(current_path) if current_path.exists() else set()
    print(f"[HB] current-828 n={len(current_ids)}", flush=True)

    target, snap_prov = build_target_ids(api, snap_dir, current_ids)
    target_ids = sorted(target)
    (root / "target_ids.json").write_text(json.dumps({"n": len(target_ids), "ids": target_ids}))
    commit()
    print(f"[HB] target union n={len(target_ids)}", flush=True)
    seed_from_existing_panel(Path("/data/quant/btcb/cmc_panel.parquet"), ohlcv_dir, set(target_ids))
    commit()

    cached = {p.stem for p in ohlcv_dir.glob("*.parquet") if p.stem.isdigit()}
    cached |= {p.stem for p in ohlcv_dir.glob("*.empty") if p.stem.isdigit()}
    guard = credit_guard(n_ids=len(target_ids), n_cached=len(cached), n_snapshots=len(snap_prov))
    print(f"[HB] CREDIT GUARD plan={guard['plan']}", flush=True)
    print(
        f"[HB] credits_projected={guard['credits_projected']} available={guard['credits_available']} "
        f"http_remaining={guard['http_remaining']} hard_stop={guard['hard_stop']}",
        flush=True,
    )
    if guard["hard_stop"]:
        print("[HB] HARD-STOP credit/HTTP guard. Reduction proposal:", flush=True)
        print(json.dumps(guard["reduction_proposal"], indent=2), flush=True)
        commit()
        return {"hard_stop": True, "guard": guard, "n_target": len(target_ids)}

    state = download_ohlcv(
        api,
        cmap,
        target_ids,
        ohlcv_dir,
        root / "download_state.json",
        commit_fn=commit,
        save_every=25,
    )
    print(
        f"[HB] download done completed={len(state.get('completed', []))} "
        f"empty={len(state.get('empty', []))} failed={len(state.get('failed', []))} "
        f"http={api.http_count} credits={api.credit_count}",
        flush=True,
    )

    panel_path = root / "panel.parquet"
    print("[HB] assembling panel...", flush=True)
    panel = assemble_panel(ohlcv_dir, cmap, panel_path)
    commit()

    schema = schema_report(panel)
    span = per_coin_span(panel)
    data_end = pd.Timestamp(panel["date"].max())
    if data_end.tzinfo is None:
        data_end = data_end.tz_localize("UTC")
    span["terminal"] = [_terminal(r["last"], data_end) for _, r in span.iterrows()]
    cutoff = pd.Timestamp(f"{ENDED_BEFORE_YEAR}-01-01", tz="UTC")
    n_ended = int((span["last"] < cutoff).sum())
    ended = {
        "n_ended": n_ended,
        "n_ids": int(len(span)),
        "n_survivor": int((span["terminal"] == "SURVIVOR").sum()),
        "before": str(cutoff.date()),
        "fail": n_ended == 0,
    }
    if ended["fail"]:
        print(f"ENDED-COUNT FAIL: ended=0 / {ended['n_ids']} (survivorship still present)", flush=True)
    else:
        print(f"ENDED-COUNT: {n_ended}/{ended['n_ids']} histories end before {cutoff.date()}", flush=True)

    graveyard = find_named_0c(span, data_end)
    n_g = sum(1 for r in graveyard if r.get("present_with_terminal"))
    print(f"Graveyard: {n_g}/{len(graveyard)} present-with-terminal", flush=True)

    quality = quality_flags(panel, span)

    print("[HB] trailing PIT by id...", flush=True)
    score, _, method, last_sym = trailing_rank_frame_by_id(panel)
    uni_dir = Path("/data/quant/btcb/universe")
    uni_dir.mkdir(parents=True, exist_ok=True)
    pit_files = {}
    for n in PIT_NS:
        pit = build_pit_topn_ids(score, n, last_sym)
        dest = uni_dir / f"btcb_top{n}_pit.parquet"
        pit.to_parquet(dest, index=False)
        pit_files[n] = str(dest)
        print(f"[HB] PIT top-{n} rows={len(pit)} dates={pit['date'].nunique() if len(pit) else 0}", flush=True)

    gate = scan_usable_from_snapshots(panel, snap_dir)
    print(f"[HB] GATE {gate['verdict']}", flush=True)

    extra0 = {
        "plan": guard["plan"],
        "credits_projected": guard["credits_projected"],
        "credits_available": guard["credits_available"],
        "credit_count": api.credit_count,
        "http_remaining": guard["http_remaining"],
        "http_count": api.http_count,
        "hard_stop": False,
        "n_target": len(target_ids),
        "n_cached": len(cached),
        "n_map": int(len(cmap)),
        "n_active": n_active,
        "n_inactive": n_inactive,
        "n_untracked": n_untracked,
        "pit_method": method,
        "dv_window": PIT_DV_WINDOW,
        "snapshot_provenance": snap_prov,
        "elapsed_sec": time.time() - t0,
        "download_empty": len(state.get("empty", [])),
        "download_failed": state.get("failed", []),
    }
    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_phase0c(
        rep_dir / "btcb_phase0c_rebuild.md",
        schema=schema,
        graveyard=graveyard,
        ended=ended,
        quality=quality,
        gate=gate,
        extra=extra0,
    )
    plot_coverage(gate, chart_dir / "btcb_coverage.png")

    naive = control = None
    skipped = bool(gate.get("blocked") or not gate.get("usable_from"))
    extra1 = {"elapsed_sec": time.time() - t0, "skipped": skipped}
    if not skipped:
        print(f"[HB] Phase 1 v3 window from {gate['usable_from']}", flush=True)
        pit50 = pd.read_parquet(uni_dir / "btcb_top50_pit.parquet")
        start = gate["usable_from"]
        naive = naive_rotation_v3(panel, pit50, start, degenerate_btc=False)
        control = naive_rotation_v3(panel, pit50, start, degenerate_btc=True)
        extra1["elapsed_sec"] = time.time() - t0
        plot_benchmark_v3(naive, gate.get("usable_from"), chart_dir / "btcb_benchmark_v3.png")
        fe = naive.get("forced_exits") or {}
        print(
            f"[HB] naive rel_sharpe={naive.get('rel_sharpe')} book={naive.get('book_total')} "
            f"btc={naive.get('btc_total')} live={naive.get('live_benchmark')} "
            f"forced={fe.get('n_events')}",
            flush=True,
        )
    else:
        print("[HB] Phase 1 v3 skipped (BLOCKED)", flush=True)

    write_phase1_v3(
        rep_dir / "btcb_phase1_benchmark_v3.md",
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
        "ended": ended,
        "quality": {k: v for k, v in quality.items() if k != "redenom_suspects"},
        "gate": _jsonable(gate),
        "extra": extra0,
        "pit_files": pit_files,
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
    (rep_dir / "btcb_phase0c_rebuild.json").write_text(json.dumps(p0j, indent=2, default=str))
    (rep_dir / "btcb_phase1_benchmark_v3.json").write_text(json.dumps(p1j, indent=2, default=str))
    commit()

    print(f"ENDED-COUNT: {ended['n_ended']}/{ended['n_ids']}" + (" FAIL" if ended["fail"] else ""), flush=True)
    print(f"Graveyard: {n_g}/{len(graveyard)} present-with-terminal", flush=True)
    print(f"USABLE-FROM: {gate.get('verdict')}", flush=True)
    if naive:
        fe = naive.get("forced_exits") or {}
        print(
            f"Benchmark relative-Sharpe={naive.get('rel_sharpe'):.3f} "
            f"book={naive.get('book_total'):.3f} vs BTC={naive.get('btc_total'):.3f} "
            f"label={'LIVE BENCHMARK' if naive.get('live_benchmark') else 'NOT A LIVE BENCHMARK'}",
            flush=True,
        )
        print(f"Forced-exits: n_events={fe.get('n_events')} n_ids={fe.get('n_ids')}", flush=True)
    else:
        print("Benchmark: SKIPPED", flush=True)
        print("Forced-exits: SKIPPED", flush=True)
    print("COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s", flush=True)
    return {
        "verdict": gate.get("verdict"),
        "usable_from": gate.get("usable_from"),
        "blocked": gate.get("blocked"),
        "n_ended": ended["n_ended"],
        "ended_fail": ended["fail"],
        "graveyard_ok": n_g,
        "graveyard_n": len(graveyard),
        "rel_sharpe": None if not naive else naive.get("rel_sharpe"),
        "forced_n": None if not naive else (naive.get("forced_exits") or {}).get("n_events"),
        "http_count": api.http_count,
        "credit_count": api.credit_count,
        "n_target": len(target_ids),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] starting BTC-BEATER P0c...", flush=True)
    summary = run_btcb_0c.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    Path("universe").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase0c_rebuild.md", "reports"),
        ("reports/btcb_phase0c_rebuild.json", "reports"),
        ("reports/btcb_phase1_benchmark_v3.md", "reports"),
        ("reports/btcb_phase1_benchmark_v3.json", "reports"),
        ("charts/btcb_benchmark_v3.png", "charts"),
        ("charts/btcb_coverage.png", "charts"),
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
        for src in (art / "reports").glob("btcb_phase0c*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase1_benchmark_v3*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] BTC-BEATER P0c complete.", flush=True)
