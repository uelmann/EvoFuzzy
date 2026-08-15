"""
BTC-BEATER ORACLE LADDER — perfect-foresight ceiling and IC-degraded oracles.

ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Usage: modal run btcb_oracle_ladder_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-oracle-ladder"
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
        "reports/btcb_oracle_ladder_addendum.md",
        remote_path="/root/btcb_oracle_ladder_addendum.md",
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

    drop = drop or {"daily_ret", "equity", "id_to_sym"}
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
def run_oracle_ladder() -> dict:
    import pandas as pd

    from baseline.data import load_panel
    from btcb.academic_factor import pit_members, spread_wide
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        ORACLE_LADDER_CRITERION,
        ORACLE_LADDER_H,
        ORACLE_LADDER_H_SEC,
        ORACLE_LADDER_MOM_DAYS,
        ORACLE_LADDER_SEEDS,
        ORACLE_LADDER_TARGETS,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.oracle_ladder import (
        _as_utc,
        build_mom_scores,
        build_noisy_scores,
        build_oracle_scores,
        build_spread_scores,
        efficiency_verdict,
        ffill_members,
        formation_dates,
        interpolate_cagr,
        run_periodic_long,
        summarize_seeds,
    )
    from btcb.oracle_ladder_report import plot_curve, update_ledger_oracle, write_oracle_ladder
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache

    t0 = time.time()
    addendum = Path("/root/btcb_oracle_ladder_addendum.md").read_text()
    for txt in (ORACLE_LADDER_CRITERION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"oracle-ladder addendum missing freeze text: {txt[:80]}")
    print("[HB] ORACLE LADDER ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {ORACLE_LADDER_CRITERION}", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
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

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    twin = load_twin_from_cache(pred_dir, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()
    swide = spread_wide(twin)

    print("[HB] loading Binance spot (canonical)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    if not (spot_dir / "BTCUSDT.parquet").exists():
        from baseline.data import download_spot_symbol_months, month_range

        print("[HB] BTCUSDT spot missing; downloading...", flush=True)
        download_spot_symbol_months("BTCUSDT", month_range("2017-08"), spot_dir)
        quant_vol.commit()
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet"))
    spot_panel = load_panel(spot_dir, spot_syms)
    spot_panel["date"] = pd.to_datetime(spot_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    spot_panel["symbol"] = spot_panel["symbol"].astype(str).str.upper()
    all_ids = sorted(set(int(i) for i in pit["id"].unique()) | {int(btc_id)})
    nonempty = set(spot_panel["symbol"].unique())
    id_to_spot = build_id_symbol_map(all_ids, cleaned, nonempty, nonempty)
    id_to_spot[int(btc_id)] = "BTCUSDT"
    close = close_wide_from_panel(spot_panel, id_to_spot)
    if int(btc_id) not in close.columns:
        raise RuntimeError("BTCUSDT spot missing from close wide")
    close = close[close.index <= end].sort_index()
    print(f"[HB] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    close.index = pd.DatetimeIndex([_as_utc(d) for d in close.index])
    dates = list(close.index)
    pos = {d: i for i, d in enumerate(dates)}
    members = ffill_members(pit_members(pit, btc_id), dates)
    swide = swide.copy()
    swide.index = pd.DatetimeIndex([_as_utc(d) for d in swide.index])
    swide = swide.reindex(close.index).ffill()
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(ORACLE_LADDER_H))
    pairs7 = formation_dates(oos, int(ORACLE_LADDER_H_SEC))
    print(f"[HB] formations h=14 n={len(pairs14)} h=7 n={len(pairs7)}", flush=True)

    def _run(scores, pairs, cost, label):
        packed = run_periodic_long(
            close, members, btc_id, scores, pairs, cost_bps=float(cost), label=label
        )
        if packed.get("error"):
            raise RuntimeError(f"{label} failed: {packed}")
        print(
            f"[HB] {label} RankIC={packed.get('rankic')} total={packed.get('total')} "
            f"CAGR={packed.get('cagr')} MaxDD={packed.get('maxdd')}",
            flush=True,
        )
        return packed

    print("[HB] ORACLE h=14...", flush=True)
    sc_or, ic_or = build_oracle_scores(close, members, btc_id, pairs14)
    oracle_gross = _run(sc_or, pairs14, 0.0, "oracle-gross-h14")
    oracle_net = _run(sc_or, pairs14, float(ALT_BPS), "oracle-net-h14")
    oracle_gross["rankic"] = float(sum(ic_or) / len(ic_or)) if ic_or else float("nan")
    oracle_net["rankic"] = oracle_gross["rankic"]

    print("[HB] ORACLE h=7 (secondary)...", flush=True)
    sc_or7, ic_or7 = build_oracle_scores(close, members, btc_id, pairs7)
    oracle7_gross = _run(sc_or7, pairs7, 0.0, "oracle-gross-h7")
    oracle7_net = _run(sc_or7, pairs7, float(ALT_BPS), "oracle-net-h7")
    oracle7_gross["rankic"] = float(sum(ic_or7) / len(ic_or7)) if ic_or7 else float("nan")
    oracle7_net["rankic"] = oracle7_gross["rankic"]

    print("[HB] OUR MODEL (frozen spread, same 14d book)...", flush=True)
    sc_m, ic_m = build_spread_scores(close, members, btc_id, pairs14, swide)
    model = _run(sc_m, pairs14, float(ALT_BPS), "model-spread-h14")
    model["rankic"] = float(sum(ic_m) / len(ic_m)) if ic_m else float(model.get("rankic") or float("nan"))

    print("[HB] NAIVE 90d excess...", flush=True)
    sc_n, ic_n = build_mom_scores(
        close, members, btc_id, pairs14, dates, pos, int(ORACLE_LADDER_MOM_DAYS)
    )
    naive = _run(sc_n, pairs14, float(ALT_BPS), "naive-90d-h14")
    naive["rankic"] = float(sum(ic_n) / len(ic_n)) if ic_n else float(naive.get("rankic") or float("nan"))

    print("[HB] RANDOM IC≈0 (5 seeds)...", flush=True)
    rand_runs = []
    for seed in ORACLE_LADDER_SEEDS:
        sc_r, ic_r, _ = build_noisy_scores(close, members, btc_id, pairs14, 0.0, int(seed))
        packed = _run(sc_r, pairs14, float(ALT_BPS), f"random-seed{seed}")
        packed["rankic"] = float(sum(ic_r) / len(ic_r)) if ic_r else float(packed.get("rankic") or float("nan"))
        rand_runs.append(packed)
    random = summarize_seeds(rand_runs)
    random["n_formations"] = rand_runs[0].get("n_formations") if rand_runs else None
    random["forced_n"] = rand_runs[0].get("forced_n") if rand_runs else None
    random["avg_n_names"] = rand_runs[0].get("avg_n_names") if rand_runs else None

    print("[HB] degraded-oracle ladder...", flush=True)
    ladder = {}
    ladder_raw = {}
    for tgt in ORACLE_LADDER_TARGETS:
        runs = []
        for seed in ORACLE_LADDER_SEEDS:
            sc, ics, sigs = build_noisy_scores(close, members, btc_id, pairs14, float(tgt), int(seed))
            packed = _run(sc, pairs14, float(ALT_BPS), f"ladder-{tgt:.2f}-s{seed}")
            packed["rankic"] = float(sum(ics) / len(ics)) if ics else float(packed.get("rankic") or float("nan"))
            packed["sigma_mean"] = float(sum(sigs) / len(sigs)) if sigs else float("nan")
            runs.append(packed)
            print(
                f"[HB] target={tgt:.2f} seed={seed} realizedIC={packed['rankic']:.4f} "
                f"CAGR={packed.get('cagr')}",
                flush=True,
            )
        ladder[float(tgt)] = summarize_seeds(runs)
        ladder_raw[str(tgt)] = [_jsonable(r) for r in runs]

    xs = [float(random.get("rankic") or 0.0)]
    ys = [float(random.get("cagr") or float("nan"))]
    for tgt in sorted(ORACLE_LADDER_TARGETS):
        xs.append(float(ladder[float(tgt)].get("rankic") or float("nan")))
        ys.append(float(ladder[float(tgt)].get("cagr") or float("nan")))
    xs.append(float(oracle_net.get("rankic") or 1.0))
    ys.append(float(oracle_net.get("cagr") or float("nan")))
    curve = interpolate_cagr(xs, ys, float(model.get("rankic") or float("nan")))
    verdict = efficiency_verdict(
        float(model.get("cagr") or float("nan")),
        curve,
        float(oracle_net.get("cagr") or float("nan")),
        float(model.get("rankic") or float("nan")),
    )
    print(
        f"[HB] VERDICT {verdict.get('label')} modelIC={verdict.get('model_rankic')} "
        f"modelCAGR={verdict.get('model_cagr')} curveCAGR={verdict.get('curve_cagr')}",
        flush=True,
    )

    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    if cmc_panel_sha1 != cmc_panel_sha0:
        raise RuntimeError("CMC panel mutated during oracle ladder")
    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "pred_sha256": pred_hash["sha256"],
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_readonly_ok": True,
        "n_pairs_h14": int(len(pairs14)),
        "n_pairs_h7": int(len(pairs7)),
        "interp_xs": xs,
        "interp_ys": ys,
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_oracle(ledger_path, verdict=verdict, oracle_net=oracle_net, model=model)

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_oracle_ladder(
        rep_dir / "btcb_oracle_ladder.md",
        oracle_gross=oracle_gross,
        oracle_net=oracle_net,
        oracle7_gross=oracle7_gross,
        oracle7_net=oracle7_net,
        ladder=ladder,
        model=model,
        naive=naive,
        random=random,
        verdict=verdict,
        extra=extra,
    )
    plot_curve(
        oracle_net,
        ladder,
        model,
        naive,
        random,
        verdict,
        chart_dir / "btcb_oracle_ladder_curve.png",
    )
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": ORACLE_LADDER_CRITERION,
        "verdict": verdict,
        "oracle_gross": _jsonable(oracle_gross),
        "oracle_net": _jsonable(oracle_net),
        "oracle7_gross": _jsonable(oracle7_gross),
        "oracle7_net": _jsonable(oracle7_net),
        "ladder": {str(k): v for k, v in ladder.items()},
        "ladder_seeds": ladder_raw,
        "model": _jsonable(model),
        "naive": _jsonable(naive),
        "random": random,
        "extra": extra,
    }
    (rep_dir / "btcb_oracle_ladder.json").write_text(json.dumps(payload, indent=2, default=str))
    quant_vol.commit()

    def _c(x):
        v = x.get("cagr")
        return f"{100.0 * v:.1f}%" if v is not None and pd.notna(v) else "nan"

    def _t(x):
        v = x.get("total")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "nan"
        return f"{v:.3e}" if abs(float(v)) >= 100 else f"{100.0 * float(v):.1f}%"

    import numpy as np

    print(
        f"CEILING h=14 NET total={_t(oracle_net)} CAGR={_c(oracle_net)} "
        f"MaxDD={oracle_net.get('maxdd')} GROSS total={_t(oracle_gross)} CAGR={_c(oracle_gross)}",
        flush=True,
    )
    for tgt in ORACLE_LADDER_TARGETS:
        b = ladder[float(tgt)]
        print(
            f"LADDER IC={tgt:.2f} realized={b.get('rankic'):.4f} "
            f"[{b.get('rankic_lo'):.4f},{b.get('rankic_hi'):.4f}] "
            f"CAGR={_c(b)} total={_t(b)} MaxDD={b.get('maxdd')}",
            flush=True,
        )
    print(
        f"MODEL RankIC={model.get('rankic')} CAGR={_c(model)} total={_t(model)} "
        f"verdict={verdict.get('label')} capture={verdict.get('capture_of_oracle_cagr')}",
        flush=True,
    )
    print("NOTHING ADOPTED. Frozen products untouched. GPU=false.", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "verdict": verdict.get("label"),
        "constraint": verdict.get("binding_constraint"),
        "oracle_cagr": oracle_net.get("cagr"),
        "oracle_total": oracle_net.get("total"),
        "model_rankic": model.get("rankic"),
        "model_cagr": model.get("cagr"),
        "curve_cagr": verdict.get("curve_cagr"),
        "capture": verdict.get("capture_of_oracle_cagr"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] ORACLE LADDER...", flush=True)
    fc = run_oracle_ladder.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_oracle_ladder.md", "reports"),
        ("reports/btcb_oracle_ladder.json", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_oracle_ladder_curve.png", "charts"),
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
        src = art / "charts" / "btcb_oracle_ladder_curve.png"
        if src.exists():
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_oracle_ladder*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] ORACLE LADDER complete.", flush=True)
