"""
BTC-BEATER ORACLE LADDER 2 — tail-blindness vs translation slack.

ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Usage: modal run btcb_oracle_ladder2_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-oracle-ladder2"
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
        "reports/btcb_oracle_ladder2_addendum.md",
        remote_path="/root/btcb_oracle_ladder2_addendum.md",
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

    drop = drop or {"daily_ret", "equity", "id_to_sym", "btc_ret", "equity_btc", "rel_equity", "w_btc", "n_names", "gate_on", "alt_gross"}
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


def _alias_prod(packed: dict) -> dict:
    packed = dict(packed)
    packed.setdefault("cagr", packed.get("book_cagr"))
    packed.setdefault("total", packed.get("book_total"))
    packed.setdefault("sharpe", packed.get("book_sharpe"))
    packed.setdefault("n_formations", packed.get("n_days"))
    return packed


@app.function(
    timeout=60 * 60,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def run_oracle_ladder2() -> dict:
    import pandas as pd

    from baseline.data import load_panel
    from btcb.academic_factor import pit_members, spread_wide
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        LONGTIDE_H,
        ORACLE_LADDER2_CRITERION,
        ORACLE_LADDER2_IC_EQ,
        ORACLE_LADDER2_IC_REF,
        ORACLE_LADDER_H,
        ORACLE_LADDER_SEEDS,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.longtide import run_long_tide
    from btcb.oracle_ladder import (
        _as_utc,
        build_noisy_scores,
        build_oracle_scores,
        build_spread_scores,
        ffill_members,
        formation_dates,
        run_periodic_long,
        summarize_seeds,
    )
    from btcb.oracle_ladder2 import (
        decompose_gap,
        formation_diagnostics,
        scores_to_daily_twin,
        summarize_diag_seeds,
        weights_v1,
        weights_v2,
        weights_v3,
    )
    from btcb.oracle_ladder2_report import (
        plot_overlap,
        plot_variants,
        update_ledger_ladder2,
        write_ladder2,
    )
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache

    t0 = time.time()
    addendum = Path("/root/btcb_oracle_ladder2_addendum.md").read_text()
    for txt in (ORACLE_LADDER2_CRITERION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"ladder2 addendum missing freeze text: {txt[:80]}")
    print("[HB] ORACLE LADDER 2 ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {ORACLE_LADDER2_CRITERION}", flush=True)

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

    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if not feat_path.exists():
        raise RuntimeError("missing feat_s")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)

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
    close.index = pd.DatetimeIndex([_as_utc(d) for d in close.index])
    print(f"[HB] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    dates = list(close.index)
    members = ffill_members(pit_members(pit, btc_id), dates)
    swide = swide.copy()
    swide.index = pd.DatetimeIndex([_as_utc(d) for d in swide.index])
    swide = swide.reindex(close.index).ffill()
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(ORACLE_LADDER_H))
    print(f"[HB] formations h=14 n={len(pairs14)}", flush=True)
    id_to_sym = cleaned.sort_values("date").groupby("id")["symbol"].last().to_dict()

    def _crude(scores, label, weight_fn=None):
        packed = run_periodic_long(
            close,
            members,
            btc_id,
            scores,
            pairs14,
            cost_bps=float(ALT_BPS),
            label=label,
            weight_fn=weight_fn,
        )
        if packed.get("error"):
            raise RuntimeError(f"{label} failed: {packed}")
        print(
            f"[HB] {label} RankIC={packed.get('rankic')} total={packed.get('total')} "
            f"CAGR={packed.get('cagr')} MaxDD={packed.get('maxdd')} TO={packed.get('ann_turnover')}",
            flush=True,
        )
        return packed

    def _prod(scores, label):
        preds = scores_to_daily_twin(scores, close.index)
        packed = run_long_tide(
            close,
            pit,
            preds,
            feat,
            btc_id,
            h=int(LONGTIDE_H),
            gate_on=None,
            park_btc=True,
            spot_filter=True,
            id_to_sym=id_to_sym,
        )
        if packed.get("error"):
            raise RuntimeError(f"{label} failed: {packed}")
        packed = _alias_prod(packed)
        packed["label"] = label
        print(
            f"[HB] {label} total={packed.get('total')} CAGR={packed.get('cagr')} "
            f"MaxDD={packed.get('maxdd')} TO={packed.get('ann_turnover')}",
            flush=True,
        )
        return packed

    print("[HB] scores: oracle / model / ladder-0.116 / ladder-0.16...", flush=True)
    sc_or, ic_or = build_oracle_scores(close, members, btc_id, pairs14)
    sc_m, ic_m = build_spread_scores(close, members, btc_id, pairs14, swide)

    print("[HB] §1 diagnostics...", flush=True)
    d_model = formation_diagnostics(close, members, btc_id, sc_m, pairs14, "model")
    d_or = formation_diagnostics(close, members, btc_id, sc_or, pairs14, "oracle")
    d_model["rankic"] = float(sum(ic_m) / len(ic_m)) if ic_m else d_model.get("rankic")
    d_or["rankic"] = float(sum(ic_or) / len(ic_or)) if ic_or else d_or.get("rankic")
    print(
        f"[HB] MODEL overlap={d_model.get('overlap'):.4f} tailIC top={d_model.get('tail_ic_top'):.4f} "
        f"bot={d_model.get('tail_ic_bot'):.4f} monster={d_model.get('monster'):.4f}",
        flush=True,
    )

    d116, d016 = [], []
    sc116, sc016 = [], []
    for seed in ORACLE_LADDER_SEEDS:
        s116, ic116, _ = build_noisy_scores(close, members, btc_id, pairs14, float(ORACLE_LADDER2_IC_EQ), int(seed))
        s016, ic016, _ = build_noisy_scores(close, members, btc_id, pairs14, float(ORACLE_LADDER2_IC_REF), int(seed))
        sc116.append(s116)
        sc016.append(s016)
        a = formation_diagnostics(close, members, btc_id, s116, pairs14, f"l0116-s{seed}")
        b = formation_diagnostics(close, members, btc_id, s016, pairs14, f"l016-s{seed}")
        a["rankic"] = float(sum(ic116) / len(ic116)) if ic116 else a.get("rankic")
        b["rankic"] = float(sum(ic016) / len(ic016)) if ic016 else b.get("rankic")
        d116.append(a)
        d016.append(b)
        print(
            f"[HB] seed={seed} l0.116 IC={a['rankic']:.4f} ov={a['overlap']:.4f} "
            f"l0.16 IC={b['rankic']:.4f} ov={b['overlap']:.4f}",
            flush=True,
        )
    diag116 = summarize_diag_seeds(d116)
    diag016 = summarize_diag_seeds(d016)

    print("[HB] §2 crude books + V1–V3...", flush=True)
    base = _crude(sc_m, "model-crude")
    v1 = _crude(sc_m, "model-v1", weight_fn=weights_v1)
    v2 = _crude(sc_m, "model-v2", weight_fn=weights_v2)
    v3 = _crude(sc_m, "model-v3", weight_fn=weights_v3)
    ora_book = _crude(sc_or, "oracle-crude")

    crude116, prod116 = [], []
    for seed, s116 in zip(ORACLE_LADDER_SEEDS, sc116):
        c = _crude(s116, f"ladder0116-crude-s{seed}")
        p = _prod(s116, f"ladder0116-prod-s{seed}")
        crude116.append(c)
        prod116.append(p)
    lad_crude = summarize_seeds(crude116)
    lad_prod = summarize_seeds(prod116)
    lad_crude["n_formations"] = crude116[0].get("n_formations") if crude116 else None
    lad_crude["avg_n_names"] = crude116[0].get("avg_n_names") if crude116 else None
    lad_crude["ann_turnover"] = float(
        sum(float(x.get("ann_turnover") or 0) for x in crude116) / max(len(crude116), 1)
    )
    lad_prod["n_days"] = prod116[0].get("n_days") if prod116 else None
    lad_prod["avg_n_names"] = prod116[0].get("avg_n_names") if prod116 else None
    lad_prod["ann_turnover"] = float(
        sum(float(x.get("ann_turnover") or 0) for x in prod116) / max(len(prod116), 1)
    )
    # representative cycles from seed 101
    lad_crude["cycles"] = crude116[0].get("cycles") if crude116 else {}
    lad_crude["start"] = crude116[0].get("start") if crude116 else None
    lad_crude["end"] = crude116[0].get("end") if crude116 else None
    lad_crude["n_days"] = crude116[0].get("n_days") if crude116 else None
    lad_prod["cycles"] = prod116[0].get("cycles") if prod116 else {}
    lad_prod["start"] = prod116[0].get("start") if prod116 else None
    lad_prod["end"] = prod116[0].get("end") if prod116 else None

    # ladder-0.16 crude CAGR for overlap curve (seed mean)
    crude016 = []
    for seed, s016 in zip(ORACLE_LADDER_SEEDS, sc016):
        crude016.append(_crude(s016, f"ladder016-crude-s{seed}"))
    lad16_book = summarize_seeds(crude016)

    diags = {"model": d_model, "oracle": d_or, "ladder0116": diag116, "ladder016": diag016}
    books = {
        "base": base,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "ladder_crude": lad_crude,
        "ladder_prod": lad_prod,
        "oracle": ora_book,
        "ladder16": lad16_book,
    }
    verdict = decompose_gap(
        model_cagr=float(base.get("cagr") or float("nan")),
        model_maxdd=float(base.get("maxdd") or float("nan")),
        model_overlap=float(d_model.get("overlap") or float("nan")),
        ladder_cagr=float(lad_crude.get("cagr") or float("nan")),
        ladder_overlap=float(diag116.get("overlap") or float("nan")),
        ladder16_cagr=float(lad16_book.get("cagr") or float("nan")),
        ladder16_overlap=float(diag016.get("overlap") or float("nan")),
        oracle_cagr=float(ora_book.get("cagr") or float("nan")),
        oracle_overlap=float(d_or.get("overlap") or float("nan")),
        ladder_prod_cagr=float(lad_prod.get("cagr") or float("nan")),
        variants={"V1": v1, "V2": v2, "V3": v3},
    )
    print(
        f"[HB] VERDICT {verdict.get('priority')} gap={verdict.get('gap_pp')} "
        f"tail={verdict.get('tail_pp')} constr={verdict.get('construction_pp')} "
        f"unexpl={verdict.get('unexplained_pp')}",
        flush=True,
    )

    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    if cmc_panel_sha1 != cmc_panel_sha0:
        raise RuntimeError("CMC panel mutated during oracle ladder 2")
    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "pred_sha256": pred_hash["sha256"],
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_readonly_ok": True,
        "n_pairs_h14": int(len(pairs14)),
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_ladder2(ledger_path, verdict=verdict, extra=extra)

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_ladder2(
        rep_dir / "btcb_oracle_ladder2.md",
        diags=diags,
        books=books,
        verdict=verdict,
        extra=extra,
    )
    plot_overlap(diags, chart_dir / "btcb_oracle_ladder2_overlap.png")
    plot_variants(books, chart_dir / "btcb_oracle_ladder2_variants.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": ORACLE_LADDER2_CRITERION,
        "verdict": verdict,
        "diags": {k: _jsonable(v) for k, v in diags.items()},
        "books": {k: _jsonable(v) for k, v in books.items()},
        "extra": extra,
    }
    (rep_dir / "btcb_oracle_ladder2.json").write_text(json.dumps(payload, indent=2, default=str))
    quant_vol.commit()

    def _c(x):
        v = x.get("cagr")
        return f"{100.0 * v:.1f}%" if v is not None and pd.notna(v) else "nan"

    print(
        f"OVERLAP model={d_model.get('overlap'):.4f} ladder0116={diag116.get('overlap'):.4f} "
        f"ladder016={diag016.get('overlap'):.4f} oracle={d_or.get('overlap'):.4f}",
        flush=True,
    )
    print(
        f"TAIL-IC model top={d_model.get('tail_ic_top'):.4f} bot={d_model.get('tail_ic_bot'):.4f} "
        f"Δ={d_model.get('bottom_minus_top'):.4f}",
        flush=True,
    )
    print(
        f"BEST {verdict.get('variant_best')} CAGR={_c({'cagr': verdict.get('variant_best_cagr')})} "
        f"Δ={verdict.get('variant_best_delta_pp')} vs base {_c(base)}",
        flush=True,
    )
    print(f"PRIORITY {verdict.get('priority')} nothing_adopted=true GPU=false", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "priority": verdict.get("priority"),
        "gap_pp": verdict.get("gap_pp"),
        "tail_pp": verdict.get("tail_pp"),
        "construction_pp": verdict.get("construction_pp"),
        "unexplained_pp": verdict.get("unexplained_pp"),
        "variant_best": verdict.get("variant_best"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] ORACLE LADDER 2...", flush=True)
    fc = run_oracle_ladder2.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_oracle_ladder2.md", "reports"),
        ("reports/btcb_oracle_ladder2.json", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_oracle_ladder2_overlap.png", "charts"),
        ("charts/btcb_oracle_ladder2_variants.png", "charts"),
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
        for src in (art / "charts").glob("btcb_oracle_ladder2*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_oracle_ladder2*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] ORACLE LADDER 2 complete.", flush=True)
