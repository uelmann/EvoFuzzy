"""
BTC-BEATER LONG-TIDE — full-size long leg + frozen Stage-T gate, BTC parking.

BACKTEST ONLY. CPU only. Zero GPU. One shot.
Usage: modal run btcb_longtide_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-longtide"
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
        "reports/btcb_longtide_addendum.md",
        remote_path="/root/btcb_longtide_addendum.md",
    )
    .add_local_file(
        "reports/btcb_phase3e_forensics.md",
        remote_path="/root/btcb_phase3e_forensics.md",
    )
    .add_local_file(
        "reports/btcb_phase3e_forensics.json",
        remote_path="/root/btcb_phase3e_forensics.json",
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

    drop = drop or {
        "daily_ret",
        "btc_ret",
        "equity",
        "equity_btc",
        "rel_equity",
        "w_btc",
        "n_names",
        "gate_on",
        "alt_gross",
        "contrib",
        "id_to_sym",
        "daily",
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
def run_longtide() -> dict:
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from btcb.binance_replay import (
        build_id_symbol_map,
        close_wide_from_panel,
        funding_wide_from_panel,
        replay_long_leg,
    )
    from btcb.book import run_hysteresis_book
    from btcb.constants import (
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        LONGTIDE_CRITERION,
        LONGTIDE_H,
        LONGTIDE_PRECONDITION,
        LONGTIDE_V1_P_ENTER,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_POSITION_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        REGIME_BREADTH,
        REGIME_OFF_HYSTERESIS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.longtide import (
        btc_bh_book,
        control_from_rets,
        enrich_book,
        gate_params_ok,
        load_phase2_preds,
        longtide_verdicts,
        read_phase3e_verdict,
        run_long_tide,
        series_corr,
        slice_book,
    )
    from btcb.universe import build_pit_topn_ids, trailing_rank_frame_by_id
    from btcb.longtide_report import (
        gate_stretch_stats,
        plot_gate_ribbon,
        plot_longtide_equity,
        update_ledger_longtide,
        write_longtide,
    )
    from btcb.spread_ls import (
        _hash_position_log,
        ew_basket,
        hash_pred_dir,
        load_twin_from_cache,
        squeeze_table,
    )
    from btcb.timing import breadth_top100, ew_top50_btc_ratio, regime_on_off

    t0 = time.time()
    addendum = Path("/root/btcb_longtide_addendum.md").read_text()
    for txt in (LONGTIDE_CRITERION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"LONG-TIDE addendum missing freeze text: {txt[:80]}")

    p3e_md = Path("/root/btcb_phase3e_forensics.md")
    p3e_js = Path("/root/btcb_phase3e_forensics.json")
    p3e_verdict = read_phase3e_verdict(p3e_md, p3e_js)
    print(f"PRECONDITION: 3.e verdict is {p3e_verdict}", flush=True)
    if p3e_verdict != LONGTIDE_PRECONDITION:
        msg = f"BLOCKED-BY-SUSPENSION: 3.e verdict is {p3e_verdict}"
        print(msg, flush=True)
        return {"blocked": True, "precondition": p3e_verdict, "message": msg}

    print("[HB] BTC-BEATER LONG-TIDE BACKTEST ONLY; zero GPU; frozen products untouched", flush=True)
    print(f"[HB] {LONGTIDE_CRITERION}", flush=True)
    gp = gate_params_ok()
    if not gp["byte_identical"]:
        raise RuntimeError(f"Stage-T gate params mutated: {gp}")
    print(
        f"[HB] gate byte-identical breadth={REGIME_BREADTH} off_hyst={REGIME_OFF_HYSTERESIS}",
        flush=True,
    )

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    panel = pd.read_parquet(CMC_PANEL)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id} rows={len(panel)}", flush=True)

    def _load_pit(name: str, required: bool = True):
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
        if required:
            raise RuntimeError(f"missing PIT {name}")
        return None

    pit50 = _load_pit("btcb_top50_floor.parquet")
    pit100 = _load_pit("btcb_top100_floor.parquet")
    pit50_v1 = _load_pit("btcb_top50_pit.parquet", required=False)
    if pit50_v1 is None:
        print("[HB] unfloored PIT missing on volume; rebuilding from uncleaned panel for v1 replay", flush=True)
        score, _, method, last_sym = trailing_rank_frame_by_id(panel)
        pit50_v1 = build_pit_topn_ids(score, 50, last_sym)
        pit50_v1 = pit50_v1[pit50_v1["date"] <= end].copy()
        print(f"[HB] rebuilt unfloored top-50 method={method} rows={len(pit50_v1)}", flush=True)

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

    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if not feat_path.exists():
        raise RuntimeError("missing feat_s")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)

    print("[HB] Stage-T regime gate (frozen params, CMC panel)...", flush=True)
    ratio = ew_top50_btc_ratio(cleaned, pit50, btc_id)
    breadth = breadth_top100(cleaned, pit100)
    regime = regime_on_off(ratio, breadth, breadth_thr=REGIME_BREADTH, off_hyst=REGIME_OFF_HYSTERESIS)
    gate_on = regime["gate_on"]

    print("[HB] loading Binance spot (no new downloads unless BTCUSDT missing)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    spot_dir.mkdir(parents=True, exist_ok=True)
    if not (spot_dir / "BTCUSDT.parquet").exists():
        print("[HB] BTCUSDT spot cache missing; downloading BTCUSDT only...", flush=True)
        from baseline.data import download_spot_symbol_months, month_range

        download_spot_symbol_months("BTCUSDT", month_range("2017-08"), spot_dir)
        quant_vol.commit()
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet"))
    if not spot_syms:
        raise RuntimeError(f"no spot klines in {spot_dir}")
    if "BTCUSDT" not in spot_syms:
        raise RuntimeError("BTCUSDT spot still missing after download")
    spot_panel = load_panel(spot_dir, spot_syms)
    spot_panel["date"] = pd.to_datetime(spot_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    spot_panel["symbol"] = spot_panel["symbol"].astype(str).str.upper()
    all_ids = sorted(set(int(i) for i in pit100["id"].unique()) | {int(btc_id)})
    nonempty_spot = set(spot_panel["symbol"].unique())
    id_to_spot = build_id_symbol_map(all_ids, cleaned, nonempty_spot, nonempty_spot)
    id_to_spot[int(btc_id)] = "BTCUSDT"
    spot_wide = close_wide_from_panel(spot_panel, id_to_spot)
    if int(btc_id) not in spot_wide.columns:
        raise RuntimeError("BTCUSDT spot missing from Binance close wide")
    id_to_sym = cleaned.sort_values("date").groupby("id")["symbol"].last().to_dict()
    print(
        f"[HB] spot symbols={len(spot_syms)} mapped={sum(1 for v in id_to_spot.values() if v)} "
        f"wide={spot_wide.shape}",
        flush=True,
    )

    print("[HB] LONG-TIDE primary (Binance spot, gate, BTC park)...", flush=True)
    tide = run_long_tide(
        spot_wide,
        pit100,
        twin,
        feat,
        btc_id,
        h=int(LONGTIDE_H),
        gate_on=gate_on,
        park_btc=True,
        spot_filter=True,
        id_to_sym=id_to_sym,
    )
    if "error" in tide:
        raise RuntimeError(f"LONG-TIDE failed: {tide}")
    print(
        f"[HB] TIDE sharpe={tide.get('book_sharpe'):.3f} rel={tide.get('rel_sharpe'):.3f} "
        f"alt={tide.get('avg_alt_deployment'):.3f} on={tide.get('gate_on_frac'):.3f}",
        flush=True,
    )

    print("[HB] NAKED LONG LEG (same selection, no gate, cash idle)...", flush=True)
    naked = run_long_tide(
        spot_wide,
        pit100,
        twin,
        feat,
        btc_id,
        h=int(LONGTIDE_H),
        gate_on=None,
        park_btc=False,
        spot_filter=True,
        id_to_sym=id_to_sym,
    )
    print(
        f"[HB] NAKED sharpe={naked.get('book_sharpe'):.3f} rel={naked.get('rel_sharpe'):.3f} "
        f"alt={naked.get('avg_alt_deployment'):.3f} maxdd={naked.get('maxdd')}",
        flush=True,
    )

    print("[HB] CMC unrestricted reference (not judged)...", flush=True)
    cmc_ref = run_long_tide(
        cmc_close,
        pit100,
        twin,
        feat,
        btc_id,
        h=int(LONGTIDE_H),
        gate_on=gate_on,
        park_btc=True,
        spot_filter=False,
        id_to_sym=id_to_sym,
    )

    print("[HB] BTC-BEATER v1 replay (read-only, CMC, p_enter=0.60)...", flush=True)
    v1_pred_dir = Path("/data/quant/btcb/phase2/preds")
    v1_preds = load_phase2_preds(v1_pred_dir, horizon=14)
    feat_v1_path = Path("/data/quant/btcb/phase2/feat_v1.parquet")
    if feat_v1_path.exists():
        feat_v1 = pd.read_parquet(feat_v1_path)
        feat_v1["date"] = pd.to_datetime(feat_v1["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        feat_v1["id"] = feat_v1["id"].astype(int)
    else:
        feat_v1 = feat
        print("[WARN] feat_v1 missing; falling back to feat_s for blowoff", flush=True)
    v1 = run_hysteresis_book(
        panel,
        pit50_v1,
        v1_preds,
        feat_v1,
        btc_id,
        p_enter=float(LONGTIDE_V1_P_ENTER),
        h=14,
    )
    v1 = enrich_book(v1)
    print(
        f"[HB] v1 rel={v1.get('rel_sharpe'):.3f} total={v1.get('book_total'):.3f} maxdd={v1.get('maxdd')}",
        flush=True,
    )

    idx = pd.DatetimeIndex(tide["daily_ret"].index).sort_values()
    idx_common = idx.intersection(v1["daily_ret"].index)
    print(
        f"[HB] identical window n={len(idx)} {idx.min().date()}→{idx.max().date()}; "
        f"common-with-v1 n={len(idx_common)}",
        flush=True,
    )
    tide_c = slice_book(tide, idx_common)
    v1_c = slice_book(v1, idx_common)
    v1_same = slice_book(v1, idx.intersection(v1["daily_ret"].index))
    btc_simple_bn = spot_wide[int(btc_id)].astype(float).pct_change()
    btc_c = btc_bh_book(btc_simple_bn, idx)

    members = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit100.groupby("date")["id"]
    }
    dates = list(cmc_close.index)
    ew_rets = ew_basket(cmc_close, members, dates)
    ew_c = control_from_rets(ew_rets.reindex(idx).fillna(0.0), btc_simple_bn)
    squeeze = squeeze_table(tide["daily_ret"], ew_rets)

    print("[HB] SPREAD-LS hybrid daily for correlation...", flush=True)
    plog_path = Path("/data/quant/btcb/phase3c/position_log.parquet")
    if not plog_path.exists():
        raise RuntimeError("missing frozen 3.c position log")
    plog = pd.read_parquet(plog_path)
    pos_sha = _hash_position_log(plog)
    if pos_sha != PHASE3C_POSITION_SHA256:
        raise RuntimeError(f"position log mutated {pos_sha}")
    raw_dir = Path("/data/quant/raw/klines")
    fund_dir = Path("/data/quant/raw/funding")
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    fund_syms = sorted(p.stem for p in fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    kline_panel = load_panel(raw_dir, kline_syms)
    kline_panel["date"] = pd.to_datetime(kline_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    kline_panel["symbol"] = kline_panel["symbol"].astype(str).str.upper()

    funding = (
        load_funding_panel(fund_dir, fund_syms)
        if fund_syms
        else pd.DataFrame(columns=["date", "symbol", "funding_rate", "n_events"])
    )
    if not funding.empty:
        funding["date"] = pd.to_datetime(funding["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        funding["symbol"] = funding["symbol"].astype(str).str.upper()
    plog_ids = sorted(set(int(i) for i in plog["id"].unique()) | {int(btc_id)})
    nonempty_perp = set(kline_panel["symbol"].unique())
    id_to_perp = build_id_symbol_map(plog_ids, cleaned, nonempty_perp, nonempty_perp)
    id_to_spot_pl = build_id_symbol_map(plog_ids, cleaned, nonempty_spot, nonempty_spot)
    spot_pl = close_wide_from_panel(spot_panel, id_to_spot_pl)
    perp_wide = close_wide_from_panel(kline_panel, id_to_perp)
    fund_wide = funding_wide_from_panel(funding, id_to_perp)
    hyb = replay_long_leg(plog, cmc_close, spot_pl, perp_wide, fund_wide)
    ls_daily = hyb["full_net"]

    corr = {
        "vs_v1": series_corr(tide["daily_ret"], v1["daily_ret"]),
        "vs_spread_ls": series_corr(tide["daily_ret"], ls_daily),
    }
    print(
        f"[HB] corr vs v1={corr['vs_v1'].get('corr')} vs SPREAD-LS={corr['vs_spread_ls'].get('corr')}",
        flush=True,
    )

    # (a–c) judged on LONG-TIDE's native window vs Binance BTC.
    # (d) judged on the common window with v1.
    verdicts = longtide_verdicts(tide, v1_c)
    d_rel = float(tide_c.get("rel_sharpe") or 0.0)
    d_v1 = float(v1_c.get("rel_sharpe") or 0.0)
    verdicts["rel_sharpe_common"] = d_rel
    verdicts["v1_rel_sharpe"] = d_v1
    verdicts["need_supersede_rel"] = d_v1 + 0.15
    verdicts["d_rel_ge_v1_plus_margin"] = bool(d_rel >= d_v1 + 0.15)
    verdicts["supersedes"] = bool(
        verdicts["viable"]
        and verdicts["d_rel_ge_v1_plus_margin"]
        and verdicts["e_alt_deployment_ge_15pct"]
        and verdicts["f_no_cycle_rel_below_floor"]
    )
    if verdicts["viable"] and not verdicts["supersedes"]:
        verdicts["status"] = "PARALLEL-VARIANT"
    elif verdicts["supersedes"]:
        verdicts["status"] = "SUPERSEDES-V1"
    else:
        verdicts["status"] = "NOT-VIABLE"
    verdicts["rel_sharpe"] = float(tide.get("rel_sharpe") or 0.0)

    stretches = gate_stretch_stats(tide.get("gate_on"))
    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    if cmc_panel_sha1 != cmc_panel_sha0:
        raise RuntimeError("CMC panel mutated during LONG-TIDE")

    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "pred_sha256": pred_hash["sha256"],
        "position_sha256": pos_sha,
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_readonly_ok": True,
        "common_start": str(idx_common.min().date()) if len(idx_common) else None,
        "common_end": str(idx_common.max().date()) if len(idx_common) else None,
        "common_n": int(len(idx_common)),
        **stretches,
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_longtide(ledger_path, verdicts=verdicts, tide=tide, extra=extra)

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_longtide(
        rep_dir / "btcb_longtide.md",
        precondition=p3e_verdict,
        verdicts=verdicts,
        tide=tide,
        naked=naked,
        v1=v1_same,
        btc=btc_c,
        ew=ew_c,
        cmc_ref=cmc_ref,
        corr=corr,
        squeeze=squeeze,
        gate=gp,
        extra=extra,
    )
    plot_longtide_equity(tide, naked, v1, chart_dir / "btcb_longtide_equity.png")
    plot_gate_ribbon(tide, chart_dir / "btcb_longtide_gate_ribbon.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": LONGTIDE_CRITERION,
        "precondition": p3e_verdict,
        "verdicts": verdicts,
        "tide": _jsonable(tide),
        "naked": _jsonable(naked),
        "v1": _jsonable(v1),
        "btc": _jsonable(btc_c),
        "ew": _jsonable(ew_c),
        "cmc_ref": _jsonable(cmc_ref),
        "tide_common": _jsonable(tide_c),
        "v1_common": _jsonable(v1_c),
        "corr": corr,
        "squeeze": squeeze,
        "gate": gp,
        "extra": extra,
    }
    (rep_dir / "btcb_longtide.json").write_text(json.dumps(payload, indent=2, default=str))
    quant_vol.commit()

    dd_tide = float(tide.get("maxdd") or float("nan"))
    dd_naked = float(naked.get("maxdd") or float("nan"))
    print(f"PRECONDITION: 3.e verdict is {p3e_verdict}", flush=True)
    print(f"VIABLE: {verdicts.get('viable')}", flush=True)
    print(f"SUPERSEDES: {verdicts.get('supersedes')}", flush=True)
    print(f"rel-line Sharpe: {verdicts.get('rel_sharpe')}", flush=True)
    print(f"avg alt deployment: {verdicts.get('avg_alt_deployment')}", flush=True)
    print(
        f"GATE VALUE (MaxDD tide vs naked): tide MaxDD={dd_tide:.4f} vs naked MaxDD={dd_naked:.4f} "
        f"(Δ={dd_tide - dd_naked:.4f})",
        flush=True,
    )
    print("COMBO untouched (v2.0-combo-final). SPREAD-LS BOOK-HYBRID untouched. v1 replayed read-only.", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "blocked": False,
        "precondition": p3e_verdict,
        "viable": verdicts.get("viable"),
        "supersedes": verdicts.get("supersedes"),
        "status": verdicts.get("status"),
        "rel_sharpe": verdicts.get("rel_sharpe"),
        "avg_alt_deployment": verdicts.get("avg_alt_deployment"),
        "maxdd": dd_tide,
        "naked_maxdd": dd_naked,
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] LONG-TIDE backtest...", flush=True)
    fc = run_longtide.spawn()
    summary = fc.get()
    if summary.get("blocked"):
        print(summary.get("message"), flush=True)
        print(json.dumps(summary, indent=2, default=str))
        return
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_longtide.md", "reports"),
        ("reports/btcb_longtide.json", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_longtide_equity.png", "charts"),
        ("charts/btcb_longtide_gate_ribbon.png", "charts"),
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
        for src in (art / "charts").glob("btcb_longtide*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_longtide*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] LONG-TIDE complete.", flush=True)
