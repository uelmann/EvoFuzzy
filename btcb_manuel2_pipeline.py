"""
BTC-BEATER MANUEL-2 — gauss-momentum falsification.

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Frozen products untouched. Master only.
Usage: modal run btcb_manuel2_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-manuel2"
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
    .add_local_file("reports/manuel2_addendum.md", remote_path="/root/manuel2_addendum.md")
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
        "equity",
        "btc_ret",
        "equity_btc",
        "rel_equity",
        "w_btc",
        "n_names",
        "corr_btc_roll90",
        "id_to_sym",
        "gate_on",
        "alt_gross",
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


def _persist_daily(books: dict, out_dir: Path) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, packed in books.items():
        rets = packed.get("daily_ret") if isinstance(packed, dict) else None
        if rets is None:
            continue
        s = pd.Series(rets, dtype=float)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).tz_convert("UTC").normalize()
        s = s.sort_index()
        s.name = str(name)
        frames.append(s)
    if frames:
        wide = pd.concat(frames, axis=1).sort_index()
        wide.index.name = "date"
        wide.to_parquet(out_dir / "daily_ret_all.parquet")


@app.function(
    timeout=60 * 60 * 3,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=65536,
)
def run_manuel2() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd

    from btcb.academic_factor import pit_members
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        LS_TRAIL_DAYS,
        MANUEL2_BEST_RULE,
        MANUEL2_BTC_COMPLETION,
        MANUEL2_CRITERION,
        MANUEL2_FORMULA,
        MANUEL2_GAUSS_COMPLETION,
        MANUEL2_MAXDD_RULE,
        MANUEL2_STABLE_COMPLETION,
        MANUEL2_START,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
    )
    from btcb.book import summarize_book as _sum
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.manuel2 import (
        _sharpe,
        btc_bh_book,
        build_members,
        cmc_close_wide,
        filter_members,
        heuristic_drop_stats,
        hybrid_close_wide,
        mcap_wide,
        mechanical_verdicts,
        peg_heuristic_mask,
        price_features,
        run_long_book,
        tagged_peg_ids,
    )
    from btcb.manuel2_report import plot_manuel2, update_ledger_manuel2, write_manuel2
    from btcb.oracle_ladder import _utc_idx, ffill_members, formation_dates, run_periodic_long
    from btcb.phase4v2 import collapse_fold_preds, preds_to_score_at
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache, trail_slice

    t0 = time.time()
    addendum = Path("/root/manuel2_addendum.md").read_text()
    for txt in (
        MANUEL2_CRITERION,
        MANUEL2_FORMULA,
        MANUEL2_STABLE_COMPLETION,
        MANUEL2_BTC_COMPLETION,
        MANUEL2_GAUSS_COMPLETION,
        MANUEL2_BEST_RULE,
        MANUEL2_MAXDD_RULE,
        DEATH_CONVENTION,
    ):
        if txt not in addendum:
            raise RuntimeError(f"manuel2 addendum missing freeze text: {txt[:80]}")
    print("[HB] MANUEL-2 ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {MANUEL2_FORMULA}", flush=True)
    print(f"[HB] {MANUEL2_CRITERION}", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    start = pd.Timestamp(MANUEL2_START, tz="UTC")
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

    tagged, tag_rows = tagged_peg_ids(cleaned)
    print(f"[HB] peg tags n_ids={len(tagged)} sample={[r['symbol'] for r in tag_rows[:12]]}", flush=True)

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")

    print("[HB] loading Binance spot close (hybrid)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet")) if spot_dir.exists() else []
    all_ids = sorted(set(int(i) for i in cleaned["id"].unique()) | {int(btc_id)})
    id_to_spot = build_id_symbol_map(all_ids, cleaned, set(spot_syms), set(spot_syms))
    id_to_spot[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in set(spot_syms) else id_to_spot.get(int(btc_id))
    spot_needed = sorted({s for s in id_to_spot.values() if s})
    spot_long = _load_close_long(spot_dir, spot_needed)
    spot_wide = close_wide_from_panel(spot_long.rename(columns={"close": "close"}), id_to_spot)
    cmc_px = cmc_close_wide(cleaned)
    close = hybrid_close_wide(cmc_px, spot_wide)
    if int(btc_id) not in close.columns:
        raise RuntimeError("BTC missing from hybrid close")
    close = close[close.index <= end].sort_index()
    close.index = _utc_idx(close.index)
    # drop dates without BTC print
    btc_ok = close[int(btc_id)].astype(float)
    close = close.loc[np.isfinite(btc_ok) & (btc_ok > 0)].copy()
    print(f"[HB] hybrid close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    mcap = mcap_wide(cleaned).reindex(close.index)
    feat_cmc = price_features(cmc_px.reindex(close.index))
    peg_heu = peg_heuristic_mask(feat_cmc["ret90"])
    heu_stats = heuristic_drop_stats(peg_heu, list(close.index))
    print(
        f"[HB] peg heuristic median/date={heu_stats.get('median_flagged_per_date')} "
        f"mean={heu_stats.get('mean_flagged_per_date')}",
        flush=True,
    )

    feat = price_features(close)
    dates = [d for d in list(close.index) if d >= start]
    if len(dates) < 10:
        raise RuntimeError("eval window too short")
    id_to_sym = cleaned.sort_values("date").groupby("id")["symbol"].last().to_dict()

    print("[HB] building mcap top-50 members (btc-in / btc-ex)...", flush=True)
    mem_in = build_members(list(close.index), mcap, tagged, peg_heu, btc_id, btc_ex=False)
    mem_ex = build_members(list(close.index), mcap, tagged, peg_heu, btc_id, btc_ex=True)
    n_in = float(np.mean([len(mem_in[d]) for d in dates if d in mem_in]))
    n_ex = float(np.mean([len(mem_ex[d]) for d in dates if d in mem_ex]))
    print(f"[HB] mean universe size btc-in={n_in:.1f} btc-ex={n_ex:.1f}", flush=True)

    specs = (
        ("daily_btc_in", "daily", mem_in),
        ("daily_btc_ex", "daily", mem_ex),
        ("weekly_btc_in", "weekly", mem_in),
        ("weekly_btc_ex", "weekly", mem_ex),
    )
    books = {}
    for name, mode, mem in specs:
        print(f"[HB] book {name}...", flush=True)
        packed = run_long_book(
            close,
            btc_id,
            mem,
            feat,
            dates,
            mode=mode,
            label=name,
            selection="score",
            id_to_sym=id_to_sym,
        )
        if packed.get("error"):
            raise RuntimeError(f"{name} failed: {packed}")
        books[name] = packed
        print(
            f"[HB] {name} total={packed.get('total')} CAGR={packed.get('cagr')} "
            f"Sharpe={packed.get('sharpe')} corr={packed.get('corr_btc')} "
            f"tilt={packed.get('score_vol_tilt')}",
            flush=True,
        )

    print("[HB] EW mcap-50 no-selection (btc-in)...", flush=True)
    ew = run_long_book(
        close,
        btc_id,
        mem_in,
        feat,
        dates,
        mode="daily",
        label="ew_mcap50",
        selection="ew",
        id_to_sym=id_to_sym,
    )
    print(f"[HB] EW total={ew.get('total')} corr={ew.get('corr_btc')}", flush=True)

    print("[HB] informational DV100 daily btc-ex...", flush=True)
    pit_mem = ffill_members(pit_members(pit, btc_id), list(close.index))
    pit_mem = filter_members(pit_mem, tagged, peg_heu, btc_id, btc_ex=True)
    dv = run_long_book(
        close,
        btc_id,
        pit_mem,
        feat,
        dates,
        mode="daily",
        label="daily_dv100_btc_ex",
        selection="score",
        id_to_sym=id_to_sym,
    )
    books["daily_dv100_btc_ex"] = dv
    print(f"[HB] DV100 total={dv.get('total')} corr={dv.get('corr_btc')}", flush=True)

    idx = books["daily_btc_ex"]["daily_ret"].index
    btc = btc_bh_book(close, btc_id, idx)
    print(f"[HB] BTC B&H total={btc.get('total')} CAGR={btc.get('cagr')} MaxDD={btc.get('maxdd')}", flush=True)

    print("[HB] frozen-spread crude book (reference, 2.c cache)...", flush=True)
    twin = load_twin_from_cache(pred_dir_2c, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()
    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
    members_pit = ffill_members(pit_members(pit, btc_id), list(close.index))
    oos = [d for d in list(close.index) if d >= pd.Timestamp(PHASE3C_REF_START, tz="UTC")]
    pairs14 = formation_dates(oos, 14)
    scores = preds_to_score_at(frozen, "spread", [t for t, _, _ in pairs14])
    spr = run_periodic_long(
        close,
        members_pit,
        btc_id,
        scores,
        pairs14,
        cost_bps=float(ALT_BPS),
        label="frozen_spread",
    )
    # attach BTC relative / corr on the common idx
    spr_rets = spr["daily_ret"]
    spr_rets.index = _utc_idx(spr_rets.index)
    btc_r = btc["daily_ret"].reindex(spr_rets.index).fillna(0.0)
    m = spr_rets.notna() & btc_r.notna()
    spr["corr_btc"] = float(spr_rets[m].corr(btc_r[m])) if int(m.sum()) >= 10 else float("nan")
    spr["total"] = spr.get("total")
    spr["sharpe"] = spr.get("sharpe")
    spr["net_sharpe_trail18m"] = spr.get("sharpe")
    from btcb.book import summarize_book as _sum

    z = pd.Series(0.0, index=spr_rets.index)
    nn = pd.Series(float("nan"), index=spr_rets.index)
    spr_sum = _sum(spr_rets, btc_r, z, nn, [])
    spr["rel_sharpe"] = spr_sum.get("rel_sharpe")
    spr["rel_total"] = spr_sum.get("rel_total")
    spr["book_sharpe"] = spr_sum.get("book_sharpe")
    spr["sharpe"] = spr_sum.get("book_sharpe")
    spr["net_sharpe_trail18m"] = _sharpe(trail_slice(spr_rets, LS_TRAIL_DAYS))
    spr["avg_n_names"] = spr.get("avg_n_names")
    spr["equity"] = spr.get("equity")
    spr["daily_ret"] = spr_rets
    print(f"[HB] frozen-spread total={spr.get('total')} CAGR={spr.get('cagr')} corr={spr.get('corr_btc')}", flush=True)

    verdict = mechanical_verdicts(books, btc)
    best_key = verdict.get("best_book")
    best = books[best_key]
    tilt = best.get("score_vol_tilt")
    plain = (
        f"Best book {best_key}: total {best.get('total')} vs BTC {btc.get('total')} "
        f"(clause1={verdict.get('clause1_total_ge_btc')}); daily corr vs BTC {best.get('corr_btc')} "
        f"(clause2={verdict.get('clause2_corr_le_070')}). Rel-line Sharpe {best.get('rel_sharpe')}; "
        f"MaxDD {best.get('maxdd')} vs BTC {btc.get('maxdd')}. "
        f"Score-vol tilt (Spearman score vs std63) = {tilt}. "
        f"Verdict {verdict.get('label')}"
        + (f" ({verdict.get('partial_which')})" if verdict.get("partial_which") else "")
        + ". Nothing adopted."
    )
    mapped_days = 0
    univ_days = 0
    for d in dates:
        for iid in mem_in.get(d, []):
            univ_days += 1
            if d in spot_wide.index and int(iid) in spot_wide.columns:
                v = spot_wide.at[d, int(iid)]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = float("nan")
                if np.isfinite(v) and v > 0:
                    mapped_days += 1
    bn_share = mapped_days / max(1, univ_days)

    extra = {
        "pred_sha256": pred_hash["sha256"],
        "cmc_panel_sha256": cmc_panel_sha0,
        "cmc_readonly_ok": cmc_panel_sha0 == CMC_PANEL_SHA256,
        "start": str(idx.min().date()) if len(idx) else None,
        "end": str(idx.max().date()) if len(idx) else None,
        "n_days": int(len(idx)),
        "gpu_used": False,
        "elapsed_sec": time.time() - t0,
        "plain": plain,
        "hybrid_bn_share": bn_share,
        "mean_univ_btc_in": n_in,
        "mean_univ_btc_ex": n_ex,
    }
    peg = {
        "n_tagged": int(len(tagged)),
        "sample_symbols": [r["symbol"] for r in tag_rows[:40]],
        "tagged_rows": tag_rows[:80],
        **heu_stats,
    }

    bench = {"btc_bh": btc, "ew_mcap50": ew, "frozen_spread": spr}

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_manuel2(
        ledger_path,
        verdict=verdict,
        extra={
            "best_total": best.get("total"),
            "btc_total": btc.get("total"),
            "best_corr": best.get("corr_btc"),
            "tilt": tilt,
        },
    )

    work = Path("/data/quant/btcb/manuel2")
    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (work, work / "books", rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    all_books = {**books, **bench}
    _persist_daily(all_books, work / "books")
    write_manuel2(
        rep_dir / "manuel2_gaussmom.md",
        books=books,
        bench=bench,
        verdict=verdict,
        peg=peg,
        extra=extra,
    )
    plot_manuel2(
        best,
        btc,
        ew,
        chart_dir / "manuel2_equity.png",
        best_label=str(best_key),
    )
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": MANUEL2_CRITERION,
        "formula": MANUEL2_FORMULA,
        "verdict": _jsonable(verdict),
        "books": {k: _jsonable(v) for k, v in books.items()},
        "bench": {k: _jsonable(v) for k, v in bench.items()},
        "peg": _jsonable(peg),
        "extra": extra,
    }
    (rep_dir / "manuel2_gaussmom.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "manuel2_addendum.md").write_text(addendum)
    quant_vol.commit()

    top5 = ", ".join(
        f"{r.get('symbol')}:{r.get('share'):.2%}" if r.get("share") is not None else str(r.get("symbol"))
        for r in (best.get("top5") or [])
    )
    print(f"{verdict.get('label')}", flush=True)
    print(
        f"CLAIM total={best.get('total')} vs BTC={btc.get('total')} "
        f"(clause1={verdict.get('clause1_total_ge_btc')}); "
        f"corr={best.get('corr_btc')} (clause2={verdict.get('clause2_corr_le_070')})",
        flush=True,
    )
    print(f"score-vol tilt Spearman(score, std63)={tilt}", flush=True)
    print(f"top-5 concentration share={best.get('top5_share')} names={top5}", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {
        "label": verdict.get("label"),
        "best_book": best_key,
        "best_total": best.get("total"),
        "btc_total": btc.get("total"),
        "best_corr": best.get("corr_btc"),
        "tilt": tilt,
        "top5_share": best.get("top5_share"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] MANUEL-2 falsification...", flush=True)
    fc = run_manuel2.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/manuel2_gaussmom.md", "reports"),
        ("reports/manuel2_gaussmom.json", "reports"),
        ("reports/manuel2_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/manuel2_equity.png", "charts"),
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
        png = Path("charts/manuel2_equity.png")
        if png.exists():
            (opt / "charts" / png.name).write_bytes(png.read_bytes())
            (opt / "screenshots" / png.name).write_bytes(png.read_bytes())
        for src in Path("reports").glob("manuel2*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = Path("reports/numbers_ledger.md")
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] MANUEL-2 complete.", flush=True)
