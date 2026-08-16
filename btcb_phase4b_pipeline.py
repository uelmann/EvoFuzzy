"""
BTC-BEATER Phase 4.b — TWIN-RANK + vol-matched null + DIR arm.

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Frozen products untouched. Master only.
Usage: modal run btcb_phase4b_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p4b-twinrank"
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
    .add_local_file(
        "reports/btcb_phase4b_addendum.md",
        remote_path="/root/btcb_phase4b_addendum.md",
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
        "equity",
        "id_to_sym",
        "btc_ret",
        "equity_btc",
        "rel_equity",
        "w_btc",
        "n_names",
        "gate_on",
        "alt_gross",
        "aucs",
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


def _load_tagged_preds(out_dir: Path, tag: str, horizon: int):
    import pandas as pd

    files = sorted(out_dir.glob(f"preds_{tag}_h{horizon}_fold*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    return df


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


def _twinrank_with_folds(top, bot):
    from btcb.phase4b import twinrank_from_heads

    parts = []
    if "fold_id" not in top.columns or "fold_id" not in bot.columns:
        s = twinrank_from_heads(top, bot, "p", "p", "twinrank")
        return s
    for fid, gt in top.groupby("fold_id"):
        gb = bot[bot["fold_id"] == fid]
        s = twinrank_from_heads(gt, gb, "p", "p", "twinrank")
        if s.empty:
            continue
        s["fold_id"] = int(fid)
        parts.append(s)
    if not parts:
        return twinrank_from_heads(top, bot, "p", "p", "twinrank")
    import pandas as pd

    return pd.concat(parts, ignore_index=True)


@app.function(
    timeout=60 * 60 * 10,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p4b() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd

    from btcb.academic_factor import pit_members
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        NULL_FOLD_IDS_2C,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
        PHASE4B_CRITERION,
        PHASE4B_DIR_RATIONALE,
        PHASE4B_H,
        PHASE4B_NULL_REGISTRATION,
        PHASE4V2_PI_SCOPE,
        SEED,
        STAGE_S_COLS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.gates import assert_no_context, pick_folds_by_id
    from btcb.hygiene import clean_panel
    from btcb.labels import add_dir_top_decile_weights, add_rank_grade_labels, add_twin_quintile_labels
    from btcb.model import make_expanding_folds, train_all_folds, train_all_rank_folds
    from btcb.oracle_ladder import _as_utc, ffill_members, formation_dates, run_periodic_long
    from btcb.phase4b import (
        attach_vol_corr,
        gate_vol_matched_dir_null,
        gate_vol_matched_rank_null,
        gate_vol_matched_twinrank_null,
        mechanical_verdicts,
        merge_dir_spread,
        real_fold_metrics,
        twinrank_from_heads,
        vol_col_name,
    )
    from btcb.phase4v2 import (
        collapse_fold_preds,
        cs_rank_blend,
        per_date_tail_metrics,
        preds_to_score_at,
        restrict_eval_frame,
    )
    from btcb.phase4b_report import (
        plot_overlap_cycles,
        plot_tail_ic_bars,
        update_ledger_phase4b,
        write_phase4b,
    )
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache
    from baseline.seedutil import seed_everything

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase4b_addendum.md").read_text()
    for txt in (
        PHASE4B_CRITERION,
        PHASE4B_NULL_REGISTRATION,
        PHASE4B_DIR_RATIONALE,
        PHASE4V2_PI_SCOPE,
        DEATH_CONVENTION,
    ):
        if txt not in addendum:
            raise RuntimeError(f"phase4b addendum missing freeze text: {txt[:80]}")
    print("[HB] PHASE 4.b TWIN-RANK ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {PHASE4B_NULL_REGISTRATION}", flush=True)
    print(f"[HB] {PHASE4B_CRITERION}", flush=True)

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

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    twin = load_twin_from_cache(pred_dir_2c, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()

    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if not feat_path.exists():
        raise RuntimeError("missing feat_s")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)
    from btcb.constants import CTX_COLS

    leaked = [c for c in CTX_COLS if c in feat.columns]
    if leaked:
        raise RuntimeError(f"context leaked into feat_s: {leaked}")

    work = Path("/data/quant/btcb/phase4b")
    pred_out = work / "preds"
    for d in (work, pred_out):
        d.mkdir(parents=True, exist_ok=True)

    print("[HB] labels h=14 twin + rank grades + DIR weights...", flush=True)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(PHASE4B_H,))
    labeled = add_rank_grade_labels(labeled, horizon=PHASE4B_H, n_grades=5)
    labeled = add_dir_top_decile_weights(labeled, horizon=PHASE4B_H, decile=10, boost=2.0)
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    labeled["date"] = pd.to_datetime(labeled["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    labeled["id"] = labeled["id"].astype(int)
    volc = vol_col_name(labeled)
    print(
        f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} "
        f"ids={labeled['id'].nunique()} vol_col={volc}",
        flush=True,
    )

    feats_s = list(STAGE_S_COLS)
    assert_no_context(feats_s)

    print("[HB] loading Binance spot close (canonical books/eval)...", flush=True)
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

    y_rank = f"y_rank_h{PHASE4B_H}"
    y_rank_bot = f"y_rank_bot_h{PHASE4B_H}"
    y_top = f"y_h{PHASE4B_H}"
    w_dir = f"w_dir_h{PHASE4B_H}"

    p4v2_pred = Path("/data/quant/btcb/phase4v2/preds")
    rank_top = _load_tagged_preds(p4v2_pred, "rank_s", PHASE4B_H)
    rank_cache_reused = bool(rank_top is not None and not rank_top.empty)
    if rank_cache_reused:
        print(f"[HB] reuse 4v2 RANK top cache rows={len(rank_top)} folds={rank_top['fold_id'].nunique()}", flush=True)
    else:
        print("[HB] train RANK top (STAGE_S) — 4v2 cache missing...", flush=True)
        rank_top, _m, _f = train_all_rank_folds(
            labeled,
            PHASE4B_H,
            out_dir=pred_out,
            feature_cols=feats_s,
            ycol=y_rank,
            tag="rank_s",
            commit_fn=quant_vol.commit,
        )

    print("[HB] train RANK bottom (inverted grades)...", flush=True)
    rank_bot, _mb, _fb = train_all_rank_folds(
        labeled,
        PHASE4B_H,
        out_dir=pred_out,
        feature_cols=feats_s,
        ycol=y_rank_bot,
        tag="rank_bot_s",
        commit_fn=quant_vol.commit,
    )

    print("[HB] train DIR top (weights on realized top decile)...", flush=True)
    dir_top, _md, _fd = train_all_folds(
        labeled,
        PHASE4B_H,
        out_dir=pred_out,
        feature_cols=feats_s,
        early_stop="per_date_auc",
        ycol=y_top,
        tag="dir_top",
        commit_fn=quant_vol.commit,
        weight_col=w_dir,
    )

    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
    top_c = collapse_fold_preds(rank_top, "p")
    bot_c = collapse_fold_preds(rank_bot, "p")
    twin_sig = twinrank_from_heads(top_c, bot_c, "p", "p", "twinrank")
    twin_raw = _twinrank_with_folds(rank_top, rank_bot)
    blend = cs_rank_blend(frozen, twin_sig, "spread", "twinrank", "spread_twinrank")
    dir_raw = merge_dir_spread(dir_top, twin)
    print(f"[HB] DIR merge rows={len(dir_raw)} dir_top={len(dir_top)} twin={len(twin)}", flush=True)
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

    grid = {}
    for name, (df, col) in signals.items():
        ev = restrict_eval_frame(df, labeled, close, btc_id, col)
        met = per_date_tail_metrics(ev, col)
        met = attach_vol_corr(met, ev, col, labeled, volc)
        grid[name] = met
        print(
            f"[HB] {name} tailIC_top={met.get('tail_ic_top')} overlap={met.get('overlap')} "
            f"rankic={met.get('rankic')} vol={met.get('vol_rank_corr')} n={met.get('n_dates')}",
            flush=True,
        )

    def _vol_of(df, col):
        ev = restrict_eval_frame(df, labeled, close, btc_id, col)
        return attach_vol_corr({}, ev, col, labeled, volc).get("vol_rank_corr")

    vol_diag = {
        "rank_top": _vol_of(top_c.rename(columns={"p": "rank_top"}), "rank_top"),
        "rank_bot": _vol_of(bot_c.rename(columns={"p": "rank_bot"}), "rank_bot"),
        "twinrank": (grid.get("twinrank") or {}).get("vol_rank_corr"),
        "frozen_spread": (grid.get("frozen_spread") or {}).get("vol_rank_corr"),
        "dir_spread": (grid.get("dir_spread") or {}).get("vol_rank_corr"),
    }
    print(
        f"[HB] vol-corr top={vol_diag['rank_top']} bot={vol_diag['rank_bot']} "
        f"twin={vol_diag['twinrank']}",
        flush=True,
    )

    folds_all = make_expanding_folds(pd.DatetimeIndex(labeled["date"].unique()), horizon=PHASE4B_H)
    null_folds = pick_folds_by_id(folds_all, NULL_FOLD_IDS_2C)
    print(f"[HB] null folds {[f.fold_id for f in null_folds]}", flush=True)

    real_twin = real_fold_metrics(twin_raw, null_folds, labeled, close, btc_id, "twinrank")
    real_rank = real_fold_metrics(rank_top, null_folds, labeled, close, btc_id, "p")
    real_dir = real_fold_metrics(dir_raw, null_folds, labeled, close, btc_id, "dir_spread")

    def _load_or_run(summary_path: Path, fn):
        if summary_path.exists():
            print(f"[HB] reuse cached null {summary_path.name}", flush=True)
            return json.loads(summary_path.read_text())
        blob = fn()
        summary_path.write_text(json.dumps(_jsonable(blob), indent=2))
        quant_vol.commit()
        return blob

    null_root = work / "null"
    null_root.mkdir(parents=True, exist_ok=True)

    print("[HB] TWIN-RANK vol-matched null (6 folds × 25 × 2 heads)...", flush=True)
    null_twin = _load_or_run(
        null_root / "twinrank_vol_matched.json",
        lambda: gate_vol_matched_twinrank_null(
            labeled,
            null_folds,
            real_twin,
            labeled,
            close,
            btc_id,
            feats_s,
            y_rank,
            y_rank_bot,
            cache_dir=null_root / "twinrank",
            commit_fn=quant_vol.commit,
            vol_col=volc,
        ),
    )

    print("[HB] retro RANK vol-matched null (6 folds × 25)...", flush=True)
    null_rank = _load_or_run(
        null_root / "rank_vol_matched.json",
        lambda: gate_vol_matched_rank_null(
            labeled,
            null_folds,
            real_rank,
            labeled,
            close,
            btc_id,
            feats_s,
            y_rank,
            cache_dir=null_root / "rank",
            commit_fn=quant_vol.commit,
            vol_col=volc,
        ),
    )

    print("[HB] DIR vol-matched null (6 folds × 25 top head)...", flush=True)
    null_dir = _load_or_run(
        null_root / "dir_vol_matched.json",
        lambda: gate_vol_matched_dir_null(
            labeled,
            null_folds,
            real_dir,
            labeled,
            close,
            btc_id,
            feats_s,
            twin,
            y_top,
            w_dir,
            cache_dir=null_root / "dir",
            commit_fn=quant_vol.commit,
            vol_col=volc,
        ),
    )

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
            f"[HB] book {name} CAGR={packed.get('cagr')} MaxDD={packed.get('maxdd')} "
            f"RankIC={packed.get('rankic')}",
            flush=True,
        )

    verdict = mechanical_verdicts(grid, null_twin, null_dir, null_rank)
    base = grid["frozen_spread"]
    twin_m = grid["twinrank"]
    dir_m = grid["dir_spread"]
    vol_line = (
        f"vol-corr collapse: top={vol_diag.get('rank_top')} bot={vol_diag.get('rank_bot')} "
        f"twinrank={vol_diag.get('twinrank')}"
    )
    retro_line = (
        f"RETRO RANK vol-matched verdict={verdict.get('retro_rank_verdict')} "
        f"skill_pass={verdict.get('retro_rank_skill_pass')} "
        f"gain_beyond_vol={'YES' if verdict.get('retro_rank_vol_matched_pass') else 'NO'}"
    )
    plain = (
        f"TWIN-RANK vs frozen spread: tail-IC(top-half) {base.get('tail_ic_top')} → "
        f"{twin_m.get('tail_ic_top')} (Δ {verdict.get('delta_twin_vs_base_tail_ic')}), "
        f"overlap {base.get('overlap')} → {twin_m.get('overlap')} "
        f"(Δ {verdict.get('delta_twin_vs_base_overlap')}). "
        f"Vol-matched null {'passed' if verdict.get('twin_null_pass') else 'did not pass'}. "
        f"DIR vs frozen: tail-IC {dir_m.get('tail_ic_top')} (Δ {verdict.get('delta_dir_vs_base_tail_ic')}), "
        f"overlap {dir_m.get('overlap')} (Δ {verdict.get('delta_dir_vs_base_overlap')}); "
        f"null {'passed' if verdict.get('dir_null_pass') else 'did not pass'}. "
        f"{vol_line}. {retro_line}. "
        f"Verdicts: {verdict.get('twinrank')}; {verdict.get('dir')}. "
        f"{verdict.get('ceiling') or ''} Nothing adopted."
    )
    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    extra = {
        "pred_sha256": pred_hash["sha256"],
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_readonly_ok": cmc_panel_sha1 == cmc_panel_sha0,
        "start": str(close.index.min().date()) if len(close) else None,
        "end": str(close.index.max().date()) if len(close) else None,
        "n_eval_dates": (grid.get("frozen_spread") or {}).get("n_dates"),
        "gpu_used": False,
        "elapsed_sec": time.time() - t0,
        "plain": plain,
        "n_pairs_h14": int(len(pairs14)),
        "rank_cache_reused": rank_cache_reused,
        "vol_col": volc,
        "vol_diag": vol_diag,
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_phase4b(
        ledger_path,
        verdict=verdict,
        extra={
            "base_tail_ic": base.get("tail_ic_top"),
            "base_overlap": base.get("overlap"),
            "twin_tail_ic": twin_m.get("tail_ic_top"),
            "twin_overlap": twin_m.get("overlap"),
            "dir_tail_ic": dir_m.get("tail_ic_top"),
            "dir_overlap": dir_m.get("overlap"),
        },
    )

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase4b(
        rep_dir / "btcb_phase4b_twinrank.md",
        grid=grid,
        books={k: _jsonable(v) for k, v in books.items()},
        null_twin=null_twin,
        null_dir=null_dir,
        null_rank=null_rank,
        vol_diag=vol_diag,
        verdict=verdict,
        extra=extra,
    )
    plot_tail_ic_bars(
        grid,
        {"twinrank": null_twin, "dir_spread": null_dir},
        chart_dir / "btcb_phase4b_tail_ic.png",
    )
    plot_overlap_cycles(grid, chart_dir / "btcb_phase4b_overlap_cycle.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": PHASE4B_CRITERION,
        "null_registration": PHASE4B_NULL_REGISTRATION,
        "verdict": _jsonable(verdict),
        "grid": {k: _jsonable(v) for k, v in grid.items()},
        "books": {k: _jsonable(v) for k, v in books.items()},
        "null_twin": _jsonable(null_twin),
        "null_dir": _jsonable(null_dir),
        "null_rank": _jsonable(null_rank),
        "vol_diag": _jsonable(vol_diag),
        "extra": extra,
    }
    (rep_dir / "btcb_phase4b_twinrank.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "btcb_phase4b_addendum.md").write_text(addendum)
    quant_vol.commit()

    print(f"{verdict.get('twinrank')}; {verdict.get('dir')}", flush=True)
    print(retro_line, flush=True)
    print(vol_line, flush=True)
    if verdict.get("ceiling"):
        print(verdict.get("ceiling"), flush=True)
    print(
        f"BASELINE tail-IC(top-half)={base.get('tail_ic_top')} overlap={base.get('overlap')} | "
        f"TWIN-RANK tail-IC={twin_m.get('tail_ic_top')} overlap={twin_m.get('overlap')} | "
        f"DIR tail-IC={dir_m.get('tail_ic_top')} overlap={dir_m.get('overlap')}",
        flush=True,
    )
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {
        "twinrank": verdict.get("twinrank"),
        "dir": verdict.get("dir"),
        "ceiling": verdict.get("ceiling"),
        "retro_rank_verdict": verdict.get("retro_rank_verdict"),
        "retro_beyond_vol": verdict.get("retro_rank_vol_matched_pass"),
        "vol_diag": vol_diag,
        "base_tail_ic": base.get("tail_ic_top"),
        "twin_tail_ic": twin_m.get("tail_ic_top"),
        "dir_tail_ic": dir_m.get("tail_ic_top"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] Phase 4.b TWIN-RANK...", flush=True)
    fc = run_btcb_p4b.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase4b_twinrank.md", "reports"),
        ("reports/btcb_phase4b_twinrank.json", "reports"),
        ("reports/btcb_phase4b_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase4b_tail_ic.png", "charts"),
        ("charts/btcb_phase4b_overlap_cycle.png", "charts"),
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
        for src in (art / "charts").glob("btcb_phase4b*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase4b*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] Phase 4.b complete.", flush=True)
