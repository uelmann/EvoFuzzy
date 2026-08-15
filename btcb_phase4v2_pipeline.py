"""
BTC-BEATER Phase 4 v2 — TAIL ROUND 1.

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Frozen products untouched. Master only.
Usage: modal run btcb_phase4v2_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p4v2-tail1"
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
        "reports/btcb_phase4v2_addendum.md",
        remote_path="/root/btcb_phase4v2_addendum.md",
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


@app.function(
    timeout=60 * 60 * 8,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p4v2() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd

    from baseline.data import load_funding_panel
    from btcb.academic_factor import pit_members
    from btcb.binance_replay import close_wide_from_panel
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
        PHASE4V2_CRITERION,
        PHASE4V2_H,
        PHASE4V2_PI_KILL,
        PHASE4V2_PI_SCOPE,
        POSITIONING_COLS,
        PRICE_ADD_COLS,
        SEED,
        STAGE_S_COLS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.gates import assert_no_context, pick_folds_by_id
    from btcb.hygiene import clean_panel
    from btcb.labels import add_rank_grade_labels, add_twin_quintile_labels
    from btcb.model import (
        make_expanding_folds,
        merge_twin_preds,
        train_all_folds,
        train_all_rank_folds,
    )
    from btcb.oracle_ladder import _as_utc, ffill_members, formation_dates, run_periodic_long
    from btcb.phase4v2 import (
        build_positioning_by_symbol,
        build_price_additions,
        collapse_fold_preds,
        coverage_tables,
        cs_rank_blend,
        fill_metrics_gaps,
        fold_tail_from_pred,
        gate_rank_tail_null,
        id_symbol_maps,
        join_price_additions,
        load_symbol_parquets,
        load_taker_panel,
        map_positioning_to_ids,
        mechanical_verdicts,
        per_date_tail_metrics,
        preds_to_score_at,
        restrict_eval_frame,
    )
    from btcb.phase4v2_report import (
        plot_overlap_cycles,
        plot_tail_ic_bars,
        update_ledger_phase4v2,
        write_phase4v2,
    )
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache
    from baseline.seedutil import seed_everything

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase4v2_addendum.md").read_text()
    for txt in (PHASE4V2_CRITERION, PHASE4V2_PI_SCOPE, PHASE4V2_PI_KILL, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"phase4v2 addendum missing freeze text: {txt[:80]}")
    print("[HB] PHASE 4 v2 TAIL ROUND 1 ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {PHASE4V2_PI_SCOPE}", flush=True)
    print(f"[HB] {PHASE4V2_PI_KILL}", flush=True)
    print(f"[HB] {PHASE4V2_CRITERION}", flush=True)

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
    leaked = [c for c in ("ctx_disp",) if c in feat.columns]
    from btcb.constants import CTX_COLS

    leaked = [c for c in CTX_COLS if c in feat.columns]
    if leaked:
        raise RuntimeError(f"context leaked into feat_s: {leaked}")

    work = Path("/data/quant/btcb/phase4v2")
    pred_out = work / "preds"
    for d in (work, pred_out):
        d.mkdir(parents=True, exist_ok=True)

    print("[HB] loading Binance caches (spot/perp/funding/metrics)...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    perp_dir = Path("/data/quant/raw/klines")
    fund_dir = Path("/data/quant/raw/funding")
    metrics_dir = Path("/data/quant/raw/metrics")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet")) if spot_dir.exists() else []
    perp_syms = sorted(p.stem.upper() for p in perp_dir.glob("*.parquet")) if perp_dir.exists() else []
    print(f"[HB] spot parquets={len(spot_syms)} perp parquets={len(perp_syms)}", flush=True)

    all_ids = sorted(set(int(i) for i in pit["id"].unique()) | {int(btc_id)})
    id_to_perp, id_to_spot = id_symbol_maps(cleaned, all_ids, set(perp_syms), set(spot_syms), btc_id)
    perp_mapped = sorted({s for s in id_to_perp.values() if s})
    print(f"[HB] perp-mapped names={sum(1 for v in id_to_perp.values() if v)} symbols={len(perp_mapped)}", flush=True)

    missing_metrics = [s for s in perp_mapped if not (metrics_dir / f"{s}.parquet").exists()]
    download_log = []
    if missing_metrics and len(missing_metrics) <= 20:
        print(f"[HB] metrics missing {len(missing_metrics)} symbols; Vision gap fill", flush=True)
        download_log.extend(fill_metrics_gaps(metrics_dir, missing_metrics, start_day="2020-09-01"))
        quant_vol.commit()
    elif missing_metrics:
        print(
            f"[HB] {len(missing_metrics)} metrics symbols missing; skipping full historical backfill "
            f"(Phase D cache expected). Sample={missing_metrics[:8]}",
            flush=True,
        )
        download_log.append(
            {
                "symbol": "*",
                "n_new_rows": 0,
                "n_todo": len(missing_metrics),
                "skipped_full_backfill": True,
                "sample": missing_metrics[:20],
                "source": "none",
            }
        )
    exist_metrics = [s for s in perp_mapped if (metrics_dir / f"{s}.parquet").exists()]
    if exist_metrics:
        recent = fill_metrics_gaps(
            metrics_dir,
            exist_metrics,
            start_day=(end - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        )
        download_log.extend([r for r in recent if int(r.get("n_new_rows") or 0) > 0])
        if any(int(r.get("n_new_rows") or 0) > 0 for r in recent):
            quant_vol.commit()

    funding = load_funding_panel(fund_dir, perp_mapped) if fund_dir.exists() else pd.DataFrame()
    metrics = load_symbol_parquets(metrics_dir, perp_mapped) if metrics_dir.exists() else pd.DataFrame()
    perp_long = _load_close_long(perp_dir, perp_mapped)
    spot_needed = sorted({s for s in list(id_to_spot.values()) + perp_mapped if s})
    spot_long = _load_close_long(spot_dir, spot_needed)
    taker = load_taker_panel(perp_dir, perp_mapped, cache_path=work / "taker_panel.parquet")
    print(
        f"[HB] funding rows={len(funding)} metrics rows={len(metrics)} "
        f"perp_close={len(perp_long)} spot_close={len(spot_long)} taker={len(taker)}",
        flush=True,
    )

    pos_raw, pos_meta = build_positioning_by_symbol(
        funding, metrics, perp_long, spot_long, taker, perp_mapped
    )
    pos_block, cov_df = map_positioning_to_ids(pos_raw, feat, id_to_perp)
    price_raw = build_price_additions(cleaned, btc_id, all_ids)
    price_block = join_price_additions(feat, price_raw)
    coverage = coverage_tables(cov_df, pit)
    oi_first = pos_meta.get("first_oi_date") or {}
    print(
        f"[HB] perp coverage from 2021={coverage.get('perp_coverage_top100_from_2021')} "
        f"n={coverage.get('n_name_days_from_2021')} oi_names={len(oi_first)}",
        flush=True,
    )
    pos_block.to_parquet(work / "positioning_id.parquet", index=False)
    price_block.to_parquet(work / "price_add_id.parquet", index=False)
    (work / "coverage.json").write_text(json.dumps(_jsonable(coverage), indent=2))
    (work / "oi_first.json").write_text(json.dumps(oi_first, indent=2))
    (work / "download_log.json").write_text(json.dumps(download_log, indent=2))
    quant_vol.commit()

    print("[HB] labels h=14 twin + rank grades...", flush=True)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(PHASE4V2_H,))
    labeled = add_rank_grade_labels(labeled, horizon=PHASE4V2_H, n_grades=5)
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    labeled["date"] = pd.to_datetime(labeled["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    labeled["id"] = labeled["id"].astype(int)
    labeled = labeled.merge(pos_block, on=["date", "id"], how="left")
    labeled = labeled.merge(price_block, on=["date", "id"], how="left")
    for c in list(POSITIONING_COLS) + list(PRICE_ADD_COLS):
        if c not in labeled.columns:
            labeled[c] = 0.0
        labeled[c] = labeled[c].fillna(0.0)
    print(
        f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} ids={labeled['id'].nunique()}",
        flush=True,
    )

    feats_s = list(STAGE_S_COLS)
    feats_pos = list(STAGE_S_COLS) + list(POSITIONING_COLS)
    feats_full = list(STAGE_S_COLS) + list(POSITIONING_COLS) + list(PRICE_ADD_COLS)
    assert_no_context(feats_s)
    assert_no_context(feats_pos)
    assert_no_context(feats_full)

    print("[HB] loading Binance spot close (canonical books/eval)...", flush=True)
    nonempty_spot = set(spot_long["symbol"].unique()) if not spot_long.empty else set()
    close = close_wide_from_panel(spot_long.rename(columns={"close": "close"}), id_to_spot)
    if int(btc_id) not in close.columns:
        raise RuntimeError("BTCUSDT spot missing from close wide")
    close = close[close.index <= end].sort_index()
    close.index = pd.DatetimeIndex([_as_utc(d) for d in close.index])
    print(f"[HB] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    def _train_rank(tag, feature_cols):
        preds, _metas, _folds = train_all_rank_folds(
            labeled,
            PHASE4V2_H,
            out_dir=pred_out,
            feature_cols=feature_cols,
            ycol=f"y_rank_h{PHASE4V2_H}",
            tag=tag,
            commit_fn=quant_vol.commit,
        )
        return collapse_fold_preds(preds, "p")

    def _train_twin(tag_top, tag_bot, feature_cols):
        top, _mt, _f = train_all_folds(
            labeled,
            PHASE4V2_H,
            out_dir=pred_out,
            feature_cols=feature_cols,
            early_stop="per_date_auc",
            ycol=f"y_h{PHASE4V2_H}",
            tag=tag_top,
            commit_fn=quant_vol.commit,
        )
        bot, _mb, _f2 = train_all_folds(
            labeled,
            PHASE4V2_H,
            out_dir=pred_out,
            feature_cols=feature_cols,
            early_stop="per_date_auc",
            ycol=f"y_bot_h{PHASE4V2_H}",
            tag=tag_bot,
            commit_fn=quant_vol.commit,
        )
        return merge_twin_preds(top, bot, PHASE4V2_H)

    print("[HB] train RANK (STAGE_S)...", flush=True)
    rank = _train_rank("rank_s", feats_s)
    print("[HB] train twin +positioning...", flush=True)
    twin_pos = _train_twin("top_pos", "bot_pos", feats_pos)
    print("[HB] train twin +positioning +price...", flush=True)
    twin_price = _train_twin("top_pos_price", "bot_pos_price", feats_full)
    print("[HB] train RANK full stack features...", flush=True)
    rank_full = _train_rank("rank_full", feats_full)

    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p")
    frozen = frozen.rename(columns={"p": "spread"})
    rank_s = rank.rename(columns={"p": "rank_score"})
    blend = cs_rank_blend(frozen, rank_s, "spread", "rank_score", "blend")
    pos_df = collapse_fold_preds(twin_pos, "spread").rename(columns={"spread": "spread_pos"})
    price_df = collapse_fold_preds(twin_price, "spread").rename(columns={"spread": "spread_pos_price"})
    rank_f = rank_full.rename(columns={"p": "rank_full"})
    full_blend = cs_rank_blend(price_df, rank_f, "spread_pos_price", "rank_full", "full_stack")

    signals = {
        "frozen_spread": (frozen, "spread"),
        "rank": (rank_s, "rank_score"),
        "spread_rank": (blend, "blend"),
        "spread_pos": (pos_df, "spread_pos"),
        "spread_pos_price": (price_df, "spread_pos_price"),
        "full_stack": (full_blend, "full_stack"),
    }

    grid = {}
    for name, (df, col) in signals.items():
        ev = restrict_eval_frame(df, labeled, close, btc_id, col)
        met = per_date_tail_metrics(ev, col)
        grid[name] = met
        print(
            f"[HB] {name} tailIC_top={met.get('tail_ic_top')} overlap={met.get('overlap')} "
            f"rankic={met.get('rankic')} n={met.get('n_dates')}",
            flush=True,
        )

    print("[HB] RANK tail null (6 folds × 25)...", flush=True)
    folds_all = make_expanding_folds(pd.DatetimeIndex(labeled["date"].unique()), horizon=PHASE4V2_H)
    null_folds = pick_folds_by_id(folds_all, NULL_FOLD_IDS_2C)
    real_tail, real_ov = {}, {}
    for fold in null_folds:
        sl = rank[(rank["date"] >= fold.val_start) & (rank["date"] <= fold.val_end)]
        sm = fold_tail_from_pred(sl, labeled, close, btc_id, "p" if "p" in sl.columns else "rank_score")
        # rank_s used rank_score; original rank still has p
        if "rank_score" in rank_s.columns:
            sl2 = rank_s[(rank_s["date"] >= fold.val_start) & (rank_s["date"] <= fold.val_end)]
            sm = fold_tail_from_pred(sl2, labeled, close, btc_id, "rank_score")
        real_tail[fold.fold_id] = sm.get("tail_ic_top")
        real_ov[fold.fold_id] = sm.get("overlap")
        print(f"[HB] real fold {fold.fold_id} tailIC={real_tail[fold.fold_id]} ov={real_ov[fold.fold_id]}", flush=True)

    null_cache = work / "null"
    null_cache.mkdir(parents=True, exist_ok=True)
    null_summary = null_cache / "rank_tail_null.json"
    if null_summary.exists():
        null = json.loads(null_summary.read_text())
        print("[HB] reuse cached RANK tail null", flush=True)
    else:
        null = gate_rank_tail_null(
            labeled,
            null_folds,
            real_tail,
            real_ov,
            labeled,
            close,
            btc_id,
            feats_s,
            ycol=f"y_rank_h{PHASE4V2_H}",
            cache_dir=null_cache,
            commit_fn=quant_vol.commit,
        )
        null_summary.write_text(json.dumps(_jsonable(null), indent=2))
        quant_vol.commit()

    dates = list(close.index)
    members = ffill_members(pit_members(pit, btc_id), dates)
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(PHASE4V2_H))
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

    verdict = mechanical_verdicts(grid, coverage, null)
    base = grid["frozen_spread"]
    best_key = verdict.get("best_tail_signal") or "rank"
    best = grid.get(best_key) or {}
    plain = (
        f"RANK/blend vs frozen spread: tail-IC(top-half) {base.get('tail_ic_top')} → "
        f"{(grid.get(verdict.get('best_a')) or {}).get('tail_ic_top')} "
        f"(Δ {verdict.get('delta_a_vs_base_tail_ic')}), overlap {base.get('overlap')} → "
        f"{(grid.get(verdict.get('best_a')) or {}).get('overlap')} "
        f"(Δ {verdict.get('delta_a_vs_base_overlap')}). Null {'passed' if verdict.get('null_pass') else 'did not pass'}. "
        f"Positioning vs best A Δ tail-IC {verdict.get('delta_pos_vs_best_a_tail_ic')} / "
        f"overlap {verdict.get('delta_pos_vs_best_a_overlap')} with "
        f"{coverage.get('perp_coverage_top100_from_2021')} perp coverage from 2021. "
        f"Price-additions vs positioning Δ tail-IC {verdict.get('delta_price_vs_pos_tail_ic')} / "
        f"overlap {verdict.get('delta_price_vs_pos_overlap')}. "
        f"Verdicts: {verdict.get('tail_loss')}; {verdict.get('positioning')}; {verdict.get('price_additions')}. "
        "Nothing adopted."
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
        "n_rank_folds": int(rank_s["date"].nunique() and rank["fold_id"].nunique()) if "fold_id" in rank.columns else None,
        "perp_mapped": int(sum(1 for v in id_to_perp.values() if v)),
        "spot_nonempty": int(len(nonempty_spot)),
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_phase4v2(
        ledger_path,
        verdict=verdict,
        extra={
            "base_tail_ic": base.get("tail_ic_top"),
            "base_overlap": base.get("overlap"),
            "best_tail_ic": best.get("tail_ic_top"),
            "best_overlap": best.get("overlap"),
            "perp_cov": coverage.get("perp_coverage_top100_from_2021"),
        },
    )

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase4v2(
        rep_dir / "btcb_phase4v2_tail1.md",
        grid=grid,
        books={k: _jsonable(v) for k, v in books.items()},
        coverage=coverage,
        null=null,
        verdict=verdict,
        extra=extra,
        oi_first=oi_first,
        download_log=download_log,
    )
    plot_tail_ic_bars(grid, chart_dir / "btcb_phase4v2_tail_ic.png")
    plot_overlap_cycles(grid, chart_dir / "btcb_phase4v2_overlap_cycle.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": PHASE4V2_CRITERION,
        "pi_scope": PHASE4V2_PI_SCOPE,
        "pi_kill": PHASE4V2_PI_KILL,
        "verdict": _jsonable(verdict),
        "grid": {k: _jsonable(v) for k, v in grid.items()},
        "books": {k: _jsonable(v) for k, v in books.items()},
        "coverage": _jsonable(coverage),
        "null": _jsonable(null),
        "extra": extra,
        "oi_first": oi_first,
        "download_log": download_log,
    }
    (rep_dir / "btcb_phase4v2_tail1.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "btcb_phase4v2_addendum.md").write_text(addendum)
    quant_vol.commit()

    print(
        f"VERDICT {verdict.get('tail_loss')}; {verdict.get('positioning')}; {verdict.get('price_additions')}",
        flush=True,
    )
    print(
        f"BASELINE tail-IC(top-half)={base.get('tail_ic_top')} overlap={base.get('overlap')} | "
        f"BEST {best_key} tail-IC={best.get('tail_ic_top')} overlap={best.get('overlap')}",
        flush=True,
    )
    print(
        f"POSITIONING coverage from 2021={coverage.get('perp_coverage_top100_from_2021')} "
        f"n={coverage.get('n_name_days_from_2021')}",
        flush=True,
    )
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {
        "tail_loss": verdict.get("tail_loss"),
        "positioning": verdict.get("positioning"),
        "price_additions": verdict.get("price_additions"),
        "best_a": verdict.get("best_a"),
        "base_tail_ic": base.get("tail_ic_top"),
        "base_overlap": base.get("overlap"),
        "best_tail_ic": best.get("tail_ic_top"),
        "best_overlap": best.get("overlap"),
        "perp_coverage_from_2021": coverage.get("perp_coverage_top100_from_2021"),
        "null_pass": verdict.get("null_pass"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] Phase 4 v2 TAIL ROUND 1...", flush=True)
    fc = run_btcb_p4v2.spawn()
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase4v2_tail1.md", "reports"),
        ("reports/btcb_phase4v2_tail1.json", "reports"),
        ("reports/btcb_phase4v2_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase4v2_tail_ic.png", "charts"),
        ("charts/btcb_phase4v2_overlap_cycle.png", "charts"),
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
        for src in (art / "charts").glob("btcb_phase4v2*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase4v2*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] Phase 4 v2 complete.", flush=True)
