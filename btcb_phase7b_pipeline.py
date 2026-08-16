"""
BTC-BEATER Phase 7.b — FUZZY-STACK.

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Frozen products untouched. Master only.
Usage: modal run --detach --timestamps btcb_phase7b_pipeline.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p7b-fuzzystack"
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
        "reports/btcb_phase7b_addendum.md",
        remote_path="/root/btcb_phase7b_addendum.md",
    )
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME, image=image)
CMC_PANEL = Path("/data/quant/btcb/full/panel.parquet")
WORK = Path("/data/quant/btcb/phase7b")
PRED_DIR = WORK / "preds"
BASE_PATH = WORK / "labeled_base.parquet"
LIB_PATH = WORK / "library.parquet"
RULES_PATH = WORK / "rules.parquet"
CLOSE_PATH = WORK / "close.parquet"
FOLDS_PATH = WORK / "folds.json"


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
        "eval_curves",
        "reliability",
        "feature_cols",
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
    if isinstance(x, bytes):
        return None
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


def _utc_frame(df):
    import pandas as pd

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    out["id"] = out["id"].astype(int)
    return out


def _assemble_frame(base_path, extras, feature_cols):
    import pandas as pd

    base = _utc_frame(pd.read_parquet(base_path))
    for spec in extras or []:
        path = Path(spec["path"])
        if not path.exists():
            continue
        cols = spec.get("columns")
        if cols:
            use = ["date", "id"] + [c for c in cols if c not in ("date", "id")]
            extra = pd.read_parquet(path, columns=use)
        else:
            extra = pd.read_parquet(path)
        extra = _utc_frame(extra)
        overlap = [c for c in extra.columns if c in base.columns and c not in ("date", "id")]
        if overlap:
            extra = extra.drop(columns=overlap)
        base = base.merge(extra, on=["date", "id"], how="left")
    for c in feature_cols:
        if (str(c).startswith("rf_") or str(c).startswith("nfn_")) and c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    missing = [c for c in feature_cols if c not in base.columns]
    if missing:
        raise RuntimeError(f"missing features {missing[:8]} n={len(missing)}")
    return base


def _write_pred_bytes(pred_df) -> bytes:
    buf = io.BytesIO()
    pred_df.to_parquet(buf, index=False)
    return buf.getvalue()


@app.function(
    timeout=60 * 90,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=49152,
    max_containers=8,
)
def train_one_head_fold(job: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd

    from btcb.model import fit_predict_fold
    from btcb.phase7b import HYGIENE, LGBM_P7B, fold_spec_from_dict

    fold = fold_spec_from_dict(job["fold"])
    feats = list(job["feature_cols"])
    df = _assemble_frame(job["base_path"], job.get("extras") or [], feats)
    pred_df, meta = fit_predict_fold(
        df,
        fold,
        feature_cols=feats,
        early_stop="per_date_auc",
        ycol=job["ycol"],
        hygiene=HYGIENE,
        lgbm_params=LGBM_P7B,
        vol_col=job.get("vol_col", "yz_vol_30"),
    )
    meta = dict(meta)
    meta["head"] = job["head"]
    meta["tag"] = job["tag"]
    blob = b"" if pred_df is None or pred_df.empty else _write_pred_bytes(pred_df)
    return {
        "meta": _jsonable(meta),
        "parquet": blob,
        "fold_id": int(fold.fold_id),
        "head": job["head"],
        "tag": job["tag"],
        "status": meta.get("status"),
        "best_iteration": meta.get("best_iteration"),
        "undertrained": meta.get("undertrained"),
        "n_pred": 0 if pred_df is None or pred_df.empty else int(len(pred_df)),
        "feature_importance_gain": meta.get("feature_importance_gain") or {},
        "eval_curves": meta.get("eval_curves"),
        "elapsed": meta.get("elapsed"),
    }


@app.function(
    timeout=60 * 90,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
    max_containers=12,
)
def null_one_job(job: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd

    from btcb.phase7b import fold_spec_from_dict, null_one_replicate

    fold = fold_spec_from_dict(job["fold"])
    feats = list(job["feature_cols"])
    df = _assemble_frame(job["base_path"], job.get("extras") or [], feats)
    labeled = df
    close = pd.read_parquet(job["close_path"])
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True)).tz_convert("UTC").normalize()
    close.columns = [int(c) if str(c).lstrip("-").isdigit() else c for c in close.columns]
    rec = null_one_replicate(
        df,
        fold,
        int(job["shuffle_seed"]),
        labeled,
        close,
        int(job["btc_id"]),
        feats,
        job["y_top"],
        job["y_bot"],
        job.get("vol_col", "yz_vol_30"),
    )
    return _jsonable(rec)


@app.function(
    timeout=60 * 60 * 14,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=65536,
)
def run_btcb_p7b() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd

    from btcb.academic_factor import pit_members
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        CTX_COLS,
        DEATH_CONVENTION,
        NULL_FOLD_IDS_2C,
        NULL_REPLICATES,
        NULL_SHUFFLE_SEEDS,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
        PHASE4B_NULL_REGISTRATION,
        PHASE7B_CRITERION,
        PHASE7B_ES_CAP,
        PHASE7B_ES_FLOOR,
        PHASE7B_ES_PATIENCE,
        PHASE7B_FIREWALL,
        PHASE7B_H,
        PHASE7B_KEEP_K,
        PHASE7B_UNDERTRAINED_LT,
        SEED,
        STAGE_S_COLS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.gates import assert_no_context, pick_folds_by_id
    from btcb.hygiene import clean_panel
    from btcb.labels import add_twin_quintile_labels
    from btcb.model import make_expanding_folds, merge_twin_preds
    from btcb.oracle_ladder import _as_utc, ffill_members, formation_dates, run_periodic_long
    from btcb.phase4b import attach_vol_corr, persist_book_daily_rets, real_fold_metrics, vol_col_name
    from btcb.phase4v2 import collapse_fold_preds, per_date_tail_metrics, preds_to_score_at, restrict_eval_frame
    from btcb.phase7b import (
        HYGIENE,
        PROVENANCE_LIBRARY,
        PROVENANCE_NFN,
        PROVENANCE_ORIG,
        PROVENANCE_RULEFORGE,
        assemble_null_from_replicates,
        assert_firewall,
        assert_library_size,
        build_product_block,
        collapse_spread,
        count_undertrained,
        fold_spec_to_dict,
        gain_share,
        hygiene_rows,
        load_rule_stack,
        mechanical_verdicts,
        pick_best_arm,
        product_catalog,
        provenance_for,
        prune_library,
        total_gain_by_feature,
    )
    from btcb.phase7b_report import (
        plot_gain_share,
        plot_tail_ic_bars,
        update_ledger_phase7b,
        write_phase7b,
    )
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache
    from baseline.seedutil import seed_everything

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase7b_addendum.md").read_text()
    for txt in (PHASE7B_CRITERION, PHASE7B_FIREWALL, PHASE4B_NULL_REGISTRATION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"phase7b addendum missing freeze text: {txt[:80]}")
    print("[HB] PHASE 7.b FUZZY-STACK ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[HB] {PHASE7B_FIREWALL}", flush=True)
    print(f"[HB] {PHASE7B_CRITERION}", flush=True)

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
    leaked = [c for c in CTX_COLS if c in feat.columns]
    if leaked:
        raise RuntimeError(f"context leaked into feat_s: {leaked}")

    for d in (WORK, PRED_DIR, WORK / "null"):
        d.mkdir(parents=True, exist_ok=True)

    print("[HB] labels h=14 twin quintiles...", flush=True)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(PHASE7B_H,))
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

    orig_cols = list(STAGE_S_COLS)
    prims, specs = product_catalog(orig_cols)
    lib_names = [s["name"] for s in specs]
    assert_library_size(len(orig_cols), len(lib_names))
    print(
        f"[HB] primitives={len(prims)} products={len(lib_names)} (C(66,2) expected 2145)",
        flush=True,
    )

    print("[HB] building CDF product library (float32)...", flush=True)
    lib_block = build_product_block(labeled, orig_cols, specs)
    lib_df = labeled[["date", "id"]].copy()
    lib_df = pd.concat([lib_df.reset_index(drop=True), lib_block.reset_index(drop=True)], axis=1)
    lib_df.to_parquet(LIB_PATH, index=False)
    print(f"[HB] library wrote {LIB_PATH} cols={len(lib_names)} rows={len(lib_df)}", flush=True)
    del lib_block, lib_df

    keep_base = ["date", "id", "symbol", volc, f"y_h{PHASE7B_H}", f"y_bot_h{PHASE7B_H}", f"excess_h{PHASE7B_H}"]
    keep_base = [c for c in keep_base if c in labeled.columns] + [c for c in orig_cols if c in labeled.columns]
    # unique preserve order
    seen = set()
    keep_u = []
    for c in keep_base:
        if c not in seen:
            seen.add(c)
            keep_u.append(c)
    labeled[keep_u].to_parquet(BASE_PATH, index=False)

    stack_pack = load_rule_stack()
    stack_rec = stack_pack["record"]
    rules_df = stack_pack["frame"]
    rule_prov = dict(stack_rec.get("rule_provenance") or {})
    rule_cols = list(rule_prov)
    if rules_df is not None and rule_cols:
        rkeep = ["date", "id"] + [c for c in rule_cols if c in rules_df.columns]
        rules_df[rkeep].to_parquet(RULES_PATH, index=False)
        print(f"[HB] rules wrote {RULES_PATH} n={len(rkeep)-2}", flush=True)
    else:
        rule_cols = []

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
    close.to_parquet(CLOSE_PATH)
    print(f"[HB] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    folds_all = make_expanding_folds(pd.DatetimeIndex(labeled["date"].unique()), horizon=PHASE7B_H)
    FOLDS_PATH.write_text(json.dumps([fold_spec_to_dict(f) for f in folds_all], indent=2, default=str))
    print(f"[HB] folds={len(folds_all)}", flush=True)
    quant_vol.commit()

    y_top = f"y_h{PHASE7B_H}"
    y_bot = f"y_bot_h{PHASE7B_H}"
    hygiene_all: list[dict] = []

    def _extras(kind: str, cols: list[str] | None = None) -> list[dict]:
        out = []
        if kind in ("library", "both"):
            out.append({"path": str(LIB_PATH), "columns": list(cols) if cols is not None else list(lib_names)})
        if kind in ("rules", "both") and RULES_PATH.exists() and rule_cols:
            out.append({"path": str(RULES_PATH), "columns": list(rule_cols)})
        return out

    def _jobs(tag_top, tag_bot, feature_cols, extras) -> list[dict]:
        jobs = []
        for fold in folds_all:
            fd = fold_spec_to_dict(fold)
            for head, tag, ycol in (("top", tag_top, y_top), ("bot", tag_bot, y_bot)):
                jobs.append(
                    {
                        "fold": fd,
                        "head": head,
                        "tag": tag,
                        "ycol": ycol,
                        "feature_cols": list(feature_cols),
                        "base_path": str(BASE_PATH),
                        "extras": extras,
                        "vol_col": volc,
                    }
                )
        return jobs

    def _persist_map(results: list[dict], tag_top: str, tag_bot: str) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
        metas_top, metas_bot = [], []
        for rec in results:
            tag = rec["tag"]
            fid = rec["fold_id"]
            stem = f"preds_{tag}_h{PHASE7B_H}_fold{fid}"
            dest = PRED_DIR / f"{stem}.parquet"
            if rec.get("parquet"):
                dest.write_bytes(rec["parquet"])
            meta = rec.get("meta") or {}
            meta["feature_importance_gain"] = rec.get("feature_importance_gain") or meta.get("feature_importance_gain") or {}
            meta["head"] = rec.get("head")
            meta["tag"] = tag
            (PRED_DIR / f"meta_{stem}.json").write_text(json.dumps(meta, indent=2, default=str))
            hygiene_all.extend(hygiene_rows([meta], tag))
            if rec.get("head") == "top":
                metas_top.append(meta)
            else:
                metas_bot.append(meta)
            print(
                f"[HB] persisted {stem} status={rec.get('status')} best_iter={rec.get('best_iteration')} "
                f"UNDERTRAINED={rec.get('undertrained')} n={rec.get('n_pred')}",
                flush=True,
            )
        quant_vol.commit()
        top_files = sorted(PRED_DIR.glob(f"preds_{tag_top}_h{PHASE7B_H}_fold*.parquet"))
        bot_files = sorted(PRED_DIR.glob(f"preds_{tag_bot}_h{PHASE7B_H}_fold*.parquet"))
        top = pd.concat([pd.read_parquet(p) for p in top_files], ignore_index=True) if top_files else pd.DataFrame()
        bot = pd.concat([pd.read_parquet(p) for p in bot_files], ignore_index=True) if bot_files else pd.DataFrame()
        if not top.empty:
            top["date"] = pd.to_datetime(top["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            top["id"] = top["id"].astype(int)
        if not bot.empty:
            bot["date"] = pd.to_datetime(bot["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            bot["id"] = bot["id"].astype(int)
        metas_top = sorted(metas_top, key=lambda m: int(m.get("fold_id") or 0))
        metas_bot = sorted(metas_bot, key=lambda m: int(m.get("fold_id") or 0))
        return top, bot, metas_top, metas_bot

    def _frames_from_pred_files(top_files, bot_files):
        top = pd.concat([pd.read_parquet(p) for p in top_files], ignore_index=True) if top_files else pd.DataFrame()
        bot = pd.concat([pd.read_parquet(p) for p in bot_files], ignore_index=True) if bot_files else pd.DataFrame()
        if not top.empty:
            top["date"] = pd.to_datetime(top["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            top["id"] = top["id"].astype(int)
        if not bot.empty:
            bot["date"] = pd.to_datetime(bot["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            bot["id"] = bot["id"].astype(int)
        return top, bot

    def _reuse_arm(tag_top, tag_bot):
        """Reuse a complete arm written before a client-disconnect kill. Not a scientific redo."""
        n_need = len(folds_all)
        top_files = sorted(PRED_DIR.glob(f"preds_{tag_top}_h{PHASE7B_H}_fold*.parquet"))
        bot_files = sorted(PRED_DIR.glob(f"preds_{tag_bot}_h{PHASE7B_H}_fold*.parquet"))
        if len(top_files) != n_need or len(bot_files) != n_need:
            return None
        metas_top, metas_bot = [], []
        for files, bucket in ((top_files, metas_top), (bot_files, metas_bot)):
            for p in files:
                mp = PRED_DIR / f"meta_{p.stem}.json"
                if not mp.exists():
                    return None
                meta = json.loads(mp.read_text())
                if int(meta.get("best_iteration") or 0) <= 5:
                    return None
                if str(meta.get("status") or "ok") != "ok":
                    return None
                bucket.append(meta)
                hygiene_all.extend(hygiene_rows([meta], meta.get("tag") or tag_top))
        print(
            f"[HB] reuse cached {tag_top}/{tag_bot} folds={n_need} "
            f"(infra retry, not a second shot)",
            flush=True,
        )
        top, bot = _frames_from_pred_files(top_files, bot_files)
        metas_top = sorted(metas_top, key=lambda m: int(m.get("fold_id") or 0))
        metas_bot = sorted(metas_bot, key=lambda m: int(m.get("fold_id") or 0))
        twin_raw = merge_twin_preds(top, bot, PHASE7B_H) if (not top.empty and not bot.empty) else pd.DataFrame()
        twin_c = collapse_spread(twin_raw)
        return {"top": top, "bot": bot, "twin_raw": twin_raw, "twin": twin_c, "metas_top": metas_top, "metas_bot": metas_bot}

    def _run_arm(tag_top, tag_bot, feature_cols, extras, provenance):
        assert_no_context(list(feature_cols))
        assert_firewall(feature_cols, provenance, orig_cols)
        reused = _reuse_arm(tag_top, tag_bot)
        if reused is not None:
            return reused
        jobs = _jobs(tag_top, tag_bot, feature_cols, extras)
        print(f"[HB] map-train {tag_top}/{tag_bot} jobs={len(jobs)} n_feat={len(feature_cols)}", flush=True)
        try:
            results = list(train_one_head_fold.map(jobs, order_outputs=True, wrap_returned_exceptions=False))
        except TypeError:
            results = list(train_one_head_fold.map(jobs, order_outputs=True))
        top, bot, mt, mb = _persist_map(results, tag_top, tag_bot)
        twin_raw = merge_twin_preds(top, bot, PHASE7B_H) if (not top.empty and not bot.empty) else pd.DataFrame()
        twin_c = collapse_spread(twin_raw)
        return {"top": top, "bot": bot, "twin_raw": twin_raw, "twin": twin_c, "metas_top": mt, "metas_bot": mb}

    quant_vol.reload()
    # Drop only the hist-argmax=1 artifacts from the broken two-phase trainer.
    # Keep complete arms so a client-disconnect retry does not retrain stage 1.
    wiped = 0
    for meta_p in list(PRED_DIR.glob("meta_preds_*.json")):
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            meta = {}
        if int(meta.get("best_iteration") or 0) > 5:
            continue
        stem = meta_p.name[len("meta_") : -len(".json")] if meta_p.name.startswith("meta_") else meta_p.stem
        (PRED_DIR / f"{stem}.parquet").unlink(missing_ok=True)
        meta_p.unlink(missing_ok=True)
        wiped += 1
    print(f"[HB] wiped {wiped} broken-shot pred artifacts in {PRED_DIR}", flush=True)
    quant_vol.commit()

    # ----- Arm A stage 1 -----
    feats_s1 = list(orig_cols) + list(lib_names)
    prov_s1 = provenance_for(feats_s1, orig_cols, lib_names, {})
    print("[HB] ARM-A stage 1: twin heads on 33 + 2145 products...", flush=True)
    arm_a_s1 = _run_arm("arma_s1_top", "arma_s1_bot", feats_s1, _extras("library"), prov_s1)
    gain_top = total_gain_by_feature(arm_a_s1["metas_top"], lib_names)
    gain_bot = total_gain_by_feature(arm_a_s1["metas_bot"], lib_names)
    prune = prune_library(gain_top, gain_bot, lib_names, k=PHASE7B_KEEP_K)
    kept = list(prune["kept"])
    print(
        f"[HB] prune union n={len(kept)} top150={prune['n_top']} bot150={prune['n_bot']} (one prune)",
        flush=True,
    )
    spec_by_name = {s["name"]: s for s in specs}
    kept_formulas = []
    for name in kept:
        spec = spec_by_name[name]
        kept_formulas.append(
            {
                "name": name,
                "formula": spec["formula"],
                "gain_top": gain_top.get(name, 0.0),
                "gain_bot": gain_bot.get(name, 0.0),
                "gain_sum": float(gain_top.get(name, 0.0)) + float(gain_bot.get(name, 0.0)),
            }
        )
    kept_formulas.sort(key=lambda r: -r["gain_sum"])
    (WORK / "kept_products.json").write_text(json.dumps(kept_formulas, indent=2))
    print("[HB] kept-products top-5:", flush=True)
    for rec in kept_formulas[:5]:
        print(f"  {rec['formula']}  gain_sum={rec['gain_sum']:.1f}", flush=True)

    # ----- Arm A stage 2 -----
    feats_a = list(orig_cols) + list(kept)
    prov_a = provenance_for(feats_a, orig_cols, kept, {})
    print(f"[HB] ARM-A stage 2: retrain on 33 + {len(kept)} kept products...", flush=True)
    arm_a = _run_arm("arma_s2_top", "arma_s2_bot", feats_a, _extras("library", kept), prov_a)

    ran = {"arm_a": True, "arm_b": False, "arm_ab": False}
    arms = {"arm_a": arm_a}

    if not stack_rec.get("skipped") and rule_cols:
        feats_b = list(orig_cols) + list(rule_cols)
        prov_b = provenance_for(feats_b, orig_cols, [], rule_prov)
        print(f"[HB] ARM-B: retrain on 33 + {len(rule_cols)} rule features...", flush=True)
        arms["arm_b"] = _run_arm("armb_top", "armb_bot", feats_b, _extras("rules"), prov_b)
        ran["arm_b"] = True
        feats_ab = list(orig_cols) + list(kept) + list(rule_cols)
        prov_ab = provenance_for(feats_ab, orig_cols, kept, rule_prov)
        print(f"[HB] ARM-A+B: retrain on 33 + {len(kept)} products + {len(rule_cols)} rules...", flush=True)
        arms["arm_ab"] = _run_arm("armab_top", "armab_bot", feats_ab, _extras("both", kept), prov_ab)
        ran["arm_ab"] = True
    else:
        print(f"STACK-SKIPPED reasons={stack_rec.get('reasons')}", flush=True)

    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
    signals = {"frozen_spread": (frozen, "spread")}
    for name, pack in arms.items():
        signals[name] = (pack["twin"], "spread")

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

    best_arm = pick_best_arm(grid, ran)
    print(f"[HB] best arm={best_arm}", flush=True)

    null_folds = pick_folds_by_id(folds_all, NULL_FOLD_IDS_2C)
    null_best = None
    judged_feats = list(feats_a)
    judged_extras = _extras("library", kept)
    judged_pack = arms.get("arm_a")
    if best_arm == "arm_b":
        judged_feats = list(orig_cols) + list(rule_cols)
        judged_extras = _extras("rules")
        judged_pack = arms.get("arm_b")
    elif best_arm == "arm_ab":
        judged_feats = list(orig_cols) + list(kept) + list(rule_cols)
        judged_extras = _extras("both", kept)
        judged_pack = arms.get("arm_ab")

    if best_arm and judged_pack is not None and not judged_pack["twin_raw"].empty:
        real = real_fold_metrics(judged_pack["twin_raw"], null_folds, labeled, close, btc_id, "spread")
        seeds = list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]
        null_jobs = []
        for fold in null_folds:
            fd = fold_spec_to_dict(fold)
            for ss in seeds:
                null_jobs.append(
                    {
                        "fold": fd,
                        "shuffle_seed": int(ss),
                        "feature_cols": list(judged_feats),
                        "base_path": str(BASE_PATH),
                        "extras": judged_extras,
                        "close_path": str(CLOSE_PATH),
                        "btc_id": int(btc_id),
                        "y_top": y_top,
                        "y_bot": y_bot,
                        "vol_col": volc,
                    }
                )
        print(f"[HB] vol-matched null best={best_arm} jobs={len(null_jobs)}", flush=True)
        try:
            null_rows = list(null_one_job.map(null_jobs, order_outputs=True, wrap_returned_exceptions=False))
        except TypeError:
            null_rows = list(null_one_job.map(null_jobs, order_outputs=True))
        for rec in null_rows:
            if rec.get("undertrained_top") or rec.get("undertrained_bot"):
                hygiene_all.append(
                    {
                        "tag": f"null_{best_arm}",
                        "fold_id": rec.get("fold_id"),
                        "head": "twin",
                        "best_iteration": rec.get("best_iteration_top"),
                        "undertrained": True,
                        "status": rec.get("status"),
                        "elapsed": None,
                    }
                )
            elif rec.get("best_iteration_top") is not None:
                hygiene_all.append(
                    {
                        "tag": f"null_{best_arm}",
                        "fold_id": rec.get("fold_id"),
                        "head": "twin",
                        "best_iteration": rec.get("best_iteration_top"),
                        "undertrained": bool(
                            rec.get("undertrained_top")
                            or (
                                rec.get("best_iteration_top") is not None
                                and int(rec["best_iteration_top"]) < PHASE7B_UNDERTRAINED_LT
                            )
                        ),
                        "status": rec.get("status"),
                        "elapsed": None,
                    }
                )
        null_best = assemble_null_from_replicates(null_rows, null_folds, real)
        (WORK / "null" / f"{best_arm}_vol_matched.json").write_text(json.dumps(_jsonable(null_best), indent=2))
        quant_vol.commit()
        print(
            f"[HB] null passed={null_best.get('passed')} judged={null_best.get('judged')} "
            f"skill={(null_best.get('tail_ic_top') or {}).get('skill_pass')}",
            flush=True,
        )

    dates = list(close.index)
    members = ffill_members(pit_members(pit, btc_id), dates)
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(PHASE7B_H))
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
    persist_book_daily_rets(books, WORK / "books")

    judged_metas = []
    judged_prov = prov_a
    if best_arm == "arm_a":
        judged_metas = (arm_a.get("metas_top") or []) + (arm_a.get("metas_bot") or [])
        judged_prov = prov_a
    elif best_arm == "arm_b":
        judged_metas = (arms["arm_b"].get("metas_top") or []) + (arms["arm_b"].get("metas_bot") or [])
        judged_prov = provenance_for(list(orig_cols) + list(rule_cols), orig_cols, [], rule_prov)
    elif best_arm == "arm_ab":
        judged_metas = (arms["arm_ab"].get("metas_top") or []) + (arms["arm_ab"].get("metas_bot") or [])
        judged_prov = provenance_for(list(orig_cols) + list(kept) + list(rule_cols), orig_cols, kept, rule_prov)
    gain = gain_share(judged_metas, judged_prov)
    gain["ruleforge"] = gain.get(PROVENANCE_RULEFORGE, 0.0)
    gain["nfn"] = gain.get(PROVENANCE_NFN, 0.0)

    verdict = mechanical_verdicts(grid, null_best, best_arm, stack_rec.get("parents") or {}, ran)
    n_under = count_undertrained(hygiene_all)
    hygiene_blob = {
        "rows": hygiene_all,
        "n_undertrained": n_under,
        "n_fits": len(hygiene_all),
        "es_floor": PHASE7B_ES_FLOOR,
        "patience": PHASE7B_ES_PATIENCE,
        "cap": PHASE7B_ES_CAP,
        "undertrained_lt": PHASE7B_UNDERTRAINED_LT,
    }

    base = grid.get("frozen_spread") or {}
    best_m = grid.get(best_arm) or {} if best_arm else {}
    per = verdict.get("per_arm") or {}
    arm_lines = []
    for key, lab in (("arm_a", "ARM-A"), ("arm_b", "ARM-B"), ("arm_ab", "ARM-A+B")):
        rec = per.get(key)
        if rec:
            arm_lines.append(f"{lab} {rec.get('label')}")
        elif key == "arm_b" and stack_rec.get("skipped"):
            arm_lines.append("ARM-B STACK-SKIPPED")
    plain = (
        f"Arm A product library ({len(kept)} kept of {len(lib_names)}) vs frozen spread: "
        f"tail-IC(top-half) {base.get('tail_ic_top')} → {(grid.get('arm_a') or {}).get('tail_ic_top')} "
        f"(Δ {(per.get('arm_a') or {}).get('delta_tail_ic_top')}), "
        f"overlap {base.get('overlap')} → {(grid.get('arm_a') or {}).get('overlap')} "
        f"(Δ {(per.get('arm_a') or {}).get('delta_overlap')}). "
        f"Best arm={best_arm}. Vol-matched null "
        f"{'passed' if verdict.get('null_pass') else 'did not pass'}. "
        f"UNDERTRAINED count={n_under}. "
        f"Gain share originals={gain.get('originals')} products={gain.get('products')} rules={gain.get('rules')}. "
        f"Verdicts: {'; '.join(arm_lines)}. "
        f"{verdict.get('closed') or ''} Nothing adopted."
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
        "vol_col": volc,
        "n_library": len(lib_names),
        "n_kept": len(kept),
        "es_floor": PHASE7B_ES_FLOOR,
        "es_patience": PHASE7B_ES_PATIENCE,
        "es_cap": PHASE7B_ES_CAP,
        "undertrained_lt": PHASE7B_UNDERTRAINED_LT,
        "firewall_passed": True,
        "hygiene": HYGIENE,
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_phase7b(
        ledger_path,
        verdict=verdict,
        extra={
            "base_tail_ic": base.get("tail_ic_top"),
            "base_overlap": base.get("overlap"),
            "best_tail_ic": best_m.get("tail_ic_top"),
            "best_overlap": best_m.get("overlap"),
            "n_undertrained": n_under,
            "gain_originals": gain.get("originals"),
            "gain_products": gain.get("products"),
            "gain_rules": gain.get("rules"),
        },
    )

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase7b(
        rep_dir / "btcb_phase7b_fuzzystack.md",
        grid=grid,
        books={k: _jsonable(v) for k, v in books.items()},
        null_best=null_best,
        stack=stack_rec,
        prune=prune,
        kept_formulas=kept_formulas,
        hygiene=hygiene_blob,
        gain=gain,
        verdict=verdict,
        extra=extra,
    )
    plot_tail_ic_bars(grid, null_best, best_arm, chart_dir / "btcb_phase7b_tail_ic.png")
    plot_gain_share(gain, chart_dir / "btcb_phase7b_gain_share.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    payload = {
        "criterion": PHASE7B_CRITERION,
        "firewall": PHASE7B_FIREWALL,
        "null_registration": PHASE4B_NULL_REGISTRATION,
        "verdict": _jsonable(verdict),
        "grid": {k: _jsonable(v) for k, v in grid.items()},
        "books": {k: _jsonable(v) for k, v in books.items()},
        "null_best": _jsonable(null_best),
        "stack": _jsonable(stack_rec),
        "prune": _jsonable(prune),
        "kept_formulas": kept_formulas[:80],
        "hygiene": _jsonable({k: v for k, v in hygiene_blob.items() if k != "rows"})
        | {"n_rows": len(hygiene_all), "rows": _jsonable(hygiene_all)},
        "gain": _jsonable(gain),
        "extra": extra,
    }
    (rep_dir / "btcb_phase7b_fuzzystack.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "btcb_phase7b_addendum.md").write_text(addendum)
    quant_vol.commit()

    for rec in kept_formulas[:5]:
        print(f"KEPT {rec['formula']}", flush=True)
    print(f"GAIN originals={gain.get('originals')} products={gain.get('products')} rules={gain.get('rules')}", flush=True)
    print(f"UNDERTRAINED count={n_under}", flush=True)
    for line in arm_lines:
        print(line, flush=True)
    if verdict.get("closed"):
        print(verdict.get("closed"), flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {
        "verdicts": {k: (per.get(k) or {}).get("label") for k in ("arm_a", "arm_b", "arm_ab")},
        "stack_skipped": bool(stack_rec.get("skipped")),
        "stack_reasons": stack_rec.get("reasons"),
        "best_arm": best_arm,
        "kept_top5": [r.get("formula") for r in kept_formulas[:5]],
        "gain": {k: gain.get(k) for k in ("originals", "products", "rules")},
        "n_undertrained": n_under,
        "closed": verdict.get("closed"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] Phase 7.b FUZZY-STACK (detach-safe spawn)...", flush=True)
    fc = run_btcb_p7b.spawn()
    print("[local] spawned run_btcb_p7b; waiting (modal run --detach keeps the app if this client drops)", flush=True)
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase7b_fuzzystack.md", "reports"),
        ("reports/btcb_phase7b_fuzzystack.json", "reports"),
        ("reports/btcb_phase7b_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase7b_tail_ic.png", "charts"),
        ("charts/btcb_phase7b_gain_share.png", "charts"),
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
        for src in (art / "charts").glob("btcb_phase7b*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase7b*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] Phase 7.b complete.", flush=True)
