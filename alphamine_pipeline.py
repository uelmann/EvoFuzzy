"""
ALPHAMINE-LO — formulaic miner + LightGBM long-only A/B vs A0.

BACKTEST ONLY. CPU only. COMBO / SPREAD-LS / LONG-TIDE untouched.
Usage: modal run --detach alphamine_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-alphamine"
VOLUME_NAME = "quant-baseline"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
    .add_local_python_source("baseline", "alphamine")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/alphamine_addendum.md", remote_path="/root/alphamine_addendum.md")
)

app = modal.App(APP_NAME, image=image)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {
        "equity",
        "daily_ret",
        "daily_gross",
        "daily_hedge",
        "daily_cost",
        "daily_funding",
        "daily_n_pos",
        "daily_n_long",
        "daily_n_short",
        "daily_flat",
        "sym_contrib",
        "side_days",
        "daily_gross_deployed",
        "daily_btc_weight",
        "name_alpha_pnl",
        "feature_importance_gain",
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


def _fold_spec(payload: dict):
    import pandas as pd

    from baseline.model import FoldSpec

    return FoldSpec(
        fold_id=int(payload["fold_id"]),
        train_start=pd.Timestamp(payload["train_start"]),
        train_end=pd.Timestamp(payload["train_end"]),
        purge_end=pd.Timestamp(payload["purge_end"]),
        embargo_end=pd.Timestamp(payload["embargo_end"]),
        val_start=pd.Timestamp(payload["val_start"]),
        val_end=pd.Timestamp(payload["val_end"]),
        horizon=int(payload["horizon"]),
    )


@app.function(
    timeout=60 * 90,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=8,
    memory=32768,
    max_containers=20,
)
def train_fold_job(payload: dict) -> dict:
    import numpy as np
    import pandas as pd

    from alphamine.arrays import build_arrays
    from alphamine.constants import BTC_SYMBOL, SEED, YCOL
    from alphamine.gp import apply_formulas, mine_fold
    from baseline.evaluate import daily_rank_ic
    from baseline.features import FEATURE_COLS
    from baseline.model import _fit_predict_fold
    from baseline.seedutil import seed_everything

    t0 = time.time()
    cfg = payload["cfg"]
    seed = int(cfg.get("seed", SEED))
    seed_everything(seed + int(payload["fold_id"]))
    fold = _fold_spec(payload)
    feat = pd.read_parquet(payload["feat_path"])
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    panel = pd.read_parquet(payload["panel_path"])
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    arr = build_arrays(panel, symbols=sorted(feat["symbol"].unique()))

    formulas = list(payload.get("formulas") or [])
    if payload.get("do_mine"):
        formulas = mine_fold(
            arr,
            feat,
            fold,
            seed=seed,
            inner_holdout_days=int(cfg["cv"]["inner_holdout_days"]),
            ycol=YCOL,
        )
    work = feat
    gp_cols: list[str] = []
    if formulas:
        work, gp_cols = apply_formulas(feat, arr, formulas, clip=float(cfg["features"]["zscore_clip"]))
    work = work[work["symbol"] != BTC_SYMBOL].copy()

    model_cfg = dict(payload["model_cfg"])
    inner = int(cfg["cv"]["inner_holdout_days"])
    shuffle = bool(payload.get("shuffle_labels", False))
    shuffle_seed = payload.get("shuffle_seed")
    arms = list(payload.get("arms") or ["a0", "mine"])
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for arm in arms:
        feats = list(FEATURE_COLS) if arm == "a0" else list(FEATURE_COLS) + list(gp_cols)
        pred_df, meta = _fit_predict_fold(
            work,
            fold,
            seed=seed,
            model_cfg=model_cfg,
            inner_holdout_days=inner,
            feature_cols=feats,
            model_name=f"alphamine_{arm}",
            shuffle_labels=shuffle,
            shuffle_seed=None if shuffle_seed is None else int(shuffle_seed),
        )
        ric = float("nan")
        if not pred_df.empty and YCOL in pred_df.columns:
            ic = daily_rank_ic(pred_df, YCOL, "score")
            ric = float(ic.mean()) if len(ic) else float("nan")
        meta["rankic_oos_raw"] = ric
        meta["arm"] = arm
        meta["n_formulas"] = int(len(formulas)) if arm == "mine" else 0
        meta["gp_cols"] = gp_cols if arm == "mine" else []
        tag = payload.get("tag") or f"h{fold.horizon}_{arm}_fold{fold.fold_id}"
        pred_path = out_dir / f"preds_{tag}_{arm}.parquet"
        if not pred_df.empty and not shuffle:
            pred_df.to_parquet(pred_path, index=False)
            meta["pred_path"] = str(pred_path)
        else:
            meta["pred_path"] = None
        metas[arm] = meta
        print(
            f"[fold] arm={arm} id={fold.fold_id} shuffle={shuffle} "
            f"status={meta.get('status')} ric={ric} iter={meta.get('best_iteration')} "
            f"n_gp={meta.get('n_formulas')}",
            flush=True,
        )

    if payload.get("check_seed") and "a0" in metas and metas["a0"].get("pred_path"):
        pred2, meta2 = _fit_predict_fold(
            work,
            fold,
            seed=seed,
            model_cfg=model_cfg,
            inner_holdout_days=inner,
            feature_cols=list(FEATURE_COLS),
            model_name="alphamine_a0",
        )
        a = pd.read_parquet(metas["a0"]["pred_path"]).sort_values(["date", "symbol"])["score"].to_numpy(dtype=float)
        b = pred2.sort_values(["date", "symbol"])["score"].to_numpy(dtype=float) if not pred2.empty else np.array([])
        n = min(len(a), len(b))
        md = float(np.max(np.abs(a[:n] - b[:n]))) if n else float("nan")
        metas["a0"]["seed_max_diff"] = md
        metas["a0"]["seed_determinism"] = bool(np.isfinite(md) and md < 1e-10)
        metas["a0"]["seed_twin_best_iteration"] = meta2.get("best_iteration")

    form_path = out_dir / f"formulas_fold{fold.fold_id}.json"
    if payload.get("do_mine") and not shuffle:
        form_path.write_text(json.dumps(formulas, indent=2, default=str))
    summary = {
        "fold_id": fold.fold_id,
        "formulas": formulas,
        "n_formulas": int(len(formulas)),
        "formula_path": str(form_path) if form_path.exists() else None,
        "arms": metas,
        "wall_elapsed": time.time() - t0,
        "shuffle_labels": shuffle,
        "tag": payload.get("tag"),
    }
    (out_dir / f"meta_{payload.get('tag') or f'fold{fold.fold_id}'}.json").write_text(
        json.dumps(_jsonable(summary), indent=2, default=str)
    )
    volume.commit()
    return summary


def _fold_payload(fr, *, cfg, model_cfg, feat_path, panel_path, out_dir, **extra) -> dict:
    p = {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "feat_path": str(feat_path),
        "panel_path": str(panel_path),
        "out_dir": str(out_dir),
        "fold_id": fr.fold_id,
        "train_start": str(fr.train_start),
        "train_end": str(fr.train_end),
        "purge_end": str(fr.purge_end),
        "embargo_end": str(fr.embargo_end),
        "val_start": str(fr.val_start),
        "val_end": str(fr.val_end),
        "horizon": int(fr.horizon),
    }
    p.update(extra)
    return p


@app.function(
    timeout=60 * 60 * 8,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=16,
    memory=65536,
)
def run_alphamine() -> dict:
    import numpy as np
    import pandas as pd

    from alphamine.book import run_long_only_topk
    from alphamine.constants import (
        BTC_SYMBOL,
        DEATH_CONVENTION,
        FALLBACK_RULE,
        FEAT_PATH,
        FIXED_TREES_FALLBACK,
        FROZEN_A0_SHA256,
        HORIZON,
        IMPROVE_CRITERION,
        NULL_ANCHOR,
        NULL_GATE,
        NULL_REPLICATES,
        NULL_SHUFFLE_SEEDS,
        OUT_ROOT,
        PIT_TOP40,
        SEED,
        VIABILITY_CRITERION,
        YCOL,
        YCOL_SIMPLE,
    )
    from alphamine.eval import (
        btc_bh_simple,
        ew_topn_simple,
        improves,
        last_fold_wins,
        null_verdict_from_cells,
        pick_null_folds,
        pooled_rankic,
        summarize_book,
        top_minus_universe,
        viable,
    )
    from alphamine.report import plot_equity, write_report
    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything

    t_pipe = time.time()
    frozen_text = Path("/root/config_frozen_a0.yaml").read_text()
    frozen_hash_file = Path("/root/config_frozen_a0.sha256").read_text().strip()
    calc = hashlib.sha256(frozen_text.encode()).hexdigest()
    if calc != frozen_hash_file:
        raise RuntimeError(f"Frozen hash mismatch file={frozen_hash_file} calc={calc}")
    live_h = hashlib.sha256(Path("/root/config.yaml").read_text().encode()).hexdigest()
    if live_h != calc:
        raise RuntimeError("config.yaml drifted from frozen A0")
    if calc != FROZEN_A0_SHA256:
        raise RuntimeError(f"Frozen A0 SHA256 constant drift: {calc}")
    addendum = Path("/root/alphamine_addendum.md").read_text()
    for needle in (IMPROVE_CRITERION, VIABILITY_CRITERION, NULL_GATE, DEATH_CONVENTION, FALLBACK_RULE):
        if needle not in addendum:
            raise RuntimeError("Addendum missing a verbatim frozen statement")
    print("[HB] ALPHAMINE-LO BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {IMPROVE_CRITERION}", flush=True)
    print(f"[HB] {VIABILITY_CRITERION}", flush=True)

    with open("/root/config.yaml") as f:
        cfg = yaml.safe_load(f)
    seed_everything(int(cfg.get("seed", SEED)))

    root = Path(cfg["paths"]["volume_root"])
    out = Path(OUT_ROOT)
    pred_dir = out / "predictions"
    null_dir = out / "null"
    rep_dir = out / "reports"
    chart_dir = out / "charts"
    for d in (out, pred_dir, null_dir, rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    feat_src = Path(FEAT_PATH)
    if not feat_src.exists():
        raise RuntimeError(f"missing features {feat_src}")
    feat = pd.read_parquet(feat_src)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    if YCOL not in feat.columns:
        raise RuntimeError(f"missing {YCOL} on features_labeled")

    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    uni_dir = root / "universe"
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    fwd = close.shift(-HORIZON) / close - 1.0
    ymap = fwd.stack(future_stack=True).rename(YCOL_SIMPLE).reset_index()
    ymap.columns = ["date", "symbol", YCOL_SIMPLE]
    ymap["date"] = pd.to_datetime(ymap["date"], utc=True)
    feat = feat.merge(ymap, on=["date", "symbol"], how="left")

    pit40_path = Path(PIT_TOP40)
    if not pit40_path.exists():
        pit40_path = uni_dir / "top40_pit.parquet"
    pit40 = pd.read_parquet(pit40_path)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)
    pit40 = pit40[pit40["symbol"] != BTC_SYMBOL].copy()
    ever = sorted(set(feat["symbol"].unique()) | {BTC_SYMBOL})
    funding = load_funding_panel(fund_dir, ever)

    feat_path = out / "features_work.parquet"
    panel_path = out / "panel_cache.parquet"
    feat.to_parquet(feat_path, index=False)
    panel.to_parquet(panel_path, index=False)
    volume.commit()
    print(f"[HB] feat rows={len(feat)} panel rows={len(panel)}", flush=True)

    folds = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=HORIZON,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    print(f"[HB] folds={len(folds)} h={HORIZON}", flush=True)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("fixed_n_estimators", None)

    def _run(mcfg: dict, tag_prefix: str, *, do_mine: bool, formulas_by_fold=None, arms=None, extra=None):
        extra = extra or {}
        payloads = []
        for fr in folds:
            forms = None if formulas_by_fold is None else formulas_by_fold.get(int(fr.fold_id), [])
            payloads.append(
                _fold_payload(
                    fr,
                    cfg=cfg,
                    model_cfg=mcfg,
                    feat_path=feat_path,
                    panel_path=panel_path,
                    out_dir=pred_dir if not extra.get("null") else null_dir,
                    tag=f"{tag_prefix}_h{HORIZON}_fold{fr.fold_id}",
                    do_mine=bool(do_mine),
                    formulas=forms or [],
                    arms=arms or ["a0", "mine"],
                    check_seed=bool(extra.get("check_seed") and fr.fold_id == 0),
                    **{k: v for k, v in extra.items() if k not in {"check_seed", "null"}},
                )
            )
        return list(train_fold_job.map(payloads))

    main_metas = _run(model_cfg, "main", do_mine=True, extra={"check_seed": True})
    volume.reload()

    def _iters(arm: str, metas: list[dict]) -> list[int]:
        out = []
        for m in metas:
            am = (m.get("arms") or {}).get(arm) or {}
            if am.get("status") == "ok" and am.get("best_iteration") is not None:
                out.append(int(am["best_iteration"]))
        return out

    formulas_by_fold = {int(m["fold_id"]): list(m.get("formulas") or []) for m in main_metas}
    used_fixed_a0 = False
    used_fixed_mine = False
    a0_iters = _iters("a0", main_metas)
    mine_iters = _iters("mine", main_metas)
    if a0_iters and float(np.median(a0_iters)) <= 1.0:
        used_fixed_a0 = True
        print("[HB] A0 median best_iteration ≤ 1; refitting A0 with fixed 500 trees", flush=True)
        mcfg = dict(model_cfg)
        mcfg["fixed_n_estimators"] = int(FIXED_TREES_FALLBACK)
        a0_refit = _run(mcfg, "fixed500_a0", do_mine=False, formulas_by_fold=formulas_by_fold, arms=["a0"])
        volume.reload()
        for m in a0_refit:
            fid = int(m["fold_id"])
            for src in main_metas:
                if int(src["fold_id"]) == fid:
                    src["arms"]["a0"] = (m.get("arms") or {}).get("a0") or src["arms"].get("a0")
        a0_iters = _iters("a0", main_metas)
    if mine_iters and float(np.median(mine_iters)) <= 1.0:
        used_fixed_mine = True
        print("[HB] MINE median best_iteration ≤ 1; refitting MINE with fixed 500 trees", flush=True)
        mcfg = dict(model_cfg)
        mcfg["fixed_n_estimators"] = int(FIXED_TREES_FALLBACK)
        mine_refit = _run(
            mcfg, "fixed500_mine", do_mine=False, formulas_by_fold=formulas_by_fold, arms=["mine"]
        )
        volume.reload()
        for m in mine_refit:
            fid = int(m["fold_id"])
            for src in main_metas:
                if int(src["fold_id"]) == fid:
                    src["arms"]["mine"] = (m.get("arms") or {}).get("mine") or src["arms"].get("mine")
        mine_iters = _iters("mine", main_metas)

    def _load_arm(arm: str) -> pd.DataFrame:
        parts = []
        for m in main_metas:
            p = ((m.get("arms") or {}).get(arm) or {}).get("pred_path")
            if p and Path(p).exists():
                parts.append(pd.read_parquet(p))
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], utc=True)
        return last_fold_wins(out)

    pred_a0 = _load_arm("a0")
    pred_mine = _load_arm("mine")
    if pred_a0.empty or pred_mine.empty:
        raise RuntimeError(f"empty preds a0={pred_a0.empty} mine={pred_mine.empty}")
    pred_a0_path = pred_dir / "lgbm_a0_h10.parquet"
    pred_mine_path = pred_dir / "lgbm_mine_h10.parquet"
    pred_a0.to_parquet(pred_a0_path, index=False)
    pred_mine.to_parquet(pred_mine_path, index=False)
    volume.commit()

    seed_meta = next(
        (
            (m.get("arms") or {}).get("a0")
            for m in main_metas
            if int(m.get("fold_id", -1)) == 0 and "seed_determinism" in ((m.get("arms") or {}).get("a0") or {})
        ),
        None,
    )
    seed_gate = {
        "name": "seed_determinism",
        "passed": bool(seed_meta.get("seed_determinism")) if seed_meta else False,
        "max_score_diff": None if seed_meta is None else seed_meta.get("seed_max_diff"),
    }

    null_folds = pick_null_folds(folds, NULL_ANCHOR)
    real_ric = {}
    for m in main_metas:
        if int(m["fold_id"]) in {fr.fold_id for fr in null_folds}:
            real_ric[int(m["fold_id"])] = float(((m.get("arms") or {}).get("mine") or {}).get("rankic_oos_raw", float("nan")))
    use_mcfg = dict(model_cfg)
    if used_fixed_mine:
        use_mcfg["fixed_n_estimators"] = int(FIXED_TREES_FALLBACK)
    null_payloads = []
    for fr in null_folds:
        for ss in list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]:
            null_payloads.append(
                _fold_payload(
                    fr,
                    cfg=cfg,
                    model_cfg=use_mcfg,
                    feat_path=feat_path,
                    panel_path=panel_path,
                    out_dir=null_dir,
                    tag=f"null_h{HORIZON}_fold{fr.fold_id}_s{ss}",
                    do_mine=False,
                    formulas=formulas_by_fold.get(int(fr.fold_id), []),
                    arms=["mine"],
                    shuffle_labels=True,
                    shuffle_seed=int(ss),
                )
            )
    print(f"[HB] null jobs={len(null_payloads)} folds={[fr.fold_id for fr in null_folds]}", flush=True)
    null_metas = list(train_fold_job.map(null_payloads)) if null_payloads else []
    volume.reload()

    cells = []
    for fr in null_folds:
        ics = []
        for m in null_metas:
            if int(m.get("fold_id", -1)) != fr.fold_id:
                continue
            ric = ((m.get("arms") or {}).get("mine") or {}).get("rankic_oos_raw", float("nan"))
            if np.isfinite(ric):
                ics.append(float(ric))
        arr = np.asarray(ics, dtype=float)
        n = int(len(arr))
        mean = float(arr.mean()) if n else float("nan")
        sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
        p95 = float(np.percentile(arr, 95)) if n else float("nan")
        se = (sd / np.sqrt(n)) if n and np.isfinite(sd) else float("nan")
        bias_lim = 2.0 * se if np.isfinite(se) else float("nan")
        bias_ok = bool(np.isfinite(mean) and np.isfinite(bias_lim) and abs(mean) <= bias_lim)
        real = float(real_ric.get(fr.fold_id, float("nan")))
        cells.append(
            {
                "fold_id": fr.fold_id,
                "n": n,
                "mean": mean,
                "sd": sd,
                "p95": p95,
                "se": float(se) if np.isfinite(se) else float("nan"),
                "bias_lim": float(bias_lim) if np.isfinite(bias_lim) else float("nan"),
                "bias_ok": bias_ok,
                "real": real,
                "exceeds_p95": bool(np.isfinite(real) and np.isfinite(p95) and real > p95),
            }
        )
    null = null_verdict_from_cells(cells)
    print(f"[HB] null verdict={null.get('verdict')}", flush=True)

    def _prep_book(pred: pd.DataFrame) -> pd.DataFrame:
        p = pred.copy()
        p["date"] = pd.to_datetime(p["date"], utc=True)
        extra_cols = ["date", "symbol"]
        if YCOL not in p.columns:
            extra_cols.append(YCOL)
        if YCOL_SIMPLE not in p.columns:
            extra_cols.append(YCOL_SIMPLE)
        extra = feat[extra_cols].drop_duplicates(["date", "symbol"], keep="last")
        extra["date"] = pd.to_datetime(extra["date"], utc=True)
        if len(extra_cols) > 2:
            p = p.merge(extra, on=["date", "symbol"], how="left")
        for src, dst in ((f"{YCOL}_x", YCOL), (f"{YCOL}_y", YCOL), (f"{YCOL_SIMPLE}_x", YCOL_SIMPLE), (f"{YCOL_SIMPLE}_y", YCOL_SIMPLE)):
            if dst not in p.columns and src in p.columns:
                p[dst] = p[src]
        p = p.merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
        p = p[p["symbol"] != BTC_SYMBOL]
        return p

    book_a0 = _prep_book(pred_a0)
    book_mine = _prep_book(pred_mine)
    ric_a0 = pooled_rankic(book_a0, YCOL)
    ric_mine = pooled_rankic(book_mine, YCOL)
    gap_a0 = top_minus_universe(book_a0, "score", YCOL_SIMPLE)
    gap_mine = top_minus_universe(book_mine, "score", YCOL_SIMPLE)

    print("[HB] running A0-LO and MINE-LO books", flush=True)
    raw_a0 = run_long_only_topk(book_a0, panel, feat, pit40, horizon=HORIZON, funding=funding, variant="a0_lo")
    raw_mine = run_long_only_topk(book_mine, panel, feat, pit40, horizon=HORIZON, funding=funding, variant="mine_lo")
    a0_sum = summarize_book(raw_a0)
    mine_sum = summarize_book(raw_mine)
    verdict = viable(mine_sum, null.get("verdict", "PARKED-NO-SKILL"))
    improve = improves(
        a0_sum,
        mine_sum,
        float(ric_a0.get("mean_ic", float("nan"))),
        float(ric_mine.get("mean_ic", float("nan"))),
        float(gap_a0.get("mean_gap", float("nan"))),
        float(gap_mine.get("mean_gap", float("nan"))),
        null.get("verdict", "PARKED-NO-SKILL"),
    )
    print(f"[HB] improve={improve.get('verdict')} viable={verdict.get('verdict')}", flush=True)

    idx = mine_sum.get("daily_ret")
    ew = ew_topn_simple(panel, pit40)
    btc = btc_bh_simple(panel)
    if isinstance(idx, pd.Series) and len(idx):
        ew = ew.reindex(idx.index).fillna(0.0) if len(ew) else ew
        btc = btc.reindex(idx.index).fillna(0.0) if len(btc) else btc
        a0r = a0_sum.get("daily_ret")
        if isinstance(a0r, pd.Series) and len(a0r):
            a0r = a0r.reindex(idx.index).fillna(0.0)
        else:
            a0r = idx * 0.0
    else:
        a0r = a0_sum.get("daily_ret")

    chart_path = chart_dir / "alphamine_equity.png"
    if isinstance(idx, pd.Series) and len(idx) and isinstance(a0r, pd.Series) and len(a0r):
        plot_equity(a0r, idx, ew if len(ew) else None, btc if len(btc) else None, chart_path)

    n_form = [int(m.get("n_formulas") or 0) for m in main_metas]
    examples = []
    for m in main_metas:
        for f in (m.get("formulas") or [])[:2]:
            examples.append(f"fold{m.get('fold_id')}: {f.get('expr')} (hoIC={f.get('holdout_ic')})")
    extra = {
        "elapsed_sec": time.time() - t_pipe,
        "used_fixed_a0": used_fixed_a0,
        "used_fixed_mine": used_fixed_mine,
        "n_folds": len(folds),
        "mean_n_formulas": float(np.mean(n_form)) if n_form else 0.0,
        "n_empty_formula_folds": int(sum(1 for x in n_form if x <= 0)),
        "formula_examples": examples,
        "seed_gate": seed_gate,
        "construction": (
            f"LightGBM A0 vs A0+GP formulas; last-fold-wins; n_folds={len(folds)}; "
            f"used_fixed_a0={used_fixed_a0} used_fixed_mine={used_fixed_mine}; "
            f"seed_determinism={seed_gate.get('passed')} max_diff={seed_gate.get('max_score_diff')}; "
            f"n_a0={len(pred_a0)} n_mine={len(pred_mine)}; write root={OUT_ROOT}."
        ),
    }
    md_path = rep_dir / "alphamine_report.md"
    text = write_report(
        md_path,
        frozen_hash=calc,
        a0_book=a0_sum,
        mine_book=mine_sum,
        improve=improve,
        verdict=verdict,
        null=null,
        ric_a0=ric_a0,
        ric_mine=ric_mine,
        gap_a0=gap_a0,
        gap_mine=gap_mine,
        benches={"ew": ew, "btc": btc},
        extra=extra,
    )
    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "improve": improve,
        "verdict": verdict,
        "null": null,
        "seed_gate": seed_gate,
        "ric_a0": ric_a0,
        "ric_mine": ric_mine,
        "gap_a0": gap_a0,
        "gap_mine": gap_mine,
        "a0_book": {k: v for k, v in a0_sum.items() if k != "daily_ret"},
        "mine_book": {k: v for k, v in mine_sum.items() if k != "daily_ret"},
        "used_fixed_a0": used_fixed_a0,
        "used_fixed_mine": used_fixed_mine,
        "n_folds": len(folds),
        "mean_n_formulas": extra["mean_n_formulas"],
        "elapsed_sec": time.time() - t_pipe,
        "a0_iters": a0_iters,
        "mine_iters": mine_iters,
    }
    (rep_dir / "alphamine_report.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "charts").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "alphamine_report.md").write_text(text)
    (root / "reports" / "alphamine_report.json").write_text((rep_dir / "alphamine_report.json").read_text())
    if chart_path.exists():
        (root / "charts" / "alphamine_equity.png").write_bytes(chart_path.read_bytes())
    volume.commit()
    print(
        f"[HB] DONE elapsed={time.time() - t_pipe:.1f}s "
        f"improve={improve.get('verdict')} viable={verdict.get('verdict')}",
        flush=True,
    )
    return {
        "frozen_sha256": calc,
        "improve": improve.get("verdict"),
        "verdict": verdict.get("verdict"),
        "sharpe_mine": verdict.get("sharpe_full"),
        "gpu_used": False,
        "elapsed_sec": time.time() - t_pipe,
    }


@app.local_entrypoint()
def main():
    print("[local] starting ALPHAMINE-LO (CPU, backtest-only, COMBO untouched)...", flush=True)
    summary = run_alphamine.remote()
    print(f"[local] remote done: {summary}", flush=True)
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/alphamine_report.md", "alphamine_report.md", "reports"),
        ("reports/alphamine_report.json", "alphamine_report.json", "reports"),
        ("charts/alphamine_equity.png", "alphamine_equity.png", "charts"),
        ("alphamine/reports/alphamine_report.md", "alphamine_report.md", "reports"),
        ("alphamine/reports/alphamine_report.json", "alphamine_report.json", "reports"),
        ("alphamine/charts/alphamine_equity.png", "alphamine_equity.png", "charts"),
    ]:
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"], check=False)
        copied = Path(kind) / name
        if dest.exists():
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(dest.read_bytes())
            print(f"[local] copied {dest} -> {copied}", flush=True)
