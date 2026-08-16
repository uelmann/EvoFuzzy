"""
RETSTD-LO — binary P(top 10% of ret/std) vs frozen A0, long-only top-decile book.

BACKTEST ONLY. CPU only. COMBO / SPREAD-LS / LONG-TIDE untouched.
Frozen A0 Huber scores and features_labeled.parquet are read-only.
Usage: modal run --detach retstd_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-retstd"
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
    .add_local_python_source("baseline", "retstd")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/retstd_addendum.md", remote_path="/root/retstd_addendum.md")
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

    from baseline.evaluate import daily_rank_ic
    from baseline.features import FEATURE_COLS
    from baseline.model import _fit_predict_fold
    from baseline.seedutil import seed_everything
    from retstd.constants import SEED, YCOL

    t0 = time.time()
    cfg = payload["cfg"]
    seed = int(cfg.get("seed", SEED))
    seed_everything(seed + int(payload["fold_id"]))
    fold = _fold_spec(payload)
    feat = pd.read_parquet(payload["feat_path"])
    feat["date"] = pd.to_datetime(feat["date"], utc=True)

    model_cfg = dict(payload["model_cfg"])
    inner = int(cfg["cv"]["inner_holdout_days"])
    shuffle = bool(payload.get("shuffle_labels", False))
    shuffle_seed = payload.get("shuffle_seed")
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df, meta = _fit_predict_fold(
        feat,
        fold,
        seed=seed,
        model_cfg=model_cfg,
        inner_holdout_days=inner,
        feature_cols=list(FEATURE_COLS),
        model_name="retstd_binary",
        shuffle_labels=shuffle,
        shuffle_seed=None if shuffle_seed is None else int(shuffle_seed),
    )
    ric = float("nan")
    if not pred_df.empty and YCOL in pred_df.columns:
        ic = daily_rank_ic(pred_df, YCOL, "score")
        ric = float(ic.mean()) if len(ic) else float("nan")
    meta["rankic_oos_raw"] = ric
    tag = payload.get("tag") or f"h{fold.horizon}_fold{fold.fold_id}"
    pred_path = out_dir / f"preds_{tag}.parquet"
    if not pred_df.empty and not shuffle:
        pred_df.to_parquet(pred_path, index=False)
        meta["pred_path"] = str(pred_path)
    else:
        meta["pred_path"] = None

    if payload.get("check_seed") and meta.get("pred_path"):
        pred2, meta2 = _fit_predict_fold(
            feat,
            fold,
            seed=seed,
            model_cfg=model_cfg,
            inner_holdout_days=inner,
            feature_cols=list(FEATURE_COLS),
            model_name="retstd_binary",
        )
        a = pd.read_parquet(meta["pred_path"]).sort_values(["date", "symbol"])["score"].to_numpy(dtype=float)
        b = pred2.sort_values(["date", "symbol"])["score"].to_numpy(dtype=float) if not pred2.empty else np.array([])
        n = min(len(a), len(b))
        md = float(np.max(np.abs(a[:n] - b[:n]))) if n else float("nan")
        meta["seed_max_diff"] = md
        meta["seed_determinism"] = bool(np.isfinite(md) and md < 1e-10)
        meta["seed_twin_best_iteration"] = meta2.get("best_iteration")

    print(
        f"[fold] id={fold.fold_id} shuffle={shuffle} status={meta.get('status')} "
        f"ric={ric} iter={meta.get('best_iteration')}",
        flush=True,
    )
    summary = {
        "fold_id": fold.fold_id,
        "meta": meta,
        "wall_elapsed": time.time() - t0,
        "shuffle_labels": shuffle,
        "tag": payload.get("tag"),
    }
    (out_dir / f"meta_{tag}.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    volume.commit()
    return summary


def _fold_payload(fr, *, cfg, model_cfg, feat_path, out_dir, **extra) -> dict:
    p = {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "feat_path": str(feat_path),
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
def run_retstd() -> dict:
    import numpy as np
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything
    from retstd.book import run_long_only_toppct
    from retstd.constants import (
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
        PRED_A0_H10,
        SEED,
        VIABILITY_CRITERION,
        YCOL,
        YCOL_RATIO,
        YCOL_SIMPLE,
    )
    from retstd.eval import (
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
    from retstd.labels import add_retstd_labels
    from retstd.report import plot_equity, write_report

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
    addendum = Path("/root/retstd_addendum.md").read_text()
    for needle in (IMPROVE_CRITERION, VIABILITY_CRITERION, NULL_GATE, DEATH_CONVENTION, FALLBACK_RULE):
        if needle not in addendum:
            raise RuntimeError("Addendum missing a verbatim frozen statement")
    print("[HB] RETSTD-LO BACKTEST ONLY; zero GPU; COMBO untouched; A0 artifacts read-only", flush=True)
    print(f"[HB] {IMPROVE_CRITERION}", flush=True)
    print(f"[HB] {VIABILITY_CRITERION}", flush=True)

    with open("/root/config.yaml") as f:
        cfg = yaml.safe_load(f)
    seed_everything(int(cfg.get("seed", SEED)))

    root = Path(cfg["paths"]["volume_root"])
    out = Path(OUT_ROOT)
    if out.resolve() != Path("/data/quant/retstd").resolve():
        raise RuntimeError(f"refusing to write outside /data/quant/retstd: {out}")
    pred_dir = out / "predictions"
    null_dir = out / "null"
    rep_dir = out / "reports"
    chart_dir = out / "charts"
    for d in (out, pred_dir, null_dir, rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    feat_src = Path(FEAT_PATH)
    if not feat_src.exists():
        raise RuntimeError(f"missing features {feat_src}")
    frozen_feat_sha = _sha256_file(feat_src)
    feat = pd.read_parquet(feat_src)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)

    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    uni_dir = root / "universe"
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)

    feat = add_retstd_labels(feat, panel, horizon=HORIZON)
    if YCOL not in feat.columns or YCOL_RATIO not in feat.columns:
        raise RuntimeError("retstd labels missing on working frame")
    mean_label = float(pd.to_numeric(feat[YCOL], errors="coerce").mean())
    print(
        f"[HB] working labels mean_y={mean_label:.4f} n={len(feat)} "
        f"frozen_feat_sha={frozen_feat_sha[:12]}",
        flush=True,
    )
    if _sha256_file(feat_src) != frozen_feat_sha:
        raise RuntimeError("features_labeled.parquet was mutated; aborting")

    a0_pred_path = Path(PRED_A0_H10)
    if not a0_pred_path.exists():
        a0_pred_path = root / "predictions" / "lgbm_price_only_h10.parquet"
    if not a0_pred_path.exists():
        raise RuntimeError("missing frozen A0 h=10 predictions")
    pred_a0 = pd.read_parquet(a0_pred_path)
    pred_a0["date"] = pd.to_datetime(pred_a0["date"], utc=True)
    drop_y = [c for c in pred_a0.columns if c.startswith("y_") or c in {YCOL_RATIO, YCOL_SIMPLE}]
    if drop_y:
        pred_a0 = pred_a0.drop(columns=drop_y)
    pred_a0 = last_fold_wins(pred_a0)

    pit40_path = Path(PIT_TOP40)
    if not pit40_path.exists():
        pit40_path = uni_dir / "top40_pit.parquet"
    pit40 = pd.read_parquet(pit40_path)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)
    ever = sorted(set(feat["symbol"].unique()) | set(panel["symbol"].unique()))
    funding = load_funding_panel(fund_dir, ever)

    feat_path = out / "features_work.parquet"
    feat.to_parquet(feat_path, index=False)
    volume.commit()
    print(f"[HB] feat rows={len(feat)} panel rows={len(panel)} a0_preds={len(pred_a0)}", flush=True)

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
    model_cfg["objective"] = "binary"

    def _run(mcfg: dict, tag_prefix: str, extra=None):
        extra = extra or {}
        payloads = [
            _fold_payload(
                fr,
                cfg=cfg,
                model_cfg=mcfg,
                feat_path=feat_path,
                out_dir=pred_dir if not extra.get("null") else null_dir,
                tag=f"{tag_prefix}_h{HORIZON}_fold{fr.fold_id}",
                check_seed=bool(extra.get("check_seed") and fr.fold_id == 0),
                **{k: v for k, v in extra.items() if k not in {"check_seed", "null"}},
            )
            for fr in folds
        ]
        return list(train_fold_job.map(payloads))

    main_metas = _run(model_cfg, "main", extra={"check_seed": True})
    volume.reload()

    def _iters(metas: list[dict]) -> list[int]:
        out = []
        for m in metas:
            am = m.get("meta") or {}
            if am.get("status") == "ok" and am.get("best_iteration") is not None:
                out.append(int(am["best_iteration"]))
        return out

    used_fixed = False
    rs_iters = _iters(main_metas)
    if rs_iters and float(np.median(rs_iters)) <= 1.0:
        used_fixed = True
        print("[HB] RETSTD median best_iteration ≤ 1; refitting with fixed 500 trees", flush=True)
        mcfg = dict(model_cfg)
        mcfg["fixed_n_estimators"] = int(FIXED_TREES_FALLBACK)
        refit = _run(mcfg, "fixed500")
        volume.reload()
        by_id = {int(m["fold_id"]): m for m in refit}
        main_metas = [by_id.get(int(m["fold_id"]), m) for m in main_metas]
        rs_iters = _iters(main_metas)

    parts = []
    for m in main_metas:
        p = (m.get("meta") or {}).get("pred_path")
        if p and Path(p).exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError("empty RETSTD preds")
    pred_rs = last_fold_wins(pd.concat(parts, ignore_index=True))
    pred_rs["date"] = pd.to_datetime(pred_rs["date"], utc=True)
    pred_rs_path = pred_dir / "lgbm_retstd_h10.parquet"
    pred_rs.to_parquet(pred_rs_path, index=False)
    volume.commit()

    seed_meta = next(
        (
            (m.get("meta") or {})
            for m in main_metas
            if int(m.get("fold_id", -1)) == 0 and "seed_determinism" in (m.get("meta") or {})
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
            real_ric[int(m["fold_id"])] = float((m.get("meta") or {}).get("rankic_oos_raw", float("nan")))
    use_mcfg = dict(model_cfg)
    if used_fixed:
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
                    out_dir=null_dir,
                    tag=f"null_h{HORIZON}_fold{fr.fold_id}_s{ss}",
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
            ric = (m.get("meta") or {}).get("rankic_oos_raw", float("nan"))
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

    extra_cols = ["date", "symbol", YCOL, YCOL_RATIO, YCOL_SIMPLE]
    extra = feat[extra_cols].drop_duplicates(["date", "symbol"], keep="last")
    extra["date"] = pd.to_datetime(extra["date"], utc=True)

    def _prep_book(pred: pd.DataFrame) -> pd.DataFrame:
        p = pred.copy()
        p["date"] = pd.to_datetime(p["date"], utc=True)
        drop = [c for c in p.columns if c in {YCOL, YCOL_RATIO, YCOL_SIMPLE} or c.endswith("_x") or c.endswith("_y")]
        if drop:
            p = p.drop(columns=[c for c in drop if c in p.columns])
        p = p.merge(extra, on=["date", "symbol"], how="left")
        p = p.merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
        return p

    book_a0 = _prep_book(pred_a0)
    book_rs = _prep_book(pred_rs)
    ric_a0_ratio = pooled_rankic(book_a0, YCOL_RATIO)
    ric_rs_ratio = pooled_rankic(book_rs, YCOL_RATIO)
    ric_a0_simple = pooled_rankic(book_a0, YCOL_SIMPLE)
    ric_rs_simple = pooled_rankic(book_rs, YCOL_SIMPLE)
    gap_a0 = top_minus_universe(book_a0, "score", YCOL_SIMPLE)
    gap_rs = top_minus_universe(book_rs, "score", YCOL_SIMPLE)

    print("[HB] running A0-LO10 and RETSTD-LO books", flush=True)
    raw_a0 = run_long_only_toppct(book_a0, panel, feat, pit40, horizon=HORIZON, funding=funding, variant="a0_lo10")
    raw_rs = run_long_only_toppct(book_rs, panel, feat, pit40, horizon=HORIZON, funding=funding, variant="retstd_lo")
    a0_sum = summarize_book(raw_a0)
    rs_sum = summarize_book(raw_rs)
    verdict = viable(rs_sum, null.get("verdict", "PARKED-NO-SKILL"))
    improve = improves(
        a0_sum,
        rs_sum,
        float(ric_a0_ratio.get("mean_ic", float("nan"))),
        float(ric_rs_ratio.get("mean_ic", float("nan"))),
        float(gap_a0.get("mean_gap", float("nan"))),
        float(gap_rs.get("mean_gap", float("nan"))),
        null.get("verdict", "PARKED-NO-SKILL"),
    )
    print(f"[HB] improve={improve.get('verdict')} viable={verdict.get('verdict')}", flush=True)

    idx = rs_sum.get("daily_ret")
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

    chart_path = chart_dir / "retstd_equity.png"
    if isinstance(idx, pd.Series) and len(idx) and isinstance(a0r, pd.Series) and len(a0r):
        plot_equity(a0r, idx, ew if len(ew) else None, btc if len(btc) else None, chart_path)

    extra_blob = {
        "elapsed_sec": time.time() - t_pipe,
        "used_fixed": used_fixed,
        "n_folds": len(folds),
        "mean_label_rate": mean_label,
        "frozen_feat_sha256": frozen_feat_sha,
        "seed_gate": seed_gate,
        "construction": (
            f"Frozen A0 Huber scores vs new binary LightGBM on P(top-decile R/STD); "
            f"last-fold-wins; n_folds={len(folds)}; used_fixed={used_fixed}; "
            f"seed_determinism={seed_gate.get('passed')} max_diff={seed_gate.get('max_score_diff')}; "
            f"n_a0={len(pred_a0)} n_retstd={len(pred_rs)}; mean_y={mean_label:.4f}; "
            f"write root={OUT_ROOT}; frozen features sha256={frozen_feat_sha}."
        ),
    }
    md_path = rep_dir / "retstd_report.md"
    text = write_report(
        md_path,
        frozen_hash=calc,
        a0_book=a0_sum,
        retstd_book=rs_sum,
        improve=improve,
        verdict=verdict,
        null=null,
        ric_a0_ratio=ric_a0_ratio,
        ric_retstd_ratio=ric_rs_ratio,
        ric_a0_simple=ric_a0_simple,
        ric_retstd_simple=ric_rs_simple,
        gap_a0=gap_a0,
        gap_retstd=gap_rs,
        benches={"ew": ew, "btc": btc},
        extra=extra_blob,
    )
    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "improve": improve,
        "verdict": verdict,
        "null": null,
        "seed_gate": seed_gate,
        "ric_a0_ratio": ric_a0_ratio,
        "ric_retstd_ratio": ric_rs_ratio,
        "ric_a0_simple": ric_a0_simple,
        "ric_retstd_simple": ric_rs_simple,
        "gap_a0": gap_a0,
        "gap_retstd": gap_rs,
        "a0_book": {k: v for k, v in a0_sum.items() if k != "daily_ret"},
        "retstd_book": {k: v for k, v in rs_sum.items() if k != "daily_ret"},
        "used_fixed": used_fixed,
        "n_folds": len(folds),
        "mean_label_rate": mean_label,
        "elapsed_sec": time.time() - t_pipe,
        "retstd_iters": rs_iters,
        "frozen_feat_sha256": frozen_feat_sha,
    }
    (rep_dir / "retstd_report.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "charts").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "retstd_report.md").write_text(text)
    (root / "reports" / "retstd_report.json").write_text((rep_dir / "retstd_report.json").read_text())
    if chart_path.exists():
        (root / "charts" / "retstd_equity.png").write_bytes(chart_path.read_bytes())
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
        "sharpe_retstd": verdict.get("sharpe_full"),
        "sharpe_a0": improve.get("sharpe_a0"),
        "gpu_used": False,
        "elapsed_sec": time.time() - t_pipe,
    }


@app.local_entrypoint()
def main():
    print("[local] starting RETSTD-LO (CPU, backtest-only, COMBO untouched)...", flush=True)
    summary = run_retstd.remote()
    print(f"[local] remote done: {summary}", flush=True)
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/retstd_report.md", "retstd_report.md", "reports"),
        ("reports/retstd_report.json", "retstd_report.json", "reports"),
        ("charts/retstd_equity.png", "retstd_equity.png", "charts"),
        ("retstd/reports/retstd_report.md", "retstd_report.md", "reports"),
        ("retstd/reports/retstd_report.json", "retstd_report.json", "reports"),
        ("retstd/charts/retstd_equity.png", "retstd_equity.png", "charts"),
    ]:
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"], check=False)
        copied = Path(kind) / name
        if dest.exists():
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(dest.read_bytes())
            print(f"[local] copied {dest} -> {copied}", flush=True)
