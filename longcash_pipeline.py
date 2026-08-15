"""
LONG-CASH — cash-financed alt-long parallel product.

BACKTEST ONLY. CPU only. COMBO / SPREAD-LS / LONG-TIDE untouched.
Usage: modal run --detach longcash_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "quant-long-cash"
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
    .add_local_python_source("baseline", "longcash")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_file("config_frozen_a0.yaml", remote_path="/root/config_frozen_a0.yaml")
    .add_local_file("config_frozen_a0.sha256", remote_path="/root/config_frozen_a0.sha256")
    .add_local_file("reports/longcash_addendum.md", remote_path="/root/longcash_addendum.md")
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


@app.function(
    timeout=60 * 90,
    retries=0,
    volumes={"/data/quant": volume},
    cpu=8,
    memory=32768,
    max_containers=20,
)
def train_one_fold_job(payload: dict) -> dict:
    import numpy as np
    import pandas as pd

    from baseline.model import FoldSpec
    from baseline.seedutil import seed_everything
    from longcash.constants import SEED
    from longcash.model import fit_predict_fold

    cfg = payload["cfg"]
    seed_everything(int(cfg.get("seed", SEED)) + int(payload["fold_id"]))
    df = pd.read_parquet(payload["feat_path"])
    fold = FoldSpec(
        fold_id=int(payload["fold_id"]),
        train_start=pd.Timestamp(payload["train_start"]),
        train_end=pd.Timestamp(payload["train_end"]),
        purge_end=pd.Timestamp(payload["purge_end"]),
        embargo_end=pd.Timestamp(payload["embargo_end"]),
        val_start=pd.Timestamp(payload["val_start"]),
        val_end=pd.Timestamp(payload["val_end"]),
        horizon=int(payload["horizon"]),
    )
    t0 = time.time()
    pred_df, meta = fit_predict_fold(
        df,
        fold,
        head=str(payload["head"]),
        seed=int(cfg.get("seed", SEED)),
        model_cfg=payload["model_cfg"],
        inner_holdout_days=int(cfg["cv"]["inner_holdout_days"]),
        shuffle_labels=bool(payload.get("shuffle_labels", False)),
        shuffle_seed=payload.get("shuffle_seed"),
    )
    if payload.get("check_seed") and not pred_df.empty:
        pred2, meta2 = fit_predict_fold(
            df,
            fold,
            head=str(payload["head"]),
            seed=int(cfg.get("seed", SEED)),
            model_cfg=payload["model_cfg"],
            inner_holdout_days=int(cfg["cv"]["inner_holdout_days"]),
        )
        if pred2.empty:
            meta["seed_max_diff"] = float("nan")
            meta["seed_determinism"] = False
        else:
            a = pred_df.sort_values(["date", "symbol"])["p"].to_numpy(dtype=float)
            b = pred2.sort_values(["date", "symbol"])["p"].to_numpy(dtype=float)
            n = min(len(a), len(b))
            md = float(np.max(np.abs(a[:n] - b[:n]))) if n else float("nan")
            meta["seed_max_diff"] = md
            meta["seed_determinism"] = bool(np.isfinite(md) and md < 1e-10)
            meta["seed_twin_best_iteration"] = meta2.get("best_iteration")
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = payload.get("tag") or f"h{fold.horizon}_{payload['head']}_fold{fold.fold_id}"
    pred_path = out_dir / f"preds_{tag}.parquet"
    if not pred_df.empty and not bool(payload.get("shuffle_labels", False)):
        pred_df.to_parquet(pred_path, index=False)
        meta["pred_path"] = str(pred_path)
    else:
        meta["pred_path"] = None
    meta["wall_elapsed"] = time.time() - t0
    meta["tag"] = tag
    (out_dir / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    volume.commit()
    print(
        f"[fold] head={payload['head']} id={fold.fold_id} shuffle={payload.get('shuffle_labels')} "
        f"status={meta.get('status')} ric={meta.get('rankic_oos_raw')} "
        f"iter={meta.get('best_iteration')} elapsed={meta['wall_elapsed']:.1f}s",
        flush=True,
    )
    return meta


def _fold_payload(fr, *, head, cfg, model_cfg, feat_path, out_dir, **extra) -> dict:
    p = {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "feat_path": str(feat_path),
        "out_dir": str(out_dir),
        "head": head,
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
def run_long_cash() -> dict:
    import numpy as np
    import pandas as pd

    from baseline.data import load_funding_panel, load_panel
    from baseline.model import make_folds
    from baseline.seedutil import seed_everything
    from longcash.book import run_long_cash_book
    from longcash.constants import (
        DEATH_CONVENTION,
        FALLBACK_RULE,
        FEAT_PATH,
        FIXED_TREES_FALLBACK,
        FROZEN_A0_SHA256,
        HORIZON,
        NULL_ANCHOR,
        NULL_GATE,
        NULL_REPLICATES,
        NULL_SHUFFLE_SEEDS,
        OUT_ROOT,
        PIT_TOP40,
        PRED_H10,
        SEED,
        VIABILITY_CRITERION,
    )
    from longcash.eval import (
        btc_bh_simple,
        ew_topn_simple,
        null_verdict_from_cells,
        summarize_book,
        top_bucket_usd_stats,
        viable,
    )
    from longcash.labels import add_usd_labels
    from longcash.model import last_fold_wins, pick_null_folds
    from longcash.report import plot_equity, write_report

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
    addendum = Path("/root/longcash_addendum.md").read_text()
    for needle in (VIABILITY_CRITERION, NULL_GATE, DEATH_CONVENTION, FALLBACK_RULE):
        if needle not in addendum:
            raise RuntimeError("Addendum missing a verbatim frozen statement")
    print("[HB] LONG-CASH BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {VIABILITY_CRITERION}", flush=True)
    print(f"[HB] {NULL_GATE}", flush=True)
    print(f"[HB] {DEATH_CONVENTION}", flush=True)
    print(f"[HB] {FALLBACK_RULE}", flush=True)

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

    feat_path = Path(FEAT_PATH)
    if not feat_path.exists():
        raise RuntimeError(f"missing features {feat_path}")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    print(f"[HB] feat rows={len(feat)}", flush=True)

    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    uni_dir = root / "universe"
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    panel = load_panel(raw_dir, kline_syms)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    pit40_path = Path(PIT_TOP40)
    if not pit40_path.exists():
        pit40_path = uni_dir / "top40_pit.parquet"
    pit40 = pd.read_parquet(pit40_path)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)
    pit40 = pit40[pit40["symbol"] != "BTCUSDT"].copy()
    ever = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    funding = load_funding_panel(fund_dir, ever)

    labeled = add_usd_labels(feat, panel, HORIZON)
    labeled_path = out / "features_usd.parquet"
    labeled.to_parquet(labeled_path, index=False)
    volume.commit()
    print(f"[HB] wrote USD labels {labeled_path} rows={len(labeled)}", flush=True)

    folds = make_folds(
        pd.DatetimeIndex(labeled["date"].unique()),
        horizon=HORIZON,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    print(f"[HB] folds={len(folds)} h={HORIZON}", flush=True)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("fixed_n_estimators", None)

    def _run(head: str, mcfg: dict, tag_prefix: str, check_seed: bool = False) -> list[dict]:
        payloads = [
            _fold_payload(
                fr,
                head=head,
                cfg=cfg,
                model_cfg=mcfg,
                feat_path=labeled_path,
                out_dir=pred_dir,
                tag=f"{tag_prefix}_h{HORIZON}_{head}_fold{fr.fold_id}",
                check_seed=bool(check_seed and fr.fold_id == 0 and head == "R"),
            )
            for fr in folds
        ]
        metas = list(train_one_fold_job.map(payloads))
        volume.reload()
        return metas

    metas_r = _run("R", model_cfg, "main", check_seed=True)
    metas_c = _run("C", model_cfg, "main")
    r_iters = [int(m["best_iteration"]) for m in metas_r if m.get("status") == "ok" and m.get("best_iteration") is not None]
    used_fixed = False
    if r_iters and float(np.median(r_iters)) <= 1.0:
        used_fixed = True
        print("[HB] Head-R median best_iteration ≤ 1; refitting with fixed 500 trees", flush=True)
        mcfg = dict(model_cfg)
        mcfg["fixed_n_estimators"] = int(FIXED_TREES_FALLBACK)
        metas_r = _run("R", mcfg, "fixed500", check_seed=True)
        r_iters = [int(m["best_iteration"]) for m in metas_r if m.get("status") == "ok" and m.get("best_iteration") is not None]

    def _load_head(head: str, metas: list[dict]) -> pd.DataFrame:
        parts = []
        for m in metas:
            p = m.get("pred_path")
            if p and Path(p).exists():
                parts.append(pd.read_parquet(p))
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    pred_r = _load_head("R", metas_r)
    pred_c = _load_head("C", metas_c)
    if pred_r.empty or pred_c.empty:
        raise RuntimeError(f"empty preds R={pred_r.empty} C={pred_c.empty}")
    pred_r = pred_r.rename(columns={"p": "er_hat", "p_raw": "er_raw"})
    pred_c = pred_c.rename(columns={"p": "p_up", "p_raw": "p_up_raw"})
    merged = pred_r.merge(
        pred_c[["date", "symbol", "fold_id", "p_up", "p_up_raw"]],
        on=["date", "symbol", "fold_id"],
        how="left",
    )
    merged["p_up"] = pd.to_numeric(merged["p_up"], errors="coerce").fillna(0.0)
    merged["date"] = pd.to_datetime(merged["date"], utc=True)
    merged = last_fold_wins(merged)
    merged_path = pred_dir / "lgbm_longcash_h10.parquet"
    merged.to_parquet(merged_path, index=False)
    volume.commit()
    print(f"[HB] merged preds rows={len(merged)} days={merged['date'].nunique()}", flush=True)

    seed_meta = next((m for m in metas_r if m.get("fold_id") == 0 and "seed_determinism" in m), None)
    seed_gate = {
        "name": "seed_determinism",
        "passed": bool(seed_meta.get("seed_determinism")) if seed_meta else False,
        "max_score_diff": None if seed_meta is None else seed_meta.get("seed_max_diff"),
    }

    null_folds = pick_null_folds(folds, NULL_ANCHOR)
    real_ric = {
        int(m["fold_id"]): float(m["rankic_oos_raw"])
        for m in metas_r
        if m.get("status") == "ok" and int(m["fold_id"]) in {fr.fold_id for fr in null_folds}
    }
    null_payloads = []
    use_mcfg = dict(model_cfg)
    if used_fixed:
        use_mcfg["fixed_n_estimators"] = int(FIXED_TREES_FALLBACK)
    for fr in null_folds:
        for ss in list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]:
            null_payloads.append(
                _fold_payload(
                    fr,
                    head="R",
                    cfg=cfg,
                    model_cfg=use_mcfg,
                    feat_path=labeled_path,
                    out_dir=null_dir,
                    tag=f"null_h{HORIZON}_R_fold{fr.fold_id}_s{ss}",
                    shuffle_labels=True,
                    shuffle_seed=int(ss),
                )
            )
    print(f"[HB] null jobs={len(null_payloads)} folds={[fr.fold_id for fr in null_folds]}", flush=True)
    null_metas = list(train_one_fold_job.map(null_payloads)) if null_payloads else []
    volume.reload()

    cells = []
    for fr in null_folds:
        ics = [
            float(m["rankic_oos_raw"])
            for m in null_metas
            if int(m.get("fold_id", -1)) == fr.fold_id and np.isfinite(m.get("rankic_oos_raw", float("nan")))
        ]
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
    print(f"[HB] null verdict={null.get('verdict')} bias={null.get('bias_pass')} skill={null.get('skill_pass')}", flush=True)

    book_preds = merged.copy()
    ycol_s = f"y_simple_h{HORIZON}"
    if ycol_s in labeled.columns:
        book_preds = book_preds.merge(
            labeled[["date", "symbol", ycol_s, f"y_usd_h{HORIZON}"]],
            on=["date", "symbol"],
            how="left",
        )
    raw_new = top_bucket_usd_stats(book_preds, "er_hat", ycol_s)
    raw_a0 = {"n_days": 0, "pct_top_pos": float("nan"), "mean_top": float("nan"), "nw_t": float("nan")}
    a0_path = Path(PRED_H10)
    if not a0_path.exists():
        a0_path = root / "predictions" / "lgbm_price_only_h10.parquet"
    if a0_path.exists() and ycol_s in labeled.columns:
        a0 = pd.read_parquet(a0_path)
        a0["date"] = pd.to_datetime(a0["date"], utc=True)
        a0 = a0.merge(labeled[["date", "symbol", ycol_s]], on=["date", "symbol"], how="left")
        a0 = a0.merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
        raw_a0 = top_bucket_usd_stats(a0, "score", ycol_s)
        print(f"[HB] A0 raw-material %top>0={raw_a0.get('pct_top_pos')}", flush=True)

    print("[HB] running LONG-CASH book", flush=True)
    raw_book = run_long_cash_book(
        book_preds,
        panel,
        labeled,
        pit40,
        horizon=HORIZON,
        funding=funding,
        apply_funding=True,
    )
    book = summarize_book(raw_book)
    verdict = viable(book, null.get("verdict", "PARKED-NO-SKILL"))
    print(
        f"[HB] {verdict.get('verdict')} sharpe={verdict.get('sharpe_full')} "
        f"trail={verdict.get('sharpe_trail18m')} gross={verdict.get('avg_gross')} "
        f"btc0={verdict.get('pass_btc0')} null={verdict.get('null_verdict')}",
        flush=True,
    )

    idx = book.get("daily_ret")
    ew = ew_topn_simple(panel, pit40)
    btc = btc_bh_simple(panel)
    if isinstance(idx, pd.Series) and len(idx):
        ew = ew.reindex(idx.index).fillna(0.0) if len(ew) else ew
        btc = btc.reindex(idx.index).fillna(0.0) if len(btc) else btc

    chart_path = chart_dir / "longcash_equity.png"
    if isinstance(idx, pd.Series) and len(idx):
        plot_equity(idx, ew if len(ew) else None, btc if len(btc) else None, chart_path)

    extra = {
        "elapsed_sec": time.time() - t_pipe,
        "used_fixed_trees": used_fixed,
        "n_folds": len(folds),
        "seed_gate": seed_gate,
        "construction": (
            f"Heads R/C LightGBM on frozen A0 33 features; last-fold-wins; "
            f"n_folds={len(folds)}; used_fixed_500={used_fixed}; "
            f"seed_determinism={seed_gate.get('passed')} max_diff={seed_gate.get('max_score_diff')}; "
            f"n_merged={len(merged)}; write root={OUT_ROOT}."
        ),
    }
    md_path = rep_dir / "longcash_report.md"
    text = write_report(
        md_path,
        frozen_hash=calc,
        book=book,
        verdict=verdict,
        null=null,
        raw_a0=raw_a0,
        raw_new=raw_new,
        benches={"ew": ew, "btc": btc},
        extra=extra,
    )
    summary = {
        "frozen_sha256": calc,
        "gpu_used": False,
        "scheduled_jobs_created": False,
        "verdict": verdict,
        "null": null,
        "seed_gate": seed_gate,
        "book": {k: v for k, v in book.items() if k != "daily_ret"},
        "raw_a0": raw_a0,
        "raw_new": raw_new,
        "used_fixed_trees": used_fixed,
        "n_folds": len(folds),
        "n_preds": int(len(merged)),
        "elapsed_sec": time.time() - t_pipe,
        "r_iters": r_iters,
        "c_iters": [int(m["best_iteration"]) for m in metas_c if m.get("status") == "ok" and m.get("best_iteration") is not None],
    }
    (rep_dir / "longcash_report.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "charts").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "longcash_report.md").write_text(text)
    (root / "reports" / "longcash_report.json").write_text((rep_dir / "longcash_report.json").read_text())
    if chart_path.exists():
        (root / "charts" / "longcash_equity.png").write_bytes(chart_path.read_bytes())
    volume.commit()
    print(f"[HB] DONE elapsed={time.time()-t_pipe:.1f}s verdict={verdict.get('verdict')}", flush=True)
    return {
        "frozen_sha256": calc,
        "verdict": verdict.get("verdict"),
        "sharpe_full": verdict.get("sharpe_full"),
        "gpu_used": False,
        "elapsed_sec": time.time() - t_pipe,
        "used_fixed_trees": used_fixed,
    }


@app.local_entrypoint()
def main():
    print("[local] starting LONG-CASH (CPU, backtest-only, COMBO untouched)...", flush=True)
    summary = run_long_cash.remote()
    print(f"[local] remote done: {summary}", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    for remote, name, kind in [
        ("reports/longcash_report.md", "longcash_report.md", "reports"),
        ("reports/longcash_report.json", "longcash_report.json", "reports"),
        ("charts/longcash_equity.png", "longcash_equity.png", "charts"),
        ("long_cash/reports/longcash_report.md", "longcash_report.md", "reports"),
        ("long_cash/reports/longcash_report.json", "longcash_report.json", "reports"),
        ("long_cash/charts/longcash_equity.png", "longcash_equity.png", "charts"),
    ]:
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, str(dest), "--force"], check=False)
        if dest.exists() and dest.is_file():
            shutil.copy2(dest, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("longcash*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("longcash*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
    print("[local] artifacts synced", flush=True)
