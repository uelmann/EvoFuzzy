"""
NASDAQ-ADAPT-1 scout — simple h=126, equity clock, long-only, 12-1 control.

BACKTEST ONLY. CPU only. COMBO / NASDAQ-LS / NASDAQ-LS21 untouched.
Usage: python nasdaq_adapt1_pipeline.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baseline.data import build_pit_topn
from baseline.model import _fit_predict_fold
from baseline.seedutil import seed_everything
from nasdaq_ls.adapt1_constants import (
    ADDENDUM_PATH,
    BOOK_START,
    CHART_PATH,
    DV_WINDOW,
    EXEC_TOP_N,
    FACTOR_CRITERION,
    FEAT_PATH,
    FIXED_TREES,
    HEADLINE_START,
    HORIZON,
    LONG_ONLY,
    MIN_TRAIN_SESSIONS,
    PRED_DIR,
    PRED_PATH,
    REPORT_JSON,
    REPORT_MD,
    SEED,
    TOP_PCT,
    TRAIN_MAX_SESSIONS,
    TRAIN_RULE,
)
from nasdaq_ls.adapt1_report import write_adapt1_report
from nasdaq_ls.adapt_features import ADAPT_FEATURE_COLS, build_adapt_feature_panel
from nasdaq_ls.book import run_ls_topn
from nasdaq_ls.constants import DEATH_CONVENTION, MARKET_PATH, PANEL_PATH, PRICE_RULE
from nasdaq_ls.download import download_all
from nasdaq_ls.eval import (
    ew_universe,
    factor_verdict,
    last_fold_wins,
    ml_claim_verdict,
    pooled_rankic,
    qqq_bh,
    sharpe,
    summarize_book,
    window_from,
)
from nasdaq_ls.folds import make_rolling_folds
from nasdaq_ls.labels import add_simple_labels
from nasdaq_ls.report import plot_equity


def _jsonable(x, drop=None):
    drop = drop or {"equity", "daily_ret", "daily_gross_pnl", "daily_cost"}
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


def _mean_gain(metas: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for m in metas:
        g = m.get("feature_importance_gain") or {}
        for k, v in g.items():
            if v is not None and np.isfinite(float(v)):
                acc[k].append(float(v))
    mean = {k: float(np.mean(vs)) for k, vs in acc.items() if vs}
    return dict(sorted(mean.items(), key=lambda kv: -kv[1]))


def main() -> dict:
    t0 = time.time()
    addendum = Path(ADDENDUM_PATH).read_text()
    for needle in (FACTOR_CRITERION, DEATH_CONVENTION, TRAIN_RULE, PRICE_RULE):
        if needle not in addendum:
            raise RuntimeError("Addendum missing a verbatim frozen statement")
    print("[HB] NASDAQ-ADAPT-1; simple h=126; long-only 10%; 12-1 control; COMBO untouched", flush=True)
    print(f"[HB] {TRAIN_RULE}", flush=True)
    print(f"[HB] {FACTOR_CRITERION}", flush=True)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    seed_everything(int(cfg.get("seed", SEED)))

    meta = download_all(force=False)
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.normalize()
    mkt_df = pd.read_parquet(MARKET_PATH)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"], utc=True).dt.normalize()
    market = mkt_df.set_index("date")["close"].sort_index()
    market = market[~market.index.duplicated(keep="last")]
    print(
        f"[HB] panel symbols={panel['symbol'].nunique()} rows={len(panel)} "
        f"{panel['date'].min().date()}→{panel['date'].max().date()} source={meta.get('source')}",
        flush=True,
    )

    feat_path = Path(FEAT_PATH)
    ycol = f"y_h{HORIZON}"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
        print(f"[HB] loaded cached ADAPT-1 features rows={len(feat)}", flush=True)
    else:
        feat = build_adapt_feature_panel(panel, market, clip=float(cfg["features"]["zscore_clip"]))
        feat = add_simple_labels(feat, panel, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(feat_path, index=False)
        print(f"[HB] features rows={len(feat)} ymean={feat[ycol].mean():.4f}", flush=True)

    if ycol not in feat.columns:
        feat = add_simple_labels(feat, panel, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))
        feat.to_parquet(feat_path, index=False)

    missing = [c for c in ADAPT_FEATURE_COLS if c not in feat.columns]
    if missing:
        raise RuntimeError(f"ADAPT features missing {missing}")

    pit = build_pit_topn(panel, n=EXEC_TOP_N, window=DV_WINDOW)
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.normalize()
    print(f"[HB] PIT top-{EXEC_TOP_N} rows={len(pit)} dates={pit['date'].nunique()}", flush=True)

    dates = pd.DatetimeIndex(sorted(feat["date"].dropna().unique()))
    folds = make_rolling_folds(
        dates,
        horizon=HORIZON,
        min_train_days=int(MIN_TRAIN_SESSIONS),
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
        train_max_sessions=int(TRAIN_MAX_SESSIONS),
        session_purge=True,
    )
    if folds:
        d0 = dates[(dates >= folds[0].train_start) & (dates <= folds[0].train_end)]
        dL = dates[(dates >= folds[-1].train_start) & (dates <= folds[-1].train_end)]
        print(
            f"[HB] folds={len(folds)} h={HORIZON} trees={FIXED_TREES} session_purge "
            f"train_sessions fold0={len(d0)} foldL={len(dL)} cap={TRAIN_MAX_SESSIONS}",
            flush=True,
        )
    else:
        raise RuntimeError("no folds")

    model_cfg = dict(cfg["model"])
    model_cfg["objective"] = "huber"
    model_cfg["fixed_n_estimators"] = int(FIXED_TREES)
    model_cfg["early_stop_metric"] = "none"
    inner = int(cfg["cv"]["inner_holdout_days"])
    seed = int(cfg.get("seed", SEED))

    pred_dir = Path(PRED_DIR)
    pred_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    metas = []
    for fr in folds:
        tag = f"h{HORIZON}_fold{fr.fold_id}"
        cache = pred_dir / f"preds_{tag}.parquet"
        meta_p = pred_dir / f"meta_{tag}.json"
        if cache.exists() and meta_p.exists():
            pred_df = pd.read_parquet(cache)
            meta_f = json.loads(meta_p.read_text())
            print(f"[fold] id={fr.fold_id} cached iter={meta_f.get('best_iteration')}", flush=True)
        else:
            pred_df, meta_f = _fit_predict_fold(
                feat,
                fr,
                seed=seed,
                model_cfg=model_cfg,
                inner_holdout_days=inner,
                feature_cols=list(ADAPT_FEATURE_COLS),
                model_name="nasdaq_adapt1_huber",
            )
            if not pred_df.empty:
                pred_df.to_parquet(cache, index=False)
            meta_p.write_text(json.dumps(_jsonable(meta_f), indent=2, default=str))
            print(
                f"[fold] id={fr.fold_id} status={meta_f.get('status')} "
                f"iter={meta_f.get('best_iteration')} train={fr.train_start.date()}→{fr.train_end.date()} "
                f"ric_rows={len(pred_df)}",
                flush=True,
            )
        metas.append(meta_f)
        if pred_df is not None and not pred_df.empty:
            parts.append(pred_df)

    iters = [int(m["best_iteration"]) for m in metas if m.get("status") == "ok" and m.get("best_iteration") is not None]
    if not iters or any(i != int(FIXED_TREES) for i in iters):
        raise RuntimeError(f"expected every ok fold to train {FIXED_TREES} trees; got {iters}")
    print(f"[HB] all folds trained {FIXED_TREES} trees n_ok={len(iters)}", flush=True)

    if not parts:
        raise RuntimeError("empty NASDAQ-ADAPT-1 preds")
    preds = last_fold_wins(pd.concat(parts, ignore_index=True))
    preds["date"] = pd.to_datetime(preds["date"], utc=True).dt.normalize()
    Path(PRED_PATH).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PRED_PATH, index=False)

    extra_cols = ["date", "symbol", ycol, f"y_simple_h{HORIZON}", "mom_252_skip21_raw"]
    extra = feat[extra_cols].drop_duplicates(["date", "symbol"], keep="last")
    extra["date"] = pd.to_datetime(extra["date"], utc=True).dt.normalize()
    book_pred = preds.drop(columns=[c for c in preds.columns if c in {ycol, f"y_simple_h{HORIZON}"}], errors="ignore")
    book_pred = book_pred.merge(extra, on=["date", "symbol"], how="left")
    book_pred = book_pred.merge(pit[["date", "symbol"]], on=["date", "symbol"], how="inner")

    ric_all = pooled_rankic(book_pred, ycol, horizon=HORIZON)
    ric_2007 = pooled_rankic(
        book_pred[book_pred["date"] >= pd.Timestamp(HEADLINE_START, tz="UTC").normalize()],
        ycol,
        horizon=HORIZON,
    )
    print(f"[HB] RankIC all={ric_all.get('mean_ic')} from2007={ric_2007.get('mean_ic')}", flush=True)

    print("[HB] running overlapping long-only GBM book h=126 top 10% from 2005", flush=True)
    raw = run_ls_topn(
        preds,
        panel,
        feat,
        pit,
        horizon=HORIZON,
        top_pct=TOP_PCT,
        book_start=BOOK_START,
        variant="nasdaq_adapt1",
        long_only=LONG_ONLY,
    )
    if raw.get("error"):
        raise RuntimeError(raw["error"])
    sum_2005 = summarize_book(raw, start=BOOK_START)
    sum_2007 = summarize_book(raw, start=HEADLINE_START)

    ctrl = extra.dropna(subset=["mom_252_skip21_raw"]).copy()
    ctrl = ctrl.merge(pit[["date", "symbol"]], on=["date", "symbol"], how="inner")
    ctrl["score"] = ctrl["mom_252_skip21_raw"]
    ric_ctrl_2007 = pooled_rankic(
        ctrl[ctrl["date"] >= pd.Timestamp(HEADLINE_START, tz="UTC").normalize()],
        ycol,
        horizon=HORIZON,
    )
    print("[HB] running 12-1 control book (same long-only overlapping mandate)", flush=True)
    raw_ctrl = run_ls_topn(
        ctrl[["date", "symbol", "score"]],
        panel,
        feat,
        pit,
        horizon=HORIZON,
        top_pct=TOP_PCT,
        book_start=BOOK_START,
        variant="nasdaq_adapt1_12_1",
        long_only=LONG_ONLY,
    )
    if raw_ctrl.get("error"):
        raise RuntimeError(raw_ctrl["error"])
    ctrl_2005 = summarize_book(raw_ctrl, start=BOOK_START)
    ctrl_2007 = summarize_book(raw_ctrl, start=HEADLINE_START)

    factor = factor_verdict(float(ric_2007.get("mean_ic", float("nan"))), float(sum_2007.get("net_sharpe_full", float("nan"))))
    ml = ml_claim_verdict(factor, float(sum_2007.get("net_sharpe_full", float("nan"))), float(ctrl_2007.get("net_sharpe_full", float("nan"))))
    print(
        f"[HB] factor={factor.get('verdict')} ml={ml.get('verdict')} "
        f"sharpe2007={sum_2007.get('net_sharpe_full')} ctrl={ctrl_2007.get('net_sharpe_full')}",
        flush=True,
    )

    idx = sum_2005.get("daily_ret")
    qqq = qqq_bh(market)
    ew = ew_universe(panel, pit)
    xs = None
    if isinstance(idx, pd.Series) and len(idx):
        qqq = qqq.reindex(idx.index).fillna(0.0) if len(qqq) else qqq
        ew = ew.reindex(idx.index).fillna(0.0) if len(ew) else ew
        gbm2007 = window_from(idx, HEADLINE_START)
        q2007 = window_from(qqq, HEADLINE_START).reindex(gbm2007.index).fillna(0.0)
        xs = gbm2007 - q2007

    chart_path = Path(CHART_PATH)
    if isinstance(idx, pd.Series) and len(idx):
        extra_series = []
        ctrl_rets = ctrl_2005.get("daily_ret")
        if isinstance(ctrl_rets, pd.Series) and len(ctrl_rets):
            extra_series.append((ctrl_rets, "12-1 control (same long-only book)"))
        plot_equity(
            idx,
            qqq if len(qqq) else None,
            ew if len(ew) else None,
            chart_path,
            treat_label="NASDAQ-ADAPT-1 GBM (long-only top 10%, h=126)",
            title="NASDAQ-ADAPT-1 vs 12-1 / QQQ (long-only PIT vol-30, h=126)",
            extra_series=extra_series,
        )

    extra_blob = {
        "elapsed_sec": time.time() - t0,
        "ticker_source": meta.get("source"),
        "n_symbols": meta.get("n_symbols"),
        "min_date": meta.get("min_date"),
        "max_date": meta.get("max_date"),
        "n_folds": len(folds),
        "iters": iters,
        "headline_start": HEADLINE_START,
        "train_note": (
            f"trees={FIXED_TREES} early_stop=off n_folds={len(folds)} "
            f"train_cap={TRAIN_MAX_SESSIONS} sessions top_pct={TOP_PCT} h={HORIZON} long_only={LONG_ONLY}"
        ),
        "qqq": qqq,
        "ew": ew,
        "excess_qqq": xs,
        "feature_gain_mean": _mean_gain(metas),
        "construction": (
            f"Yahoo Nasdaq-100 source={meta.get('source')} n={meta.get('n_symbols')}; "
            f"PIT top-{EXEC_TOP_N}; long-only top_pct={TOP_PCT}; overlapping h={HORIZON}; "
            f"rolling train ≤{TRAIN_MAX_SESSIONS} sessions; session purge/embargo; "
            f"simple (non-residual) label; equity-clock features; inv-vol; 5 bps one-way; "
            f"Adj Close; book from {BOOK_START}; FACTOR window {HEADLINE_START}; "
            f"n_preds={len(preds)}."
        ),
    }
    text = write_adapt1_report(
        Path(REPORT_MD),
        factor=factor,
        ml=ml,
        book_2005=sum_2005,
        book_2007=sum_2007,
        ctrl_2005=ctrl_2005,
        ctrl_2007=ctrl_2007,
        ric_all=ric_all,
        ric_2007=ric_2007,
        ric_ctrl_2007=ric_ctrl_2007,
        extra=extra_blob,
        factor_criterion=FACTOR_CRITERION,
    )
    summary = {
        "factor": factor,
        "ml_claim": ml,
        "book_2005": {k: v for k, v in sum_2005.items() if k != "daily_ret"},
        "book_2007": {k: v for k, v in sum_2007.items() if k != "daily_ret"},
        "ctrl_2005": {k: v for k, v in ctrl_2005.items() if k != "daily_ret"},
        "ctrl_2007": {k: v for k, v in ctrl_2007.items() if k != "daily_ret"},
        "ric_all": ric_all,
        "ric_2007": ric_2007,
        "ric_ctrl_2007": ric_ctrl_2007,
        "iters": iters,
        "n_folds": len(folds),
        "horizon": HORIZON,
        "top_pct": TOP_PCT,
        "long_only": LONG_ONLY,
        "train_max_sessions": TRAIN_MAX_SESSIONS,
        "ticker_source": meta.get("source"),
        "n_symbols": meta.get("n_symbols"),
        "feature_gain_mean": extra_blob["feature_gain_mean"],
        "excess_sharpe_qqq_2007": sharpe(xs) if isinstance(xs, pd.Series) and len(xs) else None,
        "gpu_used": False,
    }
    Path(REPORT_JSON).write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    print(text)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s factor={factor.get('verdict')} ml={ml.get('verdict')}", flush=True)
    return summary


if __name__ == "__main__":
    main()
