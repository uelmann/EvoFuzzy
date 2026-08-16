"""
NASDAQ-LS21 scout — top/worst 10%, h=21, rolling 5y train.

BACKTEST ONLY. CPU only. COMBO untouched. Does not overwrite NASDAQ-LS h=10.
Usage: python nasdaq_ls21_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baseline.data import build_pit_topn
from baseline.features import FEATURE_COLS
from baseline.model import _fit_predict_fold
from baseline.seedutil import seed_everything
from nasdaq_ls.book import run_ls_topn
from nasdaq_ls.constants import MARKET_PATH, PANEL_PATH, PRICE_RULE, DEATH_CONVENTION
from nasdaq_ls.download import download_all
from nasdaq_ls.eval import (
    ew_universe,
    factor_verdict,
    last_fold_wins,
    pooled_rankic,
    qqq_bh,
    summarize_book,
)
from nasdaq_ls.features import build_feature_panel
from nasdaq_ls.folds import make_rolling_folds
from nasdaq_ls.labels import add_labels
from nasdaq_ls.report import plot_equity, write_report
from nasdaq_ls.v21_constants import (
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


def main() -> dict:
    t0 = time.time()
    addendum = Path(ADDENDUM_PATH).read_text()
    for needle in (FACTOR_CRITERION, DEATH_CONVENTION, TRAIN_RULE, PRICE_RULE):
        if needle not in addendum:
            raise RuntimeError("Addendum missing a verbatim frozen statement")
    print("[HB] NASDAQ-LS21; top/worst 10%; h=21; rolling 5y; COMBO untouched", flush=True)
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
    old_feat = Path("data/nasdaq/features.parquet")
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
        print(f"[HB] loaded cached v21 features rows={len(feat)}", flush=True)
    elif old_feat.exists():
        feat = pd.read_parquet(old_feat)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
        feat = add_labels(feat, panel, market, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(feat_path, index=False)
        print(f"[HB] reused A0 features + y_h{HORIZON} rows={len(feat)}", flush=True)
    else:
        feat = build_feature_panel(panel, market, clip=float(cfg["features"]["zscore_clip"]))
        feat = add_labels(feat, panel, market, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(feat_path, index=False)
        print(f"[HB] features rows={len(feat)} ymean={feat[ycol].mean():.4f}", flush=True)

    if ycol not in feat.columns:
        feat = add_labels(feat, panel, market, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))

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
    )
    if folds:
        d0 = dates[(dates >= folds[0].train_start) & (dates <= folds[0].train_end)]
        dL = dates[(dates >= folds[-1].train_start) & (dates <= folds[-1].train_end)]
        print(
            f"[HB] folds={len(folds)} h={HORIZON} trees={FIXED_TREES} "
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
                feature_cols=list(FEATURE_COLS),
                model_name="nasdaq_ls21_huber",
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
        raise RuntimeError("empty NASDAQ-LS21 preds")
    preds = last_fold_wins(pd.concat(parts, ignore_index=True))
    preds["date"] = pd.to_datetime(preds["date"], utc=True).dt.normalize()
    Path(PRED_PATH).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PRED_PATH, index=False)

    extra_cols = ["date", "symbol", ycol, f"y_simple_h{HORIZON}"]
    extra = feat[extra_cols].drop_duplicates(["date", "symbol"], keep="last")
    extra["date"] = pd.to_datetime(extra["date"], utc=True).dt.normalize()
    book_pred = preds.drop(columns=[c for c in preds.columns if c in {ycol, f"y_simple_h{HORIZON}"}], errors="ignore")
    book_pred = book_pred.merge(extra, on=["date", "symbol"], how="left")
    book_pred = book_pred.merge(pit[["date", "symbol"]], on=["date", "symbol"], how="inner")

    ric_all = pooled_rankic(book_pred, ycol, horizon=HORIZON)
    ric_simple = pooled_rankic(book_pred, f"y_simple_h{HORIZON}", horizon=HORIZON)
    cut = pd.Timestamp(HEADLINE_START, tz="UTC").normalize()
    ric_2007 = pooled_rankic(book_pred[book_pred["date"] >= cut], ycol, horizon=HORIZON)
    print(f"[HB] RankIC all={ric_all.get('mean_ic')} from2007={ric_2007.get('mean_ic')}", flush=True)

    print("[HB] running overlapping LS book h=21 top/worst 10% from 2005", flush=True)
    raw = run_ls_topn(
        preds,
        panel,
        feat,
        pit,
        horizon=HORIZON,
        top_pct=TOP_PCT,
        book_start=BOOK_START,
        variant="nasdaq_ls21",
    )
    if raw.get("error"):
        raise RuntimeError(raw["error"])
    sum_2005 = summarize_book(raw, start=BOOK_START)
    sum_2007 = summarize_book(raw, start=HEADLINE_START)
    factor = factor_verdict(float(ric_2007.get("mean_ic", float("nan"))), float(sum_2007.get("net_sharpe_full", float("nan"))))
    print(f"[HB] factor={factor.get('verdict')} sharpe2007={sum_2007.get('net_sharpe_full')}", flush=True)

    idx = sum_2005.get("daily_ret")
    qqq = qqq_bh(market)
    ew = ew_universe(panel, pit)
    if isinstance(idx, pd.Series) and len(idx):
        qqq = qqq.reindex(idx.index).fillna(0.0) if len(qqq) else qqq
        ew = ew.reindex(idx.index).fillna(0.0) if len(ew) else ew

    chart_path = Path(CHART_PATH)
    if isinstance(idx, pd.Series) and len(idx):
        plot_equity(
            idx,
            qqq if len(qqq) else None,
            ew if len(ew) else None,
            chart_path,
            treat_label="NASDAQ-LS21 (top 10% − worst 10%, h=21, 5y roll)",
            title="NASDAQ-LS21 vs QQQ (top/worst 10% of PIT vol-30, h=21)",
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
            f"train_cap={TRAIN_MAX_SESSIONS} sessions top_pct={TOP_PCT} h={HORIZON}"
        ),
        "qqq": qqq,
        "ew": ew,
        "construction": (
            f"Yahoo Nasdaq-100 source={meta.get('source')} n={meta.get('n_symbols')}; "
            f"PIT top-{EXEC_TOP_N}; long/short top_pct={TOP_PCT} (k=ceil(0.10*n)); "
            f"overlapping h={HORIZON}; rolling train ≤{TRAIN_MAX_SESSIONS} sessions; "
            f"inv-vol; 5 bps one-way; Adj Close; book from {BOOK_START}; "
            f"FACTOR window {HEADLINE_START}; n_preds={len(preds)}."
        ),
    }
    md_path = Path(REPORT_MD)
    text = write_report(
        md_path,
        factor=factor,
        book_2005=sum_2005,
        book_2007=sum_2007,
        ric_all=ric_all,
        ric_2007=ric_2007,
        ric_simple=ric_simple,
        extra=extra_blob,
        factor_criterion=FACTOR_CRITERION,
        heading="NASDAQ-LS21 — top/worst 10%, h=21, 5y rolling train",
        mandate_line=(
            "PIT top 30 by 30d median dollar volume; long top 10% / short worst 10% "
            "(k=ceil(0.10*n), typically 3 names); overlapping h=21; inv-vol; 5 bps; no borrow."
        ),
        train_line="500 Huber trees, no early stop, rolling ≤1260 sessions (~5y). Residual vs spliced QQQ.",
        y_horizon=HORIZON,
        book_2005_name="NASDAQ-LS21 from 2005-01-01",
        book_2007_name="NASDAQ-LS21 from 2007-01-01 (FACTOR window)",
    )
    summary = {
        "factor": factor,
        "book_2005": {k: v for k, v in sum_2005.items() if k != "daily_ret"},
        "book_2007": {k: v for k, v in sum_2007.items() if k != "daily_ret"},
        "ric_all": ric_all,
        "ric_2007": ric_2007,
        "ric_simple": ric_simple,
        "iters": iters,
        "n_folds": len(folds),
        "horizon": HORIZON,
        "top_pct": TOP_PCT,
        "train_max_sessions": TRAIN_MAX_SESSIONS,
        "ticker_source": meta.get("source"),
        "n_symbols": meta.get("n_symbols"),
        "gpu_used": False,
    }
    Path(REPORT_JSON).write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    print(text)
    print(f"[HB] DONE elapsed={time.time() - t0:.1f}s factor={factor.get('verdict')}", flush=True)
    return summary


if __name__ == "__main__":
    main()
