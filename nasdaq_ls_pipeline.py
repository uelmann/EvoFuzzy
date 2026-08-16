"""
NASDAQ-LS scout — A0 LightGBM Huber, long 10 / short 10 on PIT vol-30.

BACKTEST ONLY. CPU only. COMBO / SPREAD-LS / LONG-TIDE untouched.
Usage: python nasdaq_ls_pipeline.py
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
from baseline.model import _fit_predict_fold, make_folds
from baseline.seedutil import seed_everything
from nasdaq_ls.book import run_ls_topn
from nasdaq_ls.constants import (
    BOOK_START,
    DEATH_CONVENTION,
    DV_WINDOW,
    EXEC_TOP_N,
    FACTOR_CRITERION,
    FEAT_PATH,
    FIXED_TREES,
    HEADLINE_START,
    HORIZON,
    K_LONG,
    K_SHORT,
    MARKET_PATH,
    PANEL_PATH,
    PRED_PATH,
    PRICE_RULE,
    SEED,
    TRAIN_RULE,
)
from nasdaq_ls.download import download_all
from nasdaq_ls.eval import (
    factor_verdict,
    last_fold_wins,
    pooled_rankic,
    qqq_bh,
    ew_universe,
    summarize_book,
    window_from,
)
from nasdaq_ls.features import build_feature_panel
from nasdaq_ls.labels import add_labels
from nasdaq_ls.report import plot_equity, write_report


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
    addendum = Path("reports/nasdaq_ls_addendum.md").read_text()
    for needle in (FACTOR_CRITERION, DEATH_CONVENTION, TRAIN_RULE, PRICE_RULE):
        if needle not in addendum:
            raise RuntimeError("Addendum missing a verbatim frozen statement")
    print("[HB] NASDAQ-LS SCOUT; 500 trees; no early stop; COMBO untouched", flush=True)
    print("[HB] returns = Yahoo Adj Close", flush=True)
    print(f"[HB] {PRICE_RULE}", flush=True)
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
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
        print(f"[HB] loaded cached features rows={len(feat)}", flush=True)
    else:
        feat = build_feature_panel(panel, market, clip=float(cfg["features"]["zscore_clip"]))
        feat = add_labels(feat, panel, market, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(feat_path, index=False)
        print(f"[HB] features rows={len(feat)} ymean={feat[f'y_h{HORIZON}'].mean():.4f}", flush=True)

    ycol = f"y_h{HORIZON}"
    if ycol not in feat.columns:
        feat = add_labels(feat, panel, market, horizons=[HORIZON], winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]))

    pit = build_pit_topn(panel, n=EXEC_TOP_N, window=DV_WINDOW)
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.normalize()
    print(f"[HB] PIT top-{EXEC_TOP_N} rows={len(pit)} dates={pit['date'].nunique()}", flush=True)

    dates = pd.DatetimeIndex(sorted(feat["date"].dropna().unique()))
    folds = make_folds(
        dates,
        horizon=HORIZON,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    print(f"[HB] folds={len(folds)} h={HORIZON} trees={FIXED_TREES} early_stop=off", flush=True)

    model_cfg = dict(cfg["model"])
    model_cfg["objective"] = "huber"
    model_cfg["fixed_n_estimators"] = int(FIXED_TREES)
    model_cfg["early_stop_metric"] = "none"
    inner = int(cfg["cv"]["inner_holdout_days"])
    seed = int(cfg.get("seed", SEED))

    pred_dir = Path("data/nasdaq/preds")
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
                model_name="nasdaq_ls_huber",
            )
            if not pred_df.empty:
                pred_df.to_parquet(cache, index=False)
            meta_p.write_text(json.dumps(_jsonable(meta_f), indent=2, default=str))
            print(
                f"[fold] id={fr.fold_id} status={meta_f.get('status')} "
                f"iter={meta_f.get('best_iteration')} ric_rows={len(pred_df)}",
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
        raise RuntimeError("empty NASDAQ-LS preds")
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
    print(
        f"[HB] RankIC all={ric_all.get('mean_ic')} from2007={ric_2007.get('mean_ic')}",
        flush=True,
    )

    print("[HB] running overlapping LS book from 2005", flush=True)
    raw = run_ls_topn(
        preds,
        panel,
        feat,
        pit,
        horizon=HORIZON,
        k_long=K_LONG,
        k_short=K_SHORT,
        book_start=BOOK_START,
        variant="nasdaq_ls",
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

    chart_path = Path("charts/nasdaq_ls_equity.png")
    if isinstance(idx, pd.Series) and len(idx):
        plot_equity(idx, qqq if len(qqq) else None, ew if len(ew) else None, chart_path)

    extra_blob = {
        "elapsed_sec": time.time() - t0,
        "ticker_source": meta.get("source"),
        "n_symbols": meta.get("n_symbols"),
        "min_date": meta.get("min_date"),
        "max_date": meta.get("max_date"),
        "n_folds": len(folds),
        "iters": iters,
        "train_note": f"trees={FIXED_TREES} early_stop=off n_folds={len(folds)}",
        "qqq": qqq,
        "ew": ew,
        "construction": (
            f"Yahoo Nasdaq-100 source={meta.get('source')} n={meta.get('n_symbols')}; "
            f"PIT top-{EXEC_TOP_N} by {DV_WINDOW}d median DV; long {K_LONG} / short {K_SHORT}; "
            f"overlapping h={HORIZON}; inv-vol; 5 bps one-way; no borrow; "
            f"returns=Yahoo Adj Close; DV=unadjusted Close×Volume; "
            f"500 Huber trees; last-fold-wins; book from {BOOK_START}; "
            f"FACTOR window from {HEADLINE_START}; market=spliced ^IXIC/QQQ; "
            f"survivorship accepted; n_preds={len(preds)}."
        ),
    }
    md_path = Path("reports/nasdaq_ls_report.md")
    text = write_report(
        md_path,
        factor=factor,
        book_2005=sum_2005,
        book_2007=sum_2007,
        ric_all=ric_all,
        ric_2007=ric_2007,
        ric_simple=ric_simple,
        extra=extra_blob,
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
        "ticker_source": meta.get("source"),
        "n_symbols": meta.get("n_symbols"),
        "gpu_used": False,
    }
    Path("reports/nasdaq_ls_report.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str))
    print(text)
    print(
        f"[HB] DONE elapsed={time.time() - t0:.1f}s factor={factor.get('verdict')}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()
