"""Walk-forward LightGBM training with purge/embargo."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_COLS
from .seedutil import seed_everything


@dataclass
class FoldSpec:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive, before purge
    purge_end: pd.Timestamp
    embargo_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    horizon: int


def make_folds(
    dates: pd.DatetimeIndex,
    horizon: int,
    min_train_days: int = 730,
    val_days: int = 90,
    step_days: int = 90,
) -> list[FoldSpec]:
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    if len(dates) < min_train_days + val_days + horizon + 10:
        return []
    start = dates[0]
    folds: list[FoldSpec] = []
    fold_id = 0
    # first train end index
    i_train_end = min_train_days - 1
    while True:
        if i_train_end >= len(dates):
            break
        train_end = dates[i_train_end]
        # purge last h days of training labels (drop from train)
        purge_cut = train_end - pd.Timedelta(days=horizon)
        embargo_end = train_end + pd.Timedelta(days=horizon + 3)
        # validation starts after embargo
        val_candidates = dates[dates > embargo_end]
        if len(val_candidates) < val_days // 2:
            break
        val_start = val_candidates[0]
        val_end_candidates = dates[dates >= val_start]
        if len(val_end_candidates) < 5:
            break
        val_end = val_end_candidates[min(val_days - 1, len(val_end_candidates) - 1)]
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=start,
                train_end=purge_cut,
                purge_end=train_end,
                embargo_end=embargo_end,
                val_start=val_start,
                val_end=val_end,
                horizon=horizon,
            )
        )
        fold_id += 1
        # step train end forward
        next_idx = i_train_end + step_days
        if next_idx >= len(dates) - 5:
            break
        i_train_end = next_idx
        if val_end >= dates[-1] - pd.Timedelta(days=horizon + 1):
            # still allow fold; next iteration may break
            if val_end == dates[min(len(dates) - 1, dates.get_indexer([val_end])[0])]:
                if len(dates[dates > val_end]) < step_days // 2:
                    break
    return folds


def _fit_predict_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int,
    model_cfg: dict,
    inner_holdout_days: int,
) -> tuple[pd.DataFrame, dict]:
    seed_everything(seed + fold.fold_id)
    ycol = f"y_h{fold.horizon}"
    feats = FEATURE_COLS
    t0 = time.time()

    train_mask = (df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)
    val_mask = (df["date"] >= fold.val_start) & (df["date"] <= fold.val_end)
    train = df.loc[train_mask].dropna(subset=feats + [ycol])
    valid = df.loc[val_mask].dropna(subset=feats + [ycol])
    if train.empty or valid.empty:
        return pd.DataFrame(), {"fold_id": fold.fold_id, "status": "empty", "elapsed": 0.0}

    # inner holdout = last inner_holdout_days of training window
    cut = fold.train_end - pd.Timedelta(days=inner_holdout_days)
    inner_tr = train[train["date"] <= cut]
    inner_ho = train[train["date"] > cut]
    if inner_tr.empty or inner_ho.empty:
        inner_tr = train.iloc[: int(len(train) * 0.85)]
        inner_ho = train.iloc[int(len(train) * 0.85) :]

    dtrain = lgb.Dataset(inner_tr[feats], label=inner_tr[ycol], free_raw_data=False)
    dvalid = lgb.Dataset(inner_ho[feats], label=inner_ho[ycol], reference=dtrain, free_raw_data=False)

    params = {
        "objective": model_cfg.get("objective", "huber"),
        "num_leaves": model_cfg.get("num_leaves", 31),
        "learning_rate": model_cfg.get("learning_rate", 0.03),
        "min_data_in_leaf": model_cfg.get("min_data_in_leaf", 200),
        "feature_fraction": model_cfg.get("feature_fraction", 0.8),
        "bagging_fraction": model_cfg.get("bagging_fraction", 0.8),
        "bagging_freq": model_cfg.get("bagging_freq", 1),
        "lambda_l2": model_cfg.get("lambda_l2", 1.0),
        "verbosity": model_cfg.get("verbosity", -1),
        "seed": seed + fold.fold_id,
        "feature_fraction_seed": seed + fold.fold_id,
        "bagging_seed": seed + fold.fold_id,
        "deterministic": True,
        "force_row_wise": True,
    }
    callbacks = [
        lgb.early_stopping(model_cfg.get("early_stopping_rounds", 100), verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=model_cfg.get("n_estimators", 3000),
        valid_sets=[dvalid],
        callbacks=callbacks,
    )
    pred = booster.predict(valid[feats], num_iteration=booster.best_iteration)
    pred_df = valid[["date", "symbol"]].copy()
    pred_df["score"] = pred
    pred_df["horizon"] = fold.horizon
    pred_df["model_name"] = "lgbm_price_only"
    pred_df["fold_id"] = fold.fold_id
    pred_df[ycol] = valid[ycol].values

    elapsed = time.time() - t0
    meta = {
        "fold_id": fold.fold_id,
        "status": "ok",
        "elapsed": elapsed,
        "best_iteration": int(booster.best_iteration or 0),
        "n_train": int(len(inner_tr)),
        "n_holdout": int(len(inner_ho)),
        "n_valid": int(len(valid)),
        "train_end": str(fold.train_end.date()),
        "val_start": str(fold.val_start.date()),
        "val_end": str(fold.val_end.date()),
    }
    return pred_df, meta


def train_all_folds(
    df: pd.DataFrame,
    horizon: int,
    cfg: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(
        pd.DatetimeIndex(df["date"].unique()),
        horizon=horizon,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    print(f"[model] horizon={horizon} folds={len(folds)}", flush=True)
    all_preds = []
    metas = []
    warn_s = cfg["cv"].get("fold_warn_seconds", 1200)
    for fold in folds:
        t0 = time.time()
        print(
            f"[model] fold {fold.fold_id+1}/{len(folds)} h={horizon} "
            f"train≤{fold.train_end.date()} val={fold.val_start.date()}→{fold.val_end.date()}",
            flush=True,
        )
        pred_df, meta = _fit_predict_fold(
            df,
            fold,
            seed=cfg["seed"],
            model_cfg=cfg["model"],
            inner_holdout_days=cfg["cv"]["inner_holdout_days"],
        )
        if meta["elapsed"] > warn_s:
            print(f"[WARN] fold {fold.fold_id} took {meta['elapsed']:.0f}s > {warn_s}s", flush=True)
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            pred_df.to_parquet(out_dir / f"preds_h{horizon}_fold{fold.fold_id}.parquet", index=False)
        # heartbeat
        print(
            f"[model] fold {fold.fold_id} done status={meta['status']} "
            f"elapsed={time.time()-t0:.1f}s best_iter={meta.get('best_iteration')}",
            flush=True,
        )
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    (out_dir / f"fold_meta_h{horizon}.json").write_text(json.dumps(metas, indent=2))
    return preds, metas
