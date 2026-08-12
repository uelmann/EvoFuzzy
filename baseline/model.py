"""Walk-forward LightGBM training with purge/embargo and RankIC early stopping."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

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
    i_train_end = min_train_days - 1
    while True:
        if i_train_end >= len(dates):
            break
        train_end = dates[i_train_end]
        purge_cut = train_end - pd.Timedelta(days=horizon)
        embargo_end = train_end + pd.Timedelta(days=horizon + 3)
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
        next_idx = i_train_end + step_days
        if next_idx >= len(dates) - 5:
            break
        i_train_end = next_idx
        if val_end >= dates[-1] - pd.Timedelta(days=horizon + 1):
            if len(dates[dates > val_end]) < step_days // 2:
                break
    return folds


def _mean_daily_rank_ic(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> float:
    """Mean cross-sectional Spearman RankIC across dates (maximize)."""
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    dates = np.asarray(dates)
    ics = []
    # group by date via sort
    order = np.argsort(dates, kind="mergesort")
    preds, labels, dates = preds[order], labels[order], dates[order]
    i = 0
    n = len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        if j - i >= 5:
            x = preds[i:j]
            y = labels[i:j]
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if len(x) >= 5 and np.unique(x).size > 1 and np.unique(y).size > 1:
                res = stats.spearmanr(x, y)
                corr = getattr(res, "correlation", None)
                if corr is None:
                    corr = getattr(res, "statistic", np.nan)
                c = float(np.asarray(corr, dtype=float).reshape(-1)[0])
                if np.isfinite(c):
                    ics.append(c)
        i = j
    if not ics:
        return 0.0
    return float(np.mean(ics))


def _make_rank_ic_feval(dates: np.ndarray):
    dates = np.asarray(dates)

    def _feval(preds, dataset):
        y = dataset.get_label()
        # LightGBM may pass preds for the full dataset; align length
        n = len(y)
        d = dates[:n]
        val = _mean_daily_rank_ic(preds[:n], y, d)
        return "rank_ic", val, True

    return _feval


def _fit_predict_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int,
    model_cfg: dict,
    inner_holdout_days: int,
    log_eval_curve: bool = False,
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
        "metric": "None",
    }

    fixed_trees = model_cfg.get("fixed_n_estimators")
    use_rank_ic = str(model_cfg.get("early_stop_metric", "rank_ic")).lower() == "rank_ic"
    n_estimators = int(model_cfg.get("n_estimators", 3000))
    patience = int(model_cfg.get("early_stopping_rounds", 100))

    ho_dates = inner_ho["date"].to_numpy()
    feval = _make_rank_ic_feval(ho_dates) if use_rank_ic and fixed_trees is None else None

    evals_result: dict = {}
    callbacks = [lgb.record_evaluation(evals_result)]
    if fixed_trees is not None:
        n_estimators = int(fixed_trees)
        callbacks.append(lgb.log_evaluation(period=0))
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dvalid],
            valid_names=["inner_ho"],
            feval=feval,
            callbacks=callbacks,
        )
        best_iteration = int(n_estimators)
        early_stop_mode = f"fixed_{n_estimators}"
    else:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=patience,
                first_metric_only=True,
                verbose=False,
                min_delta=0.0,
            )
        )
        callbacks.append(lgb.log_evaluation(period=0))
        # Also record huber-like L2 for diagnosis on first rounds
        diag_params = dict(params)
        if log_eval_curve:
            diag_params["metric"] = "l2"
        booster = lgb.train(
            diag_params if log_eval_curve else params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dvalid],
            valid_names=["inner_ho"],
            feval=feval,
            callbacks=callbacks,
        )
        best_iteration = int(booster.best_iteration or 0)
        early_stop_mode = "rank_ic" if use_rank_ic else "default"

    # Diagnostic: if logging, also print early loss/IC curve points
    curve = {}
    if evals_result:
        for set_name, metrics in evals_result.items():
            for mname, vals in metrics.items():
                curve[f"{set_name}:{mname}"] = [float(v) for v in vals[: min(30, len(vals))]]
                if log_eval_curve:
                    print(
                        f"[model diag] fold={fold.fold_id} {set_name}:{mname} "
                        f"first10={[round(v, 6) for v in vals[:10]]} "
                        f"best_iter={best_iteration}",
                        flush=True,
                    )

    pred = booster.predict(valid[feats], num_iteration=best_iteration or -1)
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
        "best_iteration": best_iteration,
        "early_stop_mode": early_stop_mode,
        "n_train": int(len(inner_tr)),
        "n_holdout": int(len(inner_ho)),
        "n_valid": int(len(valid)),
        "train_end": str(fold.train_end.date()),
        "val_start": str(fold.val_start.date()),
        "val_end": str(fold.val_end.date()),
        "eval_curve_head": curve,
    }
    return pred_df, meta


def best_iteration_distribution(metas: list[dict]) -> dict:
    iters = [int(m["best_iteration"]) for m in metas if m.get("status") == "ok" and m.get("best_iteration") is not None]
    if not iters:
        return {"n": 0, "gt1_frac": 0.0, "iters": []}
    arr = np.asarray(iters, dtype=float)
    return {
        "n": len(iters),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "gt1_frac": float(np.mean(arr > 1)),
        "iters": iters,
    }


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
            log_eval_curve=(fold.fold_id in {0, max(0, len(folds) // 2), len(folds) - 1}),
        )
        if meta["elapsed"] > warn_s:
            print(f"[WARN] fold {fold.fold_id} took {meta['elapsed']:.0f}s > {warn_s}s", flush=True)
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            pred_df.to_parquet(out_dir / f"preds_h{horizon}_fold{fold.fold_id}.parquet", index=False)
        print(
            f"[model] fold {fold.fold_id} done status={meta['status']} "
            f"elapsed={time.time()-t0:.1f}s best_iter={meta.get('best_iteration')}",
            flush=True,
        )
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    (out_dir / f"fold_meta_h{horizon}.json").write_text(json.dumps(metas, indent=2, default=str))
    return preds, metas
