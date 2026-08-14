"""Walk-forward LightGBM binary classifier + per-fold isotonic calibration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from baseline.seedutil import seed_everything
from btcb.constants import (
    FEATURE_COLS_V1,
    INNER_HOLDOUT_CALENDAR_DAYS,
    LGBM_V1,
    MIN_TRAIN_CALENDAR_DAYS,
    PRICE_COLS,
    SEED,
    STEP_CALENDAR_DAYS,
    USABLE_FROM,
    VAL_CALENDAR_DAYS,
)


@dataclass
class FoldSpec:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive, already purged
    purge_end: pd.Timestamp
    embargo_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    horizon: int


def _utc_norm(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()


def make_expanding_folds(
    dates: pd.DatetimeIndex,
    horizon: int,
    start: str = USABLE_FROM,
    min_train_days: int = MIN_TRAIN_CALENDAR_DAYS,
    val_days: int = VAL_CALENDAR_DAYS,
    step_days: int = STEP_CALENDAR_DAYS,
) -> list[FoldSpec]:
    dates = pd.DatetimeIndex(pd.to_datetime(sorted(pd.unique(dates)), utc=True)).tz_convert("UTC").normalize()
    start_ts = _utc_norm(start)
    dates = dates[dates >= start_ts]
    if dates.empty:
        return []
    first_end_cal = start_ts + pd.Timedelta(days=int(min_train_days))
    te_cand = dates[dates <= first_end_cal]
    if te_cand.empty:
        return []
    te = te_cand[-1]
    folds: list[FoldSpec] = []
    fold_id = 0
    last = dates[-1]
    while True:
        purge_cut = te - pd.Timedelta(days=int(horizon))
        embargo_end = te + pd.Timedelta(days=int(horizon) + 3)
        val_dates = dates[dates > embargo_end]
        if len(val_dates) < 10:
            break
        val_start = val_dates[0]
        val_end_target = val_start + pd.Timedelta(days=int(val_days))
        vd = val_dates[val_dates <= val_end_target]
        val_end = vd[-1] if len(vd) else val_start
        if purge_cut < start_ts:
            break
        remaining_after = dates[dates > val_end]
        if len(remaining_after) < 20:
            val_end = last
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=start_ts,
                train_end=purge_cut,
                purge_end=te,
                embargo_end=embargo_end,
                val_start=val_start,
                val_end=val_end,
                horizon=horizon,
            )
        )
        if val_end >= last:
            break
        next_te_cal = te + pd.Timedelta(days=int(step_days))
        nxt = dates[dates <= next_te_cal]
        if nxt.empty or nxt[-1] <= te:
            break
        te = nxt[-1]
        fold_id += 1
    return folds


def pick_null_folds(folds: list[FoldSpec], anchor: str = "2022-01-01") -> list[FoldSpec]:
    if not folds:
        return []
    first = folds[0]
    target = _utc_norm(anchor)
    nearest = min(folds, key=lambda f: abs((f.val_start - target).days))
    if nearest.fold_id == first.fold_id and len(folds) > 1:
        nearest = min(folds[1:], key=lambda f: abs((f.val_start - target).days))
    return [first, nearest]


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 20 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def fit_isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[m], y[m]
    if len(y) < 50 or np.unique(y).size < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(raw, y)
    return ir


def apply_calibrator(ir: IsotonicRegression | None, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if ir is None:
        return np.clip(raw, 0.0, 1.0)
    out = ir.predict(raw)
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def fit_predict_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int = SEED,
    feature_cols: list[str] | None = None,
    inner_holdout_days: int = INNER_HOLDOUT_CALENDAR_DAYS,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    lgbm_params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    seed_everything(seed + fold.fold_id + (0 if shuffle_seed is None else int(shuffle_seed)))
    ycol = f"y_h{fold.horizon}"
    feats = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS_V1)
    cfg = dict(LGBM_V1)
    if lgbm_params:
        cfg.update(lgbm_params)
    t0 = time.time()

    train_mask = (df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)
    val_mask = (df["date"] >= fold.val_start) & (df["date"] <= fold.val_end)
    price_need = [c for c in PRICE_COLS if c in df.columns]
    train = df.loc[train_mask].dropna(subset=price_need + [ycol])
    valid = df.loc[val_mask].dropna(subset=price_need)
    if train.empty or valid.empty:
        return pd.DataFrame(), {
            "fold_id": fold.fold_id,
            "status": "empty",
            "elapsed": time.time() - t0,
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
        }

    cut = fold.train_end - pd.Timedelta(days=int(inner_holdout_days))
    inner_tr = train[train["date"] <= cut]
    inner_ho = train[train["date"] > cut]
    if inner_tr.empty or inner_ho.empty:
        n = max(1, int(len(train) * 0.85))
        inner_tr = train.iloc[:n]
        inner_ho = train.iloc[n:]

    if shuffle_labels:
        ss = int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017
        rng = np.random.default_rng(ss)

        def _shuf(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d[ycol] = d.groupby("date", sort=False)[ycol].transform(lambda s: rng.permutation(s.to_numpy()))
            return d

        inner_tr = _shuf(inner_tr)
        inner_ho = _shuf(inner_ho)

    dtrain = lgb.Dataset(inner_tr[feats], label=inner_tr[ycol], free_raw_data=False)
    dvalid = lgb.Dataset(inner_ho[feats], label=inner_ho[ycol], reference=dtrain, free_raw_data=False)

    params = {
        "objective": cfg.get("objective", "binary"),
        "metric": cfg.get("metric", "auc"),
        "num_leaves": cfg.get("num_leaves", 31),
        "learning_rate": cfg.get("learning_rate", 0.03),
        "min_data_in_leaf": cfg.get("min_data_in_leaf", 200),
        "feature_fraction": cfg.get("feature_fraction", 0.8),
        "bagging_fraction": cfg.get("bagging_fraction", 0.8),
        "bagging_freq": cfg.get("bagging_freq", 1),
        "lambda_l2": cfg.get("lambda_l2", 1.0),
        "verbosity": cfg.get("verbosity", -1),
        "seed": seed + fold.fold_id,
        "feature_fraction_seed": seed + fold.fold_id,
        "bagging_seed": seed + fold.fold_id,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 8,
    }
    n_estimators = int(cfg.get("n_estimators", 3000))
    patience = int(cfg.get("early_stopping_rounds", 100))
    evals_result: dict = {}
    callbacks = [
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(stopping_rounds=patience, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dvalid],
        valid_names=["inner_ho"],
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or 0) or n_estimators

    raw_ho = booster.predict(inner_ho[feats], num_iteration=best_iteration)
    ir = None if shuffle_labels else fit_isotonic(raw_ho, inner_ho[ycol].to_numpy())
    raw_val = booster.predict(valid[feats], num_iteration=best_iteration)
    p_val = apply_calibrator(ir, raw_val)

    pred_df = valid[["date", "id", "symbol"]].copy()
    pred_df["p_raw"] = raw_val
    pred_df["p"] = p_val
    pred_df["horizon"] = fold.horizon
    pred_df["fold_id"] = fold.fold_id
    pred_df[ycol] = valid[ycol].to_numpy()

    gain = booster.feature_importance(importance_type="gain")
    gain_map = {f: float(g) for f, g in zip(feats, gain)}
    auc_ho = _auc(inner_ho[ycol].to_numpy(), apply_calibrator(ir, raw_ho) if ir is not None else raw_ho)
    auc_val = _auc(valid[ycol].to_numpy(), p_val)
    auc_val_raw = _auc(valid[ycol].to_numpy(), raw_val)

    reliability = _reliability(valid[ycol].to_numpy(), p_val)
    elapsed = time.time() - t0
    meta = {
        "fold_id": fold.fold_id,
        "status": "ok",
        "elapsed": elapsed,
        "best_iteration": best_iteration,
        "n_train": int(len(inner_tr)),
        "n_holdout": int(len(inner_ho)),
        "n_valid": int(len(valid)),
        "train_end": str(pd.Timestamp(fold.train_end).date()),
        "val_start": str(pd.Timestamp(fold.val_start).date()),
        "val_end": str(pd.Timestamp(fold.val_end).date()),
        "auc_holdout": auc_ho,
        "auc_oos": auc_val,
        "auc_oos_raw": auc_val_raw,
        "feature_importance_gain": gain_map,
        "feature_cols": feats,
        "shuffle_labels": bool(shuffle_labels),
        "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
        "calibrated": ir is not None,
        "reliability": reliability,
        "horizon": fold.horizon,
    }
    return pred_df, meta


def _reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 20:
        return {"bins": [], "frac_pos": [], "mean_p": [], "counts": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    bins, frac, mean_p, counts = [], [], [], []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        bins.append([float(edges[b]), float(edges[b + 1])])
        frac.append(float(y[sel].mean()))
        mean_p.append(float(p[sel].mean()))
        counts.append(int(sel.sum()))
    return {"bins": bins, "frac_pos": frac, "mean_p": mean_p, "counts": counts}


def train_all_folds(
    df: pd.DataFrame,
    horizon: int,
    out_dir=None,
) -> tuple[pd.DataFrame, list[dict], list[FoldSpec]]:
    folds = make_expanding_folds(pd.DatetimeIndex(df["date"].unique()), horizon=horizon)
    print(f"[HB] h={horizon} folds={len(folds)}", flush=True)
    all_preds = []
    metas = []
    for fold in folds:
        print(
            f"[HB] fold {fold.fold_id+1}/{len(folds)} h={horizon} "
            f"train≤{pd.Timestamp(fold.train_end).date()} "
            f"val={pd.Timestamp(fold.val_start).date()}→{pd.Timestamp(fold.val_end).date()}",
            flush=True,
        )
        pred_df, meta = fit_predict_fold(df, fold)
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            if out_dir is not None:
                pred_df.to_parquet(out_dir / f"preds_h{horizon}_fold{fold.fold_id}.parquet", index=False)
        print(
            f"[HB] fold {fold.fold_id} status={meta.get('status')} "
            f"auc_oos={meta.get('auc_oos')} best_iter={meta.get('best_iteration')} "
            f"elapsed={meta.get('elapsed'):.1f}s",
            flush=True,
        )
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if out_dir is not None:
        (out_dir / f"fold_meta_h{horizon}.json").write_text(json.dumps(metas, indent=2, default=str))
    return preds, metas, folds


def mean_gain(metas: list[dict], top_n: int = 15) -> list[tuple[str, float]]:
    acc: dict[str, list[float]] = {}
    for m in metas:
        gi = m.get("feature_importance_gain") or {}
        for k, v in gi.items():
            acc.setdefault(k, []).append(float(v))
    ranked = sorted(((k, float(np.mean(vs))) for k, vs in acc.items()), key=lambda kv: -kv[1])
    return ranked[:top_n]
