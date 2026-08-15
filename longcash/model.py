"""Walk-forward LightGBM heads for LONG-CASH (Huber USD + binary up)."""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from baseline.features import FEATURE_COLS
from baseline.model import FoldSpec, _make_rank_ic_feval
from baseline.seedutil import seed_everything


def mean_per_date_rank_ic(
    pred: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    min_n: int = 8,
) -> float:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="mergesort")
    pred, y, dates = pred[order], y[order], dates[order]
    ics = []
    i, n = 0, len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        x, yy = pred[i:j], y[i:j]
        m = np.isfinite(x) & np.isfinite(yy)
        x, yy = x[m], yy[m]
        if len(x) >= min_n and np.unique(x).size > 1 and np.unique(yy).size > 1:
            res = stats.spearmanr(x, yy)
            corr = getattr(res, "correlation", None)
            if corr is None:
                corr = getattr(res, "statistic", np.nan)
            c = float(np.asarray(corr, dtype=float).reshape(-1)[0])
            if np.isfinite(c):
                ics.append(c)
        i = j
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def mean_per_date_auc(
    y: np.ndarray,
    p: np.ndarray,
    dates: np.ndarray,
    min_n: int = 8,
) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="mergesort")
    y, p, dates = y[order], p[order], dates[order]
    aucs = []
    i, n = 0, len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        yy, pp = y[i:j], p[i:j]
        m = np.isfinite(yy) & np.isfinite(pp)
        yy, pp = yy[m], pp[m]
        if len(yy) >= min_n and np.unique(yy).size > 1 and np.unique(pp).size > 1:
            try:
                aucs.append(float(roc_auc_score(yy, pp)))
            except ValueError:
                pass
        i = j
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _make_per_date_auc_feval(dates: np.ndarray):
    dates = np.asarray(dates)

    def _feval(preds, dataset):
        y = dataset.get_label()
        n = len(y)
        val = mean_per_date_auc(y, preds[:n], dates[:n])
        if not np.isfinite(val):
            val = 0.5
        return "pdauc", float(val), True

    return _feval


def fit_isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[m], y[m]
    if len(y) < 50 or np.unique(np.round(y, 12)).size < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(raw, y)
    return ir


def apply_isotonic(ir: IsotonicRegression | None, raw: np.ndarray, clip01: bool) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if ir is None:
        out = raw.copy()
    else:
        out = np.asarray(ir.predict(raw), dtype=float)
    if clip01:
        return np.clip(out, 0.0, 1.0)
    return out


def last_fold_wins(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds
    out = preds.sort_values(["date", "symbol", "fold_id"])
    return out.drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)


def pick_null_folds(folds: list[FoldSpec], anchor: str = "2022-01-01") -> list[FoldSpec]:
    if not folds:
        return []

    def _utc(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    first = folds[0]
    target = _utc(anchor)
    nearest = min(folds, key=lambda f: abs((_utc(f.val_start) - target).days))
    if nearest.fold_id == first.fold_id and len(folds) > 1:
        nearest = min(folds[1:], key=lambda f: abs((_utc(f.val_start) - target).days))
    return [first, nearest]


def fit_predict_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    *,
    head: str,
    seed: int,
    model_cfg: dict,
    inner_holdout_days: int,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    btc_symbol: str = "BTCUSDT",
) -> tuple[pd.DataFrame, dict]:
    """Train one head on one fold. Head R = Huber y_usd; Head C = binary y_up."""
    seed_everything(seed + fold.fold_id + (0 if shuffle_seed is None else int(shuffle_seed)))
    h = int(fold.horizon)
    head_u = str(head).upper()
    if head_u == "R":
        ycol = f"y_usd_h{h}"
        objective = "huber"
        clip01 = False
        use_rank_ic = True
    elif head_u == "C":
        ycol = f"y_up_h{h}"
        objective = "binary"
        clip01 = True
        use_rank_ic = False
    else:
        raise ValueError(f"unknown head {head}")

    feats = list(FEATURE_COLS)
    t0 = time.time()

    def _utc(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    train_start, train_end = _utc(fold.train_start), _utc(fold.train_end)
    val_start, val_end = _utc(fold.val_start), _utc(fold.val_end)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[d["symbol"] != btc_symbol]

    train_mask = (d["date"] >= train_start) & (d["date"] <= train_end)
    val_mask = (d["date"] >= val_start) & (d["date"] <= val_end)
    need_train = feats + [ycol]
    train = d.loc[train_mask].dropna(subset=need_train)
    valid = d.loc[val_mask].dropna(subset=feats)
    if train.empty or valid.empty:
        return pd.DataFrame(), {
            "fold_id": fold.fold_id,
            "head": head_u,
            "status": "empty",
            "elapsed": time.time() - t0,
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
        }

    cut = train_end - pd.Timedelta(days=int(inner_holdout_days))
    inner_tr = train[train["date"] <= cut]
    inner_ho = train[train["date"] > cut]
    if inner_tr.empty or inner_ho.empty:
        n = max(1, int(len(train) * 0.85))
        inner_tr = train.iloc[:n]
        inner_ho = train.iloc[n:]

    if shuffle_labels:
        ss = int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017
        rng = np.random.default_rng(ss)

        def _shuf(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame[ycol] = frame.groupby("date", sort=False)[ycol].transform(
                lambda s: rng.permutation(s.to_numpy())
            )
            return frame

        inner_tr = _shuf(inner_tr)
        inner_ho = _shuf(inner_ho)

    dtrain = lgb.Dataset(inner_tr[feats], label=inner_tr[ycol], free_raw_data=False)
    dvalid = lgb.Dataset(inner_ho[feats], label=inner_ho[ycol], reference=dtrain, free_raw_data=False)

    params = {
        "objective": objective,
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
        "num_threads": 8,
    }
    n_estimators = int(model_cfg.get("n_estimators", 3000))
    patience = int(model_cfg.get("early_stopping_rounds", 100))
    fixed_trees = model_cfg.get("fixed_n_estimators")
    ho_dates = inner_ho["date"].to_numpy()
    if use_rank_ic and fixed_trees is None:
        feval = _make_rank_ic_feval(ho_dates)
    elif (not use_rank_ic) and fixed_trees is None:
        feval = _make_per_date_auc_feval(ho_dates)
    else:
        feval = None

    evals_result: dict = {}
    callbacks = [lgb.record_evaluation(evals_result), lgb.log_evaluation(period=0)]
    if fixed_trees is not None:
        n_estimators = int(fixed_trees)
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
        early_stop_mode = "rank_ic" if use_rank_ic else "pdauc"

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dvalid],
        valid_names=["inner_ho"],
        feval=feval,
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or 0) or n_estimators

    raw_ho = booster.predict(inner_ho[feats], num_iteration=best_iteration)
    ir = None if shuffle_labels else fit_isotonic(raw_ho, inner_ho[ycol].to_numpy())
    raw_val = booster.predict(valid[feats], num_iteration=best_iteration)
    p_val = apply_isotonic(ir, raw_val, clip01=clip01)

    pred_df = valid[["date", "symbol"]].copy()
    pred_df["p_raw"] = raw_val
    pred_df["p"] = p_val
    pred_df["horizon"] = h
    pred_df["fold_id"] = fold.fold_id
    pred_df["head"] = head_u
    pred_df[ycol] = valid[ycol].to_numpy() if ycol in valid.columns else np.nan

    ric = mean_per_date_rank_ic(raw_val, valid[ycol].to_numpy() if ycol in valid.columns else np.array([]), valid["date"].to_numpy())
    elapsed = time.time() - t0
    gain = booster.feature_importance(importance_type="gain")
    meta = {
        "fold_id": fold.fold_id,
        "head": head_u,
        "status": "ok",
        "elapsed": elapsed,
        "best_iteration": best_iteration,
        "early_stop_mode": early_stop_mode,
        "n_train": int(len(inner_tr)),
        "n_holdout": int(len(inner_ho)),
        "n_valid": int(len(valid)),
        "train_end": str(pd.Timestamp(fold.train_end).date()),
        "val_start": str(pd.Timestamp(fold.val_start).date()),
        "val_end": str(pd.Timestamp(fold.val_end).date()),
        "rankic_oos_raw": ric,
        "calibrated": ir is not None,
        "shuffle_labels": bool(shuffle_labels),
        "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
        "ycol": ycol,
        "feature_importance_gain": {f: float(g) for f, g in zip(feats, gain)},
        "horizon": h,
    }
    return pred_df, meta
