"""Walk-forward LightGBM binary classifier + per-fold isotonic calibration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

from baseline.seedutil import seed_everything
from btcb.constants import (
    FEATURE_COLS_V1,
    INNER_HOLDOUT_CALENDAR_DAYS,
    LGBM_RANK,
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


def mean_per_date_auc(y: np.ndarray, p: np.ndarray, dates: np.ndarray, min_n: int = 8) -> tuple[float, list[float]]:
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
        return float("nan"), []
    return float(np.mean(aucs)), aucs


def per_date_auc_series(y: np.ndarray, p: np.ndarray, dates: np.ndarray, min_n: int = 8) -> pd.Series:
    """One AUC per date (NaN if the cross-section is degenerate)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    dates = pd.to_datetime(np.asarray(dates), utc=True)
    order = np.argsort(dates, kind="mergesort")
    y, p, dates = y[order], p[order], dates[order]
    rows = []
    i, n = 0, len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        yy, pp = y[i:j], p[i:j]
        m = np.isfinite(yy) & np.isfinite(pp)
        yy, pp = yy[m], pp[m]
        val = float("nan")
        if len(yy) >= min_n and np.unique(yy).size > 1 and np.unique(pp).size > 1:
            try:
                val = float(roc_auc_score(yy, pp))
            except ValueError:
                val = float("nan")
        rows.append((pd.Timestamp(dates[i]).tz_convert("UTC").normalize(), val))
        i = j
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({d: v for d, v in rows}).sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").normalize()
    return s


def mean_per_date_rank_ic(p: np.ndarray, excess: np.ndarray, dates: np.ndarray, min_n: int = 8) -> float:
    p = np.asarray(p, dtype=float)
    excess = np.asarray(excess, dtype=float)
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="mergesort")
    p, excess, dates = p[order], excess[order], dates[order]
    ics = []
    i, n = 0, len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        x, y = p[i:j], excess[i:j]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) >= min_n and np.unique(x).size > 1 and np.unique(y).size > 1:
            res = stats.spearmanr(x, y)
            c = getattr(res, "correlation", None)
            if c is None:
                c = getattr(res, "statistic", np.nan)
            c = float(np.asarray(c, dtype=float).reshape(-1)[0])
            if np.isfinite(c):
                ics.append(c)
        i = j
    return float(np.mean(ics)) if ics else float("nan")


def per_date_rank_ic_series(p: np.ndarray, excess: np.ndarray, dates: np.ndarray, min_n: int = 8) -> pd.Series:
    p = np.asarray(p, dtype=float)
    excess = np.asarray(excess, dtype=float)
    dates = pd.to_datetime(np.asarray(dates), utc=True)
    order = np.argsort(dates, kind="mergesort")
    p, excess, dates = p[order], excess[order], dates[order]
    rows = []
    i, n = 0, len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        x, y = p[i:j], excess[i:j]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        val = float("nan")
        if len(x) >= min_n and np.unique(x).size > 1 and np.unique(y).size > 1:
            res = stats.spearmanr(x, y)
            c = getattr(res, "correlation", None)
            if c is None:
                c = getattr(res, "statistic", np.nan)
            c = float(np.asarray(c, dtype=float).reshape(-1)[0])
            if np.isfinite(c):
                val = c
        rows.append((pd.Timestamp(dates[i]).tz_convert("UTC").normalize(), val))
        i = j
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({d: v for d, v in rows}).sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").normalize()
    return s


def _make_per_date_auc_feval(dates: np.ndarray):
    dates = np.asarray(dates)

    def _feval(preds, dataset):
        y = dataset.get_label()
        n = len(y)
        val, _ = mean_per_date_auc(y, preds[:n], dates[:n])
        if not np.isfinite(val):
            val = 0.5
        return "pdauc", float(val), True

    return _feval


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
    early_stop: str = "auc",
    ycol: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    seed_everything(seed + fold.fold_id + (0 if shuffle_seed is None else int(shuffle_seed)))
    ycol = ycol or f"y_h{fold.horizon}"
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
        "metric": "None" if str(early_stop) == "per_date_auc" else cfg.get("metric", "auc"),
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
    feval = None
    if str(early_stop) == "per_date_auc":
        feval = _make_per_date_auc_feval(inner_ho["date"].to_numpy())
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
    p_val = apply_calibrator(ir, raw_val)

    pred_df = valid[["date", "id", "symbol"]].copy()
    pred_df["p_raw"] = raw_val
    pred_df["p"] = p_val
    pred_df["horizon"] = fold.horizon
    pred_df["fold_id"] = fold.fold_id
    pred_df[ycol] = valid[ycol].to_numpy()
    excol = f"excess_h{fold.horizon}"
    if excol in valid.columns:
        pred_df[excol] = valid[excol].to_numpy()

    gain = booster.feature_importance(importance_type="gain")
    gain_map = {f: float(g) for f, g in zip(feats, gain)}
    auc_ho = _auc(inner_ho[ycol].to_numpy(), apply_calibrator(ir, raw_ho) if ir is not None else raw_ho)
    auc_val = _auc(valid[ycol].to_numpy(), p_val)
    auc_val_raw = _auc(valid[ycol].to_numpy(), raw_val)
    pdauc, pdauc_list = mean_per_date_auc(valid[ycol].to_numpy(), p_val, valid["date"].to_numpy())
    pdauc_raw, _ = mean_per_date_auc(valid[ycol].to_numpy(), raw_val, valid["date"].to_numpy())
    ric = float("nan")
    if excol in valid.columns:
        ric = mean_per_date_rank_ic(p_val, valid[excol].to_numpy(), valid["date"].to_numpy())

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
        "pdauc_oos": pdauc,
        "pdauc_oos_raw": pdauc_raw,
        "pdauc_n_days": int(len(pdauc_list)),
        "rankic_oos": ric,
        "feature_importance_gain": gain_map,
        "feature_cols": feats,
        "ycol": ycol,
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
    feature_cols: list[str] | None = None,
    early_stop: str = "auc",
    ycol: str | None = None,
    tag: str | None = None,
    commit_fn=None,
) -> tuple[pd.DataFrame, list[dict], list[FoldSpec]]:
    folds = make_expanding_folds(pd.DatetimeIndex(df["date"].unique()), horizon=horizon)
    print(f"[HB] h={horizon} folds={len(folds)} early_stop={early_stop} ycol={ycol or f'y_h{horizon}'} tag={tag}", flush=True)
    all_preds = []
    metas = []
    for fold in folds:
        print(
            f"[HB] fold {fold.fold_id+1}/{len(folds)} h={horizon} tag={tag} "
            f"train≤{pd.Timestamp(fold.train_end).date()} "
            f"val={pd.Timestamp(fold.val_start).date()}→{pd.Timestamp(fold.val_end).date()}",
            flush=True,
        )
        if out_dir is not None:
            stem = f"preds_{tag}_h{horizon}_fold{fold.fold_id}" if tag else f"preds_h{horizon}_fold{fold.fold_id}"
            dest = Path(out_dir) / f"{stem}.parquet"
            if dest.exists():
                pred_df = pd.read_parquet(dest)
                all_preds.append(pred_df)
                metas.append({"fold_id": fold.fold_id, "status": "cached", "horizon": horizon})
                print(f"[HB] fold {fold.fold_id} cached {dest.name}", flush=True)
                continue
        pred_df, meta = fit_predict_fold(
            df, fold, feature_cols=feature_cols, early_stop=early_stop, ycol=ycol
        )
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            if out_dir is not None:
                stem = f"preds_{tag}_h{horizon}_fold{fold.fold_id}" if tag else f"preds_h{horizon}_fold{fold.fold_id}"
                pred_df.to_parquet(Path(out_dir) / f"{stem}.parquet", index=False)
                if commit_fn is not None:
                    commit_fn()
        print(
            f"[HB] fold {fold.fold_id} status={meta.get('status')} "
            f"pdauc={meta.get('pdauc_oos')} auc_oos={meta.get('auc_oos')} "
            f"best_iter={meta.get('best_iteration')} elapsed={meta.get('elapsed'):.1f}s",
            flush=True,
        )
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if out_dir is not None:
        meta_name = f"fold_meta_{tag}_h{horizon}.json" if tag else f"fold_meta_h{horizon}.json"
        (out_dir / meta_name).write_text(json.dumps(metas, indent=2, default=str))
    return preds, metas, folds


def merge_twin_preds(top: pd.DataFrame, bot: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Join calibrated heads; spread = p_top − p_bottom."""
    y_top = f"y_h{horizon}"
    y_bot = f"y_bot_h{horizon}"
    excol = f"excess_h{horizon}"
    t = top.rename(columns={"p": "p_top", "p_raw": "p_top_raw"})
    if y_top in t.columns:
        t = t.rename(columns={y_top: "y_top"})
    b = bot.rename(columns={"p": "p_bot", "p_raw": "p_bot_raw"})
    if y_bot in b.columns:
        b = b.rename(columns={y_bot: "y_bot"})
    keep_b = ["date", "id", "fold_id", "p_bot", "p_bot_raw"]
    if "y_bot" in b.columns:
        keep_b.append("y_bot")
    m = t.merge(b[keep_b], on=["date", "id", "fold_id"], how="inner")
    m["spread"] = m["p_top"].astype(float) - m["p_bot"].astype(float)
    m["spread_raw"] = m["p_top_raw"].astype(float) - m["p_bot_raw"].astype(float)
    m["uncertainty"] = m["p_top"].astype(float) + m["p_bot"].astype(float)
    m["horizon"] = int(horizon)
    return m


def _group_sizes(dates) -> list[int]:
    dates = pd.to_datetime(np.asarray(dates), utc=True)
    if hasattr(dates, "tz_convert"):
        dates = dates.tz_convert("UTC").normalize()
    sizes = []
    i, n = 0, len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        sizes.append(int(j - i))
        i = j
    return sizes


def fit_predict_rank_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int = SEED,
    feature_cols: list[str] | None = None,
    inner_holdout_days: int = INNER_HOLDOUT_CALENDAR_DAYS,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    lgbm_params: dict | None = None,
    ycol: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """LightGBM LambdaRank on per-date groups. Raw scores; no isotonic calibration."""
    seed_everything(seed + fold.fold_id + (0 if shuffle_seed is None else int(shuffle_seed)))
    ycol = ycol or f"y_rank_h{fold.horizon}"
    feats = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS_V1)
    cfg = dict(LGBM_RANK)
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

    inner_tr = inner_tr.sort_values(["date", "id"]).reset_index(drop=True)
    inner_ho = inner_ho.sort_values(["date", "id"]).reset_index(drop=True)
    y_tr = inner_tr[ycol].astype(int)
    y_ho = inner_ho[ycol].astype(int)
    g_tr = _group_sizes(inner_tr["date"].to_numpy())
    g_ho = _group_sizes(inner_ho["date"].to_numpy())
    if int(sum(g_tr)) != len(inner_tr) or int(sum(g_ho)) != len(inner_ho):
        return pd.DataFrame(), {
            "fold_id": fold.fold_id,
            "status": "group_mismatch",
            "elapsed": time.time() - t0,
        }

    dtrain = lgb.Dataset(inner_tr[feats], label=y_tr, group=g_tr, free_raw_data=False)
    dvalid = lgb.Dataset(inner_ho[feats], label=y_ho, group=g_ho, reference=dtrain, free_raw_data=False)

    ndcg_at = list(cfg.get("ndcg_eval_at", (10,)))
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": ndcg_at,
        "eval_at": ndcg_at,
        "lambdarank_truncation_level": int(cfg.get("lambdarank_truncation_level", 10)),
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

    raw_val = booster.predict(valid[feats], num_iteration=best_iteration)
    pred_df = valid[["date", "id", "symbol"]].copy()
    pred_df["p_raw"] = raw_val
    pred_df["p"] = raw_val
    pred_df["horizon"] = fold.horizon
    pred_df["fold_id"] = fold.fold_id
    pred_df[ycol] = valid[ycol].to_numpy()
    excol = f"excess_h{fold.horizon}"
    if excol in valid.columns:
        pred_df[excol] = valid[excol].to_numpy()

    gain = booster.feature_importance(importance_type="gain")
    gain_map = {f: float(g) for f, g in zip(feats, gain)}
    ric = float("nan")
    if excol in valid.columns:
        ric = mean_per_date_rank_ic(raw_val, valid[excol].to_numpy(), valid["date"].to_numpy())
    ndcg_ho = float("nan")
    try:
        ho_hist = (evals_result.get("inner_ho") or {}).get("ndcg@10") or (evals_result.get("inner_ho") or {}).get("ndcg")
        if ho_hist:
            ndcg_ho = float(ho_hist[best_iteration - 1] if best_iteration - 1 < len(ho_hist) else ho_hist[-1])
    except Exception:
        ndcg_ho = float("nan")

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
        "ndcg_holdout": ndcg_ho,
        "rankic_oos": ric,
        "feature_importance_gain": gain_map,
        "feature_cols": feats,
        "ycol": ycol,
        "shuffle_labels": bool(shuffle_labels),
        "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
        "calibrated": False,
        "horizon": fold.horizon,
        "objective": "lambdarank",
    }
    return pred_df, meta


def train_all_rank_folds(
    df: pd.DataFrame,
    horizon: int,
    out_dir=None,
    feature_cols: list[str] | None = None,
    ycol: str | None = None,
    tag: str | None = None,
    commit_fn=None,
) -> tuple[pd.DataFrame, list[dict], list[FoldSpec]]:
    folds = make_expanding_folds(pd.DatetimeIndex(df["date"].unique()), horizon=horizon)
    print(
        f"[HB] RANK h={horizon} folds={len(folds)} ycol={ycol or f'y_rank_h{horizon}'} tag={tag}",
        flush=True,
    )
    all_preds = []
    metas = []
    for fold in folds:
        print(
            f"[HB] rank fold {fold.fold_id+1}/{len(folds)} h={horizon} tag={tag} "
            f"train≤{pd.Timestamp(fold.train_end).date()} "
            f"val={pd.Timestamp(fold.val_start).date()}→{pd.Timestamp(fold.val_end).date()}",
            flush=True,
        )
        if out_dir is not None:
            stem = f"preds_{tag}_h{horizon}_fold{fold.fold_id}" if tag else f"preds_rank_h{horizon}_fold{fold.fold_id}"
            dest = Path(out_dir) / f"{stem}.parquet"
            if dest.exists():
                pred_df = pd.read_parquet(dest)
                all_preds.append(pred_df)
                metas.append({"fold_id": fold.fold_id, "status": "cached", "horizon": horizon})
                print(f"[HB] rank fold {fold.fold_id} cached {dest.name}", flush=True)
                continue
        pred_df, meta = fit_predict_rank_fold(df, fold, feature_cols=feature_cols, ycol=ycol)
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            if out_dir is not None:
                stem = f"preds_{tag}_h{horizon}_fold{fold.fold_id}" if tag else f"preds_rank_h{horizon}_fold{fold.fold_id}"
                pred_df.to_parquet(Path(out_dir) / f"{stem}.parquet", index=False)
                if commit_fn is not None:
                    commit_fn()
        print(
            f"[HB] rank fold {fold.fold_id} status={meta.get('status')} "
            f"rankic={meta.get('rankic_oos')} best_iter={meta.get('best_iteration')} "
            f"elapsed={meta.get('elapsed'):.1f}s",
            flush=True,
        )
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if out_dir is not None:
        meta_name = f"fold_meta_{tag}_h{horizon}.json" if tag else f"fold_meta_rank_h{horizon}.json"
        (out_dir / meta_name).write_text(json.dumps(metas, indent=2, default=str))
    return preds, metas, folds


def mean_gain(metas: list[dict], top_n: int = 15) -> list[tuple[str, float]]:
    acc: dict[str, list[float]] = {}
    for m in metas:
        gi = m.get("feature_importance_gain") or {}
        for k, v in gi.items():
            acc.setdefault(k, []).append(float(v))
    ranked = sorted(((k, float(np.mean(vs))) for k, vs in acc.items()), key=lambda kv: -kv[1])
    return ranked[:top_n]
