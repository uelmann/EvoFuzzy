"""Causal static/dynamic decomposition of existing OOS A0 scores.

No Stage A. No pre-reg edit. Re-fits each frozen A0 fold (same seed/HP) only
to score the TRAIN window; val scores are checked against the stored OOS parquet.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("GATING_LGB_LOG_PERIOD", "25")

from baseline.data import load_funding_panel, load_panel
from baseline.evaluate import daily_rank_ic, summarize_ic
from baseline.features import FEATURE_COLS
from baseline.model import FoldSpec, _make_rank_ic_feval, make_folds
from baseline.portfolio import run_tranche_portfolio
from baseline.seedutil import seed_everything
from gating_ladder.fase1 import _book
from gating_ladder.metrics import TRAIL_DAYS, slim_portfolio, trail_mask


H = 7
YCOL = "y_h7"
ALT_KS = (0, 10, 60)
DIP_KS = (12, 15, 20, 25, 30)
MU_FEATS = [
    ("yz_vol_30_raw", "raw"),
    ("amihud_14", "cs_z_as_stored"),
    ("dollar_volume", "raw"),
    ("beta_btc_60_raw", "raw"),
    ("idio_vol_60", "cs_z_as_stored"),
]
BASELINE_NET = 1.8220094307388697
BASELINE_TRAIL = 1.0969670423963036
BASELINE_TO = 24.519270899695773


    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    res = stats.spearmanr(x, y)
    c = getattr(res, "correlation", None)
    if c is None:
        c = getattr(res, "statistic", np.nan)
    return float(np.asarray(c, dtype=float).reshape(-1)[0])


def _ic_pack(df: pd.DataFrame, ycol: str, score_col: str = "score") -> dict:
    ic = daily_rank_ic(df.dropna(subset=[score_col, ycol]), ycol, score_col=score_col)
    s = summarize_ic(ic, H)
    if len(ic):
        tr = summarize_ic(ic.iloc[trail_mask(ic.index, TRAIL_DAYS)], H)
        s["trail18m_mean_ic"] = tr.get("mean_ic")
        s["trail18m_nw_t"] = tr.get("nw_tstat")
    return s


def _shifted_ic(df: pd.DataFrame, ycol: str, k: int, score_col: str = "score") -> dict:
    tmp = df.dropna(subset=[score_col, ycol]).sort_values(["symbol", "date"]).copy()
    if k == 0:
        tmp["y_shift"] = tmp[ycol]
    else:
        tmp["y_shift"] = tmp.groupby("symbol", sort=False)[ycol].shift(-int(k))
    return summarize_ic(
        daily_rank_ic(tmp.dropna(subset=["y_shift"]), "y_shift", score_col=score_col),
        H,
    )


def _fit_fold_booster(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int,
    model_cfg: dict,
    inner_holdout_days: int,
):
    """Same A0 fit as baseline.model._fit_predict_fold; returns booster + frames."""
    seed_everything(seed + fold.fold_id)
    ycol = f"y_h{fold.horizon}"
    feats = list(FEATURE_COLS)
    train_mask = (df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)
    val_mask = (df["date"] >= fold.val_start) & (df["date"] <= fold.val_end)
    need = feats + [ycol]
    train = df.loc[train_mask].dropna(subset=need)
    valid = df.loc[val_mask].dropna(subset=need)
    if train.empty or valid.empty:
        return None, None, train, valid
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
    n_estimators = int(model_cfg.get("n_estimators", 3000))
    patience = int(model_cfg.get("early_stopping_rounds", 100))
    ho_dates = inner_ho["date"].to_numpy()
    feval = _make_rank_ic_feval(ho_dates)
    log_period = int(os.environ.get("GATING_LGB_LOG_PERIOD", "0"))
    t_boost = time.time()

    def _hb_cb(env):
        it = int(getattr(env, "iteration", 0) or 0)
        if log_period > 0 and it % log_period == 0:
            print(
                f"[lgbm] fold={fold.fold_id} iter={it} elapsed={time.time() - t_boost:.1f}s",
                flush=True,
            )

    callbacks = [
        _hb_cb,
        lgb.early_stopping(
            stopping_rounds=patience, first_metric_only=True, verbose=False, min_delta=0.0
        ),
        lgb.log_evaluation(period=log_period),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dvalid],
        valid_names=["inner_ho"],
        feval=feval,
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or 0)
    return booster, best_iteration, train, valid


def _vol_bucket_rank(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dt, g in df.groupby("date", sort=True):
        gg = g.dropna(subset=[YCOL, "yz_vol_30_raw"])
        if len(gg) < 9:
            continue
        try:
            q = pd.qcut(gg["yz_vol_30_raw"], 3, labels=False, duplicates="drop")
        except ValueError:
            continue
        tmp = gg.copy()
        tmp["bucket"] = q
        tmp["y_volrank"] = tmp.groupby("bucket")[YCOL].rank(method="average")
        parts.append(tmp[["date", "symbol", "y_volrank"]])
        if int(pd.Timestamp(dt).toordinal()) % 400 == 0:
            print(f"[diag] volrank date={pd.Timestamp(dt).date()}", flush=True)
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "y_volrank"])
    return pd.concat(parts, ignore_index=True)


def _book_metrics(res: dict) -> dict:
    slim = slim_portfolio(res)
    return {
        "net_sharpe": slim.get("net_sharpe"),
        "net_sharpe_trail18m": slim.get("net_sharpe_trail18m"),
        "ann_turnover": slim.get("ann_turnover"),
        "n_days": slim.get("n_days"),
        "n_days_trail18m": slim.get("n_days_trail18m"),
        "max_drawdown": slim.get("max_drawdown"),
        "n_flat_days": slim.get("n_flat_days"),
    }


def main() -> int:
    t0 = time.time()
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    root = Path(cfg["paths"]["volume_root"])
    feat = pd.read_parquet(root / "features" / "features_labeled_h7.parquet")
    pred = pd.read_parquet(root / "predictions" / "lgbm_price_only_h7.parquet")
    pit40 = pd.read_parquet(root / "universe" / "top40_pit.parquet")
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    pred["date"] = pd.to_datetime(pred["date"], utc=True)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)

    print("[diag] loading panel + funding", flush=True)
    symbols = sorted(set(feat["symbol"].unique()) | {"BTCUSDT"})
    panel = load_panel(root / "raw" / "klines", symbols)
    funding = load_funding_panel(root / "raw" / "funding", symbols)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)

    folds = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=H,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    print(f"[diag] n_folds={len(folds)} feat_rows={len(feat)}", flush=True)

    train_parts = []
    val_abs = []
    for fold in folds:
        print(
            f"[diag] refit fold {fold.fold_id+1}/{len(folds)} "
            f"train≤{pd.Timestamp(fold.train_end).date()} "
            f"val={pd.Timestamp(fold.val_start).date()}→{pd.Timestamp(fold.val_end).date()}",
            flush=True,
        )
        booster, best_it, train, valid = _fit_fold_booster(
            feat, fold, int(cfg["seed"]), cfg["model"], int(cfg["cv"]["inner_holdout_days"])
        )
        if booster is None:
            print(f"[diag] fold {fold.fold_id} empty", flush=True)
            continue
        feats = list(FEATURE_COLS)
        niter = best_it or -1
        tr = train[["date", "symbol"]].copy()
        tr["score"] = booster.predict(train[feats], num_iteration=niter)
        tr["fold_id"] = int(fold.fold_id)
        tr["n_train_rows_name"] = tr.groupby("symbol")["score"].transform("size")
        train_parts.append(tr)
        va = valid[["date", "symbol"]].copy()
        va["score_refit"] = booster.predict(valid[feats], num_iteration=niter)
        stored = pred[pred["fold_id"] == fold.fold_id][["date", "symbol", "score"]]
        m = va.merge(stored, on=["date", "symbol"], how="inner")
        if len(m):
            val_abs.append(float(np.nanmax(np.abs(m["score_refit"] - m["score"]))))
        print(
            f"[diag] fold {fold.fold_id} best_it={best_it} train_rows={len(tr)} "
            f"val_max_abs_diff={val_abs[-1] if val_abs else float('nan')}",
            flush=True,
        )

    train_scores = pd.concat(train_parts, ignore_index=True)
    mu = (
        train_scores.groupby(["fold_id", "symbol"], sort=False)["score"]
        .agg(mu="mean", n_train_bars="size")
        .reset_index()
    )

    oos = pred.merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
    oos = oos.merge(mu, on=["fold_id", "symbol"], how="left")
    excluded = oos[oos["mu"].isna()][["fold_id", "symbol", "date"]].copy()
    excl_names = (
        excluded.groupby(["fold_id", "symbol"], sort=False)
        .size()
        .reset_index(name="n_oos_rows")
    )
    keep = oos.dropna(subset=["mu"]).copy()
    keep["f"] = keep["score"] - keep["mu"]
    print(
        f"[diag] oos_top40={len(oos)} keep={len(keep)} excluded_rows={len(excluded)} "
        f"excluded_fold_symbol={len(excl_names)} val_max_abs={max(val_abs) if val_abs else float('nan')}",
        flush=True,
    )

    ic_s = _ic_pack(keep, YCOL, "score")
    ic_mu = _ic_pack(keep, YCOL, "mu")
    ic_f = _ic_pack(keep, YCOL, "f")
    print(f"[diag] IC score={ic_s} mu={ic_mu} f={ic_f}", flush=True)

    pred_mu = keep[["date", "symbol", "fold_id"]].copy()
    pred_mu["score"] = keep["mu"].to_numpy()
    pred_f = keep[["date", "symbol", "fold_id"]].copy()
    pred_f["score"] = keep["f"].to_numpy()

    print("[diag] book MU-ONLY", flush=True)
    book_mu = _book(
        pred_mu, panel, feat, pit40, folds, H, 70, 0, funding, 5.0, 3.0, 10.0, 8.0, 1.0
    )
    print("[diag] book F-ONLY", flush=True)
    book_f = _book(
        pred_f, panel, feat, pit40, folds, H, 70, 0, funding, 5.0, 3.0, 10.0, 8.0, 1.0
    )
    mu_m = _book_metrics(book_mu)
    f_m = _book_metrics(book_f)
    print(f"[diag] MU {mu_m}", flush=True)
    print(f"[diag] F  {f_m}", flush=True)

    # What is mu: per-fold CS Spearman of mu_i vs train-mean features.
    feat_means = []
    for fold in folds:
        mask = (feat["date"] >= fold.train_start) & (feat["date"] <= fold.train_end)
        sl = feat.loc[mask]
        if sl.empty:
            continue
        cols = ["symbol"] + [c for c, _ in MU_FEATS if c in feat.columns]
        g = sl.groupby("symbol", sort=False)[cols[1:]].mean().reset_index()
        g["fold_id"] = int(fold.fold_id)
        feat_means.append(g)
    fm = pd.concat(feat_means, ignore_index=True)
    mm = mu.merge(fm, on=["fold_id", "symbol"], how="inner")
    mu_corr_folds = []
    mu_corr_pooled = {}
    for col, kind in MU_FEATS:
        if col not in mm.columns:
            mu_corr_pooled[col] = {"kind": kind, "pooled": float("nan"), "mean_fold": float("nan")}
            continue
        per = []
        for fid, g in mm.groupby("fold_id"):
            per.append(_spearman(g["mu"], g[col]))
        mu_corr_folds.append({"feature": col, "kind": kind, "per_fold": per})
        mu_corr_pooled[col] = {
            "kind": kind,
            "pooled": _spearman(mm["mu"], mm[col]),
            "mean_fold": float(np.nanmean(per)) if per else float("nan"),
            "n_folds": int(sum(np.isfinite(per))),
            "n_pairs_pooled": int(mm[["mu", col]].dropna().shape[0]),
        }
        print(f"[diag] mu corr {col} pooled={mu_corr_pooled[col]['pooled']:.4f} "
              f"mean_fold={mu_corr_pooled[col]['mean_fold']:.4f}", flush=True)

    # Alternative labels on original OOS scores (top-40), not mu/f.
    print("[diag] alternative y", flush=True)
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    fwd = np.log(close.shift(-H) / close)
    y_raw = fwd.stack().rename("y_raw").reset_index()
    y_raw.columns = ["date", "symbol", "y_raw"]
    y_raw["date"] = pd.to_datetime(y_raw["date"], utc=True)
    y_raw["y_dm"] = y_raw.groupby("date", sort=False)["y_raw"].transform(
        lambda s: s - s.mean()
    )
    oos_y = oos.merge(y_raw, on=["date", "symbol"], how="left")
    oos_y = oos_y.merge(
        feat[["date", "symbol", "yz_vol_30_raw"]].drop_duplicates(["date", "symbol"]),
        on=["date", "symbol"],
        how="left",
    )
    volr = _vol_bucket_rank(oos_y)
    oos_y = oos_y.merge(volr, on=["date", "symbol"], how="left")

    alt_table = {}
    for name, col in [("y_h7", YCOL), ("y_raw", "y_raw"), ("y_dm", "y_dm"), ("y_volrank", "y_volrank")]:
        alt_table[name] = {}
        for k in ALT_KS:
            print(f"[diag] IC_k {name} k={k}", flush=True)
            alt_table[name][str(k)] = _shifted_ic(oos_y, col, k, "score")

    dip = {}
    for k in DIP_KS:
        print(f"[diag] dip k={k}", flush=True)
        dip[str(k)] = _shifted_ic(oos_y, YCOL, k, "score")

    excl_by_fold = {
        str(int(fid)): int(n)
        for fid, n in excl_names.groupby("fold_id").size().items()
    }
    excluded_list = [
        {
            "fold_id": int(r.fold_id),
            "symbol": r.symbol,
            "n_oos_rows": int(r.n_oos_rows),
        }
        for r in excl_names.sort_values(["fold_id", "symbol"]).itertuples()
    ]

    out = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_stage_a": True,
        "pre_reg_untouched": True,
        "mu_definition": (
            "per fold, mean of the fold model's score on that fold's TRAIN window "
            "[train_start, train_end] (purged). Names with zero train rows excluded."
        ),
        "val_score_max_abs_diff_vs_stored": float(max(val_abs) if val_abs else float("nan")),
        "n_oos_top40_rows": int(len(oos)),
        "n_keep_rows": int(len(keep)),
        "n_excluded_rows": int(len(excluded)),
        "n_excluded_fold_symbol": int(len(excl_names)),
        "excluded_fold_symbol_count_by_fold": excl_by_fold,
        "excluded_fold_symbol": excluded_list,
        "ic": {
            "score_on_keep": ic_s,
            "mu": ic_mu,
            "f": ic_f,
        },
        "books": {
            "baseline_tau70_1x": {
                "net_sharpe": BASELINE_NET,
                "net_sharpe_trail18m": BASELINE_TRAIL,
                "ann_turnover": BASELINE_TO,
            },
            "mu_only": mu_m,
            "f_only": f_m,
        },
        "mu_feature_spearman": mu_corr_pooled,
        "alt_y_ic_k": alt_table,
        "dip_ic_k": dip,
        "elapsed_sec": time.time() - t0,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/fase1_static_dyn.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("[diag] wrote results/fase1_static_dyn.json", flush=True)
    print("[diag] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
