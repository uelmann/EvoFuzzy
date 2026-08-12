"""Phase B ablation: Model A (frozen A0) vs Model B (A0 + Kronos features)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, evaluate_predictions, newey_west_t, summarize_ic
from baseline.features import FEATURE_COLS
from baseline.model import FoldSpec, _fit_predict_fold, make_folds
from baseline.seedutil import seed_everything
from phase_b.kronos_features import KRONOS_FEATURE_COLS

CUTOFF = pd.Timestamp("2025-08-17", tz="UTC")

KILL_CRITERION = (
    "Kronos features are KEPT only if post-cutoff top-20 ΔRankIC ≥ +0.005 at h=7 or h=10 "
    "AND Δ is positive in ≥60% of post-cutoff folds. Otherwise the verdict is KILL."
)


def _window_mask(dates: pd.Series, window: str) -> pd.Series:
    d = pd.to_datetime(dates, utc=True)
    if window == "pre":
        return d < CUTOFF
    if window == "post":
        return d >= CUTOFF
    return pd.Series(True, index=dates.index)


def assign_fold_id(dates: pd.Series, folds: list[FoldSpec]) -> pd.Series:
    d = pd.to_datetime(dates, utc=True)
    out = pd.Series(-1, index=dates.index, dtype=int)
    for fr in folds:
        m = (d >= fr.val_start) & (d <= fr.val_end)
        out.loc[m] = fr.fold_id
    return out


def merge_kronos_features(
    feat: pd.DataFrame,
    kronos: pd.DataFrame,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Join Kronos raw features and CS-zscore them (same protocol as price features)."""
    f = feat.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    k = kronos.copy()
    k["date"] = pd.to_datetime(k["date"], utc=True)
    cols = ["date", "symbol"] + [c for c in KRONOS_FEATURE_COLS if c in k.columns]
    f = f.merge(k[cols], on=["date", "symbol"], how="left")
    for c in KRONOS_FEATURE_COLS:
        if c not in f.columns:
            f[c] = np.nan
        f[f"{c}_raw"] = f[c].astype(float)

        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return ((s - mu) / sd).clip(-clip, clip)

        f[c] = f.groupby("date", sort=False)[c].transform(_z)
    return f


def evaluate_split(
    pred: pd.DataFrame,
    horizon: int,
    universe: pd.DataFrame,
    label: str,
    window: str,
) -> dict:
    df = pred.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.loc[_window_mask(df["date"], window)].copy()
    if df.empty:
        return {
            "universe": label,
            "horizon": horizon,
            "window": window,
            "n_days": 0,
            "mean_ic": float("nan"),
            "icir": float("nan"),
            "nw_tstat": float("nan"),
        }
    ev = evaluate_predictions(df, horizon, universe=universe, label=label)
    out = {k: v for k, v in ev.items() if k != "ic_series"}
    out["window"] = window
    out["ic_series"] = ev["ic_series"]
    return out


def fold_delta_stats(
    ic_a: pd.Series,
    ic_b: pd.Series,
    folds: list[FoldSpec],
    window: str,
) -> dict:
    """Per-fold mean IC Δ and fraction of folds with positive Δ."""
    if ic_a.empty or ic_b.empty:
        return {"n_folds": 0, "frac_positive": float("nan"), "per_fold": []}
    delta = (ic_b - ic_a).dropna()
    if window == "pre":
        delta = delta[delta.index < CUTOFF]
    elif window == "post":
        delta = delta[delta.index >= CUTOFF]
    per = []
    for fr in folds:
        seg = delta[(delta.index >= fr.val_start) & (delta.index <= fr.val_end)]
        if window == "post":
            seg = seg[seg.index >= CUTOFF]
        elif window == "pre":
            seg = seg[seg.index < CUTOFF]
        if len(seg) < 3:
            continue
        dmean = float(seg.mean())
        per.append({"fold_id": fr.fold_id, "delta_mean_ic": dmean, "n_days": int(len(seg))})
    if not per:
        return {"n_folds": 0, "frac_positive": float("nan"), "per_fold": []}
    pos = sum(1 for r in per if r["delta_mean_ic"] > 0)
    return {
        "n_folds": len(per),
        "frac_positive": float(pos / len(per)),
        "n_positive": int(pos),
        "per_fold": per,
    }


def paired_nw_delta(ic_a: pd.Series, ic_b: pd.Series, horizon: int, window: str) -> dict:
    delta = (ic_b - ic_a).dropna()
    if window == "pre":
        delta = delta[delta.index < CUTOFF]
    elif window == "post":
        delta = delta[delta.index >= CUTOFF]
    vals = delta.values.astype(float)
    return {
        "window": window,
        "n_days": int(len(vals)),
        "mean_delta_ic": float(np.mean(vals)) if len(vals) else float("nan"),
        "nw_tstat": newey_west_t(vals, lag=horizon) if len(vals) else float("nan"),
    }


def aggregate_kronos_importance(metas: list[dict]) -> dict:
    acc: dict[str, list[float]] = {c: [] for c in KRONOS_FEATURE_COLS}
    for m in metas:
        gi = m.get("feature_importance_gain") or {}
        for c in KRONOS_FEATURE_COLS:
            if c in gi:
                acc[c].append(float(gi[c]))
    out = {}
    for c, vals in acc.items():
        out[c] = {
            "mean_gain": float(np.mean(vals)) if vals else 0.0,
            "median_gain": float(np.median(vals)) if vals else 0.0,
            "n_folds": len(vals),
        }
    return out


def apply_kill_criterion(results_by_h: dict) -> dict:
    """
    Pre-registered:
    KEEP iff post-cutoff top-20 ΔRankIC ≥ +0.005 at h=7 OR h=10
    AND Δ positive in ≥60% of post-cutoff folds (for that horizon).
    """
    keep_reasons = []
    details = {}
    for h in (7, 10):
        r = results_by_h.get(h, {})
        d_ic = float(r.get("delta_top20_post", float("nan")))
        frac = float(r.get("frac_pos_folds_post", float("nan")))
        ok = np.isfinite(d_ic) and np.isfinite(frac) and (d_ic >= 0.005) and (frac >= 0.60)
        details[f"h{h}"] = {
            "delta_top20_post": d_ic,
            "frac_pos_folds_post": frac,
            "passes": bool(ok),
        }
        if ok:
            keep_reasons.append(f"h={h}")
    verdict = "KEEP" if keep_reasons else "KILL"
    return {
        "verdict": verdict,
        "criterion": KILL_CRITERION,
        "keep_reasons": keep_reasons,
        "details": details,
    }


def train_model_b_folds(
    feat_b: pd.DataFrame,
    horizon: int,
    cfg: dict,
    out_dir: Path,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    folds = make_folds(
        pd.DatetimeIndex(feat_b["date"].unique()),
        horizon=horizon,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    # Match A0 h=10 fixed-500 fallback if that was used historically
    model_cfg = dict(cfg["model"])
    if horizon == 10 and model_cfg.get("fixed_n_estimators") is None:
        # A0 used fixed 500 for h=10 when rank-ic early-stop failed; keep frozen behavior
        model_cfg["fixed_n_estimators"] = 500
        model_cfg["early_stop_metric"] = "none"

    all_preds = []
    metas = []
    for fold in folds:
        print(
            f"[ablation B] fold {fold.fold_id+1}/{len(folds)} h={horizon} "
            f"val={fold.val_start.date()}→{fold.val_end.date()}",
            flush=True,
        )
        pred_df, meta = _fit_predict_fold(
            feat_b,
            fold,
            seed=cfg["seed"],
            model_cfg=model_cfg,
            inner_holdout_days=cfg["cv"]["inner_holdout_days"],
            feature_cols=feature_cols,
            model_name="lgbm_a0_plus_kronos",
        )
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            pred_df.to_parquet(out_dir / f"preds_h{horizon}_fold{fold.fold_id}.parquet", index=False)
        print(
            f"[ablation B] fold {fold.fold_id} status={meta['status']} "
            f"best_iter={meta.get('best_iteration')} elapsed={meta.get('elapsed'):.1f}s",
            flush=True,
        )
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if not preds.empty:
        preds = preds.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
            ["date", "symbol"], keep="first"
        )
        preds.to_parquet(out_dir / f"lgbm_a0_plus_kronos_h{horizon}.parquet", index=False)
    (out_dir / f"fold_meta_h{horizon}.json").write_text(json.dumps(metas, indent=2, default=str))
    return preds, metas


def run_ablation_for_horizon(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    feat: pd.DataFrame,
    pit120: pd.DataFrame,
    pit20: pd.DataFrame,
    horizon: int,
    folds: list[FoldSpec],
    metas_b: list[dict],
) -> dict:
    ycol = f"y_h{horizon}"
    a = pred_a.copy()
    b = pred_b.copy()
    a["date"] = pd.to_datetime(a["date"], utc=True)
    b["date"] = pd.to_datetime(b["date"], utc=True)
    if ycol not in a.columns:
        a = a.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    if ycol not in b.columns:
        b = b.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")

    tables = []
    ic_store = {}
    for uni_name, uni in [("top20", pit20), ("pit120", pit120)]:
        for window in ("full", "pre", "post"):
            eva = evaluate_split(a, horizon, uni, uni_name, window)
            evb = evaluate_split(b, horizon, uni, uni_name, window)
            tables.append(
                {
                    "horizon": horizon,
                    "universe": uni_name,
                    "window": window,
                    "A_mean_ic": eva.get("mean_ic"),
                    "B_mean_ic": evb.get("mean_ic"),
                    "delta_ic": (
                        float(evb["mean_ic"] - eva["mean_ic"])
                        if np.isfinite(eva.get("mean_ic", np.nan))
                        and np.isfinite(evb.get("mean_ic", np.nan))
                        else float("nan")
                    ),
                    "A_nw_t": eva.get("nw_tstat"),
                    "B_nw_t": evb.get("nw_tstat"),
                    "n_days": evb.get("n_days"),
                }
            )
            if uni_name == "top20":
                ic_store[f"A_{window}"] = eva.get("ic_series", pd.Series(dtype=float))
                ic_store[f"B_{window}"] = evb.get("ic_series", pd.Series(dtype=float))

    paired = {
        w: paired_nw_delta(ic_store.get(f"A_{w}", pd.Series(dtype=float)), ic_store.get(f"B_{w}", pd.Series(dtype=float)), horizon, w)
        for w in ("full", "pre", "post")
    }
    fold_stats = {
        w: fold_delta_stats(
            ic_store.get("A_full", pd.Series(dtype=float)),
            ic_store.get("B_full", pd.Series(dtype=float)),
            folds,
            w,
        )
        for w in ("full", "pre", "post")
    }
    # top20 post delta
    top20_rows = [t for t in tables if t["universe"] == "top20"]
    delta_post = next((t["delta_ic"] for t in top20_rows if t["window"] == "post"), float("nan"))
    delta_pre = next((t["delta_ic"] for t in top20_rows if t["window"] == "pre"), float("nan"))
    delta_full = next((t["delta_ic"] for t in top20_rows if t["window"] == "full"), float("nan"))

    # daily delta series for charts
    ic_a = ic_store.get("A_full", pd.Series(dtype=float))
    ic_b = ic_store.get("B_full", pd.Series(dtype=float))
    delta_daily = (ic_b - ic_a).dropna().sort_index()

    return {
        "horizon": horizon,
        "tables": tables,
        "paired_nw": paired,
        "fold_stats": fold_stats,
        "delta_top20_full": delta_full,
        "delta_top20_pre": delta_pre,
        "delta_top20_post": delta_post,
        "frac_pos_folds_post": fold_stats["post"].get("frac_positive"),
        "kronos_importance": aggregate_kronos_importance(metas_b),
        "delta_daily_ic": delta_daily,
        "ic_a_full": ic_a,
        "ic_b_full": ic_b,
    }
