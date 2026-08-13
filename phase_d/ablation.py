"""Phase D ablation: A0 vs A0+microstructure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, evaluate_predictions, newey_west_t
from baseline.features import FEATURE_COLS
from baseline.model import _fit_predict_fold, make_folds
from baseline.portfolio import run_tranche_portfolio
from baseline.seedutil import seed_everything
from phase_d.micro_data import MICRO_FEATURE_COLS

KEEP_CRITERION = (
    "The microstructure block is KEPT only if trailing-18-month top-20 ΔRankIC ≥ +0.005 "
    "at h=7 or h=10 AND full-OOS ΔRankIC ≥ 0 AND Δ is positive in ≥60% of trailing-18-month folds. "
    "Otherwise verdict = KILL."
)


def merge_micro(feat: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True)
    m = micro.copy()
    m["date"] = pd.to_datetime(m["date"], utc=True)
    cols = ["date", "symbol"] + [c for c in MICRO_FEATURE_COLS if c in m.columns]
    return f.merge(m[cols], on=["date", "symbol"], how="left")


def _window_mask(dates: pd.Series, window: str, end: pd.Timestamp | None = None) -> pd.Series:
    d = pd.to_datetime(dates, utc=True)
    if window == "full":
        return pd.Series(True, index=dates.index)
    if window == "trail18m":
        end = end or d.max()
        start = end - pd.Timedelta(days=365 * 1.5)
        return (d >= start) & (d <= end)
    if window.startswith("y"):
        y = int(window[1:])
        return d.dt.year == y
    raise ValueError(window)


def paired_delta_ic(ic_a: pd.Series, ic_b: pd.Series, horizon: int) -> dict:
    delta = (ic_b - ic_a).dropna()
    vals = delta.values.astype(float)
    return {
        "n_days": int(len(vals)),
        "mean_delta_ic": float(np.mean(vals)) if len(vals) else float("nan"),
        "nw_tstat": newey_west_t(vals, lag=horizon) if len(vals) else float("nan"),
    }


def fold_frac_positive(ic_a: pd.Series, ic_b: pd.Series, folds, date_mask: pd.Series | None = None) -> dict:
    delta = (ic_b - ic_a).dropna()
    if date_mask is not None:
        # date_mask indexed like original dates — align
        pass
    per = []
    for fr in folds:
        seg = delta[(delta.index >= fr.val_start) & (delta.index <= fr.val_end)]
        if date_mask is not None and len(seg):
            # keep only dates in trail window
            allowed = set(pd.to_datetime(date_mask.index[date_mask], utc=True))
            seg = seg[[d for d in seg.index if d in allowed]]
        if len(seg) < 3:
            continue
        per.append({"fold_id": fr.fold_id, "delta": float(seg.mean()), "n": int(len(seg))})
    if not per:
        return {"n_folds": 0, "frac_positive": float("nan"), "per_fold": []}
    pos = sum(1 for r in per if r["delta"] > 0)
    return {"n_folds": len(per), "frac_positive": float(pos / len(per)), "n_positive": pos, "per_fold": per}


def apply_keep_criterion(results_by_h: dict) -> dict:
    keep_reasons = []
    details = {}
    for h, r in results_by_h.items():
        d18 = float(r.get("delta_top20_trail18m", float("nan")))
        dfull = float(r.get("delta_top20_full", float("nan")))
        frac = float(r.get("frac_pos_folds_trail18m", float("nan")))
        ok = (
            np.isfinite(d18)
            and np.isfinite(dfull)
            and np.isfinite(frac)
            and d18 >= 0.005
            and dfull >= 0.0
            and frac >= 0.60
        )
        details[f"h{h}"] = {"delta_trail18m": d18, "delta_full": dfull, "frac_pos_folds_trail18m": frac, "passes": bool(ok)}
        if ok:
            keep_reasons.append(f"h={h}")
    return {
        "verdict": "KEEP" if keep_reasons else "KILL",
        "criterion": KEEP_CRITERION,
        "keep_reasons": keep_reasons,
        "details": details,
    }


def train_model_d(
    feat_d: pd.DataFrame,
    horizon: int,
    cfg: dict,
    out_dir: Path,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    folds = make_folds(
        pd.DatetimeIndex(feat_d["date"].unique()),
        horizon=horizon,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    model_cfg = dict(cfg["model"])
    if horizon == 10:
        model_cfg["fixed_n_estimators"] = 500
        model_cfg["early_stop_metric"] = "none"
    all_preds, metas = [], []
    for fold in folds:
        print(
            f"[ablation D] fold {fold.fold_id+1}/{len(folds)} h={horizon} "
            f"val={fold.val_start.date()}→{fold.val_end.date()}",
            flush=True,
        )
        pred_df, meta = _fit_predict_fold(
            feat_d,
            fold,
            seed=cfg["seed"],
            model_cfg=model_cfg,
            inner_holdout_days=cfg["cv"]["inner_holdout_days"],
            feature_cols=feature_cols,
            model_name="lgbm_a0_plus_micro",
        )
        metas.append(meta)
        if not pred_df.empty:
            all_preds.append(pred_df)
            pred_df.to_parquet(out_dir / f"preds_h{horizon}_fold{fold.fold_id}.parquet", index=False)
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if not preds.empty:
        preds = preds.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(["date", "symbol"], keep="first")
        preds.to_parquet(out_dir / f"lgbm_a0_plus_micro_h{horizon}.parquet", index=False)
    (out_dir / f"fold_meta_h{horizon}.json").write_text(json.dumps(metas, indent=2, default=str))
    return preds, metas


def aggregate_micro_importance(metas: list[dict]) -> dict:
    acc = {c: [] for c in MICRO_FEATURE_COLS}
    for m in metas:
        gi = m.get("feature_importance_gain") or {}
        for c in MICRO_FEATURE_COLS:
            if c in gi:
                acc[c].append(float(gi[c]))
    return {
        c: {"mean_gain": float(np.mean(v)) if v else 0.0, "median_gain": float(np.median(v)) if v else 0.0, "n": len(v)}
        for c, v in acc.items()
    }


def evaluate_ablation_horizon(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    feat: pd.DataFrame,
    pit20: pd.DataFrame,
    pit120: pd.DataFrame,
    panel: pd.DataFrame,
    funding: pd.DataFrame | None,
    horizon: int,
    folds,
    metas_b: list[dict],
    cfg: dict,
    coverage_by_date: pd.Series | None = None,
) -> dict:
    ycol = f"y_h{horizon}"
    a = pred_a.copy()
    b = pred_b.copy()
    if a.empty or b.empty:
        return {
            "horizon": horizon,
            "tables": [],
            "paired_nw": {},
            "fold_stats": {},
            "coverage_conditional_delta": {},
            "sharpe_delta": {},
            "delta_top20_full": float("nan"),
            "delta_top20_trail18m": float("nan"),
            "frac_pos_folds_trail18m": float("nan"),
            "micro_importance": aggregate_micro_importance(metas_b),
            "delta_daily_ic": pd.Series(dtype=float),
            "error": "empty_predictions",
        }
    a["date"] = pd.to_datetime(a["date"], utc=True)
    b["date"] = pd.to_datetime(b["date"], utc=True)
    if ycol not in a.columns:
        a = a.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    if ycol not in b.columns:
        b = b.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")

    end = max(a["date"].max(), b["date"].max())
    tables = []
    ic_store = {}
    years = sorted(set(a["date"].dt.year.unique()) | set(b["date"].dt.year.unique()))
    windows = ["full", "trail18m"] + [f"y{y}" for y in years if y >= 2022]

    for uni_name, uni in [("top20", pit20), ("pit120", pit120)]:
        for window in windows:
            ma = _window_mask(a["date"], window, end=end)
            mb = _window_mask(b["date"], window, end=end)
            aa = a.loc[ma].copy()
            bb = b.loc[mb].copy()
            if aa.empty or bb.empty:
                tables.append(
                    {
                        "horizon": horizon,
                        "universe": uni_name,
                        "window": window,
                        "A_ic": float("nan"),
                        "B_ic": float("nan"),
                        "delta_ic": float("nan"),
                        "n_days": 0,
                    }
                )
                continue
            eva = evaluate_predictions(aa, horizon, universe=uni, label=uni_name)
            evb = evaluate_predictions(bb, horizon, universe=uni, label=uni_name)
            tables.append(
                {
                    "horizon": horizon,
                    "universe": uni_name,
                    "window": window,
                    "A_ic": eva.get("mean_ic"),
                    "B_ic": evb.get("mean_ic"),
                    "delta_ic": float(evb["mean_ic"] - eva["mean_ic"])
                    if np.isfinite(eva.get("mean_ic", np.nan)) and np.isfinite(evb.get("mean_ic", np.nan))
                    else float("nan"),
                    "A_nw": eva.get("nw_tstat"),
                    "B_nw": evb.get("nw_tstat"),
                    "n_days": evb.get("n_days"),
                }
            )
            if uni_name == "top20" and window in ("full", "trail18m"):
                ic_store[f"A_{window}"] = eva.get("ic_series", pd.Series(dtype=float))
                ic_store[f"B_{window}"] = evb.get("ic_series", pd.Series(dtype=float))

    paired = {
        w: paired_delta_ic(ic_store.get(f"A_{w}", pd.Series(dtype=float)), ic_store.get(f"B_{w}", pd.Series(dtype=float)), horizon)
        for w in ("full", "trail18m")
    }

    # trail18m fold frac
    trail_dates = a.loc[_window_mask(a["date"], "trail18m", end=end), "date"]
    trail_mask = pd.Series(True, index=pd.DatetimeIndex(pd.to_datetime(trail_dates.unique(), utc=True)))
    fold_stats = {
        "trail18m": fold_frac_positive(
            ic_store.get("A_full", pd.Series(dtype=float)),
            ic_store.get("B_full", pd.Series(dtype=float)),
            folds,
            date_mask=trail_mask,
        ),
        "full": fold_frac_positive(
            ic_store.get("A_full", pd.Series(dtype=float)),
            ic_store.get("B_full", pd.Series(dtype=float)),
            folds,
            date_mask=None,
        ),
    }

    # coverage-conditional ΔIC (≥80% book coverage)
    cov_delta = {}
    if coverage_by_date is not None and not coverage_by_date.empty:
        cov = coverage_by_date.copy()
        cov.index = pd.to_datetime(cov.index, utc=True)
        good = cov[cov >= 0.80].index
        for w in ("full", "trail18m"):
            ia = ic_store.get(f"A_{w}", pd.Series(dtype=float))
            ib = ic_store.get(f"B_{w}", pd.Series(dtype=float))
            if ia.empty or ib.empty:
                cov_delta[w] = {"mean_delta_ic": float("nan"), "n_days": 0}
                continue
            delta = (ib - ia).dropna()
            delta = delta[delta.index.isin(good)]
            if w == "trail18m":
                start = end - pd.Timedelta(days=365 * 1.5)
                delta = delta[(delta.index >= start) & (delta.index <= end)]
            cov_delta[w] = {
                "mean_delta_ic": float(delta.mean()) if len(delta) else float("nan"),
                "n_days": int(len(delta)),
                "nw_tstat": newey_west_t(delta.values.astype(float), lag=horizon) if len(delta) else float("nan"),
            }

    # Sharpe delta median-τ tranche
    port = cfg["portfolio"]
    sharpe_rows = []
    for label, pdf in [("A", a), ("B", b)]:
        tres = run_tranche_portfolio(
            pdf,
            panel,
            feat,
            pit20,
            horizon=horizon,
            tau_pct=60.0,
            exit_hysteresis=port.get("exit_hysteresis", 0.6),
            gross_limit=port.get("gross_limit", 1.0),
            fee_bps=port.get("taker_fee_bps", 5.0),
            slip_bps=port.get("slippage_bps", 3.0),
            lag=0,
            apply_funding=True,
            funding=funding,
        )
        sharpe_rows.append({"model": label, "net_sharpe": tres.get("net_sharpe"), "equity": tres.get("equity"), "daily_ret": tres.get("daily_ret")})
    # paired day sharpe on intersection
    ra = sharpe_rows[0]["daily_ret"]
    rb = sharpe_rows[1]["daily_ret"]
    if isinstance(ra, pd.Series) and isinstance(rb, pd.Series):
        idx = ra.index.intersection(rb.index)
        def _sh(x):
            return float(x.mean() / x.std() * np.sqrt(365)) if len(x) and x.std() > 0 else float("nan")
        sharpe_delta = {
            "A_sharpe": _sh(ra.loc[idx]),
            "B_sharpe": _sh(rb.loc[idx]),
            "delta_sharpe": _sh(rb.loc[idx]) - _sh(ra.loc[idx]) if len(idx) else float("nan"),
            "n_days": int(len(idx)),
        }
    else:
        sharpe_delta = {"A_sharpe": sharpe_rows[0]["net_sharpe"], "B_sharpe": sharpe_rows[1]["net_sharpe"], "delta_sharpe": float("nan")}

    top20 = [t for t in tables if t["universe"] == "top20"]
    d_full = next((t["delta_ic"] for t in top20 if t["window"] == "full"), float("nan"))
    d_18 = next((t["delta_ic"] for t in top20 if t["window"] == "trail18m"), float("nan"))

    ic_a = ic_store.get("A_full", pd.Series(dtype=float))
    ic_b = ic_store.get("B_full", pd.Series(dtype=float))
    delta_daily = (ic_b - ic_a).dropna().sort_index()

    return {
        "horizon": horizon,
        "tables": tables,
        "paired_nw": paired,
        "fold_stats": fold_stats,
        "coverage_conditional_delta": cov_delta,
        "sharpe_delta": sharpe_delta,
        "delta_top20_full": d_full,
        "delta_top20_trail18m": d_18,
        "frac_pos_folds_trail18m": fold_stats["trail18m"].get("frac_positive"),
        "micro_importance": aggregate_micro_importance(metas_b),
        "delta_daily_ic": delta_daily,
    }
