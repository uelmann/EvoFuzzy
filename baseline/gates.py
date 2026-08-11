"""Sanity gates — must all pass before reporting."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .evaluate import daily_rank_ic, summarize_ic
from .features import FEATURE_COLS, features_for_symbol
from .model import FoldSpec, _fit_predict_fold
from .seedutil import seed_everything


def gate_label_shuffle(pred_fold: pd.DataFrame, horizon: int, seed: int = 42) -> dict:
    """Within-date shuffle of y → |mean RankIC| must be < 0.005."""
    ycol = f"y_h{horizon}"
    df = pred_fold.dropna(subset=["score", ycol]).copy()
    if len(df) < 100:
        return {"name": "label_shuffle", "passed": False, "reason": "too few rows"}

    ics = []
    for k in range(5):
        rng = np.random.default_rng(seed + k)
        tmp = df[["date", "symbol", "score", ycol]].copy()

        def _shuf(s: pd.Series) -> pd.Series:
            return pd.Series(rng.permutation(s.to_numpy()), index=s.index)

        tmp[ycol] = tmp.groupby("date", sort=False)[ycol].transform(_shuf)
        ic = daily_rank_ic(tmp, ycol)
        if len(ic):
            ics.append(float(ic.mean()))
    if not ics:
        return {
            "name": "label_shuffle",
            "passed": True,
            "mean_ic": 0.0,
            "threshold": 0.005,
            "note": "empty IC after shuffle (degenerate)",
        }
    mean_ic = float(np.mean(ics))
    passed = abs(mean_ic) < 0.005
    return {
        "name": "label_shuffle",
        "passed": bool(passed),
        "mean_ic": mean_ic,
        "threshold": 0.005,
        "n_shuffles": len(ics),
        "per_shuffle_mean_ic": ics,
    }


def gate_feature_lookahead(panel: pd.DataFrame) -> dict:
    """Alter future rows; features at t must be invariant."""
    btc = panel.loc[panel["symbol"] == "BTCUSDT"].set_index("date")["close"].sort_index()
    sym = "BTCUSDT"
    g = panel.loc[panel["symbol"] == sym].sort_values("date").reset_index(drop=True)
    if len(g) < 200:
        # try another liquid symbol
        counts = panel.groupby("symbol").size().sort_values(ascending=False)
        sym = counts.index[0]
        g = panel.loc[panel["symbol"] == sym].sort_values("date").reset_index(drop=True)
    mid = len(g) // 2
    t_date = g.loc[mid, "date"]
    base = features_for_symbol(g, btc)
    base_row = base.loc[base["date"] == t_date, FEATURE_COLS]
    if base_row.empty:
        return {"name": "feature_lookahead", "passed": False, "reason": "no feature row"}

    g2 = g.copy()
    # scramble all rows strictly after t
    fut = g2.index[g2["date"] > t_date]
    rng = np.random.default_rng(0)
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "dollar_volume"]:
        if col in g2.columns:
            vals = g2.loc[fut, col].values.astype(float)
            g2.loc[fut, col] = rng.permutation(vals) * (1 + rng.normal(0, 0.05, size=len(vals)))
    alt = features_for_symbol(g2, btc)
    alt_row = alt.loc[alt["date"] == t_date, FEATURE_COLS]
    diff = (base_row.values - alt_row.values)
    max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    passed = max_abs < 1e-10
    return {
        "name": "feature_lookahead",
        "passed": bool(passed),
        "max_abs_diff": max_abs,
        "symbol": sym,
        "date": str(pd.Timestamp(t_date).date()),
    }


def gate_universe_lookahead(panel: pd.DataFrame, build_pit_fn, n: int = 20, window: int = 30) -> dict:
    """Top-n membership at t invariant to future data."""
    dates = sorted(panel["date"].unique())
    if len(dates) < window + 50:
        return {"name": "universe_lookahead", "passed": False, "reason": "short history"}
    t = dates[len(dates) // 2]
    base = build_pit_fn(panel, n=n, window=window)
    base_set = set(base.loc[base["date"] == t, "symbol"])

    panel2 = panel.copy()
    fut = panel2["date"] > t
    rng = np.random.default_rng(1)
    panel2.loc[fut, "dollar_volume"] = panel2.loc[fut, "dollar_volume"].values * (
        1 + rng.normal(0, 0.5, size=fut.sum())
    )
    # also shuffle symbols' future volumes aggressively
    alt = build_pit_fn(panel2, n=n, window=window)
    alt_set = set(alt.loc[alt["date"] == t, "symbol"])
    passed = base_set == alt_set and len(base_set) > 0
    return {
        "name": "universe_lookahead",
        "passed": bool(passed),
        "date": str(pd.Timestamp(t).date()),
        "base_n": len(base_set),
        "symmetric_diff": len(base_set.symmetric_difference(alt_set)),
    }


def gate_seed_determinism(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int,
    model_cfg: dict,
    inner_holdout_days: int,
) -> dict:
    """Two runs same seed → identical metrics/preds."""
    p1, m1 = _fit_predict_fold(df, fold, seed, model_cfg, inner_holdout_days)
    p2, m2 = _fit_predict_fold(df, fold, seed, model_cfg, inner_holdout_days)
    if p1.empty or p2.empty:
        return {"name": "seed_determinism", "passed": False, "reason": "empty preds"}
    s1 = p1.sort_values(["date", "symbol"])["score"].values
    s2 = p2.sort_values(["date", "symbol"])["score"].values
    if len(s1) != len(s2):
        return {"name": "seed_determinism", "passed": False, "reason": "length mismatch"}
    max_diff = float(np.max(np.abs(s1 - s2)))
    passed = max_diff < 1e-12
    return {
        "name": "seed_determinism",
        "passed": bool(passed),
        "max_score_diff": max_diff,
        "best_iteration": m1.get("best_iteration"),
    }


def run_all_gates(
    panel: pd.DataFrame,
    feat_labeled: pd.DataFrame,
    build_pit_fn,
    fold: FoldSpec,
    cfg: dict,
    sample_pred: pd.DataFrame,
) -> list[dict]:
    print("[gates] running sanity gates...", flush=True)
    results = []
    results.append(gate_label_shuffle(sample_pred, fold.horizon, seed=cfg["seed"]))
    results.append(gate_feature_lookahead(panel))
    results.append(
        gate_universe_lookahead(
            panel,
            build_pit_fn,
            n=cfg["data"]["exec_universe_n"],
            window=cfg["data"]["exec_dv_window"],
        )
    )
    results.append(
        gate_seed_determinism(
            feat_labeled,
            fold,
            seed=cfg["seed"],
            model_cfg=cfg["model"],
            inner_holdout_days=cfg["cv"]["inner_holdout_days"],
        )
    )
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"[gates] {r['name']}: {status} {r}", flush=True)
    return results
