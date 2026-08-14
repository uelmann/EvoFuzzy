"""Phase 2 gates: lookahead, PIT, seed determinism, E.1b-style label-shuffle null."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import FEATURE_COLS_V1, NULL_REPLICATES, NULL_SHUFFLE_SEEDS, PIT_DV_WINDOW, SEED
from btcb.features import btc_id_from_panel, features_for_id
from btcb.model import FoldSpec, _auc, fit_predict_fold
from btcb.universe import build_pit_topn_ids, trailing_rank_frame_by_id


def gate_feature_lookahead(panel: pd.DataFrame, btc_id: int) -> dict:
    btc_close = panel.loc[panel["id"] == btc_id].sort_values("date").set_index("date")["close"]
    btc_close.index = pd.to_datetime(btc_close.index, utc=True).tz_convert("UTC").normalize()
    counts = panel.groupby("id").size().sort_values(ascending=False)
    iid = None
    for cand in counts.index:
        if int(cand) == int(btc_id):
            continue
        g = panel.loc[panel["id"] == cand].sort_values("date")
        if len(g) >= 200:
            iid = int(cand)
            break
    if iid is None:
        return {"name": "feature_lookahead", "passed": False, "reason": "no long-history id"}
    g = panel.loc[panel["id"] == iid].sort_values("date").reset_index(drop=True)
    g["date"] = pd.to_datetime(g["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    mid = len(g) // 2
    t_date = g.loc[mid, "date"]
    cols = [c for c in FEATURE_COLS_V1 if c in features_for_id(g, btc_close).columns]
    base = features_for_id(g, btc_close)
    base_row = base.loc[base["date"] == t_date, cols]
    if base_row.empty:
        return {"name": "feature_lookahead", "passed": False, "reason": "no feature row"}
    g2 = g.copy()
    fut = g2.index[g2["date"] > t_date]
    rng = np.random.default_rng(0)
    for col in ["open", "high", "low", "close", "volume", "dv", "mcap", "marketCap"]:
        if col in g2.columns:
            vals = g2.loc[fut, col].to_numpy(dtype=float)
            g2.loc[fut, col] = rng.permutation(vals) * (1 + rng.normal(0, 0.05, size=len(vals)))
    alt = features_for_id(g2, btc_close)
    alt_row = alt.loc[alt["date"] == t_date, cols]
    diff = base_row.to_numpy(dtype=float) - alt_row.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    passed = max_abs < 1e-10
    return {
        "name": "feature_lookahead",
        "passed": bool(passed),
        "max_abs_diff": max_abs,
        "id": iid,
        "date": str(pd.Timestamp(t_date).date()),
    }


def gate_universe_lookahead(panel: pd.DataFrame, n: int, window: int = PIT_DV_WINDOW) -> dict:
    name = f"universe_lookahead_top{n}"
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    dates = sorted(p["date"].unique())
    if len(dates) < window + 50:
        return {"name": name, "passed": False, "reason": "short history"}
    t = dates[len(dates) // 2]
    score, _, _, last_sym = trailing_rank_frame_by_id(p, window=window)
    base = build_pit_topn_ids(score, n=n, last_sym=last_sym)
    base_set = set(int(x) for x in base.loc[base["date"] == t, "id"])
    p2 = p.copy()
    fut = p2["date"] > t
    rng = np.random.default_rng(1)
    p2.loc[fut, "dv"] = p2.loc[fut, "dv"].to_numpy(dtype=float) * (1 + rng.normal(0, 0.5, size=int(fut.sum())))
    score2, _, _, last2 = trailing_rank_frame_by_id(p2, window=window)
    alt = build_pit_topn_ids(score2, n=n, last_sym=last2)
    alt_set = set(int(x) for x in alt.loc[alt["date"] == t, "id"])
    passed = base_set == alt_set and len(base_set) > 0
    return {
        "name": name,
        "passed": bool(passed),
        "n": n,
        "date": str(pd.Timestamp(t).date()),
        "base_n": len(base_set),
        "symmetric_diff": len(base_set.symmetric_difference(alt_set)),
    }


def gate_seed_determinism(df: pd.DataFrame, fold: FoldSpec, seed: int = SEED) -> dict:
    p1, m1 = fit_predict_fold(df, fold, seed=seed)
    p2, m2 = fit_predict_fold(df, fold, seed=seed)
    if p1.empty or p2.empty:
        return {"name": "seed_determinism", "passed": False, "reason": "empty preds"}
    s1 = p1.sort_values(["date", "id"])["p"].to_numpy()
    s2 = p2.sort_values(["date", "id"])["p"].to_numpy()
    if len(s1) != len(s2):
        return {"name": "seed_determinism", "passed": False, "reason": "length mismatch"}
    max_diff = float(np.max(np.abs(s1 - s2)))
    passed = max_diff < 1e-10
    return {
        "name": "seed_determinism",
        "passed": bool(passed),
        "max_score_diff": max_diff,
        "best_iteration": m1.get("best_iteration"),
        "fold_id": fold.fold_id,
        "horizon": fold.horizon,
    }


def _cell_stats(ics: list[float], center: float = 0.5) -> dict:
    arr = np.asarray([x for x in ics if np.isfinite(x)], dtype=float)
    n = int(len(arr))
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    p95 = float(np.percentile(arr, 95)) if n else float("nan")
    se = (sd / np.sqrt(n)) if n and np.isfinite(sd) else float("nan")
    bias_lim = 2.0 * se if np.isfinite(se) else float("nan")
    bias_ok = bool(np.isfinite(mean) and np.isfinite(bias_lim) and abs(mean - center) <= bias_lim)
    return {
        "n": n,
        "mean": mean,
        "center": float(center),
        "sd": sd,
        "p95": p95,
        "se": float(se) if np.isfinite(se) else float("nan"),
        "bias_lim": float(bias_lim) if np.isfinite(bias_lim) else float("nan"),
        "bias_ok": bias_ok,
        "aucs": [float(x) for x in arr],
    }


def gate_label_shuffle_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real_aucs: dict[int, float],
    n_replicates: int = NULL_REPLICATES,
    seeds: tuple[int, ...] = NULL_SHUFFLE_SEEDS,
) -> dict:
    """Retrain with within-date shuffled train labels; compare real OOS AUC to null."""
    cells = []
    for fold in folds:
        aucs = []
        use_seeds = list(seeds)[: int(n_replicates)]
        for i, ss in enumerate(use_seeds):
            print(
                f"[HB] null fold={fold.fold_id} h={fold.horizon} rep={i+1}/{len(use_seeds)} seed={ss}",
                flush=True,
            )
            pred, meta = fit_predict_fold(df, fold, seed=SEED, shuffle_labels=True, shuffle_seed=int(ss))
            if pred.empty or meta.get("status") != "ok":
                aucs.append(float("nan"))
                continue
            ycol = f"y_h{fold.horizon}"
            aucs.append(_auc(pred[ycol].to_numpy(), pred["p_raw"].to_numpy()))
        st = _cell_stats(aucs)
        real = float(real_aucs.get(fold.fold_id, float("nan")))
        st.update(
            {
                "fold_id": fold.fold_id,
                "horizon": fold.horizon,
                "real_auc": real,
                "exceeds_p95": bool(np.isfinite(real) and np.isfinite(st["p95"]) and real > st["p95"]),
            }
        )
        cells.append(st)
        print(
            f"[HB] null fold={fold.fold_id} mean={st['mean']:.4f} p95={st['p95']:.4f} "
            f"real={real:.4f} bias_ok={st['bias_ok']} skill={st['exceeds_p95']}",
            flush=True,
        )
    n_violate = sum(1 for c in cells if not c.get("bias_ok"))
    bias_pass = n_violate == 0
    skill_pass = bool(cells) and all(c.get("exceeds_p95") for c in cells)
    if not bias_pass:
        verdict = "CONTAMINATED"
    elif not skill_pass:
        verdict = "PARKED-NO-SKILL"
    else:
        verdict = "GREEN"
    passed = bool(bias_pass and skill_pass)
    return {
        "name": "label_shuffle_null",
        "passed": passed,
        "verdict": verdict,
        "bias_pass": bias_pass,
        "skill_pass": skill_pass,
        "n_violate": n_violate,
        "n_folds": len(cells),
        "n_replicates": int(n_replicates),
        "cells": cells,
    }


def run_cheap_gates(panel: pd.DataFrame, btc_id: int | None = None) -> list[dict]:
    if btc_id is None:
        btc_id = btc_id_from_panel(panel)
    results = [
        gate_feature_lookahead(panel, btc_id),
        gate_universe_lookahead(panel, n=50),
        gate_universe_lookahead(panel, n=100),
    ]
    for r in results:
        print(f"[gates] {r['name']}: {'PASS' if r.get('passed') else 'FAIL'} {r}", flush=True)
    return results
