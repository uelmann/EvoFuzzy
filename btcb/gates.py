"""Phase 2 gates: lookahead, PIT, seed determinism, E.1b-style label-shuffle null."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import FEATURE_COLS_V1, NULL_REPLICATES, NULL_SHUFFLE_SEEDS, PIT_DV_WINDOW, SEED, STAGE_S_COLS
from btcb.features import btc_id_from_panel, features_for_id
from btcb.model import FoldSpec, _auc, fit_predict_fold, mean_per_date_auc, mean_per_date_rank_ic
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
    cols = [c for c in STAGE_S_COLS if c in features_for_id(g, btc_close).columns]
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


def gate_universe_lookahead(panel: pd.DataFrame, n: int, window: int = PIT_DV_WINDOW, floored: bool = False) -> dict:
    name = f"universe_lookahead_top{n}" + ("_floor" if floored else "")
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    dates = sorted(p["date"].unique())
    if len(dates) < window + 50:
        return {"name": name, "passed": False, "reason": "short history"}
    t = dates[len(dates) // 2]
    if floored:
        from btcb.hygiene import build_floored_pit
        base, _ = build_floored_pit(p, n=n)
        p2 = p.copy()
        fut = p2["date"] > t
        rng = np.random.default_rng(1)
        p2.loc[fut, "dv"] = p2.loc[fut, "dv"].to_numpy(dtype=float) * (1 + rng.normal(0, 0.5, size=int(fut.sum())))
        alt, _ = build_floored_pit(p2, n=n)
        base_set = set(int(x) for x in base.loc[base["date"] == t, "id"])
        alt_set = set(int(x) for x in alt.loc[alt["date"] == t, "id"])
    else:
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
        "floored": bool(floored),
    }


def gate_seed_determinism(
    df: pd.DataFrame,
    fold: FoldSpec,
    seed: int = SEED,
    feature_cols: list[str] | None = None,
    early_stop: str = "auc",
    ycol: str | None = None,
) -> dict:
    p1, m1 = fit_predict_fold(df, fold, seed=seed, feature_cols=feature_cols, early_stop=early_stop, ycol=ycol)
    p2, m2 = fit_predict_fold(df, fold, seed=seed, feature_cols=feature_cols, early_stop=early_stop, ycol=ycol)
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
    feature_cols: list[str] | None = None,
    early_stop: str = "per_date_auc",
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
            pred, meta = fit_predict_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                feature_cols=feature_cols,
                early_stop=early_stop,
            )
            if pred.empty or meta.get("status") != "ok":
                aucs.append(float("nan"))
                continue
            ycol = f"y_h{fold.horizon}"
            val, _ = mean_per_date_auc(pred[ycol].to_numpy(), pred["p_raw"].to_numpy(), pred["date"].to_numpy())
            aucs.append(val)
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


def assert_no_context(feature_cols: list[str]) -> dict:
    from btcb.constants import CTX_COLS

    leaked = sorted(set(feature_cols) & set(CTX_COLS))
    passed = len(leaked) == 0
    rec = {"name": "no_context_in_stage_s", "passed": passed, "leaked": leaked, "n_feat": len(feature_cols)}
    print(f"[gates] {rec['name']}: {'PASS' if passed else 'FAIL'} {rec}", flush=True)
    if not passed:
        raise RuntimeError(f"context features in Stage S: {leaked}")
    return rec


def pick_folds_by_id(folds: list[FoldSpec], ids: tuple[int, ...]) -> list[FoldSpec]:
    by_id = {f.fold_id: f for f in folds}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(f"null fold ids missing (no substitution): {missing}")
    return [by_id[i] for i in ids]


def _stouffer_z(cells: list[dict], real_key: str) -> float:
    zs = []
    for c in cells:
        sd = c.get("sd")
        mu = c.get("mean")
        real = c.get(real_key)
        if not (np.isfinite(sd) and sd > 0 and np.isfinite(mu) and np.isfinite(real)):
            return float("nan")
        zs.append((float(real) - float(mu)) / float(sd))
    if len(zs) != len(cells) or not zs:
        return float("nan")
    return float(np.sum(zs) / np.sqrt(len(zs)))


def _metric_verdict(cells: list[dict], real_key: str, k_exceed: int, z_min: float) -> dict:
    n_exceed = int(sum(1 for c in cells if c.get("exceeds_p95")))
    n_bias = int(sum(1 for c in cells if c.get("bias_ok")))
    z = _stouffer_z(cells, real_key)
    bias_pass = bool(cells) and n_bias == len(cells)
    skill_pass = bool(n_exceed >= int(k_exceed) or (np.isfinite(z) and z >= float(z_min)))
    if not bias_pass:
        verdict = "CONTAMINATED"
    elif not skill_pass:
        verdict = "PARKED"
    else:
        verdict = "GREEN"
    return {
        "bias_pass": bias_pass,
        "skill_pass": skill_pass,
        "passed": bool(bias_pass and skill_pass),
        "verdict": verdict,
        "n_exceed": n_exceed,
        "n_folds": len(cells),
        "stouffer_z": z,
        "cells": cells,
    }


def gate_twin_spread_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real_auc_top: dict[int, float],
    real_rankic_spread: dict[int, float],
    n_replicates: int = NULL_REPLICATES,
    seeds: tuple[int, ...] = NULL_SHUFFLE_SEEDS,
    feature_cols: list[str] | None = None,
    early_stop: str = "per_date_auc",
) -> dict:
    """6-fold E.1b null for p_top AUC and spread RankIC. No fold substitution."""
    from btcb.constants import NULL_K_EXCEED, STOUFFER_Z_MIN

    auc_cells = []
    ric_cells = []
    use_seeds = list(seeds)[: int(n_replicates)]
    for fold in folds:
        y_top = f"y_h{fold.horizon}"
        y_bot = f"y_bot_h{fold.horizon}"
        excol = f"excess_h{fold.horizon}"
        aucs, rics = [], []
        for i, ss in enumerate(use_seeds):
            print(
                f"[HB] twin-null fold={fold.fold_id} h={fold.horizon} rep={i+1}/{len(use_seeds)} seed={ss}",
                flush=True,
            )
            pred_t, meta_t = fit_predict_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                feature_cols=feature_cols,
                early_stop=early_stop,
                ycol=y_top,
            )
            pred_b, meta_b = fit_predict_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss) + 50_000,
                feature_cols=feature_cols,
                early_stop=early_stop,
                ycol=y_bot,
            )
            if pred_t.empty or meta_t.get("status") != "ok":
                aucs.append(float("nan"))
                rics.append(float("nan"))
                continue
            val, _ = mean_per_date_auc(
                pred_t[y_top].to_numpy(), pred_t["p_raw"].to_numpy(), pred_t["date"].to_numpy()
            )
            aucs.append(val)
            if pred_b.empty or meta_b.get("status") != "ok":
                rics.append(float("nan"))
                continue
            m = pred_t.merge(
                pred_b[["date", "id", "p_raw"]].rename(columns={"p_raw": "p_bot_raw"}),
                on=["date", "id"],
                how="inner",
            )
            if m.empty or excol not in m.columns:
                rics.append(float("nan"))
                continue
            spread = m["p_raw"].to_numpy(dtype=float) - m["p_bot_raw"].to_numpy(dtype=float)
            rics.append(mean_per_date_rank_ic(spread, m[excol].to_numpy(), m["date"].to_numpy()))
        st_a = _cell_stats(aucs, center=0.5)
        real_a = float(real_auc_top.get(fold.fold_id, float("nan")))
        st_a.update(
            {
                "fold_id": fold.fold_id,
                "horizon": fold.horizon,
                "real_auc": real_a,
                "exceeds_p95": bool(np.isfinite(real_a) and np.isfinite(st_a["p95"]) and real_a > st_a["p95"]),
            }
        )
        auc_cells.append(st_a)
        st_r = _cell_stats(rics, center=0.0)
        real_r = float(real_rankic_spread.get(fold.fold_id, float("nan")))
        st_r.update(
            {
                "fold_id": fold.fold_id,
                "horizon": fold.horizon,
                "real_rankic": real_r,
                "exceeds_p95": bool(np.isfinite(real_r) and np.isfinite(st_r["p95"]) and real_r > st_r["p95"]),
            }
        )
        ric_cells.append(st_r)
        print(
            f"[HB] twin-null fold={fold.fold_id} auc mean={st_a['mean']:.4f} p95={st_a['p95']:.4f} "
            f"real={real_a:.4f} | ric mean={st_r['mean']:.4f} p95={st_r['p95']:.4f} real={real_r:.4f}",
            flush=True,
        )
    auc_v = _metric_verdict(auc_cells, "real_auc", NULL_K_EXCEED, STOUFFER_Z_MIN)
    ric_v = _metric_verdict(ric_cells, "real_rankic", NULL_K_EXCEED, STOUFFER_Z_MIN)
    print(
        f"[gates] twin_null AUC {auc_v['verdict']} {auc_v['n_exceed']}/6 z={auc_v['stouffer_z']} | "
        f"spread RankIC {ric_v['verdict']} {ric_v['n_exceed']}/6 z={ric_v['stouffer_z']}",
        flush=True,
    )
    return {
        "name": "twin_spread_null",
        "passed": bool(ric_v["passed"]),  # judged signal = spread RankIC
        "auc": {k: v for k, v in auc_v.items() if k != "cells"},
        "rankic": {k: v for k, v in ric_v.items() if k != "cells"},
        "auc_cells": auc_cells,
        "rankic_cells": ric_cells,
        "n_replicates": int(n_replicates),
        "fold_ids": [f.fold_id for f in folds],
    }


def run_cheap_gates(panel: pd.DataFrame, btc_id: int | None = None, floored: bool = False) -> list[dict]:
    if btc_id is None:
        btc_id = btc_id_from_panel(panel)
    results = [
        gate_feature_lookahead(panel, btc_id),
        gate_universe_lookahead(panel, n=50, floored=floored),
        gate_universe_lookahead(panel, n=100, floored=floored),
    ]
    for r in results:
        print(f"[gates] {r['name']}: {'PASS' if r.get('passed') else 'FAIL'} {r}", flush=True)
    return results
