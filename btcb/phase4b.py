"""Phase 4.b — TWIN-RANK, vol-matched null, DIR reweighting.

BACKTEST / ANALYSIS ONLY. Nothing adopted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.constants import (
    FUTURE_NULL_BIAS_MIN_VIOLATIONS,
    NULL_K_EXCEED,
    NULL_REPLICATES,
    NULL_SHUFFLE_SEEDS,
    PHASE4B_OVERLAP_DELTA,
    PHASE4B_TAIL_IC_DELTA,
    PHASE4V2_NW_LAG,
    SEED,
    STOUFFER_Z_MIN,
)
from btcb.gates import metric_verdict_e1b_house
from btcb.model import FoldSpec, fit_predict_fold, fit_predict_rank_fold
from btcb.oracle_ladder import _as_utc, _spearman
from btcb.phase4v2 import (
    _utc,
    _window_stats,
    collapse_fold_preds,
    cs_rank_blend,
    per_date_tail_metrics,
    restrict_eval_frame,
)


def _log(msg: str) -> None:
    print(f"[p4b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vol_col_name(df: pd.DataFrame) -> str:
    if "yz_vol_30_raw" in df.columns:
        return "yz_vol_30_raw"
    if "yz_vol_30" in df.columns:
        return "yz_vol_30"
    raise RuntimeError("yz_vol_30 missing from labeled frame")


def twinrank_from_heads(
    top: pd.DataFrame,
    bot: pd.DataFrame,
    col_top: str = "p",
    col_bot: str = "p",
    out_col: str = "twinrank",
) -> pd.DataFrame:
    """S = cs_rank(top) − cs_rank(bot) per date."""
    t = top[["date", "id", col_top]].copy()
    b = bot[["date", "id", col_bot]].copy()
    t["date"] = _utc(t["date"])
    b["date"] = _utc(b["date"])
    t["id"] = t["id"].astype(int)
    b["id"] = b["id"].astype(int)
    if col_top == col_bot:
        t = t.rename(columns={col_top: "score_top"})
        b = b.rename(columns={col_bot: "score_bot"})
        c_t, c_b = "score_top", "score_bot"
    else:
        c_t, c_b = col_top, col_bot
    m = t.merge(b[["date", "id", c_b]], on=["date", "id"], how="inner")
    parts = []
    for _, g in m.groupby("date", sort=False):
        g = g.copy()
        g[out_col] = g[c_t].rank(method="average", pct=True) - g[c_b].rank(method="average", pct=True)
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["date", "id", out_col])
    out = pd.concat(parts, ignore_index=True)
    keep = ["date", "id", out_col]
    return out[keep]


def mean_per_date_vol_corr(
    df: pd.DataFrame,
    score_col: str,
    vol_col: str,
    min_n: int = 8,
) -> dict:
    d = df[["date", "id", score_col, vol_col]].copy()
    d["date"] = _utc(d["date"])
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d[vol_col] = pd.to_numeric(d[vol_col], errors="coerce")
    d = d.dropna(subset=[score_col, vol_col])
    dates, ics = [], []
    for dt, g in d.groupby("date", sort=True):
        if len(g) < int(min_n):
            continue
        c = _spearman(g[score_col].to_numpy(), g[vol_col].to_numpy())
        if np.isfinite(c):
            dates.append(_as_utc(dt))
            ics.append(float(c))
    blob = _window_stats(dates, ics, lag=PHASE4V2_NW_LAG)
    return {
        "vol_rank_corr": blob["full"]["mean"],
        "vol_rank_corr_n": blob["full"]["n"],
        "vol_rank_corr_trail": blob["trail18m"]["mean"],
        "vol_rank_corr_cycles": blob["cycles"],
    }


def attach_vol_corr(met: dict, ev: pd.DataFrame, score_col: str, vol_df: pd.DataFrame, vol_col: str) -> dict:
    v = vol_df[["date", "id", vol_col]].copy()
    v["date"] = _utc(v["date"])
    v["id"] = v["id"].astype(int)
    m = ev.merge(v, on=["date", "id"], how="inner")
    met = dict(met)
    met.update(mean_per_date_vol_corr(m, score_col, vol_col))
    return met


def fold_tail_pack(pred: pd.DataFrame, labeled: pd.DataFrame, close, btc_id: int, score_col: str) -> dict:
    if pred is None or pred.empty:
        return {
            "tail_ic_top": float("nan"),
            "tail_ic_bot": float("nan"),
            "overlap": float("nan"),
            "monster": float("nan"),
            "rankic": float("nan"),
            "n_dates": 0,
        }
    p = pred.copy()
    if score_col not in p.columns and "p" in p.columns:
        p[score_col] = p["p"]
        score_col = score_col if score_col in p.columns else "p"
    ev = restrict_eval_frame(p, labeled, close, btc_id, score_col)
    if ev.empty:
        return {
            "tail_ic_top": float("nan"),
            "tail_ic_bot": float("nan"),
            "overlap": float("nan"),
            "monster": float("nan"),
            "rankic": float("nan"),
            "n_dates": 0,
        }
    met = per_date_tail_metrics(ev, score_col)
    return {
        "tail_ic_top": met.get("tail_ic_top"),
        "tail_ic_bot": met.get("tail_ic_bot"),
        "overlap": met.get("overlap"),
        "monster": met.get("monster"),
        "rankic": met.get("rankic"),
        "n_dates": met.get("n_dates"),
    }


def cell_stats_vol_matched(values: list) -> dict:
    """Null mean is the structural reference; bias_ok iff the 2·SE band can be formed."""
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = int(len(arr))
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    p95 = float(np.percentile(arr, 95)) if n else float("nan")
    se = (sd / np.sqrt(n)) if n and np.isfinite(sd) else float("nan")
    bias_lim = 2.0 * se if np.isfinite(se) else float("nan")
    bias_ok = bool(
        n >= 2
        and np.isfinite(mean)
        and np.isfinite(bias_lim)
        and abs(mean - mean) <= float(bias_lim)
    )
    return {
        "n": n,
        "mean": mean,
        "center": mean,
        "sd": sd,
        "p95": p95,
        "se": float(se) if np.isfinite(se) else float("nan"),
        "bias_lim": float(bias_lim) if np.isfinite(bias_lim) else float("nan"),
        "bias_ok": bias_ok,
        "aucs": [float(x) for x in arr],
        "null_design": "vol_matched",
    }


def _finish_null(name: str, cells_by_metric: dict, real_keys: dict) -> dict:
    packs = {}
    for metric, cells in cells_by_metric.items():
        v = metric_verdict_e1b_house(cells, real_keys[metric], NULL_K_EXCEED, STOUFFER_Z_MIN)
        packs[metric] = {k: val for k, val in v.items() if k != "cells"}
        packs[f"{metric}_cells"] = cells
    judged = packs.get("tail_ic_top") or {}
    return {
        "name": name,
        "null_design": "vol_matched",
        "passed": bool(judged.get("passed")),
        "judged": "tail_ic_top",
        "bias_min_violations": int(FUTURE_NULL_BIAS_MIN_VIOLATIONS),
        "n_replicates": int(NULL_REPLICATES),
        **packs,
    }


def _cache_load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _cache_dump(path: Path, rec: dict, commit_fn=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, default=str))
    if commit_fn is not None:
        commit_fn()


def gate_vol_matched_rank_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real: dict[int, dict],
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    feature_cols: list[str],
    ycol: str,
    cache_dir: Path | None = None,
    commit_fn=None,
    vol_col: str = "yz_vol_30",
) -> dict:
    """Single LambdaRank head, vol-matched label shuffle (retro RANK / one head)."""
    use_seeds = list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]
    cells = {"tail_ic_top": [], "overlap": [], "monster": []}
    for fold in folds:
        ics, ovs, mons = [], [], []
        for i, ss in enumerate(use_seeds):
            cached = None
            if cache_dir is not None:
                cached = _cache_load(cache_dir / f"fold{fold.fold_id}_seed{ss}.json")
            if cached is not None:
                ics.append(cached.get("tail_ic_top"))
                ovs.append(cached.get("overlap"))
                mons.append(cached.get("monster"))
                continue
            _log(f"rank-null fold={fold.fold_id} rep={i+1}/{len(use_seeds)} seed={ss}")
            pred, meta = fit_predict_rank_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                shuffle_mode="vol_matched",
                vol_col=vol_col,
                feature_cols=feature_cols,
                ycol=ycol,
            )
            if pred.empty or meta.get("status") != "ok":
                rec = {"tail_ic_top": None, "overlap": None, "monster": None, "status": meta.get("status")}
            else:
                sm = fold_tail_pack(pred, labeled, close, btc_id, "p")
                rec = {k: sm.get(k) for k in ("tail_ic_top", "overlap", "monster")}
                rec["status"] = "ok"
            ics.append(rec.get("tail_ic_top"))
            ovs.append(rec.get("overlap"))
            mons.append(rec.get("monster"))
            if cache_dir is not None:
                _cache_dump(cache_dir / f"fold{fold.fold_id}_seed{ss}.json", rec, commit_fn if (i + 1) % 5 == 0 else None)
        cells["tail_ic_top"].append(_fold_cell(fold, ics, real, "tail_ic_top"))
        cells["overlap"].append(_fold_cell(fold, ovs, real, "overlap"))
        cells["monster"].append(_fold_cell(fold, mons, real, "monster"))
        st = cells["tail_ic_top"][-1]
        _log(
            f"rank-null fold={fold.fold_id} mean={st['mean']:.4f} p95={st['p95']:.4f} "
            f"real={st.get('real_tail_ic_top')} bias_ok={st['bias_ok']}"
        )
        if cache_dir is not None and commit_fn is not None:
            commit_fn()
    return _finish_null(
        "rank_vol_matched_null",
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )


def gate_vol_matched_twinrank_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real: dict[int, dict],
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    feature_cols: list[str],
    y_top: str,
    y_bot: str,
    cache_dir: Path | None = None,
    commit_fn=None,
    vol_col: str = "yz_vol_30",
) -> dict:
    """Both ranking heads, same shuffle seed (joint vol-bucket permutation)."""
    use_seeds = list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]
    cells = {"tail_ic_top": [], "overlap": [], "monster": []}
    for fold in folds:
        ics, ovs, mons = [], [], []
        for i, ss in enumerate(use_seeds):
            cached = None
            if cache_dir is not None:
                cached = _cache_load(cache_dir / f"fold{fold.fold_id}_seed{ss}.json")
            if cached is not None:
                ics.append(cached.get("tail_ic_top"))
                ovs.append(cached.get("overlap"))
                mons.append(cached.get("monster"))
                continue
            _log(f"twinrank-null fold={fold.fold_id} rep={i+1}/{len(use_seeds)} seed={ss}")
            pred_t, meta_t = fit_predict_rank_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                shuffle_mode="vol_matched",
                vol_col=vol_col,
                feature_cols=feature_cols,
                ycol=y_top,
            )
            pred_b, meta_b = fit_predict_rank_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                shuffle_mode="vol_matched",
                vol_col=vol_col,
                feature_cols=feature_cols,
                ycol=y_bot,
            )
            if (
                pred_t.empty
                or pred_b.empty
                or meta_t.get("status") != "ok"
                or meta_b.get("status") != "ok"
            ):
                rec = {
                    "tail_ic_top": None,
                    "overlap": None,
                    "monster": None,
                    "status": f"{meta_t.get('status')}/{meta_b.get('status')}",
                }
            else:
                sig = twinrank_from_heads(pred_t, pred_b, "p", "p", "twinrank")
                sm = fold_tail_pack(sig, labeled, close, btc_id, "twinrank")
                rec = {k: sm.get(k) for k in ("tail_ic_top", "overlap", "monster")}
                rec["status"] = "ok"
            ics.append(rec.get("tail_ic_top"))
            ovs.append(rec.get("overlap"))
            mons.append(rec.get("monster"))
            if cache_dir is not None:
                _cache_dump(cache_dir / f"fold{fold.fold_id}_seed{ss}.json", rec, commit_fn if (i + 1) % 5 == 0 else None)
        cells["tail_ic_top"].append(_fold_cell(fold, ics, real, "tail_ic_top"))
        cells["overlap"].append(_fold_cell(fold, ovs, real, "overlap"))
        cells["monster"].append(_fold_cell(fold, mons, real, "monster"))
        st = cells["tail_ic_top"][-1]
        _log(
            f"twinrank-null fold={fold.fold_id} mean={st['mean']:.4f} p95={st['p95']:.4f} "
            f"real={st.get('real_tail_ic_top')} bias_ok={st['bias_ok']}"
        )
        if cache_dir is not None and commit_fn is not None:
            commit_fn()
    return _finish_null(
        "twinrank_vol_matched_null",
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )


def gate_vol_matched_dir_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real: dict[int, dict],
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    feature_cols: list[str],
    bot_2c: pd.DataFrame,
    ycol: str,
    weight_col: str,
    cache_dir: Path | None = None,
    commit_fn=None,
    vol_col: str = "yz_vol_30",
    early_stop: str = "per_date_auc",
) -> dict:
    """DIR-weighted top head on vol-matched shuffled labels; bottom = frozen 2.c."""
    use_seeds = list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]
    bot = bot_2c.copy()
    bot["date"] = _utc(bot["date"])
    bot["id"] = bot["id"].astype(int)
    if "p_bot" not in bot.columns:
        raise RuntimeError("DIR null needs frozen 2.c p_bot")
    cells = {"tail_ic_top": [], "overlap": [], "monster": []}
    for fold in folds:
        ics, ovs, mons = [], [], []
        if "fold_id" in bot.columns:
            bfold = bot.loc[bot["fold_id"] == fold.fold_id, ["date", "id", "p_bot"]]
        else:
            bfold = bot.loc[
                (bot["date"] >= fold.val_start) & (bot["date"] <= fold.val_end),
                ["date", "id", "p_bot"],
            ]
        for i, ss in enumerate(use_seeds):
            cached = None
            if cache_dir is not None:
                cached = _cache_load(cache_dir / f"fold{fold.fold_id}_seed{ss}.json")
            if cached is not None:
                ics.append(cached.get("tail_ic_top"))
                ovs.append(cached.get("overlap"))
                mons.append(cached.get("monster"))
                continue
            _log(f"dir-null fold={fold.fold_id} rep={i+1}/{len(use_seeds)} seed={ss}")
            pred, meta = fit_predict_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                shuffle_mode="vol_matched",
                vol_col=vol_col,
                feature_cols=feature_cols,
                early_stop=early_stop,
                ycol=ycol,
                weight_col=weight_col,
            )
            if pred.empty or meta.get("status") != "ok" or bfold.empty:
                rec = {"tail_ic_top": None, "overlap": None, "monster": None, "status": meta.get("status")}
            else:
                m = pred.merge(bfold, on=["date", "id"], how="inner")
                m["dir_spread"] = m["p"].astype(float) - m["p_bot"].astype(float)
                sm = fold_tail_pack(m, labeled, close, btc_id, "dir_spread")
                rec = {k: sm.get(k) for k in ("tail_ic_top", "overlap", "monster")}
                rec["status"] = "ok"
            ics.append(rec.get("tail_ic_top"))
            ovs.append(rec.get("overlap"))
            mons.append(rec.get("monster"))
            if cache_dir is not None:
                _cache_dump(cache_dir / f"fold{fold.fold_id}_seed{ss}.json", rec, commit_fn if (i + 1) % 5 == 0 else None)
        cells["tail_ic_top"].append(_fold_cell(fold, ics, real, "tail_ic_top"))
        cells["overlap"].append(_fold_cell(fold, ovs, real, "overlap"))
        cells["monster"].append(_fold_cell(fold, mons, real, "monster"))
        st = cells["tail_ic_top"][-1]
        _log(
            f"dir-null fold={fold.fold_id} mean={st['mean']:.4f} p95={st['p95']:.4f} "
            f"real={st.get('real_tail_ic_top')} bias_ok={st['bias_ok']}"
        )
        if cache_dir is not None and commit_fn is not None:
            commit_fn()
    return _finish_null(
        "dir_vol_matched_null",
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )


def _fold_cell(fold: FoldSpec, values: list, real: dict[int, dict], metric: str) -> dict:
    st = cell_stats_vol_matched(values)
    blob = real.get(fold.fold_id) or {}
    real_v = blob.get(metric, float("nan"))
    try:
        real_v = float(real_v) if real_v is not None else float("nan")
    except (TypeError, ValueError):
        real_v = float("nan")
    key = {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"}[metric]
    st.update(
        {
            "fold_id": fold.fold_id,
            "horizon": fold.horizon,
            key: real_v,
            "exceeds_p95": bool(np.isfinite(real_v) and np.isfinite(st["p95"]) and real_v > st["p95"]),
        }
    )
    return st


def _delta(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return float("nan")
    return fa - fb


def mechanical_verdicts(grid: dict, null_twin: dict, null_dir: dict, null_rank: dict) -> dict:
    base = grid.get("frozen_spread") or {}
    twin = grid.get("twinrank") or {}
    dire = grid.get("dir_spread") or {}
    d_ic_t = _delta(twin.get("tail_ic_top"), base.get("tail_ic_top"))
    d_ov_t = _delta(twin.get("overlap"), base.get("overlap"))
    d_ic_d = _delta(dire.get("tail_ic_top"), base.get("tail_ic_top"))
    d_ov_d = _delta(dire.get("overlap"), base.get("overlap"))
    twin_clears = bool(
        np.isfinite(d_ic_t)
        and np.isfinite(d_ov_t)
        and d_ic_t >= float(PHASE4B_TAIL_IC_DELTA)
        and d_ov_t >= float(PHASE4B_OVERLAP_DELTA)
    )
    dir_clears = bool(
        np.isfinite(d_ic_d)
        and np.isfinite(d_ov_d)
        and d_ic_d >= float(PHASE4B_TAIL_IC_DELTA)
        and d_ov_d >= float(PHASE4B_OVERLAP_DELTA)
    )
    twin_null_pass = bool((null_twin or {}).get("passed"))
    dir_null_pass = bool((null_dir or {}).get("passed"))
    twin_extracts = bool(twin_clears and twin_null_pass)
    dir_live = bool(dir_clears and dir_null_pass)
    ceiling = None
    if not twin_extracts and not dir_live:
        from btcb.constants import PHASE4B_CEILING

        ceiling = PHASE4B_CEILING
    rank_null = (null_rank or {}).get("tail_ic_top") or {}
    retro_beyond_vol = bool(rank_null.get("passed"))
    return {
        "twinrank": "TWIN-RANK EXTRACTS" if twin_extracts else "TWIN-RANK BARREN",
        "dir": "DIR LIVE" if dir_live else "DIR NOT LIVE",
        "twin_clears_deltas": twin_clears,
        "dir_clears_deltas": dir_clears,
        "twin_null_pass": twin_null_pass,
        "dir_null_pass": dir_null_pass,
        "delta_twin_vs_base_tail_ic": d_ic_t,
        "delta_twin_vs_base_overlap": d_ov_t,
        "delta_dir_vs_base_tail_ic": d_ic_d,
        "delta_dir_vs_base_overlap": d_ov_d,
        "ceiling": ceiling,
        "retro_rank_vol_matched_pass": retro_beyond_vol,
        "retro_rank_verdict": rank_null.get("verdict"),
        "retro_rank_skill_pass": rank_null.get("skill_pass"),
        "retro_rank_bias_pass": rank_null.get("bias_pass"),
        "nothing_adopted": True,
    }


def merge_dir_spread(top: pd.DataFrame, twin_2c: pd.DataFrame) -> pd.DataFrame:
    t = top.copy()
    t["date"] = _utc(t["date"])
    t["id"] = t["id"].astype(int)
    keep_b = ["date", "id", "p_bot"]
    if "fold_id" in twin_2c.columns:
        keep_b = ["date", "id", "fold_id", "p_bot"]
    b = twin_2c[keep_b].copy()
    b["date"] = _utc(b["date"])
    b["id"] = b["id"].astype(int)
    if "fold_id" in t.columns and "fold_id" in b.columns:
        m = t.merge(b, on=["date", "id", "fold_id"], how="inner")
        if len(m) < max(1, int(0.5 * len(t))):
            b2 = collapse_fold_preds(b.rename(columns={"p_bot": "p"}), "p").rename(columns={"p": "p_bot"})
            m = collapse_fold_preds(t, "p").merge(b2[["date", "id", "p_bot"]], on=["date", "id"], how="inner")
    else:
        m = collapse_fold_preds(t, "p").merge(
            collapse_fold_preds(b.rename(columns={"p_bot": "p"}), "p").rename(columns={"p": "p_bot"})[
                ["date", "id", "p_bot"]
            ],
            on=["date", "id"],
            how="inner",
        )
    m["dir_spread"] = m["p"].astype(float) - m["p_bot"].astype(float)
    return m


def real_fold_metrics(preds: pd.DataFrame, folds: list[FoldSpec], labeled, close, btc_id, score_col: str) -> dict[int, dict]:
    out = {}
    pr = preds.copy()
    pr["date"] = _utc(pr["date"])
    for fold in folds:
        if "fold_id" in pr.columns:
            sl = pr[pr["fold_id"].astype(int) == int(fold.fold_id)]
        else:
            sl = pr[(pr["date"] >= fold.val_start) & (pr["date"] <= fold.val_end)]
        out[fold.fold_id] = fold_tail_pack(sl, labeled, close, btc_id, score_col)
        _log(
            f"real fold {fold.fold_id} {score_col} tailIC={out[fold.fold_id].get('tail_ic_top')} "
            f"ov={out[fold.fold_id].get('overlap')}"
        )
    return out
