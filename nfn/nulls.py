"""Vol-matched null for NFN. Folds {5,15,21,24} × 15. House bias; skill ≥3/4 or Stouffer≥3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import FUTURE_NULL_BIAS_MIN_VIOLATIONS, STOUFFER_Z_MIN
from btcb.gates import metric_verdict_e1b_house
from btcb.model import FoldSpec
from btcb.phase4b import cell_stats_vol_matched, fold_tail_pack
from nfn.constants import NULL_K_EXCEED, NULL_REPLICATES, NULL_STOUFFER_Z, NULL_TRAIN_SEED
from nfn.data import PackedPanel
from nfn.train import train_one_fold


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


def assemble_null(cells_by_metric: dict, real_keys: dict) -> dict:
    packs = {}
    for metric, cells in cells_by_metric.items():
        v = metric_verdict_e1b_house(
            cells,
            real_keys[metric],
            int(NULL_K_EXCEED),
            float(NULL_STOUFFER_Z if metric == "tail_ic_top" else STOUFFER_Z_MIN),
        )
        packs[metric] = {k: val for k, val in v.items() if k != "cells"}
        packs[f"{metric}_cells"] = cells
    judged = packs.get("tail_ic_top") or {}
    return {
        "name": "nfn_vol_matched_null",
        "null_design": "vol_matched",
        "passed": bool(judged.get("passed")),
        "judged": "tail_ic_top",
        "bias_min_violations": int(FUTURE_NULL_BIAS_MIN_VIOLATIONS),
        "n_replicates": int(NULL_REPLICATES),
        "k_exceed": int(NULL_K_EXCEED),
        "n_folds": 4,
        **packs,
    }


def run_null_cell(
    pack: PackedPanel,
    fold: FoldSpec,
    shuffle_seed: int,
    labeled: pd.DataFrame,
    close,
    btc_id: int,
) -> dict:
    pred, meta = train_one_fold(
        pack,
        fold,
        seed=int(NULL_TRAIN_SEED),
        warm_blob=None,
        shuffle_labels=True,
        shuffle_seed=int(shuffle_seed),
    )
    if pred is None or pred.empty or meta.get("status") != "ok":
        return {
            "fold_id": int(fold.fold_id),
            "shuffle_seed": int(shuffle_seed),
            "tail_ic_top": None,
            "overlap": None,
            "monster": None,
            "status": meta.get("status", "empty"),
        }
    sm = fold_tail_pack(pred, labeled, close, btc_id, "spread")
    return {
        "fold_id": int(fold.fold_id),
        "shuffle_seed": int(shuffle_seed),
        "tail_ic_top": sm.get("tail_ic_top"),
        "overlap": sm.get("overlap"),
        "monster": sm.get("monster"),
        "status": "ok",
        "best_epoch": meta.get("best_epoch"),
    }


def cells_from_replicates(folds: list[FoldSpec], recs: list[dict], real: dict[int, dict]) -> dict:
    by_fold = {}
    for r in recs:
        by_fold.setdefault(int(r["fold_id"]), []).append(r)
    cells = {"tail_ic_top": [], "overlap": [], "monster": []}
    for fold in folds:
        rows = by_fold.get(int(fold.fold_id), [])
        ics = [r.get("tail_ic_top") for r in rows]
        ovs = [r.get("overlap") for r in rows]
        mons = [r.get("monster") for r in rows]
        cells["tail_ic_top"].append(_fold_cell(fold, ics, real, "tail_ic_top"))
        cells["overlap"].append(_fold_cell(fold, ovs, real, "overlap"))
        cells["monster"].append(_fold_cell(fold, mons, real, "monster"))
    return assemble_null(
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )
