"""Vol-matched null for NFN v1. Same folds × 15; v1 craft, 5-init bag, cold per fold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.model import FoldSpec
from btcb.phase4b import fold_tail_pack
from nfn.constants_v1 import N_BAG, NULL_TRAIN_SEED
from nfn.data import PackedPanel
from nfn.nulls import assemble_null, _fold_cell
from nfn.train_v1 import train_one_fold_v1


def _mean_bag_spreads(frames: list[pd.DataFrame]) -> pd.DataFrame:
    acc = None
    for i, df in enumerate(frames):
        if df is None or df.empty:
            continue
        sl = df[["date", "id", "fold_id", "spread"]].copy()
        sl = sl.rename(columns={"spread": f"s{i}"})
        acc = sl if acc is None else acc.merge(sl, on=["date", "id", "fold_id"], how="outer")
    if acc is None:
        return pd.DataFrame(columns=["date", "id", "fold_id", "spread"])
    cols = [c for c in acc.columns if c.startswith("s") and c[1:].isdigit()]
    acc["spread"] = acc[cols].mean(axis=1)
    return acc[["date", "id", "fold_id", "spread"]]


def run_null_cell_v1(
    pack: PackedPanel,
    fold: FoldSpec,
    shuffle_seed: int,
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    n_bag: int = N_BAG,
) -> dict:
    frames, metas = [], []
    for bag in range(int(n_bag)):
        pred, meta, _ = train_one_fold_v1(
            pack,
            fold,
            seed=int(NULL_TRAIN_SEED),
            bag=int(bag),
            prev_state=None,
            shuffle_labels=True,
            shuffle_seed=int(shuffle_seed) * 1009 + int(bag),
        )
        metas.append(meta)
        if pred is not None and not pred.empty and meta.get("status") == "ok":
            frames.append(pred)
    if not frames:
        return {
            "fold_id": int(fold.fold_id),
            "shuffle_seed": int(shuffle_seed),
            "tail_ic_top": None,
            "overlap": None,
            "monster": None,
            "status": "empty",
            "n_bags_ok": 0,
        }
    bagged = _mean_bag_spreads(frames)
    sm = fold_tail_pack(bagged, labeled, close, btc_id, "spread")
    windows = [m.get("selected_epoch_window") for m in metas if m.get("status") == "ok"]
    epochs = [m.get("selected_epoch") for m in metas if m.get("status") == "ok"]
    return {
        "fold_id": int(fold.fold_id),
        "shuffle_seed": int(shuffle_seed),
        "tail_ic_top": sm.get("tail_ic_top"),
        "overlap": sm.get("overlap"),
        "monster": sm.get("monster"),
        "status": "ok",
        "n_bags_ok": int(len(frames)),
        "selected_epoch_mean": float(np.nanmean(np.asarray(epochs, dtype=float))) if epochs else None,
        "windows": windows,
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
    out = assemble_null(
        cells,
        {"tail_ic_top": "real_tail_ic_top", "overlap": "real_overlap", "monster": "real_monster"},
    )
    out["name"] = "nfn_v1_vol_matched_null"
    return out
