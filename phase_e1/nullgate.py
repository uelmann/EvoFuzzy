"""Phase E.1b empirical-null bias/skill gate (pre-registered)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import evaluate_predictions

E1B_GATE = (
    "(a) BIAS TEST: for each fold×h cell, the null mean across replicates must satisfy "
    "|mean| ≤ 2·(null SD / √R) on the primary universe. If violated on ≥2 of the 8 cells, "
    "verdict = CONTAMINATED: the GRU line is closed pending a dedicated bug hunt; no further "
    "GRU work in this task. (b) SKILL TEST: the real 3-seed GRU ensemble's outer-fold IC "
    "(taken from existing Phase E artifacts — no retraining) must exceed the 95th percentile "
    "of the corresponding null on ≥3 of 4 folds at h=7 or at h=10 on the primary universe. "
    "DECISION: if (a) passes and (b) passes → gates GREEN: immediately resume Phase E.1 §2–§4 "
    "exactly as originally written (9 seeds, three disjoint 3-seed ensembles, NW-t table, "
    "A0↔S correlations, per-year tables, score autocorrelation, portfolio translation with "
    "both τ conventions) and apply the original confirmation criterion clauses (ii)–(iv) with "
    "clause (i) replaced by this gate. If (a) passes but (b) fails → verdict = PARKED-NO-SKILL: "
    "stop, no adoption, no retuning. No other outcomes exist."
)

FOLDS_FULL = [2, 9, 15, 17]
FOLDS_BUDGET = [9, 17]
SHUFFLE_SEEDS = list(range(101, 111))  # 10 replicates
GRU_TRAIN_SEED = 42
PRIMARY_UNI = "pit120"
R = 10


def cell_stats(ics: list[float]) -> dict:
    arr = np.asarray([x for x in ics if np.isfinite(x)], dtype=float)
    n = int(len(arr))
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    p95 = float(np.percentile(arr, 95)) if n else float("nan")
    se = (sd / np.sqrt(n)) if n and np.isfinite(sd) else float("nan")
    bias_lim = 2.0 * se if np.isfinite(se) else float("nan")
    bias_ok = bool(np.isfinite(mean) and np.isfinite(bias_lim) and abs(mean) <= bias_lim)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "p95": p95,
        "se": float(se) if np.isfinite(se) else float("nan"),
        "bias_lim": float(bias_lim) if np.isfinite(bias_lim) else float("nan"),
        "bias_ok": bias_ok,
        "ics": [float(x) for x in arr],
    }


def bias_skill_verdict(cells: list[dict], n_folds_planned: int) -> dict:
    """cells have horizon, fold_id, universe=primary, mean, sd, p95, bias_ok, real_ic."""
    prim = [c for c in cells if c.get("universe") == PRIMARY_UNI]
    n_violate = sum(1 for c in prim if not c.get("bias_ok"))
    n_cells = len(prim)
    bias_pass = n_violate < 2
    skill_by_h = {}
    need = 3 if n_folds_planned >= 4 else 2
    for h in (7, 10):
        hs = [c for c in prim if int(c["horizon"]) == h]
        n_ex = sum(
            1
            for c in hs
            if np.isfinite(c.get("real_ic", np.nan))
            and np.isfinite(c.get("p95", np.nan))
            and float(c["real_ic"]) > float(c["p95"])
        )
        skill_by_h[h] = {"n_exceed": n_ex, "n_folds": len(hs), "need": need, "pass": n_ex >= need}
    skill_pass = bool(skill_by_h[7]["pass"] or skill_by_h[10]["pass"])
    if not bias_pass:
        verdict = "CONTAMINATED"
    elif skill_pass:
        verdict = "GREEN"
    else:
        verdict = "PARKED-NO-SKILL"
    return {
        "bias_pass": bias_pass,
        "skill_pass": skill_pass,
        "n_violate": n_violate,
        "n_cells": n_cells,
        "skill_by_h": skill_by_h,
        "verdict": verdict,
        "need_folds": need,
    }


def fold_mean_ic(pred: pd.DataFrame, horizon: int, universe: pd.DataFrame, label: str) -> dict:
    if pred is None or pred.empty:
        return {"mean_ic": float("nan"), "n_days": 0, "universe": label}
    ev = evaluate_predictions(pred, horizon, universe=universe, label=label)
    return {"mean_ic": ev.get("mean_ic"), "n_days": ev.get("n_days"), "universe": label}


def assemble_fold_ensemble(gru_root: Path, horizon: int, fold_id: int, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        p = gru_root / f"h{horizon}" / f"seed{seed}" / f"fold{fold_id}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["symbol"] = df["symbol"].astype(str)
        frames.append(df[["date", "symbol", "score"]].rename(columns={"score": f"score_s{seed}"}))
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on=["date", "symbol"], how="outer")
    scols = [c for c in merged.columns if c.startswith("score_s")]
    merged["score"] = merged[scols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return merged[["date", "symbol", "score"]]


def plot_null(cells: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prim = [c for c in cells if c.get("universe") == PRIMARY_UNI]
    prim = sorted(prim, key=lambda c: (int(c["horizon"]), int(c["fold_id"])))
    n = max(len(prim), 1)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, prim):
        ics = c.get("ics") or []
        if ics:
            ax.hist(ics, bins=min(8, max(4, len(ics))), color="0.75", edgecolor="0.4")
        if np.isfinite(c.get("p95", np.nan)):
            ax.axvline(c["p95"], color="crimson", ls="--", lw=1.4, label="null 95th")
        if np.isfinite(c.get("mean", np.nan)):
            ax.axvline(c["mean"], color="0.2", ls=":", lw=1.2, label="null mean")
        if np.isfinite(c.get("real_ic", np.nan)):
            ax.axvline(c["real_ic"], color="steelblue", lw=1.8, label="real 3-seed")
        ax.set_title(f"h={c['horizon']} fold={c['fold_id']} pit-120")
        ax.set_xlabel("outer-fold mean RankIC")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[len(prim) :]:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
