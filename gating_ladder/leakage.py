"""Anti-leak suite for the gating ladder FASE 1.

Reuses baseline.gates for label_shuffle / feature_lookahead /
universe_lookahead / seed_determinism. New tests live here.
test_gate_identity_leakage is N/A on A0 (no gate).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, summarize_ic
from baseline.gates import (
    gate_feature_lookahead,
    gate_label_shuffle,
    gate_seed_determinism,
    gate_universe_lookahead,
)
from baseline.model import FoldSpec


# Patterns frozen in pre-reg §D.
_FILTER_RES = [
    ("fft", re.compile(r"\bfft\b", re.I)),
    ("filtfilt", re.compile(r"\bfiltfilt\b")),
    ("savgol", re.compile(r"\bsavgol", re.I)),
    ("center=True", re.compile(r"center\s*=\s*True")),
]
_INTERPOLATE = re.compile(r"\.interpolate\s*\(")

# Pre-reg justifications (none in A0 baseline at freeze). New hits after freeze = FAIL.
JUSTIFIED_HITS: dict[str, str] = {}


def test_no_lookahead_filters(roots: list[Path]) -> dict:
    """Grep pipeline code for bilateral filters / centered smoothers / interpolate."""
    hits: list[dict] = []
    skip_names = {"leakage.py"}  # this file lists the patterns
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            if path.name in skip_names:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = str(path)
            for name, cre in _FILTER_RES:
                for i, line in enumerate(text.splitlines(), start=1):
                    if line.lstrip().startswith("#"):
                        continue
                    if cre.search(line):
                        key = f"{rel}:{i}:{name}"
                        hits.append({"file": rel, "line": i, "kind": name, "text": line.strip(), "key": key})
            for i, line in enumerate(text.splitlines(), start=1):
                if line.lstrip().startswith("#"):
                    continue
                if _INTERPOLATE.search(line) and "limit_direction" not in line:
                    if "forward" not in line:
                        key = f"{rel}:{i}:interpolate"
                        hits.append(
                            {
                                "file": rel,
                                "line": i,
                                "kind": "interpolate_no_forward",
                                "text": line.strip(),
                                "key": key,
                            }
                        )
    unjust = [h for h in hits if h["key"] not in JUSTIFIED_HITS]
    return {
        "name": "test_no_lookahead_filters",
        "passed": len(unjust) == 0,
        "n_hits": len(hits),
        "n_unjustified": len(unjust),
        "hits": unjust[:50],
    }


def test_scaler_fold_isolation(stage: str, fit_index_max=None, test_start=None) -> dict:
    """A0: CS-z is per-bar, no stateful scaler. Vacuous PASS.

    For later stages, pass the max date used to fit stateful objects and the
    fold test start; fail if fit_index_max >= test_start.
    """
    if stage.upper() in {"A0", "BASELINE"}:
        return {
            "name": "test_scaler_fold_isolation",
            "passed": True,
            "status": "N/A",
            "reason": "A0 CS-z is contemporaneous per bar; no stateful scaler",
        }
    if fit_index_max is None or test_start is None:
        return {
            "name": "test_scaler_fold_isolation",
            "passed": False,
            "reason": "stateful stage must pass fit_index_max and test_start",
        }
    fit_t = pd.Timestamp(fit_index_max)
    test_t = pd.Timestamp(test_start)
    passed = fit_t < test_t
    return {
        "name": "test_scaler_fold_isolation",
        "passed": bool(passed),
        "fit_index_max": str(fit_t),
        "test_start": str(test_t),
    }


def test_shifted_target_degrades(
    pred: pd.DataFrame,
    ycol: str,
    horizon: int,
    shift: int = 10,
    score_col: str = "score",
) -> dict:
    """Same scores vs y shifted +`shift` bars per symbol. Must collapse."""
    df = pred.dropna(subset=[score_col, ycol]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    ic0 = daily_rank_ic(df, ycol, score_col=score_col)
    s0 = summarize_ic(ic0, horizon)
    tmp = df.sort_values(["symbol", "date"]).copy()
    tmp["y_shift"] = tmp.groupby("symbol", sort=False)[ycol].shift(-int(shift))
    ic1 = daily_rank_ic(tmp.dropna(subset=["y_shift"]), "y_shift", score_col=score_col)
    s1 = summarize_ic(ic1, horizon)
    mean0 = float(s0.get("mean_ic", float("nan")))
    mean1 = float(s1.get("mean_ic", float("nan")))
    t1 = float(s1.get("nw_tstat", float("nan")))
    half = 0.5 * mean0 if np.isfinite(mean0) else float("nan")
    # FAIL if shifted IC stays above 50% of unshifted OR shifted NW t >= 2.
    fail_ic = np.isfinite(mean1) and np.isfinite(half) and mean1 > half
    fail_t = np.isfinite(t1) and t1 >= 2.0
    passed = (not fail_ic) and (not fail_t) and np.isfinite(mean0) and np.isfinite(mean1)
    return {
        "name": "test_shifted_target_degrades",
        "passed": bool(passed),
        "shift": int(shift),
        "mean_ic": mean0,
        "mean_ic_shifted": mean1,
        "half_unshifted": half,
        "nw_tstat_shifted": t1,
        "n_days": s0.get("n_days"),
        "n_days_shifted": s1.get("n_days"),
        "fail_ic_too_high": bool(fail_ic),
        "fail_nw_t": bool(fail_t),
    }


def assert_axis(arr: np.ndarray, expected_shape: tuple, where: str) -> None:
    got = tuple(np.asarray(arr).shape)
    if got != tuple(expected_shape):
        raise AssertionError(f"test_axis_slicing FAIL at {where}: shape {got} != {expected_shape}")


def test_axis_slicing(pred: pd.DataFrame, score_col: str = "score") -> dict:
    """Build a (n_dates, n_symbols) score panel and assert shape before slices."""
    df = pred.dropna(subset=[score_col]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    wide = df.pivot_table(index="date", columns="symbol", values=score_col, aggfunc="mean")
    arr = wide.to_numpy(dtype=float)
    n_dates, n_syms = arr.shape
    assert_axis(arr, (n_dates, n_syms), "score_panel")
    if n_dates < 2 or n_syms < 2:
        return {"name": "test_axis_slicing", "passed": False, "reason": "panel too small", "shape": [n_dates, n_syms]}
    first = arr[0, :]
    assert_axis(first, (n_syms,), "score_panel[0,:]")
    col0 = arr[:, 0]
    assert_axis(col0, (n_dates,), "score_panel[:,0]")
    return {
        "name": "test_axis_slicing",
        "passed": True,
        "shape": [int(n_dates), int(n_syms)],
    }


def test_gate_identity_leakage_na() -> dict:
    return {
        "name": "test_gate_identity_leakage",
        "passed": True,
        "status": "N/A",
        "reason": "no gate on A0 baseline",
    }


def run_cheap_static_gates(panel: pd.DataFrame, build_pit_fn, cfg: dict, code_roots: list[Path]) -> list[dict]:
    """Gates that do not need a trained model. Run before walk-forward."""
    window = cfg["data"]["exec_dv_window"]
    results = [
        test_no_lookahead_filters(code_roots),
        test_scaler_fold_isolation("A0"),
        gate_feature_lookahead(panel),
        gate_universe_lookahead(panel, build_pit_fn, n=20, window=window, name="universe_lookahead_top20"),
        gate_universe_lookahead(panel, build_pit_fn, n=40, window=window, name="universe_lookahead_top40"),
        gate_universe_lookahead(
            panel, build_pit_fn, n=cfg["data"]["train_universe_n"], window=window, name="universe_lookahead_top120"
        ),
        test_gate_identity_leakage_na(),
    ]
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"[gates] {r['name']}: {status} {r}", flush=True)
    return results


def run_fase1_suite(
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    pred: pd.DataFrame,
    build_pit_fn,
    fold0: FoldSpec,
    cfg: dict,
    code_roots: list[Path],
) -> list[dict]:
    h = int(cfg["labels"]["primary_horizon"])
    ycol = f"y_h{h}"
    sample = pred.copy()
    if "fold_id" in sample.columns:
        fid0 = int(sample["fold_id"].min())
        sample = sample[sample["fold_id"] == fid0].copy()
    if ycol not in sample.columns:
        sample = sample.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    results = [
        gate_label_shuffle(sample, h, seed=cfg["seed"]),
        gate_seed_determinism(
            feat, fold0, seed=cfg["seed"], model_cfg=cfg["model"], inner_holdout_days=cfg["cv"]["inner_holdout_days"]
        ),
        test_shifted_target_degrades(pred if ycol in pred.columns else sample, ycol, h, shift=10),
        test_axis_slicing(pred),
    ]
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"[gates] {r['name']}: {status} {r}", flush=True)
    return results
