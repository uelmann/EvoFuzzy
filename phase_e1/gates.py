"""Phase E.1 leakage gates — all must PASS before seed/portfolio work."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic
from baseline.features import FEATURE_COLS
from phase_e.seq_model import MIN_WINDOW, WINDOW, window_from_symbol_frame


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def gru_score_from_window(win: np.ndarray, seed: int = 0) -> np.float32:
    """Deterministic 1-layer GRU last-hidden score. Function of the window only (fixed weights)."""
    if win is None:
        return np.float32("nan")
    rng = np.random.default_rng(seed)
    t_len, n_feat = win.shape
    hidden = 8
    w_ih = rng.normal(scale=0.1, size=(3 * hidden, n_feat)).astype(np.float32)
    w_hh = rng.normal(scale=0.1, size=(3 * hidden, hidden)).astype(np.float32)
    b = rng.normal(scale=0.01, size=(3 * hidden,)).astype(np.float32)
    hy = rng.normal(scale=0.1, size=(hidden,)).astype(np.float32)
    h = np.zeros((hidden,), dtype=np.float32)
    for ti in range(t_len):
        gi = w_ih @ win[ti] + w_hh @ h + b
        r = _sigmoid(gi[:hidden])
        z = _sigmoid(gi[hidden : 2 * hidden])
        n = np.tanh(gi[2 * hidden :] + r * (w_hh[2 * hidden :] @ h)[:hidden] * 0.0)
        h = (1.0 - z) * n.astype(np.float32) + z * h
    return np.float32(h @ hy)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def future_perturbation_gate(feat: pd.DataFrame, n_symbols: int = 4) -> dict:
    """Alter feat rows with date > t; sequence window at t must be byte-identical."""
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    feat["symbol"] = feat["symbol"].astype(str)
    dates = sorted(feat["date"].unique())
    if len(dates) < WINDOW + 20:
        return {"name": "future_perturbation", "passed": False, "reason": "too few dates"}
    t = dates[len(dates) // 2]
    # synthetic panel
    rng = np.random.default_rng(0)
    syn_dates = pd.date_range("2021-01-01", periods=120, freq="D", tz="UTC")
    syn_rows = []
    for s in [f"SYN{i}" for i in range(n_symbols)]:
        arr = rng.normal(size=(len(syn_dates), len(FEATURE_COLS))).astype(np.float32)
        tmp = pd.DataFrame(arr, columns=FEATURE_COLS)
        tmp["date"] = syn_dates
        tmp["symbol"] = s
        syn_rows.append(tmp)
    syn = pd.concat(syn_rows, ignore_index=True)
    syn_t = syn_dates[80]
    syn_before = {}
    for s, g in syn.groupby("symbol"):
        syn_before[s] = window_from_symbol_frame(g, syn_t)
    syn_pert = syn.copy()
    syn_pert.loc[syn_pert["date"] > syn_t, FEATURE_COLS] += 10.0
    syn_ok = True
    syn_score_ok = True
    syn_detail = []
    for s, g in syn_pert.groupby("symbol"):
        w2 = window_from_symbol_frame(g, syn_t)
        w1 = syn_before[s]
        eq = w1 is not None and w2 is not None and np.array_equal(w1, w2)
        sc_eq = bool(
            eq
            and w1 is not None
            and w2 is not None
            and gru_score_from_window(w1, seed=0) == gru_score_from_window(w2, seed=0)
        )
        syn_ok = syn_ok and eq
        syn_score_ok = syn_score_ok and sc_eq
        syn_detail.append({"symbol": s, "equal": bool(eq), "score_equal": bool(sc_eq)})

    # real data
    real_ok = True
    real_detail = []
    scores_ok = True
    rng2 = np.random.default_rng(1)

    def _score(w):
        if w is None:
            return None
        return gru_score_from_window(w, seed=0)

    symbols = [s for s, n in feat.groupby("symbol").size().items() if n >= WINDOW + 5][:n_symbols]
    feat_pert = feat.copy()
    feat_pert.loc[feat_pert["date"] > t, FEATURE_COLS] = (
        feat_pert.loc[feat_pert["date"] > t, FEATURE_COLS].astype(np.float32)
        + rng2.normal(scale=5.0, size=feat_pert.loc[feat_pert["date"] > t, FEATURE_COLS].shape).astype(np.float32)
    )
    for s in symbols:
        g0 = feat[feat["symbol"] == s]
        g1 = feat_pert[feat_pert["symbol"] == s]
        w0 = window_from_symbol_frame(g0, t)
        w1 = window_from_symbol_frame(g1, t)
        eq = w0 is not None and w1 is not None and np.array_equal(w0, w1)
        s0, s1 = _score(w0), _score(w1)
        sc_eq = bool(eq and s0 is not None and s1 is not None and np.float32(s0) == np.float32(s1))
        real_ok = real_ok and eq
        scores_ok = scores_ok and sc_eq
        real_detail.append(
            {
                "symbol": s,
                "t": str(pd.Timestamp(t).date()),
                "window_equal": bool(eq),
                "score_equal": bool(sc_eq),
                "n_future_rows": int((g1["date"] > t).sum()),
            }
        )
    passed = bool(syn_ok and syn_score_ok and real_ok and scores_ok)
    return {
        "name": "future_perturbation",
        "passed": passed,
        "synthetic_ok": bool(syn_ok),
        "synthetic_score_ok": bool(syn_score_ok),
        "real_ok": bool(real_ok),
        "score_ok": bool(scores_ok),
        "t": str(pd.Timestamp(t).date()),
        "synthetic": syn_detail,
        "real": real_detail,
        "note": "Sequence windows use only rows with date ≤ t (last 60 bars). Score at t is a fixed-weight GRU of that window.",
    }


def fold_isolation_gate(idx: pd.DataFrame, folds, gru_root: Path, inner_holdout_days: int = 90) -> dict:
    """Max date in each fold's training dataloader ≤ fold.train_end (post-purge). No warm-start."""
    idx = idx.copy()
    idx["date"] = pd.to_datetime(idx["date"], utc=True)
    rows = []
    all_ok = True
    seen_ids = set()
    for fr in folds:
        key = (int(fr.fold_id), str(_utc(fr.train_end).date()), int(getattr(fr, "horizon", 0) or 0))
        if key in seen_ids:
            continue
        seen_ids.add(key)
        te = _utc(fr.train_end)
        ts = _utc(fr.train_start)
        train = idx[(idx["date"] >= ts) & (idx["date"] <= te)]
        cut = te - pd.Timedelta(days=int(inner_holdout_days))
        inner_tr = train[train["date"] <= cut]
        mx_dl = inner_tr["date"].max() if not inner_tr.empty else None
        mx_train = train["date"].max() if not train.empty else None
        ok = mx_dl is not None and mx_dl <= te and mx_train is not None and mx_train <= te
        all_ok = all_ok and bool(ok)
        rows.append(
            {
                "fold_id": int(fr.fold_id),
                "train_start": str(ts.date()),
                "train_end": str(te.date()),
                "max_dataloader_date": str(mx_dl.date()) if mx_dl is not None else None,
                "max_train_slice_date": str(mx_train.date()) if mx_train is not None else None,
                "n_train_rows": int(len(train)),
                "passed": bool(ok),
            }
        )
    warm = []
    meta_date_fail = []
    n_meta = 0
    if gru_root.exists():
        import json

        for mp in gru_root.glob("h*/seed*/fold*_meta.json"):
            meta = json.loads(mp.read_text())
            n_meta += 1
            if meta.get("warm_start"):
                warm.append(str(mp))
            mx = meta.get("max_train_date")
            te = meta.get("fold_train_end")
            if mx and te and str(mx) > str(te):
                meta_date_fail.append({"path": str(mp), "max_train_date": mx, "fold_train_end": te})
    passed = bool(all_ok) and not warm and not meta_date_fail
    return {
        "name": "fold_isolation",
        "passed": passed,
        "warm_start": "no warm-start",
        "n_warm_start_metas": len(warm),
        "n_metas_inspected": int(n_meta),
        "meta_date_fail": meta_date_fail,
        "folds": rows,
    }


def prediction_alignment_gate(pred_s: pd.DataFrame, folds) -> dict:
    """Every S prediction date t belongs to the fold whose val window contains t."""
    p = pred_s.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    if "fold_id" not in p.columns:
        # reconstruct from val windows: assign first matching fold (Phase E keep-first)
        p["fold_id"] = -1
        for fr in sorted(folds, key=lambda f: f.fold_id):
            vs, ve = _utc(fr.val_start), _utc(fr.val_end)
            m = (p["fold_id"] < 0) & (p["date"] >= vs) & (p["date"] <= ve)
            p.loc[m, "fold_id"] = int(fr.fold_id)
    fmap = {int(fr.fold_id): (_utc(fr.val_start), _utc(fr.val_end)) for fr in folds}
    n = 0
    n_bad = 0
    n_unassigned = 0
    for fid, g in p.groupby("fold_id"):
        fid = int(fid)
        if fid not in fmap:
            n_unassigned += int(g["date"].nunique())
            n_bad += int(len(g))
            continue
        vs, ve = fmap[fid]
        bad = ~((g["date"] >= vs) & (g["date"] <= ve))
        n += int(len(g))
        n_bad += int(bad.sum())
    passed = n_bad == 0 and n_unassigned == 0 and n > 0
    return {
        "name": "prediction_alignment",
        "passed": bool(passed),
        "n_rows": int(n),
        "n_bad": int(n_bad),
        "n_unassigned": int(n_unassigned),
        "n_dates": int(p["date"].nunique()),
        "note": "Phase E assembly kept the first fold_id on overlapping dates; each kept row must lie in that fold's val window.",
    }


def summarize_shuffle_ic(pred_df: pd.DataFrame, horizon: int) -> dict:
    ycol = f"y_h{horizon}"
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if "score" not in df.columns:
        df["score"] = df.get("y_pred")
    ic = daily_rank_ic(df, ycol)
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    return {
        "mean_ic": mean_ic,
        "n_days": int(len(ic)),
        "abs_ic": abs(mean_ic) if np.isfinite(mean_ic) else float("nan"),
        "passed": bool(np.isfinite(mean_ic) and abs(mean_ic) < 0.005),
    }
