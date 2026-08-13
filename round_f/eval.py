"""Round F KEEP / COMBO verdicts and A0 gain ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import evaluate_predictions
from baseline.features import FEATURE_COLS
from phase_d.ablation import fold_frac_positive, paired_delta_ic
from phase_d2.constants import TRAIL_DAYS
from phase_d2.metrics import _sharpe, window_slice
from round_f.constants import KEEP_CRITERION


def rank_a0_gains(volume_root: Path):
    acc = {c: [] for c in FEATURE_COLS}
    paths = list(Path(volume_root).glob("**/*meta*.json"))
    n_used = 0
    for p in paths:
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        items = blob if isinstance(blob, list) else [blob]
        for m in items:
            if not isinstance(m, dict):
                continue
            gi = m.get("feature_importance_gain") or {}
            cols = m.get("feature_cols") or []
            # A0-only metas: feature set equals FEATURE_COLS (no extras)
            if cols and set(cols) != set(FEATURE_COLS):
                continue
            if not gi:
                continue
            hit = [c for c in FEATURE_COLS if c in gi]
            if len(hit) < 20:
                continue
            n_used += 1
            for c in FEATURE_COLS:
                if c in gi:
                    acc[c].append(float(gi[c]))
    ranked = sorted(
        ((c, float(np.mean(v)) if v else 0.0) for c, v in acc.items()),
        key=lambda kv: kv[1],
    )
    return ranked, n_used


def ic_tables_vs_a0(pred_a, pred_b, feat, universe, horizon, label, folds) -> dict:
    ycol = f"y_h{horizon}"
    a, b = pred_a.copy(), pred_b.copy()
    a["date"] = pd.to_datetime(a["date"], utc=True)
    b["date"] = pd.to_datetime(b["date"], utc=True)
    if ycol not in a.columns:
        a = a.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    if ycol not in b.columns:
        b = b.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    eva = evaluate_predictions(a, horizon, universe=universe, label=label)
    evb = evaluate_predictions(b, horizon, universe=universe, label=label)
    ia = eva.get("ic_series", pd.Series(dtype=float))
    ib = evb.get("ic_series", pd.Series(dtype=float))
    end = max(a["date"].max(), b["date"].max())
    start = end - pd.Timedelta(days=TRAIL_DAYS)

    def _win(ic, window):
        if ic is None or len(ic) == 0:
            return pd.Series(dtype=float)
        ic = ic.copy()
        ic.index = pd.DatetimeIndex(pd.to_datetime(ic.index, utc=True))
        if window == "full":
            return ic
        if window == "trail18m":
            return ic[(ic.index >= start) & (ic.index <= end)]
        y = int(window[1:])
        return ic[ic.index.year == y]

    years = sorted(set(ia.index.year) | set(ib.index.year)) if len(ia) or len(ib) else []
    tables = []
    for window in ["full", "trail18m"] + [f"y{y}" for y in years if y >= 2022]:
        sa, sb = _win(ia, window), _win(ib, window)
        ma, mb = (float(sa.mean()) if len(sa) else float("nan")), (float(sb.mean()) if len(sb) else float("nan"))
        tables.append(
            {
                "horizon": horizon,
                "universe": label,
                "window": window,
                "A_ic": ma,
                "B_ic": mb,
                "delta_ic": float(mb - ma) if np.isfinite(ma) and np.isfinite(mb) else float("nan"),
                "n_days": int(min(len(sa), len(sb))),
            }
        )
    paired = {w: paired_delta_ic(_win(ia, w), _win(ib, w), horizon) for w in ("full", "trail18m")}
    trail_idx = pd.DatetimeIndex(pd.to_datetime(_win(ia, "trail18m").index, utc=True))
    trail_mask = pd.Series(True, index=trail_idx) if len(trail_idx) else None
    frac = fold_frac_positive(ia, ib, folds, date_mask=trail_mask)
    return {
        "horizon": horizon,
        "universe": label,
        "tables": tables,
        "paired_nw": paired,
        "delta_full": next((t["delta_ic"] for t in tables if t["window"] == "full"), float("nan")),
        "delta_trail18m": next((t["delta_ic"] for t in tables if t["window"] == "trail18m"), float("nan")),
        "frac_pos_trail18m": frac.get("frac_positive"),
        "fold_stats": frac,
        "delta_daily": (ib - ia).dropna().sort_index(),
    }


def apply_keep(block: str, ic: dict, port_d18: dict, prune: bool = False) -> dict:
    """ic[(uni,h)] blobs; port_d18[uni] = trail18m Sharpe Δ vs A0 on that book's portfolio."""
    rows = []
    uni_verdicts = {}
    thr18 = 0.0 if prune else 0.005
    thrfull = -0.002 if prune else 0.0
    for uni in ("top20", "top40"):
        h_ok = []
        for h in (7, 10):
            blob = ic.get((uni, h)) or {}
            d18 = float(blob.get("delta_trail18m", float("nan")))
            dfull = float(blob.get("delta_full", float("nan")))
            frac = float(blob.get("frac_pos_trail18m", float("nan")))
            ok_h = (
                np.isfinite(d18)
                and np.isfinite(dfull)
                and np.isfinite(frac)
                and d18 >= thr18
                and dfull >= thrfull
                and frac >= 0.60
            )
            h_ok.append(ok_h)
            rows.append(
                {
                    "block": block,
                    "universe": uni,
                    "horizon": h,
                    "delta_ic_trail18m": d18,
                    "delta_ic_full": dfull,
                    "frac_pos_trail18m": frac,
                    "ic_pass_h": bool(ok_h),
                    "need_d18": thr18,
                    "need_dfull": thrfull,
                }
            )
        pdelta = float(port_d18.get(uni, float("nan")))
        kept = bool(any(h_ok) and np.isfinite(pdelta) and pdelta >= 0.0)
        uni_verdicts[uni] = {
            "verdict": "KEEP" if kept else "KILL",
            "ic_any_h": bool(any(h_ok)),
            "port_delta_trail18m": pdelta,
            "port_ok": bool(np.isfinite(pdelta) and pdelta >= 0.0),
        }
    return {
        "block": block,
        "criterion": KEEP_CRITERION,
        "prune": prune,
        "rows": rows,
        "by_universe": uni_verdicts,
    }


def combo_from_sleeves(p1: dict, p2: dict) -> dict:
    r1 = p1.get("daily_ret")
    r2 = p2.get("daily_ret")
    if not isinstance(r1, pd.Series) or not isinstance(r2, pd.Series):
        return {"error": "missing daily_ret"}
    r1 = r1.copy()
    r2 = r2.copy()
    r1.index = pd.DatetimeIndex(pd.to_datetime(r1.index, utc=True))
    r2.index = pd.DatetimeIndex(pd.to_datetime(r2.index, utc=True))
    idx = r1.index.intersection(r2.index)
    a = r1.reindex(idx).fillna(0.0)
    b = r2.reindex(idx).fillna(0.0)
    combo = 0.5 * a + 0.5 * b
    eq = (1.0 + combo).cumprod()
    maxdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")
    corr = float(a.corr(b)) if len(a) > 5 else float("nan")
    to1 = float(p1.get("ann_turnover", float("nan")))
    to2 = float(p2.get("ann_turnover", float("nan")))
    years = sorted({int(y) for y in idx.year.unique() if y >= 2022})
    return {
        "n_days": int(len(idx)),
        "net_sharpe_full": _sharpe(combo),
        "net_sharpe_trail18m": _sharpe(window_slice(combo, "trail18m")),
        "net_sharpe_by_year": {y: _sharpe(window_slice(combo, f"y{y}")) for y in years},
        "max_drawdown": maxdd,
        "sleeve_corr": corr,
        "ann_turnover": 0.5 * to1 + 0.5 * to2 if np.isfinite(to1) and np.isfinite(to2) else float("nan"),
        "p1_full": _sharpe(a),
        "p2_full": _sharpe(b),
        "p1_trail18m": _sharpe(window_slice(a, "trail18m")),
        "p2_trail18m": _sharpe(window_slice(b, "trail18m")),
        "daily_ret": combo,
        "equity": pd.DataFrame({"date": idx, "equity": eq.values}),
        "p1_equity": p1.get("equity"),
        "p2_equity": p2.get("equity"),
    }


def apply_combo_criterion(combo: dict, criterion: str) -> dict:
    c18, cfull = float(combo.get("net_sharpe_trail18m", float("nan"))), float(combo.get("net_sharpe_full", float("nan")))
    m18 = max(float(combo.get("p1_trail18m", float("nan"))), float(combo.get("p2_trail18m", float("nan"))))
    mfull = max(float(combo.get("p1_full", float("nan"))), float(combo.get("p2_full", float("nan"))))
    ok = np.isfinite(c18) and np.isfinite(cfull) and c18 >= m18 - 0.10 and cfull >= mfull - 0.10
    return {
        "verdict": "ADOPTED" if ok else "REJECTED",
        "criterion": criterion,
        "combo_trail18m": c18,
        "combo_full": cfull,
        "need_trail18m": m18 - 0.10,
        "need_full": mfull - 0.10,
        "max_p_trail18m": m18,
        "max_p_full": mfull,
        "pass": bool(ok),
    }
