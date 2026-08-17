"""Decay curve on existing OOS scores. No training, no new model.

Top-40, k in {0, 3, 5, 10, 20, 40, 60}:
  IC_k  = mean RankIC(score_t, y_{t+k -> t+k+7}), NW-t lag=h=7
  rho_k = mean_t Spearman CS(score_t, score_{t+k})
  c_k   = rho_k / (IC_k / IC_0)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, summarize_ic
from gating_ladder.persist_diag import YCOL, cs_lag_rho


KS = (0, 3, 5, 10, 20, 40, 60)
H = 7
CONFIRM_KS = (5, 10, 20, 40)
BAND = (0.32, 0.48)


def _shifted_ic(df: pd.DataFrame, k: int) -> dict:
    tmp = df.sort_values(["symbol", "date"]).copy()
    if k == 0:
        tmp["y_shift"] = tmp[YCOL]
    else:
        tmp["y_shift"] = tmp.groupby("symbol", sort=False)[YCOL].shift(-int(k))
    ic = daily_rank_ic(tmp.dropna(subset=["y_shift"]), "y_shift", score_col="score")
    return summarize_ic(ic, H)


def _is_monotonic(xs: list[float]) -> str:
    vals = [x for x in xs if np.isfinite(x)]
    if len(vals) < 3:
        return "na"
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if all(d > 1e-12 for d in diffs):
        return "increasing"
    if all(d < -1e-12 for d in diffs):
        return "decreasing"
    return "not_monotonic"


def main() -> int:
    pred = pd.read_parquet("/data/quant/predictions/lgbm_price_only_h7.parquet")
    pit40 = pd.read_parquet("/data/quant/universe/top40_pit.parquet")
    pred["date"] = pd.to_datetime(pred["date"], utc=True)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)
    df = pred.merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
    df = df.dropna(subset=["score", YCOL]).copy()
    print(f"[decay] top40 rows={len(df)} dates={df['date'].nunique()}", flush=True)

    wide = (
        df.pivot_table(index="date", columns="symbol", values="score", aggfunc="mean")
        .sort_index()
    )

    rows = []
    ic0 = float("nan")
    for k in KS:
        print(f"[decay] k={k}", flush=True)
        ic = _shifted_ic(df, k)
        if k == 0:
            rho_mean = 1.0
            rho_n = int(wide.shape[0])
            ic0 = float(ic["mean_ic"])
        else:
            rho = cs_lag_rho(wide, k, "spearman")
            rho_mean = float(rho.mean()) if len(rho) else float("nan")
            rho_n = int(len(rho))
        ic_k = float(ic["mean_ic"])
        ic_ratio = (
            float(ic_k / ic0) if np.isfinite(ic0) and ic0 != 0 else float("nan")
        )
        c = (
            float(rho_mean / ic_ratio)
            if np.isfinite(rho_mean) and np.isfinite(ic_ratio) and ic_ratio != 0
            else float("nan")
        )
        row = {
            "k": int(k),
            "ic_k": ic_k,
            "nw_t": float(ic["nw_tstat"]),
            "n_days": int(ic["n_days"]),
            "rho_k": rho_mean,
            "rho_n": rho_n,
            "ic_k_over_ic_0": ic_ratio,
            "c_rho_over_ic_ratio": c,
        }
        rows.append(row)
        print(f"[decay] {row}", flush=True)

    confirm = [r for r in rows if r["k"] in CONFIRM_KS]
    c_vals = [r["c_rho_over_ic_ratio"] for r in confirm]
    in_band = all(np.isfinite(c) and BAND[0] <= c <= BAND[1] for c in c_vals)
    mono = _is_monotonic(c_vals)
    mono_all = _is_monotonic([r["c_rho_over_ic_ratio"] for r in rows if r["k"] > 0])

    ic_high_rho_collapse = False
    collapse_notes = []
    for r in rows:
        if r["k"] == 0:
            continue
        ic_ratio = r["ic_k_over_ic_0"]
        rho = r["rho_k"]
        if np.isfinite(ic_ratio) and np.isfinite(rho) and ic_ratio >= 0.5 and rho <= 0.10:
            ic_high_rho_collapse = True
            collapse_notes.append(
                f"k={r['k']}: IC_k/IC_0={ic_ratio:.3f} rho_k={rho:.3f}"
            )

    falsified = (not in_band) or (mono in {"increasing", "decreasing"}) or ic_high_rho_collapse
    verdict = "FALSIFIED" if falsified else "CONFIRMED"

    out = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": "top40",
        "ks": list(KS),
        "confirm_ks": list(CONFIRM_KS),
        "band": list(BAND),
        "ic0": ic0,
        "rows": rows,
        "c_on_confirm_ks": c_vals,
        "c_in_band_5_10_20_40": in_band,
        "c_monotonic_confirm_ks": mono,
        "c_monotonic_k_gt_0": mono_all,
        "ic_high_rho_collapse": ic_high_rho_collapse,
        "collapse_notes": collapse_notes,
        "verdict": verdict,
        "model": (
            "score = persistent state + noise; "
            "IC_k/IC_0 = rho_theta(k); rho_k = rho_theta(k)*SNR; "
            "c = rho_k/(IC_k/IC_0) constant ~0.40"
        ),
        "no_training": True,
        "stage_a": False,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/fase1_decay_curve.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[decay] verdict={verdict} in_band={in_band} mono={mono}", flush=True)
    print("[decay] wrote results/fase1_decay_curve.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
