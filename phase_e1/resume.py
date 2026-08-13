"""Phase E.1 §2–§4 resume (extra seeds, tables, portfolio). No GRU retuning."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import evaluate_predictions
from baseline.portfolio import run_tranche_portfolio
from phase_e.evalutil import (
    apply_s_blend_criteria,
    blend_scores,
    daily_score_spearman,
    evaluate_pair,
    window_mask,
)
from phase_e1.report import CONFIRM_CRITERION, plot_blend_equity, plot_seeds, write_report

NEW_SEEDS = [45, 46, 47, 48, 49, 50]
OLD_SEEDS = [42, 43, 44]
ENSEMBLES = {
    "E42_44": [42, 43, 44],
    "E45_47": [45, 46, 47],
    "E48_50": [48, 49, 50],
    "GRAND9": [42, 43, 44, 45, 46, 47, 48, 49, 50],
}


def run_sections_2_to_4(
    *,
    feat: pd.DataFrame,
    pit20: pd.DataFrame,
    pit120: pd.DataFrame,
    panel: pd.DataFrame,
    funding,
    pred_a: dict,
    folds: dict,
    cfg: dict,
    gru_root: Path,
    extra_gpu: dict,
    horizons_trained: list[int],
    frozen_hash: str,
    gates: list,
    gates_ok: bool,
    rep_dir: Path,
    chart_dir: Path,
    volume_commit,
) -> dict:
    seed_pred: dict[tuple[int, int], pd.DataFrame] = {}

    def assemble(h: int, seeds: list[int]) -> pd.DataFrame:
        frames = []
        for seed in seeds:
            key = (int(h), int(seed))
            if key not in seed_pred:
                pieces = []
                d = gru_root / f"h{h}" / f"seed{seed}"
                for p in sorted(d.glob("fold*.parquet")):
                    pieces.append(pd.read_parquet(p))
                if not pieces:
                    seed_pred[key] = pd.DataFrame()
                else:
                    sdf = pd.concat(pieces, ignore_index=True)
                    sdf["date"] = pd.to_datetime(sdf["date"], utc=True)
                    sdf = sdf.sort_values(["date", "symbol", "fold_id"]).drop_duplicates(
                        ["date", "symbol"], keep="first"
                    )
                    sdf.to_parquet(gru_root / f"lgbm_seq_s_h{h}_seed{seed}.parquet", index=False)
                    seed_pred[key] = sdf
            sdf = seed_pred[key]
            if sdf.empty:
                continue
            frames.append(sdf[["date", "symbol", "score"]].rename(columns={"score": f"score_s{seed}"}))
        if not frames:
            return pd.DataFrame()
        merged = frames[0]
        for extra_f in frames[1:]:
            merged = merged.merge(extra_f, on=["date", "symbol"], how="outer")
        scols = [c for c in merged.columns if c.startswith("score_s")]
        merged["score"] = merged[scols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        ycol = f"y_h{h}"
        merged = merged.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
        return merged[["date", "symbol", "score", ycol]]

    ensemble_table = []
    ensemble_keep = {}
    nw_table = []
    corr_table = []
    year_table = []
    seed_points = []
    ens_hlines = []
    blend_by = {}
    s_by = {}

    for ens_name, seeds in ENSEMBLES.items():
        for h in horizons_trained:
            pred_s = assemble(h, seeds)
            if pred_s.empty:
                continue
            s_by[(ens_name, h)] = pred_s
            pred_b = blend_scores(pred_a[h], pred_s)
            blend_by[(ens_name, h)] = pred_b
            blob_s = evaluate_pair(
                pred_a[h], pred_s, feat, pit20, pit120, panel, funding, h, folds[h], cfg, b_label="S", compute_sharpe=False
            )
            blob_b = evaluate_pair(
                pred_a[h], pred_b, feat, pit20, pit120, panel, funding, h, folds[h], cfg, b_label="BLEND", compute_sharpe=False
            )
            keep = apply_s_blend_criteria({h: blob_s}, {h: blob_b})
            passing = set()
            for uni, u in (keep.get("universes") or {}).items():
                if u.get("BLEND_verdict") == "KEEP":
                    passing.add((uni, h))
            ensemble_keep.setdefault(ens_name, set()).update(passing)
            colors = {"E42_44": "orange", "E45_47": "green", "E48_50": "purple", "GRAND9": "black"}
            for uni in ("top20", "pit120"):
                u_s = (blob_s.get("by_universe") or {}).get(uni) or {}
                u_b = (blob_b.get("by_universe") or {}).get(uni) or {}
                keep_u = (keep.get("universes") or {}).get(uni) or {}
                keep_h = bool((keep_u.get("BLEND_details") or {}).get(f"h{h}", {}).get("passes"))
                for window in ("full", "trail18m"):
                    paired_s = ((blob_s.get("paired_nw") or {}).get(uni) or {}).get(window) or {}
                    paired_b = ((blob_b.get("paired_nw") or {}).get(uni) or {}).get(window) or {}
                    ensemble_table.append(
                        {
                            "ens": ens_name,
                            "horizon": h,
                            "universe": uni,
                            "window": window,
                            "A_ic": u_s.get("A_full" if window == "full" else "A_trail18m"),
                            "S_ic": u_s.get("S_full" if window == "full" else "S_trail18m"),
                            "BLEND_ic": u_b.get("BLEND_full" if window == "full" else "BLEND_trail18m"),
                            "delta_S": u_s.get("delta_full" if window == "full" else "delta_trail18m"),
                            "delta_BLEND": u_b.get("delta_full" if window == "full" else "delta_trail18m"),
                            "nw_t_BLEND": paired_b.get("nw_tstat"),
                            "frac_pos": (blob_b.get("fold_stats") or {}).get(uni, {}).get(
                                "trail18m" if window == "trail18m" else "full", {}
                            ).get("frac_positive"),
                            "keep_blend": keep_h,
                        }
                    )
                    ens_hlines.append(
                        {
                            "horizon": h,
                            "universe": uni,
                            "window": window,
                            "mean_ic": u_s.get("S_full" if window == "full" else "S_trail18m"),
                            "name": ens_name,
                            "color": colors.get(ens_name, "gray"),
                        }
                    )
                    if ens_name in ("E42_44", "GRAND9"):
                        nw_table.append({"ens": ens_name, "model": "S", "horizon": h, "universe": uni, "window": window, **paired_s})
                        nw_table.append({"ens": ens_name, "model": "BLEND", "horizon": h, "universe": uni, "window": window, **paired_b})
                        aa = pred_a[h].merge(pit20 if uni == "top20" else pit120, on=["date", "symbol"], how="inner")
                        ss = pred_s.merge(pit20 if uni == "top20" else pit120, on=["date", "symbol"], how="inner")
                        if window == "trail18m":
                            end = aa["date"].max()
                            m = window_mask(aa["date"], "trail18m", end=end)
                            aa = aa.loc[m]
                            ss = ss[ss["date"].isin(set(aa["date"]))]
                        corr_table.append(
                            {"ens": ens_name, "horizon": h, "universe": uni, "window": window, **daily_score_spearman(aa, ss)}
                        )
            if ens_name in ("E42_44", "GRAND9"):
                for t in blob_s.get("tables") or []:
                    if str(t["window"]).startswith("y"):
                        bt = next(
                            (x for x in (blob_b.get("tables") or []) if x["universe"] == t["universe"] and x["window"] == t["window"]),
                            {},
                        )
                        year_table.append(
                            {
                                "ens": ens_name,
                                "horizon": h,
                                "universe": t["universe"],
                                "year": t["window"][1:],
                                "A_ic": t.get("A_ic"),
                                "S_ic": t.get("S_ic", t.get("B_ic")),
                                "BLEND_ic": bt.get("BLEND_ic", bt.get("B_ic")),
                            }
                        )

    seed_dist = []
    for h in horizons_trained:
        for seed in OLD_SEEDS + NEW_SEEDS:
            ps = assemble(h, [seed])
            if ps.empty:
                continue
            for uni_name, uni in [("top20", pit20), ("pit120", pit120)]:
                ev_full = evaluate_predictions(ps, h, universe=uni, label=uni_name)
                end = ps["date"].max()
                m = window_mask(ps["date"], "trail18m", end=end)
                ev_18 = evaluate_predictions(ps.loc[m], h, universe=uni, label=uni_name)
                for window, ev in [("full", ev_full), ("trail18m", ev_18)]:
                    seed_points.append(
                        {"horizon": h, "seed": seed, "universe": uni_name, "window": window, "mean_ic": ev.get("mean_ic")}
                    )
    for h in horizons_trained:
        for uni in ("top20", "pit120"):
            for window in ("full", "trail18m"):
                vals = [
                    r["mean_ic"]
                    for r in seed_points
                    if r["horizon"] == h and r["universe"] == uni and r["window"] == window and np.isfinite(r.get("mean_ic", np.nan))
                ]
                if not vals:
                    continue
                arr = np.asarray(vals, float)
                seed_dist.append(
                    {
                        "horizon": h,
                        "universe": uni,
                        "window": window,
                        "min": float(arr.min()),
                        "median": float(np.median(arr)),
                        "max": float(arr.max()),
                        "n": int(len(arr)),
                    }
                )

    keep_lines = []
    passing_sets = []
    for name in ("E42_44", "E45_47", "E48_50"):
        sl = ensemble_keep.get(name, set())
        passing_sets.append(sl)
        keep_lines.append(f"{name} {'PASS' if sl else 'FAIL'} KEEP slices={sorted(list(sl)) if sl else 'NONE'}")
        print(f"[phaseE1] {name} pass/fail slices={sl}", flush=True)
    common = set.intersection(*passing_sets) if passing_sets else set()
    print(f"[phaseE1] common KEEP slices={common}", flush=True)

    def _nw(uni, h, window="trail18m"):
        for r in nw_table:
            if r.get("ens") == "GRAND9" and r.get("model") == "BLEND" and r.get("universe") == uni and r.get("horizon") == h and r.get("window") == window:
                return r.get("nw_tstat")
        return float("nan")

    confirmed_slice = None
    for uni, h in sorted(common):
        nwt = _nw(uni, h, "trail18m")
        if np.isfinite(nwt) and nwt >= 2.0:
            confirmed_slice = (uni, h, nwt)
            break

    def lag1_acf(pred, h):
        df = pred.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        if "score" not in df.columns:
            df["score"] = df["y_pred"]
        acs = []
        for _, g in df.groupby("symbol"):
            g = g.sort_values("date")
            s = g["score"].astype(float)
            if len(s) < 10:
                continue
            a = s.autocorr(lag=1)
            if np.isfinite(a):
                acs.append(float(a))
        return {"mean_acf": float(np.mean(acs)) if acs else float("nan"), "n_symbols": int(len(acs))}

    acf_table = []
    for ens_name in ("E42_44", "GRAND9"):
        for h in horizons_trained:
            acf_table.append({"ens": ens_name, "horizon": h, "model": "A0", **lag1_acf(pred_a[h], h)})
            if (ens_name, h) in s_by:
                acf_table.append({"ens": ens_name, "horizon": h, "model": "S", **lag1_acf(s_by[(ens_name, h)], h)})
            if (ens_name, h) in blend_by:
                acf_table.append({"ens": ens_name, "horizon": h, "model": "BLEND", **lag1_acf(blend_by[(ens_name, h)], h)})

    port = cfg["portfolio"]
    port_table = []
    equity_a = equity_b = None
    for ens_name in ("E42_44", "GRAND9"):
        for h in horizons_trained:
            pa, pb = pred_a[h], blend_by.get((ens_name, h))
            if pb is None:
                continue
            for tau_mode in ("pooled", "expanding"):
                print(f"[HB] portfolio ens={ens_name} h={h} tau_mode={tau_mode}", flush=True)
                ra = run_tranche_portfolio(
                    pa, panel, feat, pit20, horizon=h, tau_pct=60.0,
                    exit_hysteresis=port.get("exit_hysteresis", 0.6),
                    gross_limit=port.get("gross_limit", 1.0),
                    fee_bps=port.get("taker_fee_bps", 5.0),
                    slip_bps=port.get("slippage_bps", 3.0),
                    lag=0, apply_funding=True, funding=funding, tau_mode=tau_mode,
                )
                rb = run_tranche_portfolio(
                    pb, panel, feat, pit20, horizon=h, tau_pct=60.0,
                    exit_hysteresis=port.get("exit_hysteresis", 0.6),
                    gross_limit=port.get("gross_limit", 1.0),
                    fee_bps=port.get("taker_fee_bps", 5.0),
                    slip_bps=port.get("slippage_bps", 3.0),
                    lag=0, apply_funding=True, funding=funding, tau_mode=tau_mode,
                )
                da, db = ra.get("daily_ret"), rb.get("daily_ret")

                def _sh(x):
                    if x is None or not isinstance(x, pd.Series) or len(x) < 5 or x.std() == 0:
                        return float("nan")
                    return float(x.mean() / x.std() * np.sqrt(365))

                end = None
                if isinstance(da, pd.Series) and isinstance(db, pd.Series):
                    idxn = da.index.intersection(db.index)
                    da, db = da.loc[idxn], db.loc[idxn]
                    end = idxn.max() if len(idxn) else None
                start = end - pd.Timedelta(days=int(365 * 1.5)) if end is not None else None
                for window, mask in [
                    ("full", None),
                    ("trail18m", (da.index >= start) & (da.index <= end) if start is not None else None),
                ]:
                    xa = da if mask is None else da.loc[mask]
                    xb = db if mask is None else db.loc[mask]
                    port_table.append(
                        {
                            "ens": ens_name,
                            "horizon": h,
                            "tau_mode": tau_mode,
                            "window": window,
                            "A_sharpe": _sh(xa),
                            "B_sharpe": _sh(xb),
                            "delta_sharpe": _sh(xb) - _sh(xa) if np.isfinite(_sh(xa)) and np.isfinite(_sh(xb)) else float("nan"),
                            "A_to": ra.get("ann_turnover"),
                            "B_to": rb.get("ann_turnover"),
                            "A_npos": ra.get("avg_n_positions"),
                            "B_npos": rb.get("avg_n_positions"),
                            "A_flat": ra.get("pct_flat_days"),
                            "B_flat": rb.get("pct_flat_days"),
                        }
                    )
                if ens_name == "GRAND9" and tau_mode == "pooled" and h == (confirmed_slice[1] if confirmed_slice else horizons_trained[0]):
                    if isinstance(ra.get("equity"), pd.DataFrame):
                        ea = ra["equity"].copy()
                        ea["date"] = pd.to_datetime(ea["date"], utc=True)
                        equity_a = ea.set_index("date")["equity"]
                    if isinstance(rb.get("equity"), pd.DataFrame):
                        eb = rb["equity"].copy()
                        eb["date"] = pd.to_datetime(eb["date"], utc=True)
                        equity_b = eb.set_index("date")["equity"]

    h_iv = confirmed_slice[1] if confirmed_slice else (horizons_trained[0] if horizons_trained else 7)
    iv_ok = False
    iv_detail = []
    for tau_mode in ("pooled", "expanding"):
        full = next((r for r in port_table if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == tau_mode and r["window"] == "full"), None)
        tr = next((r for r in port_table if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == tau_mode and r["window"] == "trail18m"), None)
        if not full or not tr:
            continue
        ok = np.isfinite(full["delta_sharpe"]) and np.isfinite(tr["delta_sharpe"]) and full["delta_sharpe"] >= -0.10 and tr["delta_sharpe"] >= 0.0
        iv_detail.append({"tau_mode": tau_mode, "delta_full": full["delta_sharpe"], "delta_18": tr["delta_sharpe"], "ok": ok})
        if ok:
            iv_ok = True

    ii_ok = bool(common)
    iii_ok = confirmed_slice is not None
    verdict = "CONFIRMED" if (gates_ok and ii_ok and iii_ok and iv_ok) else "NOT CONFIRMED"
    details = {
        "i_gates": gates_ok,
        "ii_common_slices": [list(x) for x in sorted(common)],
        "iii_slice": confirmed_slice,
        "iv": iv_detail,
        "iv_ok": iv_ok,
        "dropped_h7": extra_gpu.get("dropped_h7"),
        "horizons_trained": horizons_trained,
    }
    print(f"[phaseE1] VERDICT={verdict} details={details}", flush=True)

    plot_seeds(seed_points, ens_hlines, chart_dir / "phaseE1_seeds.png")
    if equity_a is not None and equity_b is not None:
        plot_blend_equity(equity_a, equity_b, chart_dir / "phaseE1_blend_equity.png")

    write_report(
        rep_dir / "phaseE1_resume_sections.md",
        frozen_hash=frozen_hash,
        verdict=verdict,
        verdict_details=details,
        gates=gates,
        gates_ok=gates_ok,
        budget=extra_gpu.get("projection"),
        horizons_trained=horizons_trained,
        ensemble_keep_lines=keep_lines + [f"common={sorted(common)}", f"iii={confirmed_slice}", f"iv_ok={iv_ok}"],
        ensemble_table=ensemble_table,
        seed_dist=seed_dist,
        nw_table=nw_table,
        corr_table=corr_table,
        year_table=year_table,
        acf_table=acf_table,
        port_table=port_table,
    )
    pfull = next((r for r in port_table if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == "pooled" and r["window"] == "full"), {})
    p18 = next((r for r in port_table if r.get("ens") == "GRAND9" and r["horizon"] == h_iv and r["tau_mode"] == "pooled" and r["window"] == "trail18m"), {})
    stdout = {
        "keep_lines": keep_lines,
        "verdict": verdict,
        "portfolio": (
            f"PORTFOLIO h={h_iv} pooled ΔSharpe full={pfull.get('delta_sharpe')} trail18={p18.get('delta_sharpe')} "
            f"ΔTO={None if not pfull else (pfull.get('B_to') or 0) - (pfull.get('A_to') or 0)}"
        ),
    }
    for line in keep_lines:
        print(f"ENSEMBLE: {line}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(stdout["portfolio"], flush=True)
    blob = {
        "verdict": verdict,
        "details": details,
        "criterion": CONFIRM_CRITERION,
        "ensemble_table": ensemble_table,
        "seed_dist": seed_dist,
        "nw_table": nw_table,
        "corr_table": corr_table,
        "year_table": year_table,
        "acf_table": acf_table,
        "port_table": port_table,
        "keep_lines": keep_lines,
        "stdout": stdout,
        "extra_gpu": extra_gpu,
    }
    volume_commit()
    return blob
