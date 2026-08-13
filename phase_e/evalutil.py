"""Shared Phase E evaluation: both universes, mechanical KEEP/VIABLE rules."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, evaluate_predictions, newey_west_t
from baseline.portfolio import run_tranche_portfolio

SIG_CRITERION = (
    "The signature block is KEPT for universe U only if trailing-18-month ΔRankIC on U "
    "≥ +0.005 at h=7 or h=10 AND full-OOS ΔRankIC on U ≥ 0 AND Δ positive in ≥60% of "
    "trailing-18-month folds on U. Verdicts are per-universe and mechanical."
)

S_VIABLE_CRITERION = (
    "S is VIABLE on U only if standalone trailing-18-month RankIC on U ≥ A0 − 0.01. "
    "The BLEND is KEPT on U only if trailing-18-month RankIC(BLEND) on U ≥ RankIC(A0) + 0.005 "
    "AND full-OOS RankIC(BLEND) on U ≥ RankIC(A0) AND positive Δ in ≥60% of trailing-18-month "
    "folds on U. Verdicts per-universe and mechanical."
)


def window_mask(dates: pd.Series, window: str, end: pd.Timestamp | None = None) -> pd.Series:
    d = pd.to_datetime(dates, utc=True)
    if window == "full":
        return pd.Series(True, index=dates.index)
    if window == "trail18m":
        end = end or d.max()
        start = end - pd.Timedelta(days=int(365 * 1.5))
        return (d >= start) & (d <= end)
    if window.startswith("y"):
        return d.dt.year == int(window[1:])
    raise ValueError(window)


def paired_delta_ic(ic_a: pd.Series, ic_b: pd.Series, horizon: int) -> dict:
    a = ic_a.copy()
    b = ic_b.copy()
    a.index = pd.to_datetime(a.index, utc=True)
    b.index = pd.to_datetime(b.index, utc=True)
    delta = (b - a).dropna()
    vals = delta.values.astype(float)
    return {
        "n_days": int(len(vals)),
        "mean_delta_ic": float(np.mean(vals)) if len(vals) else float("nan"),
        "nw_tstat": newey_west_t(vals, lag=horizon) if len(vals) else float("nan"),
    }


def fold_frac_positive(ic_a: pd.Series, ic_b: pd.Series, folds, trail_end: pd.Timestamp | None = None) -> dict:
    a = ic_a.copy()
    b = ic_b.copy()
    a.index = pd.to_datetime(a.index, utc=True)
    b.index = pd.to_datetime(b.index, utc=True)
    delta = (b - a).dropna()
    if trail_end is not None:
        start = trail_end - pd.Timedelta(days=int(365 * 1.5))
        delta = delta[(delta.index >= start) & (delta.index <= trail_end)]
    per = []
    for fr in folds:
        vs = pd.Timestamp(fr.val_start)
        ve = pd.Timestamp(fr.val_end)
        if vs.tzinfo is None:
            vs = vs.tz_localize("UTC")
        if ve.tzinfo is None:
            ve = ve.tz_localize("UTC")
        seg = delta[(delta.index >= vs) & (delta.index <= ve)]
        if len(seg) < 3:
            continue
        per.append({"fold_id": fr.fold_id, "delta": float(seg.mean()), "n": int(len(seg))})
    if not per:
        return {"n_folds": 0, "frac_positive": float("nan"), "n_positive": 0, "per_fold": []}
    pos = sum(1 for r in per if r["delta"] > 0)
    return {"n_folds": len(per), "frac_positive": float(pos / len(per)), "n_positive": pos, "per_fold": per}


def _prep_pred(pred: pd.DataFrame, feat: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = pred.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    ycol = f"y_h{horizon}"
    if "score" not in df.columns:
        if "y_pred" in df.columns:
            df["score"] = df["y_pred"]
        else:
            raise ValueError("pred missing score")
    if ycol not in df.columns:
        f = feat[["date", "symbol", ycol]].copy()
        f["date"] = pd.to_datetime(f["date"], utc=True)
        df = df.merge(f, on=["date", "symbol"], how="left")
    return df


def evaluate_pair(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    feat: pd.DataFrame,
    pit20: pd.DataFrame,
    pit120: pd.DataFrame,
    panel: pd.DataFrame,
    funding: pd.DataFrame | None,
    horizon: int,
    folds,
    cfg: dict,
    b_label: str = "B",
    compute_sharpe: bool = True,
) -> dict:
    """Paired A vs B on both universes, full / trail18m / per-year."""
    ycol = f"y_h{horizon}"
    if pred_a is None or pred_b is None or pred_a.empty or pred_b.empty:
        return {
            "horizon": horizon,
            "tables": [],
            "error": "empty_predictions",
            "delta_daily_ic": {},
        }
    a = _prep_pred(pred_a, feat, horizon)
    b = _prep_pred(pred_b, feat, horizon)
    end = max(a["date"].max(), b["date"].max())
    years = sorted(
        {int(y) for y in set(a["date"].dt.year.unique()) | set(b["date"].dt.year.unique()) if int(y) >= 2022}
    )
    windows = ["full", "trail18m"] + [f"y{y}" for y in years]
    unis = [("top20", pit20), ("pit120", pit120)]

    tables = []
    ic_store: dict[str, pd.Series] = {}
    for uni_name, uni in unis:
        for window in windows:
            ma = window_mask(a["date"], window, end=end)
            mb = window_mask(b["date"], window, end=end)
            aa = a.loc[ma]
            bb = b.loc[mb]
            if aa.empty or bb.empty:
                tables.append(
                    {
                        "horizon": horizon,
                        "universe": uni_name,
                        "window": window,
                        "A_ic": float("nan"),
                        f"{b_label}_ic": float("nan"),
                        "delta_ic": float("nan"),
                        "n_days": 0,
                    }
                )
                continue
            eva = evaluate_predictions(aa, horizon, universe=uni, label=uni_name)
            evb = evaluate_predictions(bb, horizon, universe=uni, label=uni_name)
            dlt = (
                float(evb["mean_ic"] - eva["mean_ic"])
                if np.isfinite(eva.get("mean_ic", np.nan)) and np.isfinite(evb.get("mean_ic", np.nan))
                else float("nan")
            )
            row = {
                "horizon": horizon,
                "universe": uni_name,
                "window": window,
                "A_ic": eva.get("mean_ic"),
                f"{b_label}_ic": evb.get("mean_ic"),
                "delta_ic": dlt,
                "A_nw": eva.get("nw_tstat"),
                f"{b_label}_nw": evb.get("nw_tstat"),
                "n_days": evb.get("n_days"),
            }
            tables.append(row)
            if window in ("full", "trail18m"):
                ic_store[f"{uni_name}_A_{window}"] = eva.get("ic_series", pd.Series(dtype=float))
                ic_store[f"{uni_name}_{b_label}_{window}"] = evb.get("ic_series", pd.Series(dtype=float))

    paired = {}
    fold_stats = {}
    delta_daily = {}
    for uni_name, _ in unis:
        paired[uni_name] = {
            w: paired_delta_ic(
                ic_store.get(f"{uni_name}_A_{w}", pd.Series(dtype=float)),
                ic_store.get(f"{uni_name}_{b_label}_{w}", pd.Series(dtype=float)),
                horizon,
            )
            for w in ("full", "trail18m")
        }
        fold_stats[uni_name] = {
            "trail18m": fold_frac_positive(
                ic_store.get(f"{uni_name}_A_full", pd.Series(dtype=float)),
                ic_store.get(f"{uni_name}_{b_label}_full", pd.Series(dtype=float)),
                folds,
                trail_end=end,
            ),
            "full": fold_frac_positive(
                ic_store.get(f"{uni_name}_A_full", pd.Series(dtype=float)),
                ic_store.get(f"{uni_name}_{b_label}_full", pd.Series(dtype=float)),
                folds,
                trail_end=None,
            ),
        }
        ia = ic_store.get(f"{uni_name}_A_full", pd.Series(dtype=float))
        ib = ic_store.get(f"{uni_name}_{b_label}_full", pd.Series(dtype=float))
        ia.index = pd.to_datetime(ia.index, utc=True) if len(ia) else ia.index
        ib.index = pd.to_datetime(ib.index, utc=True) if len(ib) else ib.index
        delta_daily[uni_name] = (ib - ia).dropna().sort_index()

    sharpe_delta = {}
    if compute_sharpe:
        port = cfg["portfolio"]
        daily = {}
        for label, pdf in [("A", a), (b_label, b)]:
            tres = run_tranche_portfolio(
                pdf,
                panel,
                feat,
                pit20,
                horizon=horizon,
                tau_pct=60.0,
                exit_hysteresis=port.get("exit_hysteresis", 0.6),
                gross_limit=port.get("gross_limit", 1.0),
                fee_bps=port.get("taker_fee_bps", 5.0),
                slip_bps=port.get("slippage_bps", 3.0),
                lag=0,
                apply_funding=True,
                funding=funding,
            )
            daily[label] = tres.get("daily_ret")
        ra, rb = daily["A"], daily[b_label]

        def _sh(x):
            return float(x.mean() / x.std() * np.sqrt(365)) if len(x) and x.std() > 0 else float("nan")

        if isinstance(ra, pd.Series) and isinstance(rb, pd.Series):
            idx = ra.index.intersection(rb.index)
            sharpe_delta = {
                "A_sharpe": _sh(ra.loc[idx]),
                f"{b_label}_sharpe": _sh(rb.loc[idx]),
                "delta_sharpe": _sh(rb.loc[idx]) - _sh(ra.loc[idx]) if len(idx) else float("nan"),
                "n_days": int(len(idx)),
            }

    def _pick(uni, window, key):
        for t in tables:
            if t["universe"] == uni and t["window"] == window:
                return t.get(key, float("nan"))
        return float("nan")

    summary_keys = {}
    for uni_name, _ in unis:
        summary_keys[uni_name] = {
            "delta_full": _pick(uni_name, "full", "delta_ic"),
            "delta_trail18m": _pick(uni_name, "trail18m", "delta_ic"),
            "A_full": _pick(uni_name, "full", "A_ic"),
            f"{b_label}_full": _pick(uni_name, "full", f"{b_label}_ic"),
            "A_trail18m": _pick(uni_name, "trail18m", "A_ic"),
            f"{b_label}_trail18m": _pick(uni_name, "trail18m", f"{b_label}_ic"),
            "frac_pos_folds_trail18m": fold_stats[uni_name]["trail18m"].get("frac_positive"),
        }

    return {
        "horizon": horizon,
        "tables": tables,
        "paired_nw": paired,
        "fold_stats": fold_stats,
        "sharpe_delta": sharpe_delta,
        "by_universe": summary_keys,
        "delta_daily_ic": delta_daily,
        "b_label": b_label,
    }


def apply_sig_keep(results_by_h: dict) -> dict:
    """KEEP per universe if any horizon satisfies all three clauses."""
    out = {"criterion": SIG_CRITERION, "universes": {}}
    for uni in ("top20", "pit120"):
        reasons = []
        details = {}
        for h, blob in results_by_h.items():
            u = (blob.get("by_universe") or {}).get(uni) or {}
            d18 = float(u.get("delta_trail18m", float("nan")))
            dfull = float(u.get("delta_full", float("nan")))
            frac = float(u.get("frac_pos_folds_trail18m", float("nan")))
            ok = np.isfinite(d18) and np.isfinite(dfull) and np.isfinite(frac) and d18 >= 0.005 and dfull >= 0.0 and frac >= 0.60
            details[f"h{h}"] = {
                "delta_trail18m": d18,
                "delta_full": dfull,
                "frac_pos_folds_trail18m": frac,
                "passes": bool(ok),
            }
            if ok:
                reasons.append(f"h={h}")
        out["universes"][uni] = {
            "verdict": "KEEP" if reasons else "KILL",
            "keep_reasons": reasons,
            "details": details,
        }
    return out


def apply_s_blend_criteria(results_s: dict, results_blend: dict) -> dict:
    """S viability + BLEND KEEP, per universe, mechanical."""
    out = {"criterion": S_VIABLE_CRITERION, "universes": {}}
    for uni in ("top20", "pit120"):
        s_details = {}
        viable_h = []
        for h, blob in results_s.items():
            u = (blob.get("by_universe") or {}).get(uni) or {}
            a18 = float(u.get("A_trail18m", float("nan")))
            s18 = float(u.get("S_trail18m", u.get("B_trail18m", float("nan"))))
            ok = np.isfinite(a18) and np.isfinite(s18) and s18 >= a18 - 0.01
            s_details[f"h{h}"] = {"A_trail18m": a18, "S_trail18m": s18, "threshold": a18 - 0.01 if np.isfinite(a18) else float("nan"), "passes": bool(ok)}
            if ok:
                viable_h.append(f"h={h}")
        b_details = {}
        keep_h = []
        for h, blob in results_blend.items():
            u = (blob.get("by_universe") or {}).get(uni) or {}
            a18 = float(u.get("A_trail18m", float("nan")))
            b18 = float(u.get("BLEND_trail18m", u.get("B_trail18m", float("nan"))))
            a_full = float(u.get("A_full", float("nan")))
            b_full = float(u.get("BLEND_full", u.get("B_full", float("nan"))))
            frac = float(u.get("frac_pos_folds_trail18m", float("nan")))
            ok = (
                np.isfinite(a18)
                and np.isfinite(b18)
                and np.isfinite(a_full)
                and np.isfinite(b_full)
                and np.isfinite(frac)
                and b18 >= a18 + 0.005
                and b_full >= a_full
                and frac >= 0.60
            )
            b_details[f"h{h}"] = {
                "A_trail18m": a18,
                "BLEND_trail18m": b18,
                "A_full": a_full,
                "BLEND_full": b_full,
                "frac_pos_folds_trail18m": frac,
                "passes": bool(ok),
            }
            if ok:
                keep_h.append(f"h={h}")
        out["universes"][uni] = {
            "S_viable": "VIABLE" if viable_h else "NOT_VIABLE",
            "S_reasons": viable_h,
            "S_details": s_details,
            "BLEND_verdict": "KEEP" if keep_h else "KILL",
            "BLEND_reasons": keep_h,
            "BLEND_details": b_details,
        }
    return out


def daily_score_spearman(pred_a: pd.DataFrame, pred_s: pd.DataFrame) -> dict:
    a = pred_a.copy()
    s = pred_s.copy()
    a["date"] = pd.to_datetime(a["date"], utc=True)
    s["date"] = pd.to_datetime(s["date"], utc=True)
    if "score" not in a.columns:
        a["score"] = a["y_pred"]
    if "score" not in s.columns:
        s["score"] = s["y_pred"]
    j = a[["date", "symbol", "score"]].merge(
        s[["date", "symbol", "score"]], on=["date", "symbol"], suffixes=("_a", "_s")
    )
    rows = []
    for dt, g in j.groupby("date"):
        gg = g.dropna(subset=["score_a", "score_s"])
        if len(gg) < 5:
            continue
        if gg["score_a"].nunique() < 2 or gg["score_s"].nunique() < 2:
            continue
        rows.append(float(gg["score_a"].corr(gg["score_s"], method="spearman")))
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n_days": int(len(arr)),
        "mean_spearman": float(arr.mean()) if len(arr) else float("nan"),
        "median_spearman": float(np.median(arr)) if len(arr) else float("nan"),
        "std_spearman": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
    }


def blend_scores(pred_a: pd.DataFrame, pred_s: pd.DataFrame) -> pd.DataFrame:
    """50/50 average of per-date z-scored A0 and S scores."""
    a = pred_a.copy()
    s = pred_s.copy()
    a["date"] = pd.to_datetime(a["date"], utc=True)
    s["date"] = pd.to_datetime(s["date"], utc=True)
    if "score" not in a.columns:
        a["score"] = a["y_pred"]
    if "score" not in s.columns:
        s["score"] = s["y_pred"]
    j = a.merge(s[["date", "symbol", "score"]], on=["date", "symbol"], suffixes=("_a", "_s"))

    def _z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return x * 0.0
        return (x - mu) / sd

    j["za"] = j.groupby("date")["score_a"].transform(_z)
    j["zs"] = j.groupby("date")["score_s"].transform(_z)
    j["score"] = 0.5 * j["za"] + 0.5 * j["zs"]
    keep = [c for c in j.columns if c in ("date", "symbol", "score", "fold_id") or str(c).startswith("y_h")]
    extra = [c for c in ("y_h7", "y_h10") if c in j.columns]
    cols = ["date", "symbol", "score"] + extra
    return j[cols].copy()


def plot_delta_ic(
    series_map: dict[str, pd.Series],
    out_path: Path,
    title: str,
) -> None:
    """series_map keys like 'top20 h=7'."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    unis = ["top20", "pit120"]
    end = None
    for ser in series_map.values():
        if ser is None or len(ser) == 0:
            continue
        end = ser.index.max() if end is None else max(end, ser.index.max())
    for col, uni in enumerate(unis):
        ax = axes[0, col]
        ax2 = axes[1, col]
        for name, ser in series_map.items():
            if not name.startswith(uni) or ser is None or len(ser) == 0:
                continue
            s = ser.sort_index().fillna(0.0)
            ax.plot(s.index, s.cumsum().values, label=name.replace(uni + " ", ""))
            if end is not None:
                start = end - pd.Timedelta(days=int(365 * 1.5))
                z = s[(s.index >= start) & (s.index <= end)]
                if len(z):
                    ax2.plot(z.index, z.cumsum().values, label=name.replace(uni + " ", ""))
        ax.set_title(f"{title} — {uni} full")
        ax.set_ylabel("Cum ΔIC")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax2.set_title(f"{uni} trailing-18m")
        ax2.set_ylabel("Cum ΔIC")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def aggregate_gain(metas: list[dict], cols: list[str]) -> dict:
    acc: dict[str, list[float]] = {c: [] for c in cols}
    for m in metas:
        g = m.get("feature_importance_gain") or {}
        for c in cols:
            if c in g:
                acc[c].append(float(g[c]))
    return {
        c: {
            "mean_gain": float(np.mean(v)) if v else 0.0,
            "median_gain": float(np.median(v)) if v else 0.0,
            "n": len(v),
        }
        for c, v in acc.items()
    }
