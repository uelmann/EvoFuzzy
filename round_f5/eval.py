"""Round F5 sleeve selection, stability diagnostic, COMBO′ hurdle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase_d2.metrics import _sharpe, window_slice
from round_f5.constants import (
    COMBO_F_FULL,
    COMBO_F_TRAIL,
    COMBO_PRIME_CRITERION,
    SLEEVE_CRITERION,
)


def ic_block_pass(blob_h7: dict, blob_h10: dict) -> dict:
    """House block RankIC gate vs A0: trail ≥ +0.005, full ≥ 0, ≥60% pos folds, at h=7 or h=10."""
    rows = []
    any_ok = False
    for h, blob in ((7, blob_h7), (10, blob_h10)):
        d18 = float((blob or {}).get("delta_trail18m", float("nan")))
        dfull = float((blob or {}).get("delta_full", float("nan")))
        frac = float((blob or {}).get("frac_pos_trail18m", float("nan")))
        ok = (
            np.isfinite(d18)
            and np.isfinite(dfull)
            and np.isfinite(frac)
            and d18 >= 0.005
            and dfull >= 0.0
            and frac >= 0.60
        )
        any_ok = any_ok or ok
        rows.append(
            {
                "horizon": h,
                "delta_ic_trail18m": d18,
                "delta_ic_full": dfull,
                "frac_pos_trail18m": frac,
                "pass": bool(ok),
            }
        )
    return {"pass": bool(any_ok), "rows": rows}


def apply_sleeve_rule(cands: dict, c3_ic: dict) -> dict:
    """cands[id] has net_sharpe_full / net_sharpe_trail18m. C0 is incumbent."""
    inc = cands["C0"]
    i18 = float(inc.get("net_sharpe_trail18m", float("nan")))
    ifull = float(inc.get("net_sharpe_full", float("nan")))
    need18 = i18 + 0.15
    needfull = ifull - 0.10
    rows = []
    qualifying = []
    for cid in ("C1", "C2", "C3"):
        c = cands[cid]
        t18 = float(c.get("net_sharpe_trail18m", float("nan")))
        tfull = float(c.get("net_sharpe_full", float("nan")))
        sharpe_ok = np.isfinite(t18) and np.isfinite(tfull) and t18 >= need18 and tfull >= needfull
        ic_ok = True
        if cid == "C3":
            ic_ok = bool(c3_ic.get("pass"))
        ok = bool(sharpe_ok and ic_ok)
        rows.append(
            {
                "id": cid,
                "trail18m": t18,
                "full": tfull,
                "need_trail18m": need18,
                "need_full": needfull,
                "sharpe_ok": bool(sharpe_ok),
                "ic_ok": bool(ic_ok),
                "qualify": ok,
            }
        )
        if ok:
            qualifying.append(cid)
    if qualifying:
        selected = max(qualifying, key=lambda k: float(cands[k]["net_sharpe_trail18m"]))
        verdict = "REPLACE"
    else:
        selected = "C0"
        verdict = "INCUMBENT"
    return {
        "criterion": SLEEVE_CRITERION,
        "incumbent_trail18m": i18,
        "incumbent_full": ifull,
        "need_trail18m": need18,
        "need_full": needfull,
        "rows": rows,
        "qualifying": qualifying,
        "selected": selected,
        "verdict": verdict,
    }


def stability_vs_incumbent(sel: dict, inc: dict) -> dict:
    def _npos(blob):
        s = blob.get("daily_n_pos")
        if not isinstance(s, pd.Series) or s.empty:
            return pd.Series(dtype=float)
        s = s.copy()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
        return s.astype(float)

    def _flat(blob):
        s = blob.get("daily_flat")
        if not isinstance(s, pd.Series) or s.empty:
            n = _npos(blob)
            if n.empty:
                return pd.Series(dtype=float)
            return (n <= 0).astype(float)
        s = s.copy()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
        return s.astype(float)

    ns, ni = _npos(sel), _npos(inc)
    fs, fi = _flat(sel), _flat(inc)
    idx = ns.index.intersection(ni.index)
    ns, ni = ns.reindex(idx), ni.reindex(idx)
    fs, fi = fs.reindex(idx), fi.reindex(idx)
    dns = ns.diff().abs()
    dni = ni.diff().abs()

    def _dist(d):
        d = d.dropna()
        if d.empty:
            return {}
        return {
            "mean": float(d.mean()),
            "median": float(d.median()),
            "p90": float(np.nanpercentile(d.to_numpy(), 90)),
            "max": float(d.max()),
            "frac_ge_10": float((d >= 10).mean()),
            "frac_ge_20": float((d >= 20).mean()),
        }

    years = sorted({int(y) for y in idx.year.unique() if y >= 2022})
    by_year = []
    for y in years:
        m = idx.year == y
        by_year.append(
            {
                "year": y,
                "sel_avg_n_pos": float(ns[m].mean()) if m.any() else float("nan"),
                "inc_avg_n_pos": float(ni[m].mean()) if m.any() else float("nan"),
                "sel_pct_flat": float(fs[m].mean()) if m.any() else float("nan"),
                "inc_pct_flat": float(fi[m].mean()) if m.any() else float("nan"),
            }
        )
    return {
        "n_days": int(len(idx)),
        "sel_avg_n_pos": float(ns.mean()) if len(ns) else float("nan"),
        "inc_avg_n_pos": float(ni.mean()) if len(ni) else float("nan"),
        "sel_pct_flat": float(fs.mean()) if len(fs) else float("nan"),
        "inc_pct_flat": float(fi.mean()) if len(fi) else float("nan"),
        "sel_dpos": _dist(dns),
        "inc_dpos": _dist(dni),
        "by_year": by_year,
        "note": "Information only; no verdict. Pathological = large daily |Δn| swings vs incumbent.",
    }


def apply_combo_prime(combo_p: dict) -> dict:
    c18 = float(combo_p.get("net_sharpe_trail18m", float("nan")))
    cfull = float(combo_p.get("net_sharpe_full", float("nan")))
    need18 = COMBO_F_TRAIL - 0.05
    needfull = COMBO_F_FULL - 0.05
    ok = np.isfinite(c18) and np.isfinite(cfull) and c18 >= need18 and cfull >= needfull
    return {
        "criterion": COMBO_PRIME_CRITERION,
        "combo_prime_trail18m": c18,
        "combo_prime_full": cfull,
        "combo_f_trail18m": COMBO_F_TRAIL,
        "combo_f_full": COMBO_F_FULL,
        "need_trail18m": need18,
        "need_full": needfull,
        "pass": bool(ok),
        "verdict": "ADOPTED" if ok else "REJECTED",
        "reference": "COMBO′" if ok else "COMBO (Round F)",
    }


def year_pos_flat(blob: dict) -> list[dict]:
    n = blob.get("daily_n_pos")
    f = blob.get("daily_flat")
    r = blob.get("daily_ret")
    if not isinstance(n, pd.Series) or n.empty:
        return []
    n = n.copy()
    n.index = pd.DatetimeIndex(pd.to_datetime(n.index, utc=True))
    if isinstance(f, pd.Series) and len(f):
        f = f.copy()
        f.index = pd.DatetimeIndex(pd.to_datetime(f.index, utc=True))
    else:
        f = (n <= 0).astype(float)
    rows = []
    for y in sorted({int(x) for x in n.index.year.unique() if x >= 2022}):
        m = n.index.year == y
        rec = {
            "year": y,
            "avg_n_pos": float(n[m].mean()),
            "pct_flat": float(f.reindex(n.index)[m].mean()),
        }
        if isinstance(r, pd.Series) and len(r):
            rr = r.copy()
            rr.index = pd.DatetimeIndex(pd.to_datetime(rr.index, utc=True))
            rec["net_sharpe"] = _sharpe(window_slice(rr, f"y{y}"))
        rows.append(rec)
    return rows
