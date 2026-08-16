"""Viability, HIGH/LOW diagnostics, identical-days comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimetau.constants import (
    ANNUALIZATION,
    DELTA_FULL,
    DELTA_TRAIL_FLOOR,
    REGIME_BASE,
    REGIME_HIGH,
    REGIME_LOW,
    VIABILITY_CRITERION,
)


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION)) if len(x) and x.std() > 0 else 0.0


def apply_viability(reg_full: float, reg_trail: float, ref_full: float, ref_trail: float) -> dict:
    need_full = float(ref_full) + float(DELTA_FULL)
    need_trail = float(ref_trail) + float(DELTA_TRAIL_FLOOR)
    full_ok = np.isfinite(reg_full) and np.isfinite(need_full) and float(reg_full) >= need_full
    trail_ok = np.isfinite(reg_trail) and np.isfinite(need_trail) and float(reg_trail) >= need_trail
    ok = bool(full_ok and trail_ok)
    return {
        "criterion": VIABILITY_CRITERION,
        "label": "VIABLE" if ok else "NOT VIABLE",
        "pass": ok,
        "full_ok": bool(full_ok),
        "trail_ok": bool(trail_ok),
        "reg_full": float(reg_full),
        "reg_trail": float(reg_trail),
        "ref_full": float(ref_full),
        "ref_trail": float(ref_trail),
        "need_full": float(need_full),
        "need_trail": float(need_trail),
        "delta_full": float(reg_full) - float(ref_full) if np.isfinite(reg_full) and np.isfinite(ref_full) else float("nan"),
        "delta_trail": float(reg_trail) - float(ref_trail)
        if np.isfinite(reg_trail) and np.isfinite(ref_trail)
        else float("nan"),
    }


def _as_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    return out


def regime_slice_sharpe(rets: pd.Series, regime: pd.Series) -> dict:
    r = _as_utc(rets).astype(float)
    g = _as_utc(regime).astype(float)
    r, g = r.align(g, join="inner")
    out = {}
    for code, name in ((REGIME_HIGH, "HIGH"), (REGIME_LOW, "LOW"), (REGIME_BASE, "BASE")):
        m = g.to_numpy() == code
        sl = r[m]
        out[name] = {
            "n": int(m.sum()),
            "frac": float(m.mean()) if len(m) else 0.0,
            "sharpe": _sharpe(sl) if int(m.sum()) > 5 else float("nan"),
            "mean": float(sl.mean()) if int(m.sum()) else float("nan"),
        }
    return out
