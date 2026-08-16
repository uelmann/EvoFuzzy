"""Book stats and OOS prediction frame for FuzzyX-v1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loss import occupancy, path_loss, portfolio_returns


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5 or np.std(x, ddof=1) <= 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(365.0))


def positions_to_frame(dates, symbols, pos, mask) -> pd.DataFrame:
    rows = []
    for i, dt in enumerate(dates):
        for j, sym in enumerate(symbols):
            if not mask[i, j]:
                continue
            rows.append({"date": pd.Timestamp(dt), "symbol": sym, "pos": float(pos[i, j])})
    return pd.DataFrame(rows)


def book_from_pred(pred: dict) -> dict:
    hard = pred["hard_pos"]
    mask = pred["mask"]
    # weekly 7-day simple returns already in the packed ret used for hard_loss
    hl = pred["hard_loss"]
    sl = pred["soft_loss"]
    long_f, short_f, traded_f = occupancy(hard, mask)
    return {
        "net_sharpe_weekly": _sharpe_from_core_path(pred),
        "hard_loss": hl,
        "soft_loss": sl,
        "long_frac": long_f,
        "short_frac": short_f,
        "traded_frac": traded_f,
        "n_reb": int(hard.shape[0]),
        "n_symbols": int(hard.shape[1]),
    }


def _sharpe_from_core_path(pred: dict) -> float:
    """Rebuild weekly net returns from hard positions and stored ret_h7 if present."""
    if "ret_h7" not in pred:
        return float("nan")
    port, _ = portfolio_returns(pred["hard_pos"], pred["ret_h7"], mask=pred["mask"])
    return _sharpe(port)


def attach_returns(pred: dict, packed) -> dict:
    out = dict(pred)
    out["ret_h7"] = packed.ret_h7
    return out


def verdict(gates: list[dict], bias: dict, book: dict, a0_delta: dict | None) -> dict:
    leak_ok = all(g.get("passed") for g in gates)
    bias_ok = bool(bias.get("passed"))
    sharpe = book.get("net_sharpe_weekly", float("nan"))
    skill_ok = bool(np.isfinite(sharpe) and sharpe >= 0.0)
    if a0_delta is None or a0_delta.get("skipped"):
        vs_ok = True
        vs_skip = True
    else:
        vs_ok = bool(a0_delta.get("delta_sharpe", -1e9) >= -0.10)
        vs_skip = False
    if not leak_ok:
        label = "PARK (leakage)"
    elif not bias_ok:
        label = "CONTAMINATED"
    elif not skill_ok:
        label = "PARK"
    elif not vs_ok:
        label = "PARK"
    else:
        label = "VIABLE candidate"
    return {
        "leak_ok": leak_ok,
        "bias_ok": bias_ok,
        "skill_ok": skill_ok,
        "vs_a0_ok": vs_ok,
        "vs_a0_skip": vs_skip,
        "verdict": label,
        "net_sharpe_weekly": sharpe,
    }
