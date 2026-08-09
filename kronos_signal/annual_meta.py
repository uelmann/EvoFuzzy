"""
Annual retrain/test meta walk-forward.

For each test year Y:
  - train logistic meta on all prior feature rows (asof < Y-01-01)
  - test on rows with asof in year Y
Then stitch yearly test segments into one equity curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .backtest import StepResult, summarize_steps
from .compare import BASELINE_META_FEATURES
from .features import steps_to_frame
from .meta_model import _make_model


@dataclass
class YearResult:
    year: int
    n_train: int
    n_test: int
    n_long: int
    n_short: int
    n_hold: int
    hit_rate: float | None
    total_return: float
    buy_hold_return: float
    max_drawdown: float

    def to_dict(self) -> dict:
        return asdict(self)


def _equity_and_dd(rets: np.ndarray) -> tuple[float, float]:
    if rets.size == 0:
        return 0.0, 0.0
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min())
    return float(eq[-1] - 1.0), dd


def annual_retrain_meta(
    steps: list[dict],
    ohlcv: pd.DataFrame,
    *,
    test_years: list[int] | None = None,
    min_train: int = 30,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
    model_type: str = "logistic",
    feature_cols: list[str] | None = None,
) -> dict:
    frame = steps_to_frame(steps, ohlcv)
    frame["year"] = pd.to_datetime(frame["asof"], utc=True).dt.year
    feat_cols = feature_cols or list(BASELINE_META_FEATURES)
    feat_cols = [c for c in feat_cols if c in frame.columns]

    years = sorted(frame["year"].unique().tolist())
    if test_years is None:
        # need at least one prior year of rows for training when possible
        test_years = [y for y in years if y >= years[0] + 1] or years
    else:
        test_years = [y for y in test_years if y in years]

    X = np.nan_to_num(frame[feat_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = frame["y_up"].to_numpy(dtype=int)
    realized = frame["realized_return"].to_numpy(dtype=float)
    asofs = pd.to_datetime(frame["asof"], utc=True)

    all_steps: list[StepResult] = []
    year_rows: list[YearResult] = []

    for year in test_years:
        train_idx = np.where(asofs.dt.year < year)[0]
        test_idx = np.where(asofs.dt.year == year)[0]
        if len(test_idx) == 0:
            continue
        if len(train_idx) < min_train:
            # not enough history: skip or HOLD-all
            year_rows.append(
                YearResult(
                    year=year,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    n_long=0,
                    n_short=0,
                    n_hold=len(test_idx),
                    hit_rate=None,
                    total_return=0.0,
                    buy_hold_return=float(np.prod(1.0 + realized[test_idx]) - 1.0),
                    max_drawdown=0.0,
                )
            )
            for i in test_idx:
                all_steps.append(
                    StepResult(
                        asof=str(frame.iloc[i]["asof"]),
                        signal="HOLD",
                        p_up=0.5,
                        mean_return=0.0,
                        realized_return=float(realized[i]),
                        strategy_return=0.0,
                        last_close=float("nan"),
                        realized_close=float("nan"),
                        correct=None,
                    )
                )
            continue

        model, _ = _make_model(model_type)
        model.fit(X[train_idx], y[train_idx])

        year_step_rets = []
        n_long = n_short = n_hold = 0
        corrects = []
        for i in test_idx:
            proba = float(model.predict_proba(X[i : i + 1])[0, 1])
            real = float(realized[i])
            if proba >= proba_long:
                signal, strat, correct = "LONG", real, real > 0
                n_long += 1
            elif proba <= proba_short:
                signal, strat, correct = "SHORT", -real, real < 0
                n_short += 1
            else:
                signal, strat, correct = "HOLD", 0.0, None
                n_hold += 1
            if correct is not None:
                corrects.append(correct)
            year_step_rets.append(strat)
            all_steps.append(
                StepResult(
                    asof=str(frame.iloc[i]["asof"]),
                    signal=signal,
                    p_up=proba,
                    mean_return=proba - 0.5,
                    realized_return=real,
                    strategy_return=float(strat),
                    last_close=float("nan"),
                    realized_close=float("nan"),
                    correct=correct,
                )
            )

        rets = np.asarray(year_step_rets, dtype=float)
        total_r, max_dd = _equity_and_dd(rets)
        year_rows.append(
            YearResult(
                year=year,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_long=n_long,
                n_short=n_short,
                n_hold=n_hold,
                hit_rate=(float(np.mean(corrects)) if corrects else None),
                total_return=total_r,
                buy_hold_return=float(np.prod(1.0 + realized[test_idx]) - 1.0),
                max_drawdown=max_dd,
            )
        )

    stitched_rets = np.array([s.strategy_return for s in all_steps], dtype=float)
    total_r, max_dd = _equity_and_dd(stitched_rets)
    bh = float(np.prod(1.0 + np.array([s.realized_return for s in all_steps])) - 1.0)
    active = [s for s in all_steps if s.signal != "HOLD"]
    hits = [s for s in active if s.correct]
    summary = summarize_steps(
        all_steps,
        first_close=100.0,
        last_close=100.0 * (1.0 + bh),
        lookback=400,
        pred_len=5,
        n_paths=10,
        step=5,
        tau=0.0,
    )

    return {
        "scheme": "annual_retrain_meta",
        "data_source": "Binance BTCUSDT 1d (data-api.binance.vision / mirrors)",
        "test_years": test_years,
        "by_year": [y.to_dict() for y in year_rows],
        "overall": {
            "n_steps": len(all_steps),
            "n_long": summary.n_long,
            "n_short": summary.n_short,
            "n_hold": summary.n_hold,
            "hit_rate": (len(hits) / len(active)) if active else None,
            "total_return": total_r,
            "buy_hold_return": bh,
            "max_drawdown": max_dd,
            "start": all_steps[0].asof if all_steps else "",
            "end": all_steps[-1].asof if all_steps else "",
        },
        "raw_full_period": None,  # filled by caller if desired
        "steps": [asdict(s) for s in all_steps],
        "feature_cols": feat_cols,
    }
