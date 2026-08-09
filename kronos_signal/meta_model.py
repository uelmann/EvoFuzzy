"""
Walk-forward meta-model on Kronos + market features.

Point 1-2: do NOT trade raw Kronos; train a classifier that maps features →
LONG/SHORT/HOLD and backtest it with expanding window (no future leakage).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import StepResult, summarize_steps
from .features import FEATURE_COLS


@dataclass
class MetaBacktestResult:
    name: str
    n_steps: int
    n_long: int
    n_short: int
    n_hold: int
    hit_rate: float | None
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    avg_active_return: float | None
    start: str
    end: str
    min_train: int
    proba_long: float
    proba_short: float
    steps: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _clf() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=0.5,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def walk_forward_meta(
    frame: pd.DataFrame,
    *,
    min_train: int = 40,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
    lookback: int = 400,
    pred_len: int = 5,
    n_paths: int = 10,
    step: int = 5,
    tau: float = 0.0,
) -> MetaBacktestResult:
    """
    Expanding-window walk-forward:
    for each t >= min_train, fit on [0, t), predict P(up) at t, trade, realize label at t.
    """
    if len(frame) <= min_train:
        raise ValueError(f"Need more than min_train={min_train} rows, got {len(frame)}")

    X = frame[FEATURE_COLS].to_numpy(dtype=float)
    y = frame["y_up"].to_numpy(dtype=int)
    realized = frame["realized_return"].to_numpy(dtype=float)

    steps: list[StepResult] = []
    for t in range(min_train, len(frame)):
        model = _clf()
        model.fit(X[:t], y[:t])
        proba = float(model.predict_proba(X[t : t + 1])[0, 1])
        real = float(realized[t])

        if proba >= proba_long:
            signal = "LONG"
            strat = real
            correct = real > 0
        elif proba <= proba_short:
            signal = "SHORT"
            strat = -real
            correct = real < 0
        else:
            signal = "HOLD"
            strat = 0.0
            correct = None

        steps.append(
            StepResult(
                asof=str(frame.iloc[t]["asof"]),
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

    # buy&hold over meta evaluation window
    first_close = 1.0
    # approximate B&H from compounded realized of the evaluated steps
    bh = float(np.prod(1.0 + realized[min_train:]) - 1.0)
    # use summarize with dummy closes; override buy_hold after
    summary = summarize_steps(
        steps,
        first_close=100.0,
        last_close=100.0 * (1.0 + bh),
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
    )
    return MetaBacktestResult(
        name="meta_logistic_walkforward",
        n_steps=summary.n_steps,
        n_long=summary.n_long,
        n_short=summary.n_short,
        n_hold=summary.n_hold,
        hit_rate=summary.hit_rate,
        total_return=summary.total_return,
        buy_hold_return=bh,
        max_drawdown=summary.max_drawdown,
        avg_active_return=summary.avg_active_return,
        start=summary.start,
        end=summary.end,
        min_train=min_train,
        proba_long=proba_long,
        proba_short=proba_short,
        steps=summary.steps,
    )


def raw_rule_on_frame(frame: pd.DataFrame) -> MetaBacktestResult:
    """Baseline: original Kronos raw LONG/HOLD/SHORT on the same rows."""
    steps: list[StepResult] = []
    for _, row in frame.iterrows():
        sig = row["raw_signal"]
        real = float(row["realized_return"])
        if sig == "LONG":
            strat, correct = real, real > 0
        elif sig == "SHORT":
            strat, correct = -real, real < 0
        else:
            strat, correct = 0.0, None
        steps.append(
            StepResult(
                asof=str(row["asof"]),
                signal=sig,
                p_up=float(row["kronos_p_up"]),
                mean_return=float(row["kronos_mean_r"]),
                realized_return=real,
                strategy_return=float(strat),
                last_close=float("nan"),
                realized_close=float("nan"),
                correct=correct,
            )
        )
    bh = float(np.prod(1.0 + frame["realized_return"].to_numpy()) - 1.0)
    summary = summarize_steps(
        steps,
        first_close=100.0,
        last_close=100.0 * (1.0 + bh),
        lookback=400,
        pred_len=5,
        n_paths=10,
        step=5,
        tau=0.005,
    )
    return MetaBacktestResult(
        name="raw_kronos_rule",
        n_steps=summary.n_steps,
        n_long=summary.n_long,
        n_short=summary.n_short,
        n_hold=summary.n_hold,
        hit_rate=summary.hit_rate,
        total_return=summary.total_return,
        buy_hold_return=bh,
        max_drawdown=summary.max_drawdown,
        avg_active_return=summary.avg_active_return,
        start=summary.start,
        end=summary.end,
        min_train=0,
        proba_long=0.6,
        proba_short=0.4,
        steps=summary.steps,
    )
