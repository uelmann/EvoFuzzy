"""Walk-forward backtest for Kronos daily LONG/HOLD/SHORT signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config
from .data import window_at
from .signals import decide_signal, path_returns


ForecastFn = Callable[
    [pd.DataFrame, pd.Series, pd.Series, int],
    np.ndarray,
]


@dataclass
class StepResult:
    asof: str
    signal: str
    p_up: float
    mean_return: float
    realized_return: float
    strategy_return: float
    last_close: float
    realized_close: float
    correct: bool | None  # None when HOLD


@dataclass
class BacktestSummary:
    symbol: str
    interval: str
    model: str
    lookback: int
    pred_len: int
    n_paths: int
    step: int
    tau: float
    n_steps: int
    n_long: int
    n_short: int
    n_hold: int
    n_active: int
    hit_rate: float | None
    total_return: float
    buy_hold_return: float
    avg_active_return: float | None
    max_drawdown: float
    equity_final: float
    start: str
    end: str
    diagnostics: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_diagnostics(steps: list[StepResult]) -> dict:
    """Explain failure modes: long-bias, scale bias, weak rank correlation."""
    if not steps:
        return {}
    pred = np.array([s.mean_return for s in steps], dtype=float)
    real = np.array([s.realized_return for s in steps], dtype=float)
    p_up = np.array([s.p_up for s in steps], dtype=float)
    corr = float(np.corrcoef(pred, real)[0, 1]) if len(steps) > 1 else float("nan")
    abs_pred = float(np.mean(np.abs(pred)))
    abs_real = float(np.mean(np.abs(real)))
    return {
        "p_up_mean": float(np.mean(p_up)),
        "p_up_median": float(np.median(p_up)),
        "p_up_min": float(np.min(p_up)),
        "frac_p_up_ge_0_9": float(np.mean(p_up >= 0.9)),
        "pred_return_mean": float(np.mean(pred)),
        "realized_return_mean": float(np.mean(real)),
        "pred_vs_realized_bias": float(np.mean(pred) - np.mean(real)),
        "pred_abs_mean": abs_pred,
        "realized_abs_mean": abs_real,
        "pred_scale_ratio": abs_pred / abs_real if abs_real > 1e-12 else None,
        "corr_pred_realized": corr,
        "sign_agreement": float(np.mean(np.sign(pred) == np.sign(real))),
        "long_bias_note": (
            "Model almost always forecasts upside; thresholds rarely produce SHORT/HOLD."
            if float(np.mean(p_up)) >= 0.8
            else "p_up distribution is more balanced."
        ),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_steps(
    steps: list[StepResult],
    *,
    first_close: float,
    last_close: float,
    lookback: int,
    pred_len: int,
    n_paths: int,
    step: int,
    tau: float,
) -> BacktestSummary:
    n_long = sum(1 for s in steps if s.signal == "LONG")
    n_short = sum(1 for s in steps if s.signal == "SHORT")
    n_hold = sum(1 for s in steps if s.signal == "HOLD")
    active = [s for s in steps if s.signal != "HOLD"]
    hits = [s for s in active if s.correct]
    hit_rate = (len(hits) / len(active)) if active else None
    avg_active = float(np.mean([s.strategy_return for s in active])) if active else None

    equity = [1.0]
    for s in steps:
        equity.append(equity[-1] * (1.0 + s.strategy_return))
    equity_arr = np.asarray(equity, dtype=float)
    total_return = float(equity_arr[-1] - 1.0) if len(equity_arr) > 1 else 0.0
    buy_hold = (last_close / first_close - 1.0) if first_close > 0 else 0.0

    return BacktestSummary(
        symbol=config.SYMBOL,
        interval=config.INTERVAL,
        model=config.MODEL_ID,
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
        n_steps=len(steps),
        n_long=n_long,
        n_short=n_short,
        n_hold=n_hold,
        n_active=len(active),
        hit_rate=hit_rate,
        total_return=total_return,
        buy_hold_return=float(buy_hold),
        avg_active_return=avg_active,
        max_drawdown=_max_drawdown(equity_arr),
        equity_final=float(equity_arr[-1]),
        start=steps[0].asof if steps else "",
        end=steps[-1].asof if steps else "",
        diagnostics=compute_diagnostics(steps),
        steps=[asdict(s) for s in steps],
    )


def run_walk_forward(
    df: pd.DataFrame,
    forecast_fn: ForecastFn,
    *,
    lookback: int = config.LOOKBACK,
    pred_len: int = config.PRED_LEN,
    n_paths: int = config.N_PATHS,
    step: int | None = None,
    tau: float = config.TAU,
    max_steps: int | None = None,
    verbose: bool = True,
) -> BacktestSummary:
    """
    Non-overlapping walk-forward:
    at each decision bar t, forecast t+1..t+pred_len, take signal, realize
    close[t+pred_len] / close[t] - 1, then advance by `step` (default=pred_len).
    """
    step = pred_len if step is None else step
    end_indices = list(range(lookback - 1, len(df) - pred_len, step))
    if max_steps is not None:
        end_indices = end_indices[-max_steps:]

    if not end_indices:
        raise ValueError("No backtest steps available with current settings")

    steps: list[StepResult] = []
    for i, end_idx in enumerate(end_indices):
        x_df, x_ts, y_ts, last_close, realized_close = window_at(
            df, end_idx, lookback=lookback, pred_len=pred_len
        )
        if verbose:
            print(
                f"[{i + 1}/{len(end_indices)}] {x_ts.iloc[-1]} close={last_close:.2f}",
                flush=True,
            )

        path_closes = forecast_fn(x_df, x_ts, y_ts, pred_len)
        pred_returns = path_returns(last_close, path_closes[:, -1])
        decision = decide_signal(
            pred_returns,
            last_close=last_close,
            horizon_days=pred_len,
            tau=tau,
        )
        realized = realized_close / last_close - 1.0
        if decision.signal == "LONG":
            strat = realized
            correct = realized > 0
        elif decision.signal == "SHORT":
            strat = -realized
            correct = realized < 0
        else:
            strat = 0.0
            correct = None

        steps.append(
            StepResult(
                asof=str(x_ts.iloc[-1]),
                signal=decision.signal,
                p_up=decision.p_up,
                mean_return=decision.mean_return,
                realized_return=float(realized),
                strategy_return=float(strat),
                last_close=last_close,
                realized_close=realized_close,
                correct=correct,
            )
        )
        if verbose:
            print(
                f"  → {decision.signal} p_up={decision.p_up:.2%} "
                f"pred={decision.mean_return:.2%} real={realized:.2%} strat={strat:.2%}",
                flush=True,
            )

    first_close = float(df.iloc[end_indices[0]]["close"])
    last_close = float(df.iloc[end_indices[-1] + pred_len]["close"])
    return summarize_steps(
        steps,
        first_close=first_close,
        last_close=last_close,
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
    )
