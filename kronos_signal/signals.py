"""Convert Kronos path forecasts into long / hold / short."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from . import config


@dataclass
class SignalResult:
    signal: str  # LONG | HOLD | SHORT
    p_up: float
    mean_return: float
    median_return: float
    std_return: float
    last_close: float
    horizon_days: int
    n_paths: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def path_returns(last_close: float, path_closes: Iterable[float]) -> np.ndarray:
    closes = np.asarray(list(path_closes), dtype=float)
    if last_close <= 0:
        raise ValueError("last_close must be positive")
    return closes / last_close - 1.0


def decide_signal(
    returns: np.ndarray,
    *,
    last_close: float,
    horizon_days: int = config.PRED_LEN,
    p_up_long: float = config.P_UP_LONG,
    p_up_short: float = config.P_UP_SHORT,
    tau: float = config.TAU,
) -> SignalResult:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        raise ValueError("returns must be non-empty")

    p_up = float(np.mean(returns > 0))
    mean_r = float(np.mean(returns))
    median_r = float(np.median(returns))
    std_r = float(np.std(returns))

    if p_up >= p_up_long and mean_r >= tau:
        signal = "LONG"
        reason = f"p_up={p_up:.2%} >= {p_up_long:.0%} and mean_r={mean_r:.2%} >= {tau:.2%}"
    elif p_up <= p_up_short and mean_r <= -tau:
        signal = "SHORT"
        reason = f"p_up={p_up:.2%} <= {p_up_short:.0%} and mean_r={mean_r:.2%} <= -{tau:.2%}"
    else:
        signal = "HOLD"
        reason = (
            f"uncertain or small move (p_up={p_up:.2%}, mean_r={mean_r:.2%}; "
            f"need p_up>={p_up_long:.0%}/{p_up_short:.0%} and |mean_r|>={tau:.2%})"
        )

    return SignalResult(
        signal=signal,
        p_up=p_up,
        mean_return=mean_r,
        median_return=median_r,
        std_return=std_r,
        last_close=float(last_close),
        horizon_days=horizon_days,
        n_paths=int(returns.size),
        reason=reason,
    )
