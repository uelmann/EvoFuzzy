"""Unit tests for meta-model walk-forward (no GPU)."""

import numpy as np
import pandas as pd

from kronos_signal.features import FEATURE_COLS, steps_to_frame
from kronos_signal.meta_model import walk_forward_meta


def _fake_ohlcv(n=250, start="2024-01-01"):
    closes = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, n)) * 100
    ts = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamps": ts,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.ones(n),
            "amount": closes,
        }
    )


def test_walk_forward_meta_runs():
    ohlcv = _fake_ohlcv(250)
    steps = []
    for i in range(120, 230, 5):
        asof = ohlcv.iloc[i]["timestamps"]
        real = float(ohlcv.iloc[i + 5]["close"] / ohlcv.iloc[i]["close"] - 1)
        steps.append(
            {
                "asof": str(asof),
                "signal": "LONG" if real > 0 else "SHORT",
                "p_up": 0.7 if real > 0 else 0.3,
                "mean_return": real * 0.5 + 0.001,
                "realized_return": real,
            }
        )
    frame = steps_to_frame(steps, ohlcv)
    assert all(c in frame.columns for c in FEATURE_COLS)
    meta = walk_forward_meta(
        frame,
        min_train=8,
        proba_long=0.55,
        proba_short=0.45,
        model_type="logistic",
        embargo_steps=1,
    )
    assert meta.n_steps == len(frame) - 8
    assert meta.n_long + meta.n_short + meta.n_hold == meta.n_steps


if __name__ == "__main__":
    test_walk_forward_meta_runs()
    print("ok")
