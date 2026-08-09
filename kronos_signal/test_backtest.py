"""Unit tests for walk-forward summary math (no GPU / Kronos)."""

import numpy as np
import pandas as pd

from kronos_signal.backtest import StepResult, run_walk_forward, summarize_steps


def test_summarize_hit_rate_and_equity():
    steps = [
        StepResult("t1", "LONG", 0.7, 0.01, 0.02, 0.02, 100, 102, True),
        StepResult("t2", "HOLD", 0.5, 0.0, -0.01, 0.0, 102, 101, None),
        StepResult("t3", "SHORT", 0.3, -0.01, 0.03, -0.03, 101, 104, False),
    ]
    summary = summarize_steps(
        steps,
        first_close=100,
        last_close=104,
        lookback=400,
        pred_len=5,
        n_paths=10,
        step=5,
        tau=0.005,
    )
    assert summary.n_long == 1
    assert summary.n_short == 1
    assert summary.n_hold == 1
    assert summary.hit_rate == 0.5
    # equity: 1 * 1.02 * 1.0 * 0.97
    assert abs(summary.equity_final - 1.02 * 0.97) < 1e-9
    assert abs(summary.buy_hold_return - 0.04) < 1e-9
    assert "corr_pred_realized" in summary.diagnostics
    assert "p_up_mean" in summary.diagnostics


def test_walk_forward_with_fake_forecast():
    n = 50
    closes = np.linspace(100, 150, n)
    df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": np.ones(n),
            "amount": closes,
        }
    )

    def forecast_fn(x_df, x_ts, y_ts, pred_len):
        # Always predict +2% at horizon → LONG
        last = float(x_df["close"].iloc[-1])
        return np.full((5, pred_len), last * 1.02)

    summary = run_walk_forward(
        df,
        forecast_fn,
        lookback=20,
        pred_len=5,
        n_paths=5,
        step=5,
        tau=0.005,
        max_steps=4,
        verbose=False,
    )
    assert summary.n_steps == 4
    assert summary.n_long == 4
    assert summary.hit_rate == 1.0  # uptrend


if __name__ == "__main__":
    test_summarize_hit_rate_and_equity()
    test_walk_forward_with_fake_forecast()
    print("ok")
