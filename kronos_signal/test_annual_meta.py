"""Unit test for annual retrain meta (no GPU)."""

import numpy as np
import pandas as pd

from kronos_signal.annual_meta import annual_retrain_meta


def test_annual_retrain_runs():
    rng = np.random.default_rng(0)
    n = 500
    closes = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    ts = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    ohlcv = pd.DataFrame(
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
    steps = []
    for i in range(120, 480, 5):
        asof = ts[i]
        real = float(closes[i + 5] / closes[i] - 1)
        steps.append(
            {
                "asof": str(asof),
                "signal": "LONG" if real > 0 else "SHORT",
                "p_up": 0.65 if real > 0 else 0.35,
                "mean_return": real,
                "realized_return": real,
            }
        )
    out = annual_retrain_meta(
        steps, ohlcv, test_years=[2022, 2023], min_train=10, model_type="logistic"
    )
    assert out["overall"]["n_steps"] > 0
    assert len(out["by_year"]) >= 1


if __name__ == "__main__":
    test_annual_retrain_runs()
    print("ok")
