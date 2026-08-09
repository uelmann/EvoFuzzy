"""Unit tests for long/hold/short decision logic (no GPU)."""

import numpy as np

from kronos_signal.signals import decide_signal, path_returns


def test_path_returns():
    r = path_returns(100.0, [101.0, 99.0, 110.0])
    assert np.allclose(r, [0.01, -0.01, 0.10])


def test_long_signal():
    # 8/10 up, mean ~ +1.5%
    returns = np.array([0.02, 0.015, 0.01, 0.02, 0.03, 0.01, 0.012, 0.018, -0.002, 0.005])
    out = decide_signal(returns, last_close=50000.0, tau=0.005)
    assert out.signal == "LONG"
    assert out.p_up == 0.9


def test_short_signal():
    returns = np.array([-0.02, -0.015, -0.01, -0.02, -0.03, -0.01, -0.012, -0.018, 0.002, -0.005])
    out = decide_signal(returns, last_close=50000.0, tau=0.005)
    assert out.signal == "SHORT"


def test_hold_when_uncertain():
    returns = np.array([0.01, -0.01, 0.005, -0.004, 0.002, -0.003, 0.001, -0.002, 0.0, 0.004])
    out = decide_signal(returns, last_close=50000.0, tau=0.005)
    assert out.signal == "HOLD"


if __name__ == "__main__":
    test_path_returns()
    test_long_signal()
    test_short_signal()
    test_hold_when_uncertain()
    print("ok")
