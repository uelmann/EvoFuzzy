"""FuzzyX — fuzzy membership + AND/OR rules + cross-section encoder.

Design: reports/fuzzyx_architecture.md
Prototype only. Not a trained book. Does not replace COMBO / A0.
"""

from .constants import FEATURE_COLS, N_FEATURES, REBALANCE_DAYS, UNIVERSE_N
from .loss import path_loss
from .model import FuzzyX, hard_positions, soft_positions
from .universe import rebalance_dates

__all__ = [
    "FEATURE_COLS",
    "FuzzyX",
    "N_FEATURES",
    "REBALANCE_DAYS",
    "UNIVERSE_N",
    "hard_positions",
    "path_loss",
    "rebalance_dates",
    "soft_positions",
]
