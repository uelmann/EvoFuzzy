"""Frozen defaults for the FuzzyX prototype. No sweeps in this module.

FEATURE_COLS must stay identical to baseline.features.FEATURE_COLS (A0).
Listed here so this package imports without pandas.
"""

from __future__ import annotations

SEED = 42
UNIVERSE_N = 30
REBALANCE_DAYS = 7
N_MFS = 3  # Low / Mid / High
N_RULES = 24
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
# deepsets = default (CS residual). xsec = 1-layer asset-token attention.
ENCODER = "deepsets"
DROPOUT = 0.2
GROSS_LIMIT = 1.0
TAKER_FEE_BPS = 5.0
SLIPPAGE_BPS = 3.0
GATE_TEMPERATURE = 0.5
POS_TEMPERATURE = 0.7
OCC_LONG_MIN = 0.20
OCC_SHORT_MIN = 0.30
OCC_TRADED_MIN = 0.25
OCC_PENALTY = 1e5
MAXDD_CAP = 0.99
CS_CLIP = 5.0
MF_INIT_CENTERS = (-1.0, 0.0, 1.0)
MF_INIT_SIGMA = 0.85

# Occupancy floors copied from WRKS_L_S_NNET_Nov_22_ (loss_L / loss_S / loss_z).
FEATURE_COLS = [
    "ret_7",
    "ret_14",
    "ret_28",
    "ret_56",
    "ret_90",
    "mom_28_skip7",
    "mom_90_skip14",
    "rev_1",
    "rev_3",
    "close_sma20",
    "close_sma50",
    "close_sma100",
    "sma20_sma50",
    "ema12_ema26",
    "yz_vol_14",
    "yz_vol_30",
    "yz_vol_60",
    "pk_vol_14",
    "vol_ratio",
    "vol_of_vol_30",
    "max_ret_14",
    "min_ret_14",
    "dist_high_90",
    "dist_low_90",
    "range_pos_28",
    "skew_28",
    "skew_60",
    "beta_btc_60",
    "idio_vol_60",
    "corr_btc_28",
    "amihud_14",
    "dv_z_30",
    "dv_trend",
]
N_FEATURES = len(FEATURE_COLS)
