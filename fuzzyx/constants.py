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
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12
TURN_LAMBDA = 0.0  # v1e: corr(wealth, t) * (1 + last cumret)
BIAS_LAMBDA = 0.0
OCC_LAMBDA = 0.0
ACTIVE_LAMBDA = 0.0  # v1c: no pay-to-play
FLAT_INIT_BIAS = 0.0
OCC_NUKE = False
LEVER_UP = True  # v1c: w = p / Σ|p| so corr has a path
SHUFFLE_SEEDS = tuple(range(101, 111))
MIN_TRAIN_DAYS = 730
VAL_DAYS = 90
STEP_DAYS = 90
INNER_HOLDOUT_DAYS = 90
PURGE_H = 7
EMBARGO_EXTRA = 3
EXEC_DV_WINDOW = 30
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

# Occupancy floors copied from WRKS_L_S_NNET_Nov_22_ (diagnostics only in v1b).
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
