"""Frozen constants for the REGIME-TAU parallel book. No sweeps."""

from __future__ import annotations

VIABILITY_CRITERION = (
    "COMBO-REGIME-TAU is VIABLE if its identical-days net Sharpe ≥ reference "
    "COMBO + 0.10 AND trailing-18m net Sharpe ≥ reference COMBO − 0.05. These "
    "are viability labels for a parallel product; no outcome changes the "
    "reference book. No post-hoc adjustment."
)

DEATH_CONVENTION = (
    "A held coin whose data ends is force-exited at its last available close "
    "(no better information assumed). The count and PnL impact of such forced "
    "exits is reported in every backtest of this project."
)

DELTA_FULL = 0.10
DELTA_TRAIL_FLOOR = -0.05

P1_H = 7
P1_TAU_BASE = 80.0
P1_TAU_HIGH = 90.0
P1_TAU_LOW = 70.0
P1_COST_BPS = 5.0
P1_SLIP_BPS = 3.0

P2_H = 10
P2_TAU_BASE = 70.0
P2_TAU_HIGH = 80.0
P2_TAU_LOW = 60.0
P2_LIQ_CAP = 0.005
P2_NOM_USD = 1_000_000.0

CS_CORR_WINDOW = 60
CS_CORR_MIN_NAMES = 10
CS_CORR_MIN_OBS = 20
WARMUP_OBS = 252
PIT_N = 40
BTC_SYM = "BTCUSDT"

TRAIL_DAYS = 547
ANNUALIZATION = 365

FROZEN_A0_SHA256 = (
    "e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa"
)
FROZEN_A0_PATH = "/data/quant/models/lgbm_price_only.txt"
CONFIG_FROZEN = "config_frozen_a0.yaml"

PRED_H7 = "/data/quant/predictions/lgbm_price_only_h7.parquet"
PRED_H10 = "/data/quant/predictions/lgbm_price_only_h10.parquet"
PANEL_PATH = "/data/quant/panel/panel.parquet"

REGIME_BASE = 0
REGIME_LOW = 1
REGIME_HIGH = 2
REGIME_NAME = {REGIME_BASE: "BASE", REGIME_LOW: "LOW", REGIME_HIGH: "HIGH"}
