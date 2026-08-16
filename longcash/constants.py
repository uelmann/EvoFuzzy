"""Frozen constants for the LONG-CASH parallel product.

COMBO / SPREAD-LS / LONG-TIDE are not modified. Viability strings are quoted
verbatim in the addendum BEFORE any results.
"""

from __future__ import annotations

VIABILITY_CRITERION = (
    "LONG-CASH is VIABLE as a standalone cash-financed alt-long mandate only if ALL of: "
    "(a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; "
    "(c) full-period total return > 0; (d) average deployed alt gross ≥ 0.15; "
    "(e) BTC weight is identically 0 every day; (f) the Head-R label-shuffle null is GREEN. "
    "It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment."
)

NULL_GATE = (
    "Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). "
    "Skill passes if the real Head-R OOS RankIC exceeds the null 95th percentile on "
    "**both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, "
    "PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds."
)

DEATH_CONVENTION = (
    "A held coin whose data ends is force-exited at its last available close "
    "(no better information assumed). The count and PnL impact of such forced exits "
    "is reported in every backtest of this project."
)

FALLBACK_RULE = (
    "If Head-R median best_iteration ≤ 1, refit Head R with fixed 500 trees "
    "(A0 h=10 fallback). The judged book uses the refit. This rule is frozen before results."
)

FROZEN_A0_SHA256 = (
    "e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa"
)

HORIZON = 10
SEED = 42
BTC_SYMBOL = "BTCUSDT"

ER_HURDLE = 0.0
P_UP_HURDLE = 0.5
MIN_NAMES = 3
MAX_NAMES = 10

GROSS_LIMIT = 1.0
LIQ_CAP_ADV_FRAC = 0.005
NOMINAL_BOOK_USD = 1_000_000.0
FEE_BPS_TOP = 5.0
SLIP_BPS_TOP = 3.0
FEE_BPS_NEXT = 10.0
SLIP_BPS_NEXT = 8.0
LAG = 0

FULL_SHARPE_MIN = 0.50
TRAIL_SHARPE_MIN = 0.00
AVG_GROSS_MIN = 0.15
TRAIL_DAYS = int(365 * 1.5)
ANNUALIZATION = 365

NULL_REPLICATES = 10
NULL_SHUFFLE_SEEDS = tuple(range(101, 111))
NULL_ANCHOR = "2022-01-01"
FIXED_TREES_FALLBACK = 500

PRED_H10 = "/data/quant/predictions/lgbm_price_only_h10.parquet"
PIT_TOP40 = "/data/quant/universe/top40_pit.parquet"
FEAT_PATH = "/data/quant/features/features_labeled.parquet"

OUT_ROOT = "/data/quant/long_cash"
