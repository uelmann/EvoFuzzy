"""Frozen constants for the ALPHAMINE-LO feature A/B (long-only).

COMBO / SPREAD-LS / LONG-TIDE are not modified. Viability strings are quoted
verbatim in the addendum BEFORE any results.
"""

from __future__ import annotations

IMPROVE_CRITERION = (
    "ALPHAMINE-LO IMPROVES on A0-LO only if ALL of: "
    "(a) pooled OOS RankIC of MINE exceeds A0; "
    "(b) MINE top-quintile minus universe 10-day USDT simple return exceeds A0's; "
    "(c) MINE long-only net Sharpe exceeds A0-LO net Sharpe; "
    "(d) BTC weight is identically 0 every day on both books; "
    "(e) the MINE label-shuffle null is GREEN. "
    "This is an A/B feature test, not a replacement for COMBO. No post-hoc adjustment."
)

VIABILITY_CRITERION = (
    "ALPHAMINE-LO is VIABLE as a standalone long-only mandate only if ALL of: "
    "(a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; "
    "(c) full-period total return > 0; (d) average deployed alt gross ≥ 0.15; "
    "(e) BTC weight is identically 0 every day; (f) the MINE label-shuffle null is GREEN. "
    "It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment."
)

NULL_GATE = (
    "Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). "
    "Skill passes if the real MINE OOS RankIC exceeds the null 95th percentile on "
    "**both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, "
    "PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds."
)

DEATH_CONVENTION = (
    "A held coin whose data ends is force-exited at its last available close "
    "(no better information assumed). The count and PnL impact of such forced exits "
    "is reported in every backtest of this project."
)

FALLBACK_RULE = (
    "If an arm's median best_iteration ≤ 1, refit that arm with fixed 500 trees "
    "(A0 h=10 fallback). The judged book for that arm uses the refit. This rule is frozen before results."
)

FROZEN_A0_SHA256 = (
    "e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa"
)

HORIZON = 10
SEED = 42
BTC_SYMBOL = "BTCUSDT"
YCOL = "y_h10"
YCOL_SIMPLE = "y_simple_h10"

TOP_K = 10
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

# GP miner (small, frozen)
GP_POP = 32
GP_GENS = 6
GP_MAX_DEPTH = 3
GP_ELITE = 4
GP_TOURNAMENT = 3
GP_CX = 0.6
GP_MUT = 0.3
GP_WINDOWS = (5, 10, 20, 40)
GP_KEEP = 8
GP_MIN_HOLDOUT_IC = 0.01
GP_MAX_ABS_CORR = 0.70
GP_BUDGET_SEC = 180.0
GP_MIN_CS = 8

FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "vwap",
    "ret",
)

UNARY_OPS = ("abs", "log", "neg", "sign", "cs_rank")
TS_UNARY_OPS = ("delay", "delta", "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_rank")
BINARY_OPS = ("add", "sub", "mul", "div")
TS_BINARY_OPS = ("ts_corr",)

PIT_TOP40 = "/data/quant/universe/top40_pit.parquet"
PIT_TOP120 = "/data/quant/universe/top120_pit.parquet"
FEAT_PATH = "/data/quant/features/features_labeled.parquet"
OUT_ROOT = "/data/quant/alphamine"
