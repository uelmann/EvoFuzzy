"""Frozen constants for NASDAQ-LS21 (top/worst 10%, h=21, 5y rolling train).

Does not replace the h=10 k=10 scout record. COMBO untouched.
"""

from __future__ import annotations

from nasdaq_ls.constants import DEATH_CONVENTION, PRICE_RULE  # noqa: F401

FACTOR_CRITERION = (
    "NASDAQ-LS21 shows a cross-sectional factor if BOTH of: "
    "(a) pooled OOS RankIC of LightGBM Huber score vs residualized h=21 "
    "forward return, on the PIT top-30 by dollar volume, from 2007-01-01 "
    "onward, is > 0; "
    "(b) long-short net Sharpe from 2007-01-01 onward is > 0. "
    "This is a scout, not a product. It does not replace COMBO. "
    "Survivorship (today's Nasdaq-100 members) and missing borrow costs "
    "are accepted for this run. No post-hoc adjustment."
)

TRAIN_RULE = (
    "Every fold trains exactly 500 LightGBM trees on a rolling window of at "
    "most 1260 sessions (~5×252). No expanding 1990s window. No early stopping. "
    "Huber objective. Label and overlapping book use h=21 sessions. "
    "Legs are the top 10% and worst 10% of that day's PIT top-30 "
    "(k = ceil(0.10 × n), typically 3 names each side)."
)

HORIZON = 21
TOP_PCT = 0.10
EXEC_TOP_N = 30
DV_WINDOW = 30
SEED = 42
FIXED_TREES = 500
TRAIN_MAX_SESSIONS = 5 * 252  # 1260
MIN_TRAIN_SESSIONS = 5 * 252

PRICE_START = "1990-01-01"
BOOK_START = "2005-01-01"
HEADLINE_START = "2007-01-01"

FEAT_PATH = "data/nasdaq21/features.parquet"
PRED_PATH = "data/nasdaq21/preds_h21.parquet"
PRED_DIR = "data/nasdaq21/preds"
REPORT_MD = "reports/nasdaq_ls21_report.md"
REPORT_JSON = "reports/nasdaq_ls21_report.json"
CHART_PATH = "charts/nasdaq_ls21_equity.png"
ADDENDUM_PATH = "reports/nasdaq_ls21_addendum.md"
