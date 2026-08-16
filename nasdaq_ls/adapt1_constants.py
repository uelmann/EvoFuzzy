"""Frozen constants for NASDAQ-ADAPT-1.

COMBO / SPREAD-LS / LONG-TIDE / A0 crypto artifacts are not modified.
Does not overwrite NASDAQ-LS or NASDAQ-LS21 records.
"""

from __future__ import annotations

from nasdaq_ls.constants import DEATH_CONVENTION, PRICE_RULE  # noqa: F401

FACTOR_CRITERION = (
    "NASDAQ-ADAPT-1 shows a cross-sectional factor if BOTH of: "
    "(a) pooled OOS RankIC of LightGBM Huber score vs the simple (non-residual) "
    "h=126 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 "
    "onward, is > 0; "
    "(b) long-only top 10% net Sharpe from 2007-01-01 onward is > 0. "
    "NASDAQ-ADAPT-1 has an ML claim if additionally the LightGBM book's "
    "2007-onward net Sharpe exceeds the 12-1 momentum control book built with "
    "the identical long-only overlapping h=126 mandate. "
    "This is a scout, not a product. It does not replace COMBO. "
    "Survivorship (today's Nasdaq-100 members) and missing borrow costs "
    "are accepted for this run. No post-hoc adjustment."
)

TRAIN_RULE = (
    "Every fold trains exactly 500 LightGBM trees on a rolling window of at "
    "most 1260 sessions (~5×252). No early stopping. Huber objective. "
    "Label is the winsorized simple 126-session forward return (not QQQ-residual). "
    "Features are the equity clock (ret_21/63/126/252, mom_252_skip21, and the "
    "other NASDAQ-ADAPT-1 columns), not A0's 7/14/28/90 crypto windows. "
    "Book is long-only top 10% of the PIT top-30 (no shorts, no QQQ overlay). "
    "Overlapping 126 tranches. Purge and embargo are counted in sessions, not "
    "calendar days. The 12-1 control uses mom_252_skip21_raw as the score with "
    "the same book."
)

HORIZON = 126
TOP_PCT = 0.10
LONG_ONLY = True
EXEC_TOP_N = 30
DV_WINDOW = 30
SEED = 42
FIXED_TREES = 500
TRAIN_MAX_SESSIONS = 5 * 252  # 1260
MIN_TRAIN_SESSIONS = 5 * 252

PRICE_START = "1990-01-01"
BOOK_START = "2005-01-01"
HEADLINE_START = "2007-01-01"

FEAT_PATH = "data/nasdaq_adapt1/features.parquet"
PRED_PATH = "data/nasdaq_adapt1/preds_h126.parquet"
PRED_DIR = "data/nasdaq_adapt1/preds"
REPORT_MD = "reports/nasdaq_adapt1_report.md"
REPORT_JSON = "reports/nasdaq_adapt1_report.json"
CHART_PATH = "charts/nasdaq_adapt1_equity.png"
ADDENDUM_PATH = "reports/nasdaq_adapt1_addendum.md"
