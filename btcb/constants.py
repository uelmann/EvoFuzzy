"""BTC-BEATER Phase 0/1 frozen constants. No sweeps."""

from __future__ import annotations

PHASE0_GATE = (
    "The dataset is USABLE-FROM-YYYY-MM if, from that date onward, ≥80% of the "
    "historical top-200 sample coins are present with correct terminal histories "
    "and a PIT universe is reconstructable. The earliest such date is the project's "
    "backtest start. If no date before 2021-01 qualifies, the 2018–2020 era is "
    "declared FICTION and excluded; if no date before 2023-01 qualifies, the "
    "project is BLOCKED pending a different data source. Mechanical, no post-hoc "
    "adjustment."
)

PHASE1_LABEL = (
    "NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe "
    "(book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its "
    "numbers become the floor every ML phase of this project must beat net of costs."
)

PHASE0C_GATE = (
    "The dataset is USABLE-FROM-YYYY-MM at the first quarterly CMC historical "
    "snapshot D whose true-top-100 coverage is ≥ 85% and remains ≥ 85% at every "
    "later snapshot, measured against the external snapshot lists. If that first "
    "D is after 2023-01, the project is BLOCKED pending a different data source. "
    "Mechanical, no post-hoc adjustment."
)

DEATH_CONVENTION = (
    "A held coin whose data ends is force-exited at its last available close "
    "(no better information assumed). The count and PnL impact of such forced "
    "exits is reported in every backtest of this project."
)

SEED = 42
SAMPLE_N = 30
SAMPLE_TOPN = 200
SAMPLE_YEARS = (2018, 2019, 2020)
PRESENT_FRAC = 0.80
PIT_MIN_NAMES = 50
PIT_DATE_FRAC = 0.80
SURVIVOR_LAG_DAYS = 14
ENDED_LAG_DAYS = 30
LISTED_SLACK_DAYS = 30
PIT_DV_WINDOW = 30
PIT_DV_MIN_PERIODS = 10
PIT_NS = (50, 100)

FICTION_BEFORE = "2021-01-01"
BLOCK_BEFORE = "2023-01-01"
SCAN_START = "2018-01-01"

# Phase 1 — frozen
LOOKBACK = 90
N_HOLD = 10
NAME_CAP = 0.10
HORIZON = 7
ALT_BPS = 10.0
BTC_BPS = 2.0
EXEC_UNIVERSE_N = 50

CORR_FLAG = 0.99
AGREE_N = 20

STABLE_OR_WRAP = frozenset(
    {
        "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDD", "TUSD",
        "GUSD", "USDP", "PYUSD", "FRAX", "TUSD", "EURT", "EURC", "AEUR",
        "WBTC", "WETH", "STETH", "WBETH", "BTCB", "RENBTC", "CBETH", "WSTETH",
        "RETH", "SAVAX", "PAXG", "XAUT",
    }
)

NAMED_GRAVEYARD = [
    {"query": "LUNA", "slugs": ("terra-luna", "terra-luna-v2"), "event": "Terra collapse May 2022 / LUNA 2.0", "expect_crash": "2022-05"},
    {"query": "LUNC", "slugs": ("terra-luna",), "event": "Terra Classic (ex-LUNA) after May 2022", "expect_crash": "2022-05"},
    {"query": "FTT", "slugs": ("ftx-token",), "event": "FTX collapse Nov 2022", "expect_crash": "2022-11"},
    {"query": "BCC", "slugs": ("bcc",), "event": "BCC ticker / BCH-forks era 2017–18", "expect_crash": "2017-08"},
    {"query": "BCH", "slugs": ("bitcoin-cash",), "event": "Bitcoin Cash (BCHABC/BCHSV forks 2018)", "expect_crash": "2018-11"},
    {"query": "BSV", "slugs": ("bitcoin-sv", "bitcoin-cash-sv"), "event": "BCHSV fork Nov 2018", "expect_crash": "2018-11"},
    {"query": "SRM", "slugs": ("serum",), "event": "Serum / FTX complex 2022", "expect_crash": "2022-11"},
    {"query": "CEL", "slugs": ("celsius-degree-token",), "event": "Celsius bankruptcy 2022", "expect_crash": "2022-06"},
    {"query": "BTT", "slugs": ("bittorrent", "bittorrent-new"), "event": "BTT redenomination ~Jan 2022", "expect_crash": "2022-01"},
    {"query": "XEM", "slugs": ("nem",), "event": "XEM secular decline", "expect_crash": None},
]

LIQUID_AGREE = (
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "DOT", "LTC", "LINK",
    "BCH", "ATOM", "AVAX", "ETC", "FIL", "NEAR", "UNI", "AAVE", "TRX", "XLM",
    "ALGO", "VET", "APT", "ARB", "OP", "SUI",
)

CYCLES = (
    ("2018-19", "2018-01-01", "2019-12-31"),
    ("2020-21", "2020-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023-24", "2023-01-01", "2024-12-31"),
    ("2025-26", "2025-01-01", "2026-12-31"),
)

ANNUALIZATION = 365

# Phase 0.c — full map + honest window
COVERAGE_THRESH = 0.85
COVERAGE_NS = (50, 100, 200)
SNAPSHOT_TOPN = 500
SNAPSHOT_START = (2017, 1)
SNAPSHOT_END = (2025, 4)
HTTP_HARD_STOP = 100_000
DOWNLOAD_MAX_YEARS = 12
DOWNLOAD_PERIOD_DAYS = 180
DOWNLOAD_SLEEP_S = 0.12
CONVERT_ID_USD = 2781
ENDED_BEFORE_YEAR = 2026

GRAVEYARD_0C = [
    {"query": "SRM", "slugs": ("serum",), "event": "Serum / FTX complex 2022"},
    {"query": "CEL", "slugs": ("celsius", "celsius-degree-token"), "event": "Celsius bankruptcy 2022"},
    {"query": "UST", "slugs": ("terrausd",), "event": "TerraUSD collapse May 2022"},
    {"query": "ANC", "slugs": ("anchor-protocol",), "event": "Anchor Protocol / Terra 2022"},
    {"query": "SAFEMOON", "slugs": ("safemoon",), "event": "SafeMoon collapse / delist"},
    {"query": "BCC", "slugs": ("bitconnect",), "event": "BitConnect era 2017–18"},
    {"query": "XEM", "slugs": ("nem",), "event": "XEM secular decline"},
    {"query": "FTT", "slugs": ("ftx-token",), "event": "FTX collapse Nov 2022 (continuity)"},
    {"query": "LUNC", "slugs": ("terra-luna",), "event": "Terra Classic after May 2022 (continuity)"},
]

# ---------------------------------------------------------------------------
# Phase 2 — winner-tail classifier (frozen a priori; no sweeps)
# ---------------------------------------------------------------------------

PHASE2_CRITERION = (
    "MODEL-V1 is VIABLE if, on the full OOS window at the median p_enter: "
    "(a) the relative line (book/BTC) has Sharpe > 0; (b) total return ≥ BTC B&H; "
    "(c) MaxDD ≤ BTC B&H MaxDD. MODEL-V1 REPLACES the naive rotation as the project "
    "floor if additionally its relative-line Sharpe ≥ naive v3 relative-line Sharpe "
    "+ 0.15 on the same window. Per-cycle honesty: report 2019–20, 2021, 2022, "
    "2023–24, 2025–26 separately; a verdict is not overridden by any single cycle. "
    "Mechanical, no post-hoc adjustment."
)

USABLE_FROM = "2017-09-30"
PHASE2_HORIZONS = (14, 30)
PHASE2_PRIMARY_H = 14
P_ENTER_GRID = (0.55, 0.60, 0.65)
P_EXIT_GAP = 0.05
BLOWOFF_RET_7 = 0.50
REPLACES_MARGIN = 0.15
MIN_TRAIN_CALENDAR_DAYS = 730
VAL_CALENDAR_DAYS = 90
STEP_CALENDAR_DAYS = 90
INNER_HOLDOUT_CALENDAR_DAYS = 90
CS_CLIP = 5.0
CTX_OWN_Z_WINDOW = 250
CTX_OWN_Z_MINP = 60

# 25 pruned survivors (A0 FEATURE_COLS minus Round-F eight)
PRICE_MOM_TREND_DIST = (
    "ret_14",
    "ret_56",
    "ret_90",
    "mom_90_skip14",
    "close_sma20",
    "close_sma50",
    "close_sma100",
    "sma20_sma50",
    "ema12_ema26",
    "dist_high_90",
    "dist_low_90",
)
PRICE_OWN_VOL_RANGE = (
    "yz_vol_14",
    "yz_vol_30",
    "yz_vol_60",
    "pk_vol_14",
    "vol_ratio",
    "vol_of_vol_30",
    "max_ret_14",
    "min_ret_14",
    "range_pos_28",
    "skew_60",
    "beta_btc_60",
    "idio_vol_60",
    "corr_btc_28",
    "amihud_14",
)
PRICE_COLS = PRICE_MOM_TREND_DIST + PRICE_OWN_VOL_RANGE
assert len(PRICE_COLS) == 25, len(PRICE_COLS)

CTX_COLS = (
    "ctx_disp",
    "ctx_excess_disp",
    "ctx_btc_vol",
    "ctx_btc_trend",
    "ctx_breadth",
    "ctx_corr",
    "ctx_alt_btc_trend",
)
NEW_COLS = (
    "log_mcap",
    "mcap_rank",
    "d_rank_30",
    "d_rank_90",
    "log_age",
    "dist_ath",
    "turnover",
    "turnover_z30",
)
FEATURE_COLS_V1 = PRICE_COLS + CTX_COLS + NEW_COLS
assert len(FEATURE_COLS_V1) == 40, len(FEATURE_COLS_V1)

LGBM_V1 = dict(
    objective="binary",
    metric="auc",
    num_leaves=31,
    learning_rate=0.03,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=3000,
    early_stopping_rounds=100,
    verbosity=-1,
)

NULL_REPLICATES = 25
NULL_SHUFFLE_SEEDS = tuple(range(101, 126))  # 25 seeds
NULL_FOLD_ANCHORS = ("first", "2022-01-01")  # fold 0 and fold nearest 2022-01-01 val_start

PHASE2_CYCLES = (
    ("2019-20", "2019-10-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023-24", "2023-01-01", "2024-12-31"),
    ("2025-26", "2025-01-01", "2026-12-31"),
)

# ---------------------------------------------------------------------------
# Phase 2.b — hygiene + two-stage model (frozen a priori; no sweeps)
# ---------------------------------------------------------------------------

PHASE2B_CRITERION = (
    "STAGE-S has SELECTION SKILL if, at h=14, full-OOS mean per-date AUC ≥ 0.52 "
    "with the empirical-null gates passing. MODEL-V2 is VIABLE if, on the full "
    "OOS window: (a) relative-line Sharpe > 0; (b) total return ≥ BTC B&H; "
    "(c) MaxDD ≤ BTC B&H MaxDD. It REPLACES the naive v4 floor if additionally "
    "relative-line Sharpe ≥ naive v4 + 0.15. Per-cycle honesty table mandatory; "
    "no single cycle overrides. Mechanical, no post-hoc adjustment."
)

JUMP_ABS_RET = 5.0  # |daily ret| > 5 flags redenom/split suspects
FLOOR_DV_MED_30 = 2_000_000.0
FLOOR_MIN_PRICE = 1e-6
FLOOR_MIN_SESSIONS = 60
FLOOR_MAX_ABS_RET_30 = 2.0  # 200%
CONTRIB_SHARE_FLAG = 0.25
STAGE_S_QUINTILE = 0.80  # y=1 if excess rank ≥ this quantile (top quintile)
REGIME_BREADTH = 0.50
REGIME_BUDGET = 0.50
REGIME_OFF_HYSTERESIS = 5
P_ENTER_V2 = 0.55  # names must clear this calibrated p to fill K
STAGE_S_AUC_SKILL = 0.52
AUTOPSY_START = "2019-10-19"  # the +9.98M% same-window naive v3 book

STAGE_S_COLS = PRICE_COLS + NEW_COLS
assert len(STAGE_S_COLS) == 33, len(STAGE_S_COLS)
assert not set(CTX_COLS) & set(STAGE_S_COLS)

# ---------------------------------------------------------------------------
# Phase 2.c — twin-head spread + repowered skill null (frozen a priori)
# ---------------------------------------------------------------------------

PHASE2C_CRITERION = (
    "SPREAD has SELECTION SKILL if, at h=14 or h=30: mean per-date RankIC(spread) "
    "≥ +0.01 AND mean per-date AUC ≥ 0.52 AND the §2 gate passes for the spread "
    "metric. MODEL-V3 is VIABLE if on the full OOS window at median θ: (a) "
    "relative-line Sharpe > 0; (b) total ≥ BTC B&H; (c) MaxDD ≤ BTC B&H. "
    "MODEL-V3 is PRODUCT-GRADE if additionally relative-line Sharpe ≥ 0.30 AND "
    "average alt allocation ≥ 5% (non-degenerate book). Per-cycle honesty table "
    "mandatory; no single cycle overrides. Mechanical, no post-hoc adjustment."
)

PHASE2C_NULL_GATE = (
    "Bias: every fold's null mean must satisfy the E.1b centering bound "
    "(AUC around 0.5, RankIC around 0). Skill passes if, for the judged signal, "
    "≥5 of 6 folds exceed their null 95th percentile OR the Stouffer-combined z "
    "across the 6 folds is ≥ 3.0. Symmetric: failure = PARKED, no override, "
    "no retest with different folds."
)

STAGE_S_BOT_QUINTILE = 0.20  # y=1 if excess rank ≤ this quantile (bottom quintile)
THETA_GRID = (0.10, 0.15, 0.20)
NULL_FOLD_IDS_2C = (0, 5, 9, 15, 21, 24)
NULL_K_EXCEED = 5
STOUFFER_Z_MIN = 3.0
SPREAD_RANKIC_SKILL = 0.01
PRODUCT_REL_SHARPE = 0.30
PRODUCT_ALT_MIN = 0.05  # average alt allocation = 1 - avg_w_btc

# ---------------------------------------------------------------------------
# Phase 3 — SPREAD-LS challenger (frozen a priori; no sweeps)
# ---------------------------------------------------------------------------

PHASE3_CRITERION = (
    "SPREAD-LS is VIABLE if full-OOS net Sharpe ≥ 0.8 AND trailing-18m net Sharpe "
    "≥ 0.3. It is SLEEVE-GRADE (candidate third sleeve alongside the frozen COMBO) "
    "if additionally its daily PnL correlation with the COMBO on the overlapping "
    "window is < 0.5 AND its same-window net Sharpe ≥ COMBO − 0.10. It is a "
    "REPLACEMENT CANDIDATE only if same-window net Sharpe ≥ COMBO + 0.15. "
    "Verdicts mechanical; the dollar-neutral variant is the headline; the "
    "beta-matched variant is reported, not judged. No post-hoc adjustment."
)

PHASE3_FUNDING_CAVEAT = (
    "FUNDING = 0. Funding is not available in this dataset; the sign of omitted "
    "funding is unknown. This is a material caveat on SPREAD-LS net Sharpe. "
    "Shorts on USDT-M perpetuals would have paid or received funding that is "
    "not in this book."
)

LS_GROSS_LONG = 0.50
LS_GROSS_SHORT = 0.50
LS_DECILE_K = 10
LS_QUINTILE_K = 20
LS_MIN_SHORTABLE = 5
LS_LONG_BPS = 10.0
LS_SHORT_FEE_BPS = 5.0
LS_SHORT_SLIP_BPS = 3.0
LS_BETA_WINDOW = 90
LS_BETA_MATCH_LOOKBACK = 60
LS_TRAIL_DAYS = 547  # 18 months, same house convention as COMBO
LS_VIABLE_FULL = 0.8
LS_VIABLE_TRAIL = 0.3
LS_SLEEVE_CORR_MAX = 0.5
LS_SLEEVE_SHARPE_GAP = 0.10
LS_REPLACE_SHARPE_GAP = 0.15
LS_SQUEEZE_N = 20
COMBO_OVERLAP_START = "2022-01-01"
LS_OOS_START = "2019-10-01"

# ---------------------------------------------------------------------------
# Universe sensitivity — SPREAD-LS U ∈ {30, 50, 100} (frozen a priori)
# ---------------------------------------------------------------------------

UNIVERSE_SENS_CRITERION = (
    "The production universe is the SMALLEST U whose full-OOS net Sharpe ≥ "
    "(best U's Sharpe − 0.15) AND trailing-18m ≥ (best U's trailing − 0.15) — "
    "i.e., prefer concentration/tradability only when it costs less than 0.15 "
    "Sharpe. Dollar-volume ranking remains the house standard; the mcap table "
    "is informational unless mcap beats volume by ≥ 0.20 on both windows for "
    "the chosen U. Mechanical, no post-hoc adjustment."
)

UNIVERSE_NS = (30, 50, 100)
UNIVERSE_SHARPE_TOL = 0.15
UNIVERSE_MCAP_BEAT = 0.20
UNIVERSE_FUNDING_ON = False  # 3.b has not run; all books funding-off
UNIVERSE_PRIMARY_H = 14

# ---------------------------------------------------------------------------
# Horizon sweep — SPREAD-LS h ∈ {3, 7, 14, 30} (frozen a priori)
# ---------------------------------------------------------------------------

HORIZON_SWEEP_CRITERION = (
    "h=14 is the incumbent. A different horizon becomes production only if its "
    "null gate passes AND its trailing-18m net Sharpe ≥ incumbent + 0.15 AND its "
    "full-OOS net Sharpe ≥ incumbent − 0.10. Among multiple qualifiers, highest "
    "trailing wins. If none qualify, h=14 stays. Mechanical, no post-hoc adjustment."
)

HORIZON_SWEEP_HS = (3, 7, 14, 30)
HORIZON_SWEEP_TRAIN = (3, 7)
HORIZON_SWEEP_INCUMBENT = 14
HORIZON_TRAIL_BEAT = 0.15
HORIZON_FULL_TOL = 0.10
HORIZON_FUNDING_ON = False  # 3.b has not run; all books funding-off
PHASE2C_PRED_SHA256 = "28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78"
PHASE2C_PRED_N_FILES = 112
