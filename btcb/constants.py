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
UNIVERSE_FUNDING_ON = False  # 3.b dropped; 3.c decides funding-on hybrid
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
HORIZON_FUNDING_ON = False  # 3.b dropped; 3.c decides funding-on hybrid
PHASE2C_PRED_SHA256 = "28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78"
PHASE2C_PRED_N_FILES = 112

# ---------------------------------------------------------------------------
# Phase 3.c — Binance replay of SPREAD-LS (frozen a priori; no sweeps)
# Replaces Phase 3.b. MASTER book removed from scope by PI.
# ---------------------------------------------------------------------------

PHASE3C_BETA_MATCH_DESIGNATION = (
    "The Phase 3 freeze designated dollar-neutral as the judged headline and "
    "β-matched as reported-not-judged. After seeing results (DN β=−0.122, "
    "β-matched β=0.025), β-matched is designated the production SPREAD-LS book. "
    "This is disclosed, not hidden. Phase 3 mechanical verdicts (VIABLE, not "
    "SLEEVE-GRADE, not replacement) remain those of the DN headline and are not "
    "retroactively rewritten. All subsequent work (funding-on, MASTER) uses the "
    "β-matched book."
)

PHASE3C_HOUSE_RULE = (
    "The bias clause of future null gates reverts to the original E.1b "
    "tolerance: CONTAMINATED requires ≥2 fold-level violations of the 2·SE "
    "bound, not 1. The 'every fold' variant has ≈25% false-alarm probability "
    "with 6 folds. No past verdicts change."
)

PHASE3C_MASTER_NOTE = (
    "MASTER (COMBO+SPREAD-LS combination) removed from scope by PI decision; "
    "the 0.157 correlation remains on the ledger for allocation purposes."
)

PHASE3C_VALIDATION = (
    "PRICES ARE VALIDATED if, on the replayable subset, BOOK-BINANCE-ONLY daily "
    "PnL correlation with the same-days CMC-priced book is ≥ 0.95 AND its net "
    "Sharpe ≥ (same-days CMC Sharpe − 0.15). If validated, BOOK-HYBRID "
    "(funding-on) becomes the OFFICIAL SPREAD-LS record; funding-off CMC "
    "numbers are deprecated with a ledger footnote. If NOT validated, the "
    "discrepancy is quantified per year and per name-tier, the official record "
    "is suspended, and no improvement work proceeds until the pricing gap is "
    "understood. Mechanical, no post-hoc adjustment."
)

PHASE3C_CORR_MIN = 0.95
PHASE3C_SHARPE_TOL = 0.15
PHASE3C_REF_VARIANT = "bm_h14"
PHASE3C_REF_H = 14
PHASE3C_REF_START = "2019-10-19"
PHASE3C_REF_END = "2026-08-13"
PHASE3C_REF_N_DAYS = 2491
PHASE3C_REF_SHARPE = 1.8177621190077422
PHASE3C_REF_TRAIL = 2.457536436771376
PHASE3C_REF_MAXDD = -0.2581255543376627
PHASE3C_REF_BTC_HITS = 0
COMBO_SPREADLS_CORR = 0.157
# Future null gates only. Existing _metric_verdict still requires every fold.
FUTURE_NULL_BIAS_MIN_VIOLATIONS = 2
PHASE3C_NAME_TIERS = ((1, 10, "1-10"), (11, 50, "11-50"), (51, 100, "51-100"))
PHASE3C_POSITION_SHA256 = "f47f7ece40d6cee536b2a07c25961d1d69284f92ddf716447a52b5f57fcc232b"
PHASE3C_REF_HYBRID_SHARPE = 1.5551874624157

# ---------------------------------------------------------------------------
# Academic factor — unconstrained D10−D1 on CMC + implementation-tax waterfall
# (frozen a priori; analysis only; nothing adopted)
# ---------------------------------------------------------------------------

ACADEMIC_FACTOR_CRITERION = (
    "PAPER ALPHA EXISTS if FACTOR-JT top-100 gross Sharpe ≥ 1.0 with NW-t ≥ 3.0 "
    "on the full OOS window. The IMPLEMENTATION TAX is recorded as (paper gross "
    "Sharpe − implementable hybrid Sharpe), decomposed per the waterfall. Labels "
    "are diagnostic; nothing is adopted or changed. Mechanical, no post-hoc "
    "adjustment."
)

ACADEMIC_FACTOR_H = 14
ACADEMIC_FACTOR_NW_LAG = 14
ACADEMIC_FACTOR_SHARPE_MIN = 1.0
ACADEMIC_FACTOR_NW_T_MIN = 3.0
ACADEMIC_FACTOR_NAIVE_BPS = 10.0
ACADEMIC_FACTOR_NS = (50, 100)

# ---------------------------------------------------------------------------
# Phase 3.e — pricing-gap forensics (frozen a priori; analysis only)
# Mandated by the 3.c suspension. No book redesign.
# ---------------------------------------------------------------------------

PHASE3E_OUTCOME = (
    "SIGNAL-CONFIRMED if RankIC(spread vs Binance returns) ≥ RankIC(same-names CMC) "
    "− 0.02 on both full and trailing windows: the gap is then an execution/pricing-level "
    "effect; the official record RESUMES as BOOK-HYBRID funding-on, with the suspension "
    "footnote replaced by the forensic decomposition, and Binance-priced numbers become "
    "canonical for all future phases. SIGNAL-PARTLY-ARTIFACT if RankIC drops > 0.02 on "
    "either window: the record stays suspended, the artifact share is quantified, and "
    "the next phase MUST re-derive the book on Binance-only pricing before anything else. "
    "Mechanical, no post-hoc adjustment."
)

PHASE3E_RANKIC_TOL = 0.02
PHASE3E_H = 14
PHASE3E_NAME_TIERS = ((1, 30, "1-30"), (31, 60, "31-60"), (61, 100, "61-100"))
PHASE3E_TOP_N = 30
PHASE3E_STALE_CMC_BPS = 10.0
PHASE3E_STALE_BN_MOVE = 0.02
PHASE3E_REF_BN_SHARPE = 1.2956392949142752
PHASE3E_REF_CMC_SUB_SHARPE = 1.5585902582860225
PHASE3E_REF_CORR = 0.9838
CMC_PANEL_SHA256 = "c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249"

# ---------------------------------------------------------------------------
# LONG-TIDE — full-size long leg + frozen Stage-T gate, BTC parking
# (frozen a priori; backtest only; no signal changes)
# ---------------------------------------------------------------------------

LONGTIDE_CRITERION = (
    "LONG-TIDE is VIABLE if: (a) total return ≥ BTC B&H; (b) relative-line (book/BTC) Sharpe > 0; "
    "(c) MaxDD ≤ BTC B&H MaxDD. It SUPERSEDES BTC-BEATER v1 as the official long product only if "
    "additionally: (d) relative-line Sharpe ≥ v1's + 0.15 on the common window; (e) average alt "
    "deployment ≥ 15%; (f) no cycle with relative-line Sharpe < −0.30. If (a–c) pass but (d–f) "
    "do not, LONG-TIDE is recorded as a parallel long variant and v1 stays official. Mechanical, "
    "no post-hoc adjustment."
)

LONGTIDE_H = 14
LONGTIDE_BUDGET = 1.0
LONGTIDE_K = 10
LONGTIDE_REL_MARGIN = 0.15
LONGTIDE_ALT_MIN = 0.15
LONGTIDE_CYCLE_REL_FLOOR = -0.30
LONGTIDE_V1_P_ENTER = 0.60
LONGTIDE_PRECONDITION = "SIGNAL-CONFIRMED"

# ---------------------------------------------------------------------------
# ORACLE LADDER — perfect-foresight ceiling + IC-degraded oracles
# (frozen a priori; analysis only; nothing adopted)
# ---------------------------------------------------------------------------

ORACLE_LADDER_CRITERION = (
    "MODEL EFFICIENCY verdict: our model sits ON-CURVE if its CAGR is within ±20% "
    "(in log terms) of the ladder interpolation at its own realized RankIC — conclusion: "
    "the binding constraint is INFORMATION, and improvement means new data, not new "
    "architecture. It sits BELOW-CURVE if lower than that band — conclusion: TRANSLATION "
    "slack exists in the signal→book layer, quantified as the CAGR gap. Mechanical, "
    "no post-hoc adjustment."
)

ORACLE_LADDER_H = 14
ORACLE_LADDER_H_SEC = 7
ORACLE_LADDER_TARGETS = (0.50, 0.30, 0.20, 0.16, 0.10, 0.05)
ORACLE_LADDER_SEEDS = (101, 102, 103, 104, 105)
ORACLE_LADDER_MOM_DAYS = 90
ORACLE_LADDER_LOG_BAND = 0.20
ORACLE_LADDER_MIN_N = 8
ORACLE_LADDER_DECILE = 10

# ---------------------------------------------------------------------------
# ORACLE LADDER 2 — decompose BELOW-CURVE: tail-blindness vs translation
# (frozen a priori; analysis only; nothing adopted)
# ---------------------------------------------------------------------------

ORACLE_LADDER2_CRITERION = (
    "The gap decomposition is: TAIL-INFORMATION share = the part explained by "
    "overlap/tail-IC deficits vs the equal-IC ladder; CONSTRUCTION share = the "
    "best of V1–V3 minus the crude base, plus the production-construction delta "
    "measured on the ladder signal. No variant is adopted here; any adoption "
    "requires a fresh pre-registered phase with the house criteria. If the best "
    "translation variant improves CAGR by ≥ +10pp over the crude base at "
    "comparable MaxDD, translation work is declared the next priority; otherwise "
    "the right-tail information hunt (catalysts/attention data) is declared the "
    "next priority. Mechanical, no post-hoc adjustment."
)

ORACLE_LADDER2_IC_EQ = 0.116
ORACLE_LADDER2_IC_REF = 0.16
ORACLE_LADDER2_PP_MIN = 0.10
ORACLE_LADDER2_MAXDD_SLACK = 0.10
ORACLE_LADDER2_V1_CAP = 0.15
ORACLE_LADDER2_V2_N = 5
ORACLE_LADDER2_V2_CAP = 0.20
ORACLE_LADDER2_V3_Q = 0.95
ORACLE_LADDER2_V3_CAP = 0.10
ORACLE_LADDER2_MONSTER_K = 3

# ---------------------------------------------------------------------------
# Phase 4 v2 — TAIL ROUND 1: LambdaRank head + positioning block
# (frozen a priori; backtest/analysis only; nothing adopted)
# ---------------------------------------------------------------------------

PHASE4V2_PI_SCOPE = (
    "Catalyst and attention data families (unlocks, listing announcements, search "
    "volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume "
    "plus derivatives data already retrievable (funding, open interest, basis, "
    "taker flows)."
)

PHASE4V2_PI_KILL = (
    "The old project's microstructure KILL applied to the mean-regression label "
    "on the old system; this phase judges a positioning block on the NEW system's "
    "tail metrics — fresh pre-registration, not a kill-list retest."
)

PHASE4V2_CRITERION = (
    "TAIL-LOSS EXTRACTS if RANK or blend improves tail-IC(top-half) ≥ +0.010 AND "
    "overlap ≥ +0.015 vs baseline with the null passing; BARREN otherwise. "
    "POSITIONING LIVE if the positioning block adds tail-IC(top-half) ≥ +0.010 "
    "OR overlap ≥ +0.015 on top of the best A-signal, with ≥50% perp coverage of "
    "top-100 name-days from 2021. PRICE-ADDITIONS LIVE at the same thresholds. "
    "Verdicts mechanical; nothing adopted; any production change requires a "
    "fresh pre-registered phase. No post-hoc adjustment."
)

PHASE4V2_H = 14
PHASE4V2_RANK_GRADES = 5
PHASE4V2_TRUNCATION = 10
PHASE4V2_NDCG_AT = 10
PHASE4V2_TAIL_IC_DELTA = 0.010
PHASE4V2_OVERLAP_DELTA = 0.015
PHASE4V2_PERP_COV_MIN = 0.50
PHASE4V2_PERP_COV_FROM = "2021-01-01"
PHASE4V2_OVERLAP_NULL_CENTER = 0.10  # 1 / top-decile
PHASE4V2_NW_LAG = 14

LGBM_RANK = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=(10,),
    lambdarank_truncation_level=10,
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

POSITIONING_COLS = (
    "funding_z_7",
    "funding_z_30",
    "funding_level_3d",
    "dOI_7",
    "dOI_30",
    "basis",
    "taker_imbalance_7",
    "pos_missing",
)
PRICE_ADD_COLS = (
    "past_alpha_60",
    "trend_composite",
)
assert len(POSITIONING_COLS) == 8, len(POSITIONING_COLS)
assert len(PRICE_ADD_COLS) == 2, len(PRICE_ADD_COLS)
assert not set(POSITIONING_COLS) & set(STAGE_S_COLS)
assert not set(PRICE_ADD_COLS) & set(STAGE_S_COLS)

# ---------------------------------------------------------------------------
# Phase 4.b — TWIN-RANK + vol-matched null (new house standard) + DIR arm
# (frozen a priori; backtest/analysis only; nothing adopted)
# ---------------------------------------------------------------------------

PHASE4B_NULL_REGISTRATION = (
    "For tail metrics (tail-IC top-half, top-decile overlap, monster capture), "
    "the empirical null shuffles labels WITHIN vol-quintile buckets per date "
    "(yz_vol_30 quintiles), preserving the vol→outcome loading. Folds {0,5,9,15,21,24} "
    "× 25 replicates. The null mean per fold becomes the structural reference level; "
    "bias check = null mean stability across replicates (2·SE band around the fold's "
    "own null mean, E.1b tolerance: ≥2 fold violations for CONTAMINATED). Skill = real "
    "metric exceeds the vol-matched null 95th percentile on ≥5/6 folds OR Stouffer z ≥ 3.0. "
    "This supersedes the plain within-date shuffle for tail metrics in all future phases; "
    "plain-shuffle results remain on the record."
)

PHASE4B_CRITERION = (
    "TWIN-RANK EXTRACTS if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the "
    "frozen spread, with the vol-matched null passing. DIR LIVE at the same thresholds "
    "with the same null. If BOTH fail, the ledger records: 'PRICE-VOLUME TAIL CEILING — "
    "within this data perimeter, tail improvements beyond the frozen spread are not "
    "demonstrable under vol-matched nulls; the fork (capital phase | Phase-5 hourly "
    "attention | perimeter expansion) passes to the PI.' Verdicts mechanical; nothing "
    "adopted; any production change requires a fresh pre-registered phase. No post-hoc "
    "adjustment."
)

PHASE4B_CEILING = (
    "PRICE-VOLUME TAIL CEILING — within this data perimeter, tail improvements beyond "
    "the frozen spread are not demonstrable under vol-matched nulls; the fork "
    "(capital phase | Phase-5 hourly attention | perimeter expansion) passes to the PI."
)

PHASE4B_DIR_RATIONALE = (
    "label-distribution reweighting for rare extreme positives (Yang et al. ICML 2021 "
    "lineage; logit-adjustment Menon et al. 2021), the cheapest tail-emphasis "
    "intervention available."
)

PHASE4B_H = 14
PHASE4B_RANK_GRADES = 5
PHASE4B_VOL_BINS = 5
PHASE4B_DIR_BOOST = 2.0  # w_i = 1 + BOOST * 1[top decile]
PHASE4B_TAIL_IC_DELTA = 0.010
PHASE4B_OVERLAP_DELTA = 0.015

# ---------------------------------------------------------------------------
# MANUEL-2 — gauss(ret14)·gauss(ret28)/gauss(std63), top-50 mcap, top 5%, long-only
# (frozen a priori; backtest/analysis only; nothing adopted)
# ---------------------------------------------------------------------------

MANUEL2_CRITERION = (
    "MANUEL-2 is CONFIRMED if the best pre-declared book (daily or weekly, btc-in "
    "or btc-ex — 4 books, disclosed as a 4-way look) on Binance-hybrid pricing has: "
    "total ≥ BTC B&H AND daily correlation with BTC ≤ 0.70. It is STRONG if additionally "
    "relative-line Sharpe ≥ 0.50 and MaxDD ≤ 1.10 × BTC's. It is PARTIAL if exactly one "
    "of the two claim clauses passes (state which). REFUTED if neither. Per-cycle honesty "
    "table mandatory; no single cycle overrides. If CONFIRMED, a fresh pre-registered "
    "phase evaluates adoption. Mechanical, no post-hoc adjustment."
)

MANUEL2_FORMULA = (
    "Score = gauss(ret_14d) × gauss(ret_28d) / gauss(std_63d), "
    "gauss(x) = Φ(z_cs(x)), z_cs = cross-sectional z-score per date across that day's universe. "
    "All inputs use data ≤ t."
)

MANUEL2_STABLE_COMPLETION = (
    "STABLECOIN EXCLUSION: pegged assets (stablecoins, wrapped/pegged tokens) are excluded "
    "from the universe. Rationale on record: the formula divides by gauss(std63); std≈0 names "
    "get near-infinite scores — the literal formula would buy Tether. Exclusion list built from "
    "the panel's pegged-asset tags + |90d total return| < 2% heuristic, logged."
)

MANUEL2_BTC_COMPLETION = (
    "BTC VARIANTS: run BOTH btc-in and btc-ex universes (BTC's low std makes the denominator "
    "favor it; the PI's thesis is a non-BTC-like curve, so both are shown)."
)

MANUEL2_GAUSS_COMPLETION = (
    "gauss(x) = Φ(z_cs(x)): normal CDF of the CROSS-SECTIONAL z-score, computed per date "
    "across the day's universe. All inputs use data ≤ t."
)

MANUEL2_BEST_RULE = (
    "Best book = highest full-window total return among the four pre-declared books "
    "(daily/weekly × btc-in/btc-ex). Verdicts apply to that book. All four are reported. "
    "This is the disclosed 4-way look. No other selection."
)

MANUEL2_MAXDD_RULE = (
    "MaxDD is the house signed quantity (negative). STRONG clause 'MaxDD ≤ 1.10 × BTC's' "
    "is |book MaxDD| ≤ 1.10 × |BTC MaxDD|, i.e. book_maxdd >= 1.10 * btc_maxdd when both ≤ 0."
)

MANUEL2_UNIVERSE_N = 50
MANUEL2_TOP_PCT = 0.05
MANUEL2_RET_A = 14
MANUEL2_RET_B = 28
MANUEL2_STD_W = 63
MANUEL2_PEG_LOOKBACK = 90
MANUEL2_PEG_ABS_RET = 0.02
MANUEL2_CORR_MAX = 0.70
MANUEL2_REL_SHARPE_STRONG = 0.50
MANUEL2_MAXDD_MULT = 1.10
MANUEL2_ROLL_CORR = 90
MANUEL2_COST_BPS = 10.0
MANUEL2_WEEKDAY = 0  # Monday
MANUEL2_START = PHASE3C_REF_START
MANUEL2_STD_DDOF = 1  # pandas rolling std of daily simple returns
MANUEL2_CS_DDOF = 0  # cross-sectional z
MANUEL2_EXTRA_STABLES = frozenset(
    {
        "UST", "USTC", "USDE", "USDS", "USDY", "USDJ", "USDX", "CUSD", "SUSD", "LUSD",
        "MUSD", "OUSD", "DUSD", "HUSD", "EURS", "EUROC", "AGEUR", "USDBC", "USD0",
        "CRVUSD", "GHO", "USDA", "USR", "USD1", "RLUSD", "TBTC", "HBTC", "SBTC",
        "TETH", "WEETH", "EZETH", "RSETH", "SWETH", "METH", "CBBTC", "SOLBTC",
    }
)
MANUEL2_PEG_NEEDLES = (
    "tether",
    "trueusd",
    "usd-coin",
    "usd coin",
    "binance-usd",
    "binance usd",
    "wrapped-bitcoin",
    "wrapped bitcoin",
    "wrapped-ether",
    "wrapped ether",
    "wrapped-btc",
    "wrapped-eth",
    "binance-peg",
    "pegged",
    "stablecoin",
    "terrausd",
    "first-digital-usd",
)
MANUEL2_PEG_USD_EXCEPT = frozenset({"SAND", "BAND", "BOND", "AMP"})

# ---------------------------------------------------------------------------
# PROJECT FORGE — Phase F0 (evolutionary STRATEGY miner, compounding fitness)
# Frozen a priori. Backtest/analysis only. Nothing adopted. Zero GPU.
# ---------------------------------------------------------------------------

FORGE_NONCONTAMINATION = (
    "The PI's hand-made formulas are EXCLUDED from this project: not as seeds, "
    "not as operator biases, not as fitness shaping. The search space contains "
    "generic primitives and operators only. If the miner independently evolves "
    "similar structures, they are reported without censorship — that is evidence "
    "the method works, not contamination."
)

FORGE_FITNESS = (
    "fitness(S) = median over the perturbation set P of [ann. net log-wealth of "
    "BOOK(S, p)] − λ_c · nodes(S)/25, computed ONLY on the MINE window. BOOK(S, p): "
    "long-only, top-k by S (k per p), equal-weight, daily rebalance, 10 bps/side, "
    "always invested, death convention. Perturbation set P (fixed): k ∈ {3, 5, 8} × "
    "rebalance ∈ {daily, weekly} × the two universes = 12 books. λ_c = 0.02. A formula "
    "whose 12-book spread (max−min ann log-wealth) exceeds 1.0 is discarded regardless "
    "of median (knife-edge filter). No correlation metric appears anywhere in the fitness."
)

FORGE_CRITERION = (
    "On the JUDGE window, per champion (headline book = k=5, daily, top-100 DV): "
    "FORGE-ALIVE if ≥1 champion has (a) total return ≥ BTC B&H same-window; "
    "(b) relative-line Sharpe > 0; (c) MaxDD ≤ 1.15 × BTC's. FORGE-STRONG if ≥2 "
    "champions pass, or one passes with relative-line Sharpe ≥ 0.5. FORGE-DEAD "
    "otherwise, and the ledger records that GP-mined formulaic strategies do not "
    "survive nesting on this data. Champions' formulas, MINE/SELECT/JUDGE numbers, "
    "and the full per-cycle honesty tables are printed regardless. Benchmarks on "
    "JUDGE: BTC B&H, EW baskets (both universes), frozen-spread crude book. "
    "Mechanical, no post-hoc adjustment; no re-mining, no second look at JUDGE."
)

FORGE_MINE_START = "2019-10-20"
FORGE_MINE_END = "2022-12-31"
FORGE_SELECT_START = "2023-01-01"
FORGE_SELECT_END = "2024-12-31"
FORGE_JUDGE_START = "2025-01-01"
FORGE_JUDGE_END = PHASE3C_REF_END  # 2026-08-13

FORGE_POP = 2000
FORGE_GENS = 60
FORGE_TOURNAMENT = 20
FORGE_P_CROSSOVER = 0.70
FORGE_P_SUBTREE = 0.25
FORGE_P_POINT = 0.05
FORGE_MAX_DEPTH = 8
FORGE_MAX_NODES = 25
FORGE_INIT_DEPTH = (2, 6)
FORGE_EARLY_STOP_GENS = 10
FORGE_ELITE = 1
FORGE_LAMBDA_C = 0.02
FORGE_KNIFE_SPREAD = 1.0
FORGE_K_SET = (3, 5, 8)
FORGE_REBAL_SET = ("daily", "weekly")
FORGE_HEADLINE_K = 5
FORGE_HEADLINE_REBAL = "daily"
FORGE_HEADLINE_UNIV = "dv100"
FORGE_SELECT_TOP = 50
FORGE_N_CHAMPIONS = 5
FORGE_DIVERSITY_CORR = 0.70
FORGE_COST_BPS = 10.0
FORGE_MAXDD_MULT = 1.15
FORGE_REL_SHARPE_STRONG = 0.50
FORGE_NULL_GENS = 30  # half budget
FORGE_SEED = SEED  # 42
FORGE_WEEKDAY = 0  # Monday
FORGE_PDIV_EPS = 1e-8
FORGE_STD_DDOF = 1
FORGE_CS_DDOF = 0
FORGE_TS_K = (5, 14, 28, 63)
FORGE_RET_K = (1, 3, 7, 14, 21, 28, 63, 90)
FORGE_STD_K = (14, 30, 63, 90)
FORGE_HL_K = (28, 90)
FORGE_MCAP_N = 50
FORGE_BUDGET_SEC = 6 * 3600
FORGE_HEARTBEAT_SEC = 60
FORGE_WATCHDOG_SILENCE_SEC = 20 * 60
FORGE_MAX_CONCURRENCY = 50
FORGE_CUBE_START = "2018-01-01"

FORGE_PRIMITIVES = (
    "close",
    "volume",
    "mcap",
    "ret_1",
    "ret_3",
    "ret_7",
    "ret_14",
    "ret_21",
    "ret_28",
    "ret_63",
    "ret_90",
    "std_14",
    "std_30",
    "std_63",
    "std_90",
    "dist_high_28",
    "dist_high_90",
    "dist_low_28",
    "dist_low_90",
    "dist_ath",
    "amihud_14",
    "turnover",
    "age",
    "mcap_rank",
    "funding_z_7",
    "funding_z_30",
    "dOI_7",
    "dOI_30",
    "basis",
    "taker_imb_7",
    "pos_missing",
)
assert len(FORGE_PRIMITIVES) == 31, len(FORGE_PRIMITIVES)

FORGE_UNARY_OPS = ("abs", "neg", "log1pabs", "rank_cs", "z_cs", "phi_cs")
FORGE_BINARY_OPS = ("add", "sub", "mul", "pdiv", "min", "max")
FORGE_TS_OPS = ("ts_mean", "ts_std", "ts_max", "ts_min", "ts_rank", "lag")

# ---------------------------------------------------------------------------
# Phase 8 — MODEL-ZOO (CS-ATTN / TabPFN v2 / ridge-on-ranks)
# Frozen a priori. Backtest/analysis only. Nothing adopted. Independent of 7.c/7.d.
# GPU allowed ONLY for Arm B if TabPFN requires it (hard cap $20, logged).
# ---------------------------------------------------------------------------

PHASE8_CRITERION = (
    "An arm is LIVE if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the "
    "frozen spread with the vol-matched null passing. An arm is a WHOLE-RANKING "
    "LEAD if instead ΔRankIC ≥ +0.010 with the null passing (recorded for a fresh "
    "production-book phase, not adopted here). LINEAR-CEILING (informational, from "
    "Arm C): if Arm C's whole-list RankIC ≥ 0.90 × the frozen spread's, the ledger "
    "records that nonlinearity contributes less than 10% of the daily signal and "
    "future daily modeling effort is unjustified. If any arm's signal has "
    "correlation < 0.6 with the frozen spread while reaching RankIC ≥ 0.10, it is "
    "recorded as an ORTHOGONAL SIGNAL candidate for a fresh blending phase. "
    "Nothing adopted here. Mechanical, no post-hoc adjustment."
)

PHASE8_FIREWALL = (
    "The PI's hand-made formulas (including MANUEL-2) are quarantined from this "
    "phase: not imported, not seeded, not used as features or targets."
)

PHASE8_TABPFN_CAVEAT = (
    "TabPFN assumes i.i.d.-like structure from its synthetic prior; financial "
    "non-stationarity violates it. This arm tests whether that matters in practice."
)

PHASE8_NULL_REGISTRATION = (
    "VOL-MATCHED NULL on the single BEST zoo arm by whole-list RankIC: folds "
    "{5,15,21,24} × 15 within-vol-quintile shuffles (first 15 of NULL_SHUFFLE_SEEDS), "
    "Modal .map fan-out. Skill = real exceeds vol-matched null p95 on ≥3/4 folds "
    "OR Stouffer z ≥ 3.0. Bias = 2·SE band around the fold's own null mean; "
    "CONTAMINATED iff ≥2 fold violations. LIVE uses the tail-IC(top-half) gate; "
    "WHOLE-RANKING LEAD uses the RankIC gate on the same shuffles. Only the best "
    "arm is nulled; other arms cannot be LIVE or LEAD. CS-ATTN null cells are "
    "cold-start, primary seed 42 only (not 3-init bag; null folds are "
    "non-contiguous so warm-start does not apply). Ridge and TabPFN nulls rerun "
    "the frozen procedure. RankIC null band is recorded for the chart."
)

PHASE8_DATE_SUBSAMPLE = (
    "Primary comparison = full OOS for every arm that completes. If TabPFN cannot "
    "finish full OOS inside the $20 GPU cap, ALL arms (and the frozen spread) are "
    "judged on the pre-declared 1-in-3 OOS date subsample: sorted unique OOS "
    "dates, keep i % 3 == 0 (0-indexed). Arms A/C still report full-OOS metrics "
    "as informational. The 1-in-3 rule is frozen before results."
)

PHASE8_H = 14
PHASE8_TAIL_IC_DELTA = 0.010
PHASE8_OVERLAP_DELTA = 0.015
PHASE8_RANKIC_DELTA = 0.010
PHASE8_LINEAR_CEILING_RATIO = 0.90
PHASE8_ORTH_CORR = 0.60
PHASE8_ORTH_RANKIC = 0.10
PHASE8_NULL_FOLD_IDS = (5, 15, 21, 24)
PHASE8_NULL_REPLICATES = 15
PHASE8_NULL_K_EXCEED = 3
PHASE8_INNER_HOLDOUT_DATES = 120
PHASE8_SEEDS = (42, 43, 44)
PHASE8_ATTN_D_MODEL = 64
PHASE8_ATTN_HEADS = 4
PHASE8_ATTN_LAYERS = 2
PHASE8_ATTN_FF = 128
PHASE8_ATTN_LR = 1e-4
PHASE8_ATTN_LR_MIN = 1e-5
PHASE8_ATTN_WD = 1e-4
PHASE8_ATTN_CLIP = 1.0
PHASE8_ATTN_ES_FLOOR = 10
PHASE8_ATTN_PATIENCE = 8
PHASE8_ATTN_CAP = 40
PHASE8_ATTN_SWA = 3
PHASE8_ATTN_DATE_BATCH = 64
PHASE8_RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
PHASE8_TABPFN_CONTEXT_CAP = 10_000
PHASE8_TABPFN_N_ESTIMATORS = 8
PHASE8_GPU_USD_CAP = 20.0
PHASE8_GPU_USD_PER_HOUR_A10G = 1.10
PHASE8_DIAG_DATES = 10
PHASE8_SUBSAMPLE_STRIDE = 3
PHASE8_SUBSAMPLE_OFFSET = 0


