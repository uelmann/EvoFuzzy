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
