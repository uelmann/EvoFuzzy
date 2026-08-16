"""Frozen constants for the NASDAQ-LS scout.

COMBO / SPREAD-LS / LONG-TIDE / A0 crypto artifacts are not modified.
This is a conceptual factor scout on a survivorship-biased Nasdaq-100 list.
"""

from __future__ import annotations

FACTOR_CRITERION = (
    "NASDAQ-LS shows a cross-sectional factor if BOTH of: "
    "(a) pooled OOS RankIC of LightGBM Huber score vs residualized h=10 "
    "forward return, on the PIT top-30 by dollar volume, from 2007-01-01 "
    "onward, is > 0; "
    "(b) long-short net Sharpe from 2007-01-01 onward is > 0. "
    "This is a scout, not a product. It does not replace COMBO. "
    "Survivorship (today's Nasdaq-100 members) and missing borrow costs "
    "are accepted for this run. No post-hoc adjustment."
)

DEATH_CONVENTION = (
    "A held name whose data ends is force-exited at its last available close "
    "(no better information assumed). The count of such forced exits is reported."
)

TRAIN_RULE = (
    "Every fold trains exactly 500 LightGBM trees (the working A0 h=10 LS recipe). "
    "No early stopping. Huber objective. Market proxy is spliced Nasdaq "
    "(^IXIC then QQQ) in place of BTC. Judged book is long 10 / short 10 "
    "inside the PIT top-30 by 30d median dollar volume."
)

PRICE_RULE = (
    "All return calculations (labels, book PnL, QQQ benchmark, and price features) "
    "use Yahoo Adj Close (splits and dividends). Point-in-time dollar-volume ranks "
    "use unadjusted Close × Volume. OHLC bars are scaled by AdjClose/Close so "
    "features sit on the same total-return series."
)

HORIZON = 10
K_LONG = 10
K_SHORT = 10
EXEC_TOP_N = 30
DV_WINDOW = 30
SEED = 42
FIXED_TREES = 500

PRICE_START = "1990-01-01"
BOOK_START = "2005-01-01"
HEADLINE_START = "2007-01-01"

MARKET_QQQ = "QQQ"
MARKET_IXIC = "^IXIC"
MARKET_NAME = "MARKET"

MIN_CS = 20
MIN_HISTORY_DAYS = 100
COST_BPS = 5.0
GROSS_LIMIT = 1.0
LAG = 0
ANNUALIZATION = 252
TRAIL_DAYS = int(365 * 1.5)

# Baked-in fallback if Wikipedia is unreachable. Today's NDX is the universe
# by design (survivorship accepted). Runtime prefers a live Wikipedia scrape.
FALLBACK_TICKERS = (
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CCEP",
    "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP",
    "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST",
    "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU",
    "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDLZ",
    "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA", "NXPI",
    "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL",
    "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD",
    "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS", "AXON", "TRI",
)

DATA_DIR = "data/nasdaq"
PANEL_PATH = "data/nasdaq/panel.parquet"
MARKET_PATH = "data/nasdaq/market.parquet"
TICKERS_PATH = "data/nasdaq/tickers.json"
FEAT_PATH = "data/nasdaq/features.parquet"
PRED_PATH = "data/nasdaq/preds_h10.parquet"
