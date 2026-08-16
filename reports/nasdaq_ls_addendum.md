# NASDAQ-LS — freeze addendum

**Status:** FROZEN before results. Conceptual scout: A0-style LightGBM Huber long-short on **today's Nasdaq-100** list.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Scope:** backtest and analysis only. Local CPU. Yahoo Finance daily bars. Zero GPU. No live components. No Modal write into `/data/quant/`.

This is **not** a product. This is **not** a retrain of frozen A0 crypto. Crypto artifacts, COMBO, SPREAD-LS, and LONG-TIDE are **read-only**. Look-ahead in index membership (survivorship / “wallhooking”) is **accepted for this run**; a later pass will rebuild with point-in-time constituents.

## Pre-registered factor statement (verbatim)

> NASDAQ-LS shows a cross-sectional factor if BOTH of: (a) pooled OOS RankIC of LightGBM Huber score vs residualized h=10 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 onward, is > 0; (b) long-short net Sharpe from 2007-01-01 onward is > 0. This is a scout, not a product. It does not replace COMBO. Survivorship (today's Nasdaq-100 members) and missing borrow costs are accepted for this run. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held name whose data ends is force-exited at its last available close (no better information assumed). The count of such forced exits is reported.

## Training rule (verbatim)

> Every fold trains exactly 500 LightGBM trees (the working A0 h=10 LS recipe). No early stopping. Huber objective. Market proxy is spliced Nasdaq (^IXIC then QQQ) in place of BTC. Judged book is long 10 / short 10 inside the PIT top-30 by 30d median dollar volume.

## Price rule (verbatim)

> All return calculations (labels, book PnL, QQQ benchmark, and price features) use Yahoo Adj Close (splits and dividends). Point-in-time dollar-volume ranks use unadjusted Close × Volume. OHLC bars are scaled by AdjClose/Close so features sit on the same total-return series.

## Mandate (frozen)

- Catalogue: Nasdaq-100 tickers as of download time (Wikipedia, else baked-in fallback). **Survivorship bias accepted.**
- Prices: Yahoo Finance daily OHLCV, 1990-01-01 → today. **Returns = Adj Close.**
- Market: `^IXIC` spliced into `QQQ` **Adj Close** at QQQ inception; used for β / residual labels (A0’s BTC slot).
- Train CS: all catalogue names with features that day.
- Exec universe: PIT **top 30** by 30-day median dollar volume (**unadjusted Close × Volume**), causal rolling window.
- Book: **long 10 / short 10** by score inside that top-30. Dollar-neutral 0.5L / 0.5S. Inv-vol within each leg. Overlapping h=10 tranches. Residual is cash. **No index hedge overlay.**
- Costs: 5 bps one-way on traded notional. No borrow, no financing.
- First trade date in the chart: **2005-01-01**. FACTOR gate window: **2007-01-01** onward. Walk-forward training uses all prior history (1990s included).
- Horizon h=10 sessions. Label = residualized forward log return vs the market proxy, winsorized 1/99 per date (A0 recipe).
- Same 33 A0 feature columns. LightGBM hparams match frozen A0 except **fixed 500 trees / no early stop**.
- Annualization **252** (US sessions), not crypto 365.

## What this freeze does not do

- Does not retrain, rescore, or rewrite A0 crypto artifacts, COMBO, SPREAD-LS, or LONG-TIDE.
- Does not claim a point-in-time Nasdaq-100 membership file.
- Does not charge stock-loan or locate.
- Does not use GPU, schedules, or live components.
- Does not restore crypto funding or BTC overlay.
