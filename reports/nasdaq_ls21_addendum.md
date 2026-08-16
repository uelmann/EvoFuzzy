# NASDAQ-LS21 — freeze addendum

**Status:** FROZEN before results. Same Nasdaq-100 scout as NASDAQ-LS, with three protocol changes: **top/worst 10%** (not 10 names), **h=21** prediction and overlapping rebalance, **rolling train cap 5 years**.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Prior scout:** NASDAQ-LS (k=10, h=10, expanding train) is **not overwritten**.

This is **not** a product. Survivorship (today’s Nasdaq-100 members) is **accepted for this run**.

## Pre-registered factor statement (verbatim)

> NASDAQ-LS21 shows a cross-sectional factor if BOTH of: (a) pooled OOS RankIC of LightGBM Huber score vs residualized h=21 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 onward, is > 0; (b) long-short net Sharpe from 2007-01-01 onward is > 0. This is a scout, not a product. It does not replace COMBO. Survivorship (today's Nasdaq-100 members) and missing borrow costs are accepted for this run. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held name whose data ends is force-exited at its last available close (no better information assumed). The count of such forced exits is reported.

## Training rule (verbatim)

> Every fold trains exactly 500 LightGBM trees on a rolling window of at most 1260 sessions (~5×252). No expanding 1990s window. No early stopping. Huber objective. Label and overlapping book use h=21 sessions. Legs are the top 10% and worst 10% of that day's PIT top-30 (k = ceil(0.10 × n), typically 3 names each side).

## Price rule (verbatim)

> All return calculations (labels, book PnL, QQQ benchmark, and price features) use Yahoo Adj Close (splits and dividends). Point-in-time dollar-volume ranks use unadjusted Close × Volume. OHLC bars are scaled by AdjClose/Close so features sit on the same total-return series.

## Mandate (frozen)

- Catalogue / exec: PIT **top 30** by 30-session median dollar volume among today’s Nasdaq-100.
- Legs: **long top 10% / short worst 10%** of that day’s top-30 (`k = ceil(0.10 × n)` → **3 names** when n=30). Not fixed 10 names.
- Horizon: **21 sessions**. Label = residualized 21-session forward log return vs spliced ^IXIC/QQQ. Overlapping **21 tranches**: each session one tranche reselects and holds ~21 sessions.
- Train: rolling **≤ 1260 sessions (~5 years)**. First fold also requires 1260 sessions. Refit every 90 sessions (A0 step). 500 Huber trees, no early stop.
- Costs: 5 bps one-way. No borrow. No index overlay. Inv-vol 50/50. Adj Close returns.
- Chart from **2005-01-01**. FACTOR window from **2007-01-01**. Annualization 252.

## What this freeze does not do

- Does not rewrite the NASDAQ-LS h=10 k=10 record, COMBO, SPREAD-LS, or LONG-TIDE.
- Does not use a point-in-time Nasdaq-100 membership file.
- Does not charge stock-loan.
