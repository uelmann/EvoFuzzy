# NASDAQ-ADAPT-1 — freeze addendum

**Status:** FROZEN before results.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Prior scouts:** NASDAQ-LS and NASDAQ-LS21 records are **not overwritten**.

This is the adaptation the NASDAQ-LS forensics asked for: **simple-return label** (so the book is paid in what the model ranks), **equity clock** (21/63/126/252 and 12–1), **long-only** (no forced shorts among today’s winners), **session purge/embargo**, and a **12–1 control** on the identical book. Survivorship (today’s Nasdaq-100 members) is **accepted for this run**.

## Pre-registered factor statement (verbatim)

> NASDAQ-ADAPT-1 shows a cross-sectional factor if BOTH of: (a) pooled OOS RankIC of LightGBM Huber score vs the simple (non-residual) h=126 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 onward, is > 0; (b) long-only top 10% net Sharpe from 2007-01-01 onward is > 0. NASDAQ-ADAPT-1 has an ML claim if additionally the LightGBM book's 2007-onward net Sharpe exceeds the 12-1 momentum control book built with the identical long-only overlapping h=126 mandate. This is a scout, not a product. It does not replace COMBO. Survivorship (today's Nasdaq-100 members) and missing borrow costs are accepted for this run. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held name whose data ends is force-exited at its last available close (no better information assumed). The count of such forced exits is reported.

## Training rule (verbatim)

> Every fold trains exactly 500 LightGBM trees on a rolling window of at most 1260 sessions (~5×252). No early stopping. Huber objective. Label is the winsorized simple 126-session forward return (not QQQ-residual). Features are the equity clock (ret_21/63/126/252, mom_252_skip21, and the other NASDAQ-ADAPT-1 columns), not A0's 7/14/28/90 crypto windows. Book is long-only top 10% of the PIT top-30 (no shorts, no QQQ overlay). Overlapping 126 tranches. Purge and embargo are counted in sessions, not calendar days. The 12-1 control uses mom_252_skip21_raw as the score with the same book.

## Price rule (verbatim)

> All return calculations (labels, book PnL, QQQ benchmark, and price features) use Yahoo Adj Close (splits and dividends). Point-in-time dollar-volume ranks use unadjusted Close × Volume. OHLC bars are scaled by AdjClose/Close so features sit on the same total-return series.

## Mandate (frozen)

- Catalogue / exec: PIT **top 30** by 30-session median dollar volume among today’s Nasdaq-100.
- Book: **long-only top 10%** of that day’s top-30 (`k = ceil(0.10 × n)` → **3 names** when n=30). **No shorts. No QQQ overlay.**
- Horizon: **126 sessions** (~6 months). Label = winsorized **simple** 126-session forward return (1/99 per date). Overlapping **126 tranches**.
- Features: equity clock listed in `nasdaq_ls/adapt_features.py` (`ADAPT_FEATURE_COLS`). Includes **mom_252_skip21** (12–1). Does **not** use A0 `ret_7/14/28/90`.
- Control: same universe, same long-only overlapping h=126 book, score = **mom_252_skip21_raw** (not z-scored).
- Train: rolling **≤ 1260 sessions (~5 years)**. First fold also requires 1260 sessions. Refit every 90 sessions. 500 Huber trees, no early stop. Purge/embargo in **sessions**.
- Costs: 5 bps one-way. No borrow. Inv-vol on the long leg, gross 1.0. Adj Close returns.
- Chart from **2005-01-01**. FACTOR / ML-claim window from **2007-01-01**. Annualization 252.
- Excess vs QQQ is **informational**, not a FACTOR input.

## What this freeze does not do

- Does not rewrite NASDAQ-LS, NASDAQ-LS21, COMBO, SPREAD-LS, or LONG-TIDE.
- Does not use a point-in-time Nasdaq-100 membership file.
- Does not charge stock-loan.
- Does not claim a product if LightGBM merely copies 12–1.
