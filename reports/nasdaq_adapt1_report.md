# NASDAQ-ADAPT-1 — simple h=126, equity clock, long-only, 12-1 control

**BACKTEST AND ANALYSIS ONLY.** Adaptation of the Nasdaq scout after the LS/LS21 forensics. COMBO, SPREAD-LS, LONG-TIDE, NASDAQ-LS, and NASDAQ-LS21 are **untouched**. Survivorship (today's index members) is accepted for this scout.

**Universe source:** fallback n=100
**Price span:** 1990-01-02 00:00:00+00:00 → 2026-08-14 00:00:00+00:00
**Mandate:** PIT top 30 by 30d median dollar volume; long-only top 10%; overlapping h=126; inv-vol; 5 bps; no shorts; no QQQ overlay.
**Train:** 500 Huber trees, no early stop, rolling ≤1260 sessions. Label = winsorized simple 126-session forward return. Equity-clock features. Session purge/embargo.

## Pre-registered factor statement

> NASDAQ-ADAPT-1 shows a cross-sectional factor if BOTH of: (a) pooled OOS RankIC of LightGBM Huber score vs the simple (non-residual) h=126 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 onward, is > 0; (b) long-only top 10% net Sharpe from 2007-01-01 onward is > 0. NASDAQ-ADAPT-1 has an ML claim if additionally the LightGBM book's 2007-onward net Sharpe exceeds the 12-1 momentum control book built with the identical long-only overlapping h=126 mandate. This is a scout, not a product. It does not replace COMBO. Survivorship (today's Nasdaq-100 members) and missing borrow costs are accepted for this run. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Scout: NO FACTOR** — RankIC from 2007-01-01=-0.013 (pass=False); long-only net Sharpe from 2007-01-01=0.833 (pass=True).
- **NO ML CLAIM** — GBM Sharpe=0.833 vs 12-1 control Sharpe=0.860 (beat=False).

## Headline books (252-day Sharpe)

| book | full Sharpe | trail-18m | CAGR | MaxDD | total | avg #long | avg #short | avg |gross| | % flat | cost drag | forced | top-5 |name| PnL |
|------|-------------|-----------|------|-------|-------|-----------|------------|-------------|--------|-----------|--------|------------------|
| ADAPT-1 GBM long-only from 2005-01-01 | 0.842 | 1.508 | 20.6% | -61.9% | 5125.3% | 13.740 | 0.000 | 0.812 | 0.0% | 0.035 | 0 | NFLX=0.884, TSLA=0.826, NVDA=0.603, AMZN=0.445, AMD=0.440 |
| ADAPT-1 GBM long-only from 2007-01-01 (FACTOR window) | 0.833 | 1.508 | 20.9% | -61.9% | 3624.4% | 13.740 | 0.000 | 0.812 | 0.0% | 0.035 | 0 | NFLX=0.884, TSLA=0.826, NVDA=0.603, AMZN=0.445, AMD=0.440 |
| 12-1 control long-only from 2005-01-01 | 0.869 | 0.953 | 24.7% | -56.3% | 11573.7% | 7.770 | 0.000 | 0.855 | 0.0% | 0.032 | 0 | NVDA=1.315, MU=0.728, TSLA=0.670, PLTR=0.328, NFLX=0.303 |
| 12-1 control long-only from 2007-01-01 | 0.860 | 0.953 | 24.8% | -56.3% | 7597.2% | 7.770 | 0.000 | 0.855 | 0.0% | 0.032 | 0 | NVDA=1.315, MU=0.728, TSLA=0.670, PLTR=0.328, NFLX=0.303 |

### GBM Sharpe by year (FACTOR window)

| 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.473 | -1.271 | 1.729 | 1.003 | -0.254 | 0.007 | 3.297 | 1.118 | 1.084 | 1.333 | 1.533 | 0.800 | 2.007 | 2.233 | 0.645 | -0.516 | 1.938 | 1.165 | 1.798 | -3.052 |

## RankIC on PIT top-30 vs simple h=126 return

| window | LightGBM RankIC | ICIR | NW-t | n | 12-1 RankIC |
|--------|-----------------|------|------|---|-------------|
| all OOS | -0.011 | -0.794 | -0.666 | 7707 | — |
| from 2007-01-01 | -0.013 | -0.926 | -0.601 | 4809 | 0.017 |

### Costless benchmarks (not FACTOR inputs)

- QQQ B&H from 2007: Sharpe=0.768, trail-18m=0.937, CAGR=15.7%, MaxDD=-53.4%, total=1520.8%.
- EW PIT top-30 from 2007: Sharpe=0.849, trail-18m=1.187, CAGR=19.0%, MaxDD=-52.3%, total=2686.3%.
- GBM excess vs QQQ from 2007: Sharpe=0.378, trail-18m=1.638, CAGR=4.6%, MaxDD=-51.1%, total=135.9%.

### Mean LightGBM gain (across ok folds)

yz_vol_126=7212.0, amihud_21=6907.2, beta_mkt_126=6206.9, dist_low_252=4427.4, idio_vol_63=4222.7, yz_vol_63=3759.6, ret_252=3459.0, vol_of_vol_63=3377.0, corr_mkt_63=3080.6, dist_high_252=3070.2

## Construction notes

Yahoo Nasdaq-100 source=fallback n=100; PIT top-30; long-only top_pct=0.1; overlapping h=126; rolling train ≤1260 sessions; session purge/embargo; simple (non-residual) label; equity-clock features; inv-vol; 5 bps one-way; Adj Close; book from 2005-01-01; FACTOR window 2007-01-01; n_preds=571344.

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE, NASDAQ-LS, and NASDAQ-LS21 are not modified. This scout does not rewrite the system card.

Elapsed seconds: 360.0. GPU used: false. trees=500 early_stop=off n_folds=87 train_cap=1260 sessions top_pct=0.1 h=126 long_only=True.
