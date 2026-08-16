# NASDAQ-LS — LightGBM long 10 / short 10 on Nasdaq-100 (scout)

**BACKTEST AND ANALYSIS ONLY.** A0-style Huber LightGBM on Yahoo Nasdaq-100 bars. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**. Survivorship (today's index members) is accepted for this scout.

**Universe source:** fallback n=100
**Price span:** 1990-01-02 00:00:00+00:00 → 2026-08-14 00:00:00+00:00
**Mandate:** PIT top 30 by 30d median dollar volume; long 10 / short 10 by score; overlapping h=10; inv-vol; 5 bps one-way; no borrow; no index overlay.
**Train:** 500 trees, no early stop, Huber. Market residual = spliced ^IXIC/QQQ.

## Pre-registered factor statement

> NASDAQ-LS shows a cross-sectional factor if BOTH of: (a) pooled OOS RankIC of LightGBM Huber score vs residualized h=10 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 onward, is > 0; (b) long-short net Sharpe from 2007-01-01 onward is > 0. This is a scout, not a product. It does not replace COMBO. Survivorship (today's Nasdaq-100 members) and missing borrow costs are accepted for this run. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Scout: NO FACTOR** — RankIC from 2007-01-01=0.022 (pass=True); LS net Sharpe from 2007-01-01=-0.490 (pass=False).

## Headline books (252-day Sharpe)

| book | full Sharpe | trail-18m | CAGR | MaxDD | total | avg #long | avg #short | avg |gross| | % flat | cost drag | forced | top-5 |name| PnL |
|------|-------------|-----------|------|-------|-------|-----------|------------|-------------|--------|-----------|--------|------------------|
| NASDAQ-LS from 2005-01-01 | -0.465 | -1.449 | -3.6% | -62.0% | -54.9% | 14.078 | 14.937 | 0.824 | 0.0% | 0.274 | 0 | NVDA=-0.251, COST=0.142, TSLA=-0.116, AVGO=-0.107, AMAT=-0.106 |
| NASDAQ-LS from 2007-01-01 (FACTOR window) | -0.490 | -1.449 | -3.9% | -62.0% | -54.1% | 14.078 | 14.937 | 0.824 | 0.0% | 0.274 | 0 | NVDA=-0.251, COST=0.142, TSLA=-0.116, AVGO=-0.107, AMAT=-0.106 |

### Sharpe by year (FACTOR window)

| 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.624 | 1.829 | -0.803 | -2.481 | 1.256 | 0.045 | -0.905 | 0.032 | 0.342 | -1.035 | -1.618 | -0.270 | -0.667 | -0.941 | -0.422 | 0.968 | -1.888 | -0.058 | -2.049 | -0.909 |

## RankIC on PIT top-30

| window | RankIC vs residual y | ICIR | NW-t | n | RankIC vs simple 10d USDT-style return |
|--------|----------------------|------|------|---|----------------------------------------|
| all OOS | 0.032 | 2.232 | 5.828 | 8439 | 0.007 |
| from 2007-01-01 | 0.022 | 1.578 | 3.137 | 4897 | — |

### Costless benchmarks (not FACTOR inputs)

- QQQ B&H from 2007: Sharpe=0.788, trail-18m=1.161, CAGR=16.2%, MaxDD=-53.2%, total=1759.3%.
- EW PIT top-30 from 2007: Sharpe=0.893, trail-18m=1.588, CAGR=20.4%, MaxDD=-52.0%, total=3603.4%.

## Construction notes

Yahoo Nasdaq-100 source=fallback n=100; PIT top-30 by 30d median DV; long 10 / short 10; overlapping h=10; inv-vol; 5 bps one-way; no borrow; returns=Yahoo Adj Close; DV=unadjusted Close×Volume; 500 Huber trees; last-fold-wins; book from 2005-01-01; FACTOR window from 2007-01-01; market=spliced ^IXIC/QQQ; survivorship accepted; n_preds=618226.

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified. This scout does not rewrite the system card.

Elapsed seconds: 479.8. GPU used: false. trees=500 early_stop=off n_folds=94.
