# NASDAQ-LS21 — top/worst 10%, h=21, 5y rolling train

**BACKTEST AND ANALYSIS ONLY.** A0-style Huber LightGBM on Yahoo Nasdaq-100 bars. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**. Survivorship (today's index members) is accepted for this scout.

**Universe source:** fallback n=100
**Price span:** 1990-01-02 00:00:00+00:00 → 2026-08-14 00:00:00+00:00
**Mandate:** PIT top 30 by 30d median dollar volume; long top 10% / short worst 10% (k=ceil(0.10*n), typically 3 names); overlapping h=21; inv-vol; 5 bps; no borrow.
**Train:** 500 Huber trees, no early stop, rolling ≤1260 sessions (~5y). Residual vs spliced QQQ.

## Pre-registered factor statement

> NASDAQ-LS21 shows a cross-sectional factor if BOTH of: (a) pooled OOS RankIC of LightGBM Huber score vs residualized h=21 forward return, on the PIT top-30 by dollar volume, from 2007-01-01 onward, is > 0; (b) long-short net Sharpe from 2007-01-01 onward is > 0. This is a scout, not a product. It does not replace COMBO. Survivorship (today's Nasdaq-100 members) and missing borrow costs are accepted for this run. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Scout: NO FACTOR** — RankIC from 2007-01-01=0.008 (pass=True); LS net Sharpe from 2007-01-01=-0.581 (pass=False).

## Headline books (252-day Sharpe)

| book | full Sharpe | trail-18m | CAGR | MaxDD | total | avg #long | avg #short | avg |gross| | % flat | cost drag | forced | top-5 |name| PnL |
|------|-------------|-----------|------|-------|-------|-----------|------------|-------------|--------|-----------|--------|------------------|
| NASDAQ-LS21 from 2005-01-01 | -0.577 | -0.733 | -6.6% | -79.1% | -76.7% | 8.997 | 8.831 | 0.838 | 0.0% | 0.205 | 0 | NVDA=-0.365, NFLX=-0.299, MU=-0.263, MRVL=-0.177, INTC=0.171 |
| NASDAQ-LS21 from 2007-01-01 (FACTOR window) | -0.581 | -0.733 | -6.7% | -78.6% | -74.0% | 8.997 | 8.831 | 0.838 | 0.0% | 0.205 | 0 | NVDA=-0.365, NFLX=-0.299, MU=-0.263, MRVL=-0.177, INTC=0.171 |

### Sharpe by year (FACTOR window)

| 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.221 | 1.651 | 0.118 | -1.531 | -1.122 | -1.334 | -0.807 | -0.179 | 0.290 | -1.352 | -0.845 | -0.466 | -1.434 | -1.112 | -1.140 | -0.991 | 0.044 | -0.203 | 0.020 | -1.573 |

## RankIC on PIT top-30

| window | RankIC vs residual y | ICIR | NW-t | n | RankIC vs simple 21d return |
|--------|----------------------|------|------|---|----------------------------------------|
| all OOS | 0.019 | 1.381 | 2.526 | 7865 | 0.003 |
| from 2007-01-01 | 0.008 | 0.623 | 0.908 | 4878 | — |

### Costless benchmarks (not FACTOR inputs)

- QQQ B&H from 2007: Sharpe=0.774, trail-18m=1.115, CAGR=15.9%, MaxDD=-55.0%, total=1638.3%.
- EW PIT top-30 from 2007: Sharpe=0.881, trail-18m=1.552, CAGR=20.1%, MaxDD=-53.6%, total=3357.4%.

## Construction notes

Yahoo Nasdaq-100 source=fallback n=100; PIT top-30; long/short top_pct=0.1 (k=ceil(0.10*n)); overlapping h=21; rolling train ≤1260 sessions; inv-vol; 5 bps one-way; Adj Close; book from 2005-01-01; FACTOR window 2007-01-01; n_preds=590879.

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified. This scout does not rewrite the system card.

Elapsed seconds: 193.7. GPU used: false. trees=500 early_stop=off n_folds=88 train_cap=1260 sessions top_pct=0.1 h=21.
