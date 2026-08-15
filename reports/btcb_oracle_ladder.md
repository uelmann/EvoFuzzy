# BTC-BEATER ORACLE LADDER — perfect-foresight ceiling and IC curve

**ANALYSIS ONLY.** Nothing adopted. No retraining, no product changes. CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.

Construction is **14-day full rebalance** (not the production overlapping-tranche book). Every ladder/reference point uses the same construction.

## Pre-registered verdict (verbatim, before results)

> MODEL EFFICIENCY verdict: our model sits ON-CURVE if its CAGR is within ±20% (in log terms) of the ladder interpolation at its own realized RankIC — conclusion: the binding constraint is INFORMATION, and improvement means new data, not new architecture. It sits BELOW-CURVE if lower than that band — conclusion: TRANSLATION slack exists in the signal→book layer, quantified as the CAGR gap. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Window 2019-10-20 → 2026-07-18 n=2464 formations=176
- GPU used = `False`

## Mechanical verdict

- **MODEL EFFICIENCY = BELOW-CURVE**
- model RankIC `0.1160` CAGR `13.9%`
- ladder interpolation at that IC `119.8%` (band `99.8%` – `143.7%`)
- ratio vs curve `0.116`  log-gap `-2.156`  (need |log-gap| ≤ ln(1.20) = 0.182)
- capture of oracle CAGR `0.11%`
- TRANSLATION slack exists in the signal→book layer, quantified as the CAGR gap `13.9%` vs curve `119.8%`.

Mechanical, no post-hoc adjustment.

## Plain language

the ceiling is 1.546e+14 total / 1.254e+04% CAGR (NET h=14; GROSS 1.839e+14 / 1.287e+04%); at IC 1.0→0.16 the curve passes through oracle CAGR 1.254e+04% → ladder-0.16 CAGR 189.8% (realized RankIC 0.1601); our model captures 0.11% of the oracle CAGR, which is below its information content.

## 1 — Oracle (perfect foresight)

| book | RankIC | total | CAGR | MaxDD | Sharpe | avg #names | formations | forced |
|------|--------|-------|------|-------|--------|------------|------------|--------|
| ORACLE GROSS h=14 | 1.0000 | 1.839e+14 | 1.287e+04% | -30.4% | 7.261 | 5.9 | 176 | 0 |
| ORACLE NET h=14 (10 bps/side) | 1.0000 | 1.546e+14 | 1.254e+04% | -30.4% | 7.222 | 5.9 | 176 | 0 |
| ORACLE GROSS h=7 (secondary) | 1.0000 | 9.225e+19 | 8.884e+04% | -31.8% | 9.409 | 5.9 | 353 | 0 |
| ORACLE NET h=7 (secondary) | 1.0000 | 6.528e+19 | 8.441e+04% | -31.9% | 9.337 | 5.9 | 353 | 0 |

The h=14 NET row is the ceiling used on the curve.

## 2 — Degraded-oracle ladder (h=14, costs on, 5 seeds)

| target IC | realized RankIC mean [range] | total mean [range] | CAGR mean [range] | MaxDD mean [range] |
|-----------|------------------------------|--------------------|-------------------|--------------------|
| 0.50 | 0.5000 [0.5000, 0.5001] | 5.174e+09 [3.783e+09, 6.068e+09] | 2643.4% [2523.0%, 2713.2%] | -34.5% [-37.9%, -31.0%] |
| 0.30 | 0.3001 [0.2999, 0.3002] | 1.317e+06 [8.145e+05, 1.792e+06] | 701.5% [650.9%, 743.9%] | -41.7% [-45.7%, -37.1%] |
| 0.20 | 0.1999 [0.1998, 0.2001] | 1.036e+04 [6.631e+03, 1.593e+04] | 290.6% [268.2%, 319.3%] | -46.4% [-49.9%, -41.3%] |
| 0.16 | 0.1601 [0.1599, 0.1603] | 1.336e+03 [9.580e+02, 1.552e+03] | 189.8% [176.5%, 197.0%] | -48.5% [-51.8%, -43.1%] |
| 0.10 | 0.1000 [0.0999, 0.1000] | 8.949e+01 [6.043e+01, 1.248e+02] | 94.2% [84.0%, 104.7%] | -55.3% [-60.6%, -49.9%] |
| 0.05 | 0.0504 [0.0502, 0.0505] | 1.229e+01 [850.5%, 1.578e+01] | 46.4% [39.6%, 51.9%] | -65.5% [-73.5%, -60.4%] |

Realized RankIC is the mean per-date Spearman(score, future excess) after per-date noise calibration.

## 3 — Real reference points (same construction, same window, costs on)

| book | RankIC | total | CAGR | MaxDD | Sharpe | avg #names | formations | forced |
|------|--------|-------|------|-------|--------|------------|------------|--------|
| OUR MODEL (frozen 2.c spread) | 0.1160 | 140.3% | 13.9% | -72.1% | 0.514 | 5.9 | 176 | 0 |
| NAIVE 90d excess | -0.0236 | 2.4% | 0.4% | -84.1% | 0.294 | 5.7 | 176 | 0 |
| RANDOM (IC≈0, 5-seed mean) | -0.0009 | 74.5% | 8.1% | -79.1% | 0.418 | 5.9 | 176 | 0 |

RANDOM RankIC mean `-0.0009` [-0.0039, 0.0022]; CAGR `8.1%`.

## Per-cycle (NET)

| cycle | book | n | total | CAGR | MaxDD | Sharpe |
|-------|------|---|-------|------|-------|--------|
| 2019-20 | ORACLE NET h=14 | 439 | 1.359e+02 | 5875.2% | -20.2% | 7.933 |
| 2021 | ORACLE NET h=14 | 365 | 1.330e+04 | 1.330e+06% | -30.4% | 9.258 |
| 2022 | ORACLE NET h=14 | 365 | 3.779e+01 | 3779.3% | -22.7% | 5.146 |
| 2023-24 | ORACLE NET h=14 | 731 | 1.617e+04 | 1.253e+04% | -14.7% | 8.177 |
| 2025-26 | ORACLE NET h=14 | 564 | 1.343e+02 | 2294.5% | -12.1% | 6.665 |
| 2019-20 | OUR MODEL | 439 | 51.6% | 41.3% | -31.7% | 1.117 |
| 2021 | OUR MODEL | 365 | 212.7% | 212.7% | -40.9% | 1.909 |
| 2022 | OUR MODEL | 365 | -60.8% | -60.8% | -61.9% | -1.223 |
| 2023-24 | OUR MODEL | 731 | 80.0% | 34.1% | -25.9% | 0.981 |
| 2025-26 | OUR MODEL | 564 | -28.2% | -19.3% | -37.3% | -0.563 |
| 2019-20 | NAIVE 90d | 439 | 21.4% | 17.5% | -33.7% | 0.597 |
| 2021 | NAIVE 90d | 365 | 294.2% | 294.2% | -45.1% | 2.059 |
| 2022 | NAIVE 90d | 365 | -72.5% | -72.5% | -73.5% | -1.518 |
| 2023-24 | NAIVE 90d | 731 | 78.9% | 33.7% | -49.7% | 0.811 |
| 2025-26 | NAIVE 90d | 564 | -56.5% | -41.6% | -65.8% | -1.131 |

## Notes

- OUR MODEL here is a naked 14d-full-rebalance long book on the frozen spread. It is **not** LONG-TIDE and **not** SPREAD-LS (those are overlapping-tranche products). Its RankIC on this window/construction is the per-date Spearman of the frozen spread vs next-14d excess; that is not the same number as daily RankIC quoted for the production tranche books.
- Binance-listed PIT top-100 is ~60 names, so the top decile is ~6 names at the 10% cap (residual cash idle). Identical across every point.
- Ladder noise calibration: realized RankIC matches each target to ~0.0001 (5-seed mean); see table.
- Nothing is adopted. This is a map of information vs translation.

Elapsed s=250.2. GPU=False.

COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.

