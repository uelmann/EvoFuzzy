# BTC-BEATER Phase 3 — SPREAD-LS challenger

**BACKTEST ONLY.** Portfolio layer only. 2.c spread scores reused byte-identical. No retraining. No BTC in either leg. CPU only, zero GPU. COMBO untouched.

## Pre-registered criteria (verbatim, before results)

> SPREAD-LS is VIABLE if full-OOS net Sharpe ≥ 0.8 AND trailing-18m net Sharpe ≥ 0.3. It is SLEEVE-GRADE (candidate third sleeve alongside the frozen COMBO) if additionally its daily PnL correlation with the COMBO on the overlapping window is < 0.5 AND its same-window net Sharpe ≥ COMBO − 0.10. It is a REPLACEMENT CANDIDATE only if same-window net Sharpe ≥ COMBO + 0.15. Verdicts mechanical; the dollar-neutral variant is the headline; the beta-matched variant is reported, not judged. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

Shorts are force-covered on the same convention. Forced-exit and forced-cover counts are reported separately.

## Funding caveat (verbatim)

> FUNDING = 0. Funding is not available in this dataset; the sign of omitted funding is unknown. This is a material caveat on SPREAD-LS net Sharpe. Shorts on USDT-M perpetuals would have paid or received funding that is not in this book.

## Mechanical verdicts (dollar-neutral h=14 headline)

- **SPREAD-LS is VIABLE: True** (full 1.354 need ≥ 0.800; trail-18m 2.385 need ≥ 0.300)
- **SPREAD-LS is SLEEVE-GRADE: False** (corr=0.157 need < 0.500; same-window 1.376 vs COMBO 1.711 need ≥ 1.611)
- **SPREAD-LS is REPLACEMENT CANDIDATE: False** (need ≥ 1.861)
- OOS 2019-10-19 → 2026-08-13 n=2491
- realized beta vs BTC (full OLS) = -0.122
- avg shortable (PIT-100 ex-BTC) = 67.6; % incomplete short = 89.0%
- forced exits=14 covers=7
- BTC in book hits = 0

The beta-matched variant is reported, not judged. No post-hoc adjustment.

## Books

| book | net Sharpe | trail-18m | total | CAGR | MaxDD | #long | #short | shortable | % inc. short | ann TO | exits | covers | β vs BTC |
|------|------------|-----------|-------|------|-------|-------|--------|-----------|--------------|--------|-------|--------|----------|
| SPREAD-LS DN h=14 (headline) | 1.354 | 2.385 | 1145.4% | 44.7% | -35.9% | 26.64 | 18.78 | 67.6 | 89.0% | 11.02 | 14 | 7 | -0.122 |
| SPREAD-LS β-match h=14 | 1.818 | 2.458 | 1706.6% | 52.8% | -25.8% | 26.70 | 18.74 | 67.6 | 89.0% | 10.35 | 14 | 7 | 0.025 |
| SPREAD-LS DN h=30 | 1.095 | 1.977 | 540.0% | 31.5% | -34.2% | 31.19 | 22.81 | 68.0 | 88.6% | 6.44 | 30 | 44 | -0.101 |
| SPREAD-LS β-match h=30 | 1.394 | 2.093 | 719.4% | 36.4% | -34.6% | 31.25 | 22.75 | 68.0 | 88.6% | 6.38 | 30 | 44 | 0.018 |

## Per-cycle honesty (headline DN h=14)

| cycle | n | total | CAGR | net Sharpe | MaxDD | #long | #short |
|-------|---|-------|------|------------|-------|-------|--------|
| 2019-20 | 440 | 64.3% | 50.9% | 1.355 | -30.5% | 25.76 | 4.67 |
| 2021 | 365 | 5.2% | 5.2% | 0.324 | -29.6% | 27.67 | 20.24 |
| 2022 | 365 | 100.1% | 100.1% | 3.158 | -7.3% | 26.80 | 17.06 |
| 2023-24 | 731 | 12.1% | 5.9% | 0.346 | -21.2% | 27.33 | 22.86 |
| 2025-26 | 590 | 221.3% | 105.9% | 2.424 | -16.3% | 25.72 | 24.40 |

## vs frozen COMBO (overlap 2022-01 →)

Window 2022-01-23 → 2026-06-26 n=1616. COMBO replayed from frozen A0 scores; product untouched.

- SPREAD-LS same-window net Sharpe = 1.376 (total 348.6%)
- COMBO same-window net Sharpe = 1.711 (total 610.1%)
- daily PnL correlation = 0.157

## Squeeze-days (20 largest EW floored top-100 up-days)

| date | EW top-100 | SPREAD-LS net |
|------|------------|---------------|
| 2021-05-24 | 25.46% | -1.06% |
| 2020-03-19 | 19.91% | -0.46% |
| 2022-11-10 | 18.35% | -2.78% |
| 2021-05-20 | 17.43% | -1.32% |
| 2021-04-26 | 15.47% | -5.06% |
| 2024-11-06 | 14.34% | -5.47% |
| 2021-09-22 | 13.95% | -3.28% |
| 2024-08-08 | 13.62% | -2.29% |
| 2025-11-07 | 13.12% | -7.81% |
| 2025-04-09 | 13.04% | -3.57% |
| 2020-03-13 | 12.97% | -2.06% |
| 2025-05-08 | 12.67% | -3.54% |
| 2025-08-22 | 12.01% | -7.33% |
| 2021-03-01 | 11.85% | -5.77% |
| 2021-05-26 | 11.79% | -2.11% |
| 2021-01-28 | 11.53% | -8.46% |
| 2022-02-28 | 11.44% | -2.21% |
| 2025-10-12 | 11.31% | -4.86% |
| 2021-02-05 | 11.28% | -2.06% |
| 2025-03-02 | 11.24% | -0.73% |

Squeeze-day mean SPREAD-LS PnL = -3.61%; sum = -72.24%.

## Cache / reuse

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` n_files=112
- BTC id = 1; BTC in book hits = 0
- shortable mapped Binance perps = 831
- GPU=False. Elapsed s=238.5.

COMBO untouched (v2.0-combo-final).

