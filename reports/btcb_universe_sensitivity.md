# BTC-BEATER SPREAD-LS — universe sensitivity (top-30 / top-50 / top-100)

**BACKTEST ONLY.** Portfolio layer only. 2.c spread cache byte-identical. FUNDING=OFF (3.b has not run). β-matched. CPU only, zero GPU. COMBO untouched.

## Pre-registered reading (verbatim, before results)

> The production universe is the SMALLEST U whose full-OOS net Sharpe ≥ (best U's Sharpe − 0.15) AND trailing-18m ≥ (best U's trailing − 0.15) — i.e., prefer concentration/tradability only when it costs less than 0.15 Sharpe. Dollar-volume ranking remains the house standard; the mcap table is informational unless mcap beats volume by ≥ 0.20 on both windows for the chosen U. Mechanical, no post-hoc adjustment.

## Funding caveat (verbatim)

> FUNDING = 0. Funding is not available in this dataset; the sign of omitted funding is unknown. This is a material caveat on SPREAD-LS net Sharpe. Shorts on USDT-M perpetuals would have paid or received funding that is not in this book.

## Mechanical choice

- **Chosen production U = top-100** (fallback=False; ranking=dollar_volume)
- best full=1.818 so need ≥ 1.668; best trail-18m=2.458 so need ≥ 2.308
- mcap beats volume by ≥ 0.20 on both windows for chosen U: **False** (DV 1.818/2.458 vs mcap 0.983/2.024)

## Dollar-volume universes (house standard, β-matched, funding-off, h=14)

| U | net Sharpe | trail-18m | MaxDD | #long | #short | shortable | % inc. short | ann TO | squeeze mean | β vs BTC | top-5 PnL | RankIC |
|---|------------|-----------|-------|-------|--------|-----------|--------------|--------|--------------|----------|-----------|--------|
| DV top-30 | 1.148 | 1.116 | -27.2% | 9.04 | 7.01 | 23.4 | 45.8% | 9.25 | -0.008 | 0.036 | 56.0% | 0.150 |
| DV top-50 | 1.439 | 1.867 | -24.4% | 13.98 | 11.04 | 37.3 | 67.1% | 10.22 | -0.009 | 0.025 | 44.9% | 0.154 |
| DV top-100 | 1.818 | 2.458 | -25.8% | 26.70 | 18.74 | 67.6 | 89.0% | 10.35 | -0.012 | 0.025 | 20.7% | 0.162 |

## Per-cycle (dollar-volume)

| U | cycle | n | net Sharpe | MaxDD | #long | #short |
|---|-------|---|------------|-------|-------|--------|
| 30 | 2019-20 | 440 | 1.445 | -20.3% | 9.85 | 4.75 |
| 30 | 2021 | 365 | 0.396 | -27.2% | 9.24 | 7.23 |
| 30 | 2022 | 365 | 2.114 | -9.2% | 8.95 | 7.38 |
| 30 | 2023-24 | 731 | 1.170 | -8.6% | 8.51 | 7.76 |
| 30 | 2025-26 | 590 | 1.302 | -13.6% | 9.02 | 7.41 |
| 50 | 2019-20 | 440 | 1.293 | -24.4% | 13.84 | 5.04 |
| 50 | 2021 | 365 | 1.246 | -23.9% | 15.17 | 12.30 |
| 50 | 2022 | 365 | 1.691 | -10.8% | 14.51 | 11.42 |
| 50 | 2023-24 | 731 | 1.421 | -9.9% | 13.25 | 12.82 |
| 50 | 2025-26 | 590 | 1.931 | -11.1% | 13.93 | 12.28 |
| 100 | 2019-20 | 440 | 1.713 | -25.8% | 25.76 | 4.67 |
| 100 | 2021 | 365 | 1.299 | -19.6% | 27.75 | 20.18 |
| 100 | 2022 | 365 | 2.284 | -8.0% | 26.87 | 17.00 |
| 100 | 2023-24 | 731 | 1.402 | -9.6% | 27.41 | 22.81 |
| 100 | 2025-26 | 590 | 2.523 | -21.6% | 25.77 | 24.38 |

Equity overlays: dollar-volume `charts/btcb_universe_sens_equity.png`; market-cap (informational) `charts/btcb_universe_sens_mcap_equity.png`.

## Market-cap universes (informational, same mechanics)

| U | net Sharpe | trail-18m | MaxDD | #long | #short | shortable | % inc. short | ann TO | squeeze mean | β vs BTC | top-5 PnL | RankIC |
|---|------------|-----------|-------|-------|--------|-----------|--------------|--------|--------------|----------|-----------|--------|
| mcap top-30 | 0.952 | 1.740 | -24.7% | 9.05 | 6.97 | 21.7 | 51.6% | 9.55 | -0.001 | 0.031 | 54.3% | 0.111 |
| mcap top-50 | 1.020 | 1.238 | -30.9% | 14.05 | 10.62 | 32.6 | 67.8% | 10.43 | -0.003 | 0.019 | 40.8% | 0.119 |
| mcap top-100 | 0.983 | 2.024 | -42.0% | 25.07 | 19.34 | 50.4 | 88.8% | 9.12 | -0.013 | 0.017 | 12.5% | 0.127 |

## Within-universe RankIC (spread vs excess h=14, last-fold-wins)

| ranking | U=30 | U=50 | U=100 |
|---------|------|------|-------|
| dollar-volume | 0.1502 | 0.1539 | 0.1622 |
| market-cap | 0.1110 | 0.1190 | 0.1265 |

## Cache / reuse

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` n_files=112
- BTC in book hits (all runs) = 0
- GPU=False. Elapsed s=390.3.

COMBO untouched (v2.0-combo-final).

