# BTC-BEATER Phase 1 — dumb benchmark the ML must beat

**BACKTEST ONLY.** Parameters frozen a priori. No sweeps. COMBO untouched.

## Pre-registered label

> NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe (book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its numbers become the floor every ML phase of this project must beat net of costs.

## Mechanical verdict

- **NAIVE-ROTATION is NOT A LIVE BENCHMARK**
- Relative-line Sharpe (book/BTC) = -0.456 (need > 0: False)
- Book total return = -91.2% vs BTC B&H 375.2% (need book ≥ BTC: False)
- Usable window: 2018-01-02 → 2026-08-08 (n=3141)

These numbers are the floor every later ML phase must beat net of costs.

## Headline vs BTC B&H

| book | total | CAGR | USD Sharpe | MaxDD | rel CAGR | rel Sharpe | avg %BTC | ann TO |
|------|-------|------|------------|-------|----------|------------|----------|--------|
| naive rotation | -91.2% | -24.6% | 0.175 | -97.8% | -37.1% | -0.456 | 8.8% | 11.86 |
| BTC B&H | 375.2% | 19.9% | 0.607 | -81.5% | 0 | 0 | 100% | 0 |
| 100% BTC control | 375.2% | 19.9% | 0.607 | -81.5% | 0.0000 | 0.0000 | 100.0% | 0.0000 |

The 100% BTC control should reproduce B&H (relative line ≈ 1, rel Sharpe ≈ 0).

## Per-cycle

| cycle | n | book tot | BTC tot | book Sharpe | rel CAGR | rel Sharpe | MaxDD | avg %BTC |
|-------|---|----------|---------|-------------|----------|------------|-------|----------|
| 2018-19 | 729 | -82.4% | -47.3% | -0.449 | -42.2% | -0.801 | -93.7% | 19.8% |
| 2020-21 | 731 | 1453.6% | 543.7% | 1.819 | 55.3% | 0.948 | -68.7% | 4.4% |
| 2022 | 365 | -88.3% | -64.3% | -1.695 | -67.3% | -1.565 | -89.1% | 3.0% |
| 2023-24 | 731 | 108.8% | 464.6% | 0.855 | -39.1% | -0.653 | -57.6% | 5.7% |
| 2025-26 | 585 | -86.8% | -30.5% | -1.211 | -64.5% | -1.646 | -89.8% | 8.0% |

USD Sharpe by calendar year: 2018=-1.406, 2019=0.978, 2020=1.784, 2021=1.855, 2022=-1.695, 2023=1.205, 2024=0.571, 2025=-0.867, 2026=-2.018

Elapsed s=51.5. GPU=false.

