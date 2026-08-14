# BTC-BEATER Phase 1 v3 — honest-window naive rotation

**BACKTEST ONLY.** Parameters frozen a priori. No sweeps. Old-archive and 2018-circular benchmarks discarded unread. COMBO untouched.

## Pre-registered label

> NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe (book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its numbers become the floor every ML phase of this project must beat net of costs.

## Death-in-position convention

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical verdict

- **NAIVE-ROTATION is a LIVE BENCHMARK**
- Relative-line Sharpe (book/BTC) = 0.515 (need > 0: True)
- Book total return = 4686813.5% vs BTC B&H 1395.9% (need book ≥ BTC: True)
- Usable window: 2017-10-01 → 2026-08-13 (n=3239)
- Forced exits: n_events=7 n_ids=7 weight_sum=0.5286 cost_drag=0.000634 pnl_impact_vs_ghost=-0.0006

These numbers are the floor every later ML phase must beat net of costs.

## Headline vs BTC B&H

| book | total | CAGR | USD Sharpe | MaxDD | rel CAGR | rel Sharpe | avg %BTC | ann TO |
|------|-------|------|------------|-------|----------|------------|----------|--------|
| naive rotation v3 | 4686813.5% | 236.0% | 0.528 | -96.0% | 147.7% | 0.515 | 5.5% | 13.05 |
| BTC B&H | 1395.9% | 35.6% | 0.795 | -83.4% | 0 | 0 | 100% | 0 |
| 100% BTC control | 1395.9% | 35.6% | 0.795 | -83.4% | 0.0000 | 0.0000 | 100.0% | 0.0000 |

## Per-cycle

| cycle | n | book tot | BTC tot | book Sharpe | rel CAGR | rel Sharpe | MaxDD | avg %BTC |
|-------|---|----------|---------|-------------|----------|------------|-------|----------|
| 2018-19 | 730 | -78.4% | -49.2% | -0.139 | -34.8% | -0.131 | -96.0% | 12.4% |
| 2020-21 | 731 | 149236668.7% | 543.7% | 1.116 | 47644.2% | 1.098 | -63.2% | 4.7% |
| 2022 | 365 | -90.6% | -64.3% | -2.036 | -73.6% | -1.920 | -91.1% | 1.8% |
| 2023-24 | 731 | 126.6% | 464.6% | 0.903 | -36.6% | -0.523 | -53.8% | 3.0% |
| 2025-26 | 590 | -74.1% | -30.5% | -0.949 | -45.7% | -1.093 | -80.2% | 2.6% |

Elapsed s=120.3. GPU=false.

