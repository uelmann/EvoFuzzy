# BTC-BEATER — Academic factor (unconstrained D10−D1) + implementation tax

**ANALYSIS ONLY.** Frozen 2.c spread, CMC prices. No shortability / anti-blowoff / hysteresis on the paper factor. CPU only, zero GPU. COMBO untouched. 3.c suspension unchanged. Nothing adopted.

## Pre-registered labels (verbatim, frozen before results)

> PAPER ALPHA EXISTS if FACTOR-JT top-100 gross Sharpe ≥ 1.0 with NW-t ≥ 3.0 on the full OOS window. The IMPLEMENTATION TAX is recorded as (paper gross Sharpe − implementable hybrid Sharpe), decomposed per the waterfall. Labels are diagnostic; nothing is adopted or changed. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical verdicts

- **PAPER ALPHA EXISTS**
- FACTOR-JT top-100 GROSS Sharpe = `1.522` (need ≥ `1.000`; pass=True)
- NW-t (lag 14) = `3.97` (need ≥ `3.0`; pass=True)
- n_days = 2491 (2019-10-19 → 2026-08-13)
- **IMPLEMENTATION TAX** = paper GROSS Sharpe − 3.c hybrid Sharpe = `-0.033`

Mechanical, no post-hoc adjustment. Diagnostic only; nothing is adopted or changed.

## Construction notes

- Universe: floored PIT top-N (dollar-volume primary; mcap informational). BTC excluded.
- Rank: last-fold-wins 2.c spread, top/bottom decile (`k = n_scored // 10`).
- FACTOR-DAILY: refresh every day. FACTOR-JT: 14 overlapping cohorts, each held 14 OOS steps.
- Academic legs are 1.0 long / 1.0 short (D10−D1). Sharpe is scale-invariant vs the 0.5/0.5 book.
- GROSS = no costs. NET-NAIVE = 10 bps/side flat on combined overlay traded notional, both legs.
- Shortability step: bottom-decile ∩ live USDT-M perp; remaining short weights not renormalized; CMC; funding=0.
- Real costs: longs 10 bps, shorts 5+3 bps, same overlay.
- Waterfall terminal = frozen 3.c BOOK-HYBRID (β-matched h=14 book, Binance+funding), not a JT variant.
- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- Position-log sha256 = `f47f7ece40d6cee536b2a07c25961d1d69284f92ddf716447a52b5f57fcc232b`
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- GPU used = `False`

## FACTOR-DAILY (dollar-volume)

| series | Sharpe | trail-18m | ann mean | ann vol | NW-t | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 | MaxDD | avg nL | avg nS | ann TO |
|--------|--------|-----------|----------|---------|------|---------|------|------|---------|---------|-------|--------|--------|--------|
| top-100 GROSS | 1.517 | 2.159 | 1.247 | 0.822 | 4.31 | 1.014 | 0.807 | 2.337 | 1.077 | 2.369 | -60.4% | 9.0 | 9.0 | 189.26 |
| top-100 NET-NAIVE | 1.056 | 1.772 | 0.868 | 0.822 | 3.00 | 0.552 | 0.341 | 1.825 | 0.577 | 1.975 | -71.4% | 9.0 | 9.0 | 189.26 |
| top-50 GROSS | 0.879 | 1.056 | 0.918 | 1.044 | 2.77 | 1.401 | 0.554 | 1.142 | 0.206 | 1.310 | -95.1% | 4.0 | 4.0 | 205.64 |
| top-50 NET-NAIVE | 0.485 | 0.677 | 0.507 | 1.044 | 1.53 | 0.927 | 0.135 | 0.825 | -0.227 | 0.932 | -97.7% | 4.0 | 4.0 | 205.64 |

## FACTOR-JT (dollar-volume)

| series | Sharpe | trail-18m | ann mean | ann vol | NW-t | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 | MaxDD | avg nL | avg nS | ann TO |
|--------|--------|-----------|----------|---------|------|---------|------|------|---------|---------|-------|--------|--------|--------|
| top-100 GROSS | 1.522 | 2.573 | 0.956 | 0.628 | 3.97 | 0.608 | 1.265 | 3.038 | 0.658 | 2.647 | -50.2% | 9.0 | 9.0 | 31.21 |
| top-100 NET-NAIVE | 1.423 | 2.482 | 0.894 | 0.628 | 3.71 | 0.512 | 1.175 | 2.902 | 0.554 | 2.556 | -52.5% | 9.0 | 9.0 | 31.21 |
| top-50 GROSS | 1.225 | 1.966 | 0.853 | 0.697 | 3.61 | 1.107 | 0.792 | 1.795 | 0.584 | 2.051 | -51.9% | 4.0 | 4.0 | 32.93 |
| top-50 NET-NAIVE | 1.130 | 1.865 | 0.787 | 0.697 | 3.33 | 1.013 | 0.705 | 1.692 | 0.493 | 1.952 | -53.2% | 4.0 | 4.0 | 32.93 |

## FACTOR-JT (market-cap, informational)

| series | Sharpe | trail-18m | ann mean | ann vol | NW-t | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 | MaxDD | avg nL | avg nS | ann TO |
|--------|--------|-----------|----------|---------|------|---------|------|------|---------|---------|-------|--------|--------|--------|
| mcap top-100 GROSS | 0.843 | 1.314 | 0.538 | 0.638 | 2.36 | 1.359 | 0.389 | 1.802 | -0.274 | 1.305 | -65.6% | 6.3 | 6.3 | 31.77 |
| mcap top-100 NET-NAIVE | 0.743 | 1.232 | 0.474 | 0.638 | 2.08 | 1.258 | 0.302 | 1.685 | -0.398 | 1.222 | -66.0% | 6.3 | 6.3 | 31.77 |
| mcap top-50 GROSS | 0.913 | 0.849 | 0.660 | 0.723 | 2.61 | 2.301 | 0.191 | 2.554 | -0.288 | 0.840 | -74.6% | 3.8 | 3.8 | 32.73 |
| mcap top-50 NET-NAIVE | 0.822 | 0.779 | 0.595 | 0.723 | 2.35 | 2.196 | 0.115 | 2.444 | -0.411 | 0.769 | -77.9% | 3.8 | 3.8 | 32.73 |

## Leg decomposition (FACTOR-JT top-100 GROSS, unconstrained)

Academic identity: D10−D1 = (long − universe) + (universe − short).

| piece | Sharpe | trail-18m | ann mean | NW-t | share of factor mean |
|-------|--------|-----------|----------|------|----------------------|
| long leg | 0.732 | 0.132 | 0.492 | 1.92 | — |
| short leg | -0.466 | -1.868 | -0.464 | -1.35 | — |
| universe EW | 0.331 | -0.820 | 0.265 | 0.90 | — |
| long − universe | 0.770 | 1.953 | 0.227 | 2.07 | 0.238 |
| universe − short | 1.637 | 2.177 | 0.729 | 4.10 | 0.762 |

## Correlation vs implementable SPREAD-LS

- FACTOR-JT top-100 GROSS vs BOOK-CMC (same prices, frozen positions): `0.6930` (n=2491)
- FACTOR-JT top-100 GROSS vs BOOK-HYBRID (3.c implementable): `0.6765` (n=2491)

## Implementation-tax waterfall (FACTOR-JT top-100 spine)

| step | Sharpe | ΔSharpe | trail-18m | NW-t | ann mean |
|------|--------|---------|-----------|------|----------|
| 1. paper GROSS (FACTOR-JT top-100) | 1.522 | — | 2.573 | 3.97 | 0.956 |
| 2. NET-NAIVE (10 bps/side) | 1.423 | -0.100 | 2.482 | 3.71 | 0.894 |
| 3. + shortability filter (CMC, no funding) | 1.393 | -0.030 | 2.251 | 3.60 | 0.964 |
| 4. + real costs/tiering (10 / 5+3 bps) | 1.399 | 0.006 | 2.259 | 3.61 | 0.968 |
| 5. + Binance prices + funding (3.c hybrid book) | 1.555 | 0.156 | 1.381 | 3.71 | 0.391 |

IMPLEMENTATION TAX (paper GROSS − hybrid) = `-0.033`.

Charts: `charts/btcb_academic_factor_equity.png`, `charts/btcb_academic_factor_waterfall.png`.

## 3.c suspension (unchanged)

Official SPREAD-LS record remains SUSPENDED. This phase does not adopt the paper factor, does not change the production book, and does not lift the pricing-gap freeze.
