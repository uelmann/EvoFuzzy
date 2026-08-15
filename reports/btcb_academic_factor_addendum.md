# BTC-BEATER — Academic factor freeze addendum

**Status:** FROZEN before results. Unconstrained D10−D1 spread factor on CMC (paper alpha) plus an implementation-tax waterfall.
**Scope:** ANALYSIS ONLY. No retraining, no signal or book changes, no schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final and BTC-BEATER v1 are **untouched**. The 2.c spread cache is reused byte-identical (sha256 verified). CMC raw data is read-only.

This phase produces a **reference series**, not a product. No book redesign. The 3.c suspension clause is unaffected: this phase adds measurement, adopts nothing.

Signals are **not** recomputed. Rankings use the frozen 2.c spread (h=14, last-fold-wins).

## Pre-registered labels (verbatim, before results)

> PAPER ALPHA EXISTS if FACTOR-JT top-100 gross Sharpe ≥ 1.0 with NW-t ≥ 3.0 on the full OOS window. The IMPLEMENTATION TAX is recorded as (paper gross Sharpe − implementable hybrid Sharpe), decomposed per the waterfall. Labels are diagnostic; nothing is adopted or changed. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Paper factor construction (fixed a priori)

CMC prices. **NO** shortability filter. **NO** anti-blowoff. **NO** hysteresis.

- Universe: floored PIT top-100 by dollar volume (primary) and top-50 (secondary). Market-cap-ranked variants are informational.
- FACTOR-DAILY: each day, equal-weight LONG the top decile by spread, equal-weight SHORT the bottom decile — every name, listed or not, shortable or not. Daily refresh.
- FACTOR-JT: Jegadeesh-Titman overlapping construction at h=14 (14 cohorts, each held 14 days) — the standard academic factor form matching our horizon.
- Each: GROSS (no costs — the paper ceiling) and NET-NAIVE (10 bps/side flat, both legs) as reference.
- Legs are 1.0 long / 1.0 short (academic D10−D1). Sharpe is scale-invariant versus the 0.5/0.5 implementable book.
- BTC is never in either leg.

## Measurements (fixed)

- Factor Sharpe, annualized mean and vol, NW t-stat (lag 14), full OOS / trailing-18m / per-cycle.
- Leg decomposition academic-style: long-leg-minus-universe and universe-minus-short-leg (where does the paper alpha live, unconstrained?).
- Correlation of FACTOR-JT daily returns with the implementable SPREAD-LS book.

## Implementation-tax waterfall (fixed)

One table, FACTOR-JT top-100 as the spine:

1. paper GROSS Sharpe
2. NET-NAIVE (10 bps/side flat, both legs)
3. + shortability filter only (shorts restricted to perp-listed, still CMC prices, no funding)
4. + real costs/tiering (longs 10 bps; shorts 5 bps + 3 bps slippage)
5. + Binance pricing + funding (= the 3.c hybrid book)

Each step's ΔSharpe is labeled. This is the tax, item by item. Shortability does not fill the short leg with names outside the bottom decile and does not renormalize remaining shorts.

## What this freeze does not do

- Does not recompute signals, retrain heads, or change the 2.c spread.
- Does not change β-matched sizing, h=14, hysteresis, anti-blowoff, death convention, or the production book.
- Does not lift or rewrite the 3.c suspension.
- Does not build a MASTER / combination book.
- Does not introduce schedules or live components.
- Does not touch COMBO, the system card, frozen A0 scores, or BTC-BEATER v1.
- Does not use GPU.
- Does not adopt the paper factor as a product.
