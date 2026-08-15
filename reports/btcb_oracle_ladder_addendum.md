# BTC-BEATER — ORACLE LADDER freeze addendum

**Status:** FROZEN before results. Perfect-foresight ceiling, IC-degraded oracles, model location on the curve.
**Scope:** ANALYSIS ONLY. No retraining, no product changes, no schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final, SPREAD-LS, LONG-TIDE, and BTC-BEATER v1 are **untouched**. The 2.c spread cache is reused byte-identical (sha256 verified). CMC raw data is read-only. Pricing follows the 3.e canonical convention (Binance).

This phase produces a **map**, not a product. Nothing is adopted.

## Pre-registered verdict (verbatim, before results)

> MODEL EFFICIENCY verdict: our model sits ON-CURVE if its CAGR is within ±20% (in log terms) of the ladder interpolation at its own realized RankIC — conclusion: the binding constraint is INFORMATION, and improvement means new data, not new architecture. It sits BELOW-CURVE if lower than that band — conclusion: TRANSLATION slack exists in the signal→book layer, quantified as the CAGR gap. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Construction (fixed, identical across every point)

Floored PIT top-100 dollar-volume. Binance-spot-listed names at date t (canonical prices). BTC excluded.

- Rebalance every 14 days (7d oracle secondary). Full rebalance, no overlapping tranches, no hysteresis, no anti-blowoff, no gate, no shorts, no funding.
- Equal-weight the top decile by the point's score. 10% name cap. Residual cash (idle).
- Costs on = 10 bps/side. Oracle also reported GROSS.
- This construction **differs** from the production h=14 tranche books (LONG-TIDE, SPREAD-LS). The comparison is apples-to-apples on the ladder, not versus production.

## Ladder (fixed)

Score_i = z(future excess-vs-BTC_i) + ε_i, ε calibrated **per date** so realized per-date RankIC ≈ target. Targets {0.50, 0.30, 0.20, 0.16, 0.10, 0.05}. 5 noise seeds per target; report mean ± range of realized RankIC, total, CAGR, MaxDD. ε is rank-orthogonalized to the date's z(excess) before the σ search so low-IC targets are attainable.

## Reference points (fixed)

- ORACLE: score = realized future excess (lookahead). Ceiling.
- OUR MODEL: frozen 2.c spread as score. Report its realized per-date RankIC on this window.
- NAIVE: 90d excess-vs-BTC top decile.
- RANDOM: IC≈0, 5 seeds.

## What this freeze does not do

- Does not recompute signals, retrain, or change any frozen product.
- Does not adopt an oracle book or a new architecture.
- Does not use GPU.
