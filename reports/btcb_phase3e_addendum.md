# BTC-BEATER — Phase 3.e freeze addendum

**Status:** FROZEN before results. Pricing-gap forensics mandated by the 3.c suspension.
**Scope:** ANALYSIS ONLY. No retraining, no signal changes, no book redesign, no schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final, SPREAD-LS as a product, and BTC-BEATER v1 are **untouched**. All 3.c artifacts (position log, mappings, prices, funding) are reused. The 2.c spread cache is reused byte-identical (sha256 verified). CMC raw data is read-only.

3.c validation failed: BOOK-BINANCE-ONLY Sharpe 1.296 vs floor 1.409, correlation 0.984. The frozen 3.c clause suspends the official record until the gap is understood. This phase decomposes the gap and rules on whether the SIGNAL itself survives real exchange prices.

## Pre-registered outcomes (verbatim, before results)

> SIGNAL-CONFIRMED if RankIC(spread vs Binance returns) ≥ RankIC(same-names CMC) − 0.02 on both full and trailing windows: the gap is then an execution/pricing-level effect; the official record RESUMES as BOOK-HYBRID funding-on, with the suspension footnote replaced by the forensic decomposition, and Binance-priced numbers become canonical for all future phases. SIGNAL-PARTLY-ARTIFACT if RankIC drops > 0.02 on either window: the record stays suspended, the artifact share is quantified, and the next phase MUST re-derive the book on Binance-only pricing before anything else. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Gap decomposition (fixed)

Replayable subset, same 3.c positions. BOOK-BINANCE-ONLY with funding ON is the 3.c validation book.

- (a) FUNDING vs REPRICING: rerun BOOK-BINANCE-ONLY with funding OFF. ΔSharpe(funding) = fundingON − fundingOFF. ΔSharpe(repricing) = fundingOFF_binance − same-days CMC. Both, per year.
- (b) BY SIDE: long-leg vs short-leg contribution to the repricing gap (same-days PnL diff per leg).
- (c) BY TIER: gap contribution by PIT liquidity rank 1–30 / 31–60 / 61–100.
- (d) CONCENTRATION: top-30 name-days by |w·Δr|, with their share of the total repricing gap.

## Stale-price test (fixed)

For the top-30 disagreement name-days:

- **STALE** if the CMC close is unchanged vs the prior CMC close (or |r_CMC| < 10 bps) while |r_BN| ≥ 2%, **or** CMC's day-t return matches Binance's day-(t−1) return (lag ≥ 1 day) while the contemporaneous prints disagree.
- **LEVEL-DIFF** if both |r_CMC| and |r_BN| ≥ 2%, same sign, and not STALE.
- **OTHER** otherwise.

Quantify the % of the total repricing PnL gap attributable to STALE prints (all replayable name-days, not only the top-30).

## Decisive test (fixed)

Per-date RankIC of the frozen 2.c spread vs Binance-realized h=14 log excess-vs-BTC, replayable names only (spot close if live, else perp). Same-names CMC RankIC on the identical name-dates. Full OOS and trailing-18m. Quintile bucket curve of mean excess on Binance returns (and CMC, same names).

## Structural funding (fixed)

Funding PnL by year. Average funding rate of held shorts vs the shortable-universe average, in bps/day. Never-listed longs (3.c: 95 names): their total PnL contribution in the CMC book.

## What this freeze does not do

- Does not recompute signals, retrain heads, or change the 2.c spread.
- Does not change β-matched sizing, hysteresis, anti-blowoff, or costs.
- Does not build LONG-TIDE or any new book.
- Does not lift the 3.c suspension except via the mechanical SIGNAL-CONFIRMED outcome above.
- Does not touch COMBO, the system card, frozen A0 scores, or BTC-BEATER v1.
- Does not use GPU.
