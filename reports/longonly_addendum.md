# Long-only variants — freeze addendum

**Status:** FROZEN before results. Parallel product/mandate evaluation.
**Reference book:** COMBO v2.0-combo-final (Sleeve A C0 + Sleeve B P2, causal τ). **UNCHANGED** by this task.
**Scope:** backtest and analysis only. Portfolio layer only. Frozen A0 scores and causal median-τ reused as-is. No retraining, no feature changes, no τ re-optimization. Master only. CPU only. Zero GPU. No live components.

## Pre-registered viability statements (verbatim)

> LO-H is VIABLE as a standalone mandate only if its full-period net Sharpe ≥ 0.7 AND trailing-18m net Sharpe ≥ 0.3. LO-U is VIABLE as a standalone mandate only if its full-period regression alpha vs BTC B&H is positive with NW-t ≥ 2.0 AND trailing-18m alpha is positive. These are viability labels for a parallel product; no outcome changes the reference book. No post-hoc adjustment.

## Variant definitions

Both sleeves and both variants share identical calendar days. Base books: Sleeve A (top-20, h=7, causal median-τ=80) and Sleeve B (top-40, h=10, causal median-τ=70, tiered costs, ADV cap), tranche execution, funding accrued, lag 0 — exactly as in the system card.

### LO-H (hedged long-only)

- Entries ONLY on score > τ (no short entries on alts).
- Same exit convention as the frozen tranche engine (`_hard_threshold_state`; `exit_hysteresis` is discarded there, matching the live COMBO books).
- Per-position sizing IDENTICAL to the reference book: the long half of `_size_book` (`0.5 * tg * iv / sum(iv_longs)`). Do **not** dump the short-side 50% budget onto longs. Utilization floats when few names qualify.
- BTC-perp hedge = −Σ(w_i · beta_btc_60_i), rebalanced daily, funding accrued on the hedge leg too.

### LO-U (pure long-only)

- Same entries and sizing as LO-H.
- NO hedge of any kind.
- Uninvested capital is flat cash (0 return).

### COMBO-LO-H / COMBO-LO-U

- 50/50 of the two sleeves per variant.
- Gross convention identical to the reference COMBO (`0.5 A + 0.5 B` daily returns).

## Benchmarks and alpha

- Benchmarks (costless): BTC buy&hold; equal-weight PIT top-20 basket, daily rebalanced.
- For every LO book: daily-return OLS vs BTC B&H → annualized alpha, beta, Newey–West t-stat on alpha (HAC lag = h), full-period and trailing-18m.
- Correlation of each LO book's daily PnL with the frozen reference COMBO.

## Long/short attribution (bonus diagnostic)

For the frozen reference COMBO and each sleeve: decompose historical net PnL into long-leg, short-leg, hedge-leg, funding, and costs — full period and per calendar year. State the fraction of total net PnL contributed by the long legs.

## What this freeze does not do

- Does not retrain A0 or recompute scores.
- Does not re-optimize τ.
- Does not change the reference COMBO, sleeves, schedules, or live components.
- Does not renormalize long-only gross upward when the short side is empty.
