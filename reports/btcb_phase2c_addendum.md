# BTC-BEATER — Phase 2.c freeze addendum

**Status:** FROZEN before results. Twin-head spread signal + repowered E.1b-style skill null.
**Scope:** backtest + analysis only. No schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final is **untouched**.

Cleaned+floored panel and PIT universes from Phase 2.b are **reused as-is**. No new hygiene. No data changes. Stage-T regime gate is frozen unchanged (no sweeps). Context features remain excluded from Stage S.

## Pre-registered criteria (verbatim)

> SPREAD has SELECTION SKILL if, at h=14 or h=30: mean per-date RankIC(spread) ≥ +0.01 AND mean per-date AUC ≥ 0.52 AND the §2 gate passes for the spread metric. MODEL-V3 is VIABLE if on the full OOS window at median θ: (a) relative-line Sharpe > 0; (b) total ≥ BTC B&H; (c) MaxDD ≤ BTC B&H. MODEL-V3 is PRODUCT-GRADE if additionally relative-line Sharpe ≥ 0.30 AND average alt allocation ≥ 5% (non-degenerate book). Per-cycle honesty table mandatory; no single cycle overrides. Mechanical, no post-hoc adjustment.

## Repowered skill null (verbatim)

> Bias: every fold's null mean must satisfy the E.1b centering bound (AUC around 0.5, RankIC around 0). Skill passes if, for the judged signal, ≥5 of 6 folds exceed their null 95th percentile OR the Stouffer-combined z across the 6 folds is ≥ 3.0. Symmetric: failure = PARKED, no override, no retest with different folds.

Folds frozen: {0, 5, 9, 15, 21, 24}. 25 within-date shuffle replicates each. h=14. Metrics: (i) per-date AUC of p_top vs top-quintile membership; (ii) per-date RankIC of spread vs realized excess. The judged signal for SPREAD skill is (ii).

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Twin head

- Head TOP: y=1 iff h-day excess-vs-BTC is in the top quintile of that date’s PIT top-100. Identical to Phase 2.b Stage S.
- Head BOTTOM: y=1 iff h-day excess-vs-BTC is in the bottom quintile of that date’s PIT top-100.
- Same 33 per-coin features, same LightGBM params, same expanding folds, same inner-holdout early stopping on mean per-date AUC. Context excluded.
- Isotonic calibration per fold, train-only, independent per head.
- Signal: spread = p_top_cal − p_bottom_cal.
- Diagnostic only (no trading use): uncertainty = p_top_cal + p_bottom_cal, and its cross-sectional rank-correlation with yz_vol_30.

## Stage T (frozen)

Alt-exposure budget = 50% when [EW top-50/BTC ratio > its 90d SMA] AND [breadth top-100 > 0.5], with 5-day OFF hysteresis. No changes, no sweeps.

## Book v3

When the gate is ON: names in the floored PIT top-50 with spread ≥ θ, θ ∈ {0.10, 0.15, 0.20}, house median-θ on relative-line Sharpe as headline. Equal-weight, 10% cap, K ≤ 10, anti-blowoff (7d raw > +50%) on new entries, remainder BTC, h-tranche, 10/2 bps, death convention. Naive v4 is shown for record only; the operative floor is BTC.

## What this freeze does not do

- Does not change prices, eligibility, or PIT membership.
- Does not learn a timing model or put context features in Stage S.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
- Does not introduce schedules or live components.
