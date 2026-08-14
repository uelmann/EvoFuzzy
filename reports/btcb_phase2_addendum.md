# BTC-BEATER — Phase 2 freeze addendum

**Status:** FROZEN before results. One model. No bake-offs, no sequence models, no extras.
**Scope:** backtest + analysis only. No schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final is **untouched**. Naive rotation v3 remains the floor until (and unless) MODEL-V1 replaces it under the criterion below.

Causal thresholds everywhere (training-window only). Calibrated p is used as-is — no percentile re-mapping of OOS scores.

## Pre-registered viability / replace-floor criterion (verbatim)

> MODEL-V1 is VIABLE if, on the full OOS window at the median p_enter: (a) the relative line (book/BTC) has Sharpe > 0; (b) total return ≥ BTC B&H; (c) MaxDD ≤ BTC B&H MaxDD. MODEL-V1 REPLACES the naive rotation as the project floor if additionally its relative-line Sharpe ≥ naive v3 relative-line Sharpe + 0.15 on the same window. Per-cycle honesty: report 2019–20, 2021, 2022, 2023–24, 2025–26 separately; a verdict is not overridden by any single cycle. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim, all project backtests)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Frozen design (a priori)

- Label: y=1 iff h-day forward log-return exceeds BTC’s over the same window (excess > 0). Horizons h ∈ {14, 30} both trained and reported. **Primary headline = h=14** (more OOS days; h=30 is the robustness book and does not override the h=14 verdict).
- Universes from Phase 0.c files: train labels on PIT top-100; trade on PIT top-50. BTC is the parking asset (never a pick). Stables/wrapped already excluded from those PIT files.
- Walk-forward: expanding, initial train ≥ 730 **calendar** days from 2017-09-30 (so OOS ≈ 2019-10); refit every 90 calendar days; purge last h days of train; embargo h+3. Inner holdout = last 90 calendar days of train.
- Features (~40), all data ≤ t, clip ±5:
  - Price block = the 25 Round-F pruned survivors. Momentum / trend / distance on BTC-denominated price (close/BTC). Vol, MAX/MIN, range, skew, beta/idio/corr/amihud on own USD returns as in A0.
  - Context block (7): ctx_disp, ctx_excess_disp (replaces ctx_score_disp), ctx_btc_vol, ctx_btc_trend, ctx_breadth on PIT top-100, ctx_corr on PIT top-50, ctx_alt_btc_trend = EW top-50/BTC vs its 90d SMA. Context series are **own-z over 250d** then broadcast (cross-sectional z is degenerate for date-level features). Coin-level features are CS z-scored per date among that date’s PIT top-100.
  - New-data block: log mcap, mcap rank, Δrank 30d/90d, log coin age, PIT distance from ATH, volume/mcap turnover, turnover z vs own 30d.
- LightGBM binary, pooled, params fixed: num_leaves 31, lr 0.03, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8/1, lambda_l2 1.0, seed 42; early stopping on inner-holdout AUC, patience 100.
- Calibration: isotonic regression fitted per fold on the inner-holdout (training window only). Output = calibrated p(beats BTC).
- Entry/exit: enter if p ≥ p_enter, exit if p ≤ p_enter − 0.05. Grid p_enter ∈ {0.55, 0.60, 0.65}. Headline = house **median convention**: the grid point whose full-OOS relative-line Sharpe is the median of the three (closest to median if a tie). Thresholds applied as-is.
- Anti-blowoff (fixed): no NEW entries on names with trailing 7d raw return > +50%.
- Book: equal-weight qualifiers in PIT top-50, cap 10% per name, max 10 names, remainder in BTC, h-tranche, 10 bps alts / 2 bps BTC. Never cash.
- Same-window naive v3: recomputed from the model OOS start (not the 2017-10 full-window print) for the replace-floor clause.

## Gates (must pass before results are treated as official)

- Feature lookahead unit test (features at t invariant to future OHLC/volume/mcap).
- PIT universe lookahead on top-50 and top-100.
- Seed determinism (fold 0, h=14, two runs).
- Label-shuffle empirical null, E.1b design: within-date shuffle of **train** labels, 25 replicates (seeds 101–125) on 2 folds at h=14 (fold 0 and the fold whose val_start is nearest 2022-01-01). Metric = OOS AUC vs real labels. (a) BIAS: for each of the 2 folds, |null mean| ≤ 2·(null SD / √R). (b) SKILL: real-model OOS AUC must exceed the null 95th percentile on **both** folds. Both (a) and (b) required.

## What this freeze does not do

- Does not train a second architecture or sweep hyperparameters.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
- Does not introduce schedules or live components.
