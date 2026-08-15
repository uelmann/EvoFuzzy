# BTC-BEATER — SPREAD-LS horizon sweep freeze addendum

**Status:** FROZEN before results. Twin-head retraining at h=3 and h=7; h=14/h=30 caches reused.
**Scope:** backtest + analysis only. No schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final and BTC-BEATER v1 are **untouched**.

Production universe is the floored PIT **top-100 by dollar volume** (universe-sensitivity verdict). Same 33 context-free Stage-S features, same expanding-fold recipe, same LightGBM params, isotonic calibration per fold per head. Phase 3.b (funding-on) has **not** run; every book in this freeze is **funding-off**, flagged consistently.

## Pre-registered reading (verbatim)

> h=14 is the incumbent. A different horizon becomes production only if its null gate passes AND its trailing-18m net Sharpe ≥ incumbent + 0.15 AND its full-OOS net Sharpe ≥ incumbent − 0.10. Among multiple qualifiers, highest trailing wins. If none qualify, h=14 stays. Mechanical, no post-hoc adjustment.

## Repowered skill null (verbatim, same as 2.c)

> Bias: every fold's null mean must satisfy the E.1b centering bound (AUC around 0.5, RankIC around 0). Skill passes if, for the judged signal, ≥5 of 6 folds exceed their null 95th percentile OR the Stouffer-combined z across the 6 folds is ≥ 3.0. Symmetric: failure = PARKED, no override, no retest with different folds.

Folds frozen: {0, 5, 9, 15, 21, 24}. 25 within-date shuffle replicates each. Judged signal = per-date RankIC of the spread vs realized excess. A new horizon that fails its null is **reported but not judged** for production. h=14 null is reused from 2.c (not re-run). h=30 has no horizon-specific null in this freeze and is reported, not judged.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Twin heads (new)

- Train TOP and BOTTOM quintile heads at **h=3 and h=7 only**. h=14 and h=30 caches reused byte-identical (sha256 verified).
- Spread = p_top_cal − p_bottom_cal per horizon.
- Same 33 per-coin features, same LightGBM params, same expanding folds (native-horizon purge/embargo), same inner-holdout early stopping on mean per-date AUC. Context excluded.
- Isotonic calibration per fold, train-only, independent per head.

## Books (fixed a priori, no sweeps)

For h ∈ {3, 7, 14, 30}, β-matched SPREAD-LS on floored PIT top-100 DV:

- Long = top decile of U by that horizon's spread; short = bottom decile ∩ perp-shortable. Native **h tranches**.
- EW within each leg; 10% per-name cap; unfilled budget stays cash — never BTC.
- Quintile-exit hysteresis; anti-blowoff on new longs; death-in-position.
- Costs: longs spot 10 bps/side; shorts perp 5 bps + 3 bps slippage/side. **FUNDING = 0** (3.b has not run).

## Per-trade economics (fixed)

Per horizon: average holding period (days), round-trips per year, average GROSS edge per round-trip (bps), average cost per round-trip (bps), net edge per trade (bps), annualized cost drag. Plus: per-date RankIC of the spread; net Sharpe full / trailing-18m / per-cycle; MaxDD; turnover; squeeze-day mean.

## What this freeze does not do

- Does not retrain h=14 or h=30.
- Does not apply funding (not available; 3.b not run).
- Does not change the production universe, features, LightGBM params, or 2.b hygiene.
- Does not sweep costs, hysteresis, or gross.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
