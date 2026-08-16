# REGIME-TAU — freeze addendum

**Status:** FROZEN before results. Parallel portfolio-layer experiment.
**Reference book:** COMBO v2.0-combo-final (Sleeve A + Sleeve B, causal median-τ). **UNCHANGED.**
**Scope:** BACKTEST AND ANALYSIS ONLY. No retraining, no feature changes, no τ-grid search, no live components. Master only. CPU only. Zero GPU. Frozen COMBO, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are **untouched**.

This task does not edit existing pipelines. New files only (`regimetau/`, `regimetau_pipeline.py`, this addendum).

## Motivation (system-card shelved lead, untested)

The system card lists an untested candidate: regime-conditional τ as a portfolio-layer fix for the IC-vs-Sharpe wedge (threshold-τ books degrade in 2022/2024 even when RankIC is fine). This freeze tests **one** pre-registered rule. It is not a sweep.

## Pre-registered viability (verbatim)

> COMBO-REGIME-TAU is VIABLE if its identical-days net Sharpe ≥ reference COMBO + 0.10 AND trailing-18m net Sharpe ≥ reference COMBO − 0.05. These are viability labels for a parallel product; no outcome changes the reference book. No post-hoc adjustment.

## Frozen rule (one config, no sweeps)

- **Regime variable:** median pairwise Pearson correlation of 60 daily log-returns among PIT top-40 names excluding BTCUSDT, computed at date *t* from returns ending at *t* (lag-0, same close-*t* convention as A0). Require ≥10 names with ≥20 overlapping observations; else the date is missing and treated as warmup/BASE.
- **State:** after 252 valid past CS-corr observations, HIGH if CS-corr(*t*) > expanding median of CS-corr(*s*) for *s* < *t*; otherwise LOW. Warmup dates use BASE τ (house median-τ). Expanding median is causal (past only).
- **τ map (percentage points of |score|, fold-train / training-window, same folds as A0):**
  - Sleeve A (top-20, h=7): BASE 80, HIGH 90, LOW 70
  - Sleeve B (top-40, h=10, tiered costs, ADV cap): BASE 70, HIGH 80, LOW 60
- **Interpretation (frozen, not tuned):** HIGH cross-sectional correlation = crowded / low-dispersion → more selective (higher τ). LOW correlation = dispersion → more names (lower τ). Δ = ±10 points, one step, no other deltas.
- **Engine:** same tranche, lag-0, funding on, BTC beta hedge, hysteresis discarded (`_hard_threshold_state`), seed 42. Frozen A0 scores reused, not recomputed.
- **Death-in-position:** a held coin whose data ends is force-exited at its last available close (no better information assumed). Count and PnL impact reported.

## What is judged

Product verdict is on **COMBO-REGIME-TAU** vs the identical-days **reference COMBO** rebuilt in this run from the same scores/engine (not a pasted ledger figure). Sleeve-level Sharpe and HIGH/LOW day-counts are diagnostic only.

## What this freeze does not do

- Does not retrain A0 or recompute scores
- Does not re-optimize τ, window, or the HIGH/LOW split
- Does not change COMBO, SPREAD-LS, LONG-TIDE, schedules, or live components
- Does not use catalyst/attention data
- Does not touch existing Python modules
