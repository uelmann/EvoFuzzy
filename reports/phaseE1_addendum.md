# Phase E.1 addendum — label-shuffle gate recalibration

**Written and frozen before any empirical null is observed.** Backtest/analysis only. No schedules or live components. Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`.

This addendum does not alter the Phase E.1 record. The original gate FAIL stands as recorded. Phase E.1b replaces the miscalibrated threshold with a pre-registered empirical-null gate.

## Original E.1 gate FAIL (verbatim)

From `reports/phaseE1_report.md`:

```
### gru_label_shuffle: **FAIL**

- Threshold: `|IC| < 0.005` on outer-fold RankIC (mean of 3 seeds, labels shuffled within date).
- Folds: `[9, 17]` (mid-sample and most recent)
- Fold-mean RankIC (3 seeds), gate is `|mean IC| < 0.005`: h=7 fold9 **−0.04214 FAIL**; h=7 fold17 **+0.00511 FAIL**; h=10 fold9 **−0.04877 FAIL**; h=10 fold17 **+0.01906 FAIL**.
- 11/12 seed×fold cells exceed the threshold. §2–§4 not run.
| h | fold | seed | mean IC | |IC| | pass |
|---|------|------|---------|-----|------|
| 7 | 9 | 42 | -0.0336 | 0.0336 | False |
| 7 | 9 | 43 | -0.0563 | 0.0563 | False |
| 7 | 9 | 44 | -0.0365 | 0.0365 | False |
| 7 | 17 | 42 | 0.0213 | 0.0213 | False |
| 7 | 17 | 43 | 0.0397 | 0.0397 | False |
| 7 | 17 | 44 | -0.0457 | 0.0457 | False |
| 10 | 9 | 42 | -0.0146 | 0.0146 | False |
| 10 | 9 | 43 | -0.0517 | 0.0517 | False |
| 10 | 9 | 44 | -0.0799 | 0.0799 | False |
| 10 | 17 | 42 | 0.0007 | 0.0007 | True |
| 10 | 17 | 43 | 0.0272 | 0.0272 | False |
| 10 | 17 | 44 | 0.0293 | 0.0293 | False |
```

Phase E.1 mechanical verdict remains **NOT CONFIRMED** on that gate. Other E.1 gates (future-perturbation, fold isolation / no warm-start, prediction alignment) passed and are not reopened.

## Miscalibration rationale

The fixed `|IC| < 0.005` threshold is miscalibrated for non-degenerate models. Under within-date shuffling, LightGBM collapses to a near-constant model (IC exactly ~0), while a GRU always emits varied rankings, so its fold-mean IC under the null has SE ≈ ±0.03–0.06 with h-day overlapping labels — both signs, exactly the observed pattern (−0.042, +0.005, −0.049, +0.019).

That FAIL is left on the books. The replacement gate below is direction-symmetric (bias test) and can also kill the GRU harder than `|IC| < 0.005` (if the null is shifted, or if real IC does not clear the 95th percentile of the empirical null).

## Recalibration designed before observing the empirical null

Fold IDs, shuffle-replicate seeds, bias threshold, skill percentile, primary universe, and all decision rules below are pre-registered in this addendum. They were not chosen by looking at a 10-replicate null. The four E.1 point estimates above are the original FAIL record; they are not used to set the new threshold.

## Pre-registered design (Phase E.1b)

- **Folds:** 2 (early-sample), 9 (mid; used in E.1), 15 (recent), 17 (most recent; used in E.1). Total 4.
- **Horizons:** h=7 and h=10.
- **Replicates:** 10 per fold×h. Shuffle seeds `{101,…,110}` (within-date label permutation). GRU training seed **42** on every replicate. Same architecture and early-stopping protocol as Phase E.
- **Primary universe:** pit-120. Top-20 is informational only.
- **GPU budget:** project first; hard cap 6 GPU-h. If the projection exceeds the cap, drop to folds `{9, 17}` × 10 replicates (state it). Bias then uses the available cells (CONTAMINATED if ≥2 cells violate). Skill then requires the real ensemble to exceed the 95th percentile on both remaining folds at h=7 or at h=10.
- **Real GRU IC (skill test):** 3-seed ensemble `{42,43,44}` outer-fold RankIC from existing Phase E artifacts. No retraining of the real GRU.

## Pre-registered gate (verbatim)

> (a) BIAS TEST: for each fold×h cell, the null mean across replicates must satisfy |mean| ≤ 2·(null SD / √R) on the primary universe. If violated on ≥2 of the 8 cells, verdict = CONTAMINATED: the GRU line is closed pending a dedicated bug hunt; no further GRU work in this task. (b) SKILL TEST: the real 3-seed GRU ensemble's outer-fold IC (taken from existing Phase E artifacts — no retraining) must exceed the 95th percentile of the corresponding null on ≥3 of 4 folds at h=7 or at h=10 on the primary universe. DECISION: if (a) passes and (b) passes → gates GREEN: immediately resume Phase E.1 §2–§4 exactly as originally written (9 seeds, three disjoint 3-seed ensembles, NW-t table, A0↔S correlations, per-year tables, score autocorrelation, portfolio translation with both τ conventions) and apply the original confirmation criterion clauses (ii)–(iv) with clause (i) replaced by this gate. If (a) passes but (b) fails → verdict = PARKED-NO-SKILL: stop, no adoption, no retuning. No other outcomes exist.

A0 LightGBM is run through the same empirical-null construction on the same 4 folds (10 shuffle replicates, CPU) for house-gate calibration only. It carries no verdict for A0.
