# Phase E.1b — empirical-null GRU label-shuffle gate

**Verification only.** No GRU retuning. Backtest/analysis only.

- Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
- Addendum (written before the null was observed): `reports/phaseE1_addendum.md`
- Original E.1 `|IC|<0.005` FAIL is preserved; this gate replaces clause (i) only if GREEN.

## Pre-registered gate (verbatim, before results)

> (a) BIAS TEST: for each fold×h cell, the null mean across replicates must satisfy |mean| ≤ 2·(null SD / √R) on the primary universe. If violated on ≥2 of the 8 cells, verdict = CONTAMINATED: the GRU line is closed pending a dedicated bug hunt; no further GRU work in this task. (b) SKILL TEST: the real 3-seed GRU ensemble's outer-fold IC (taken from existing Phase E artifacts — no retraining) must exceed the 95th percentile of the corresponding null on ≥3 of 4 folds at h=7 or at h=10 on the primary universe. DECISION: if (a) passes and (b) passes → gates GREEN: immediately resume Phase E.1 §2–§4 exactly as originally written (9 seeds, three disjoint 3-seed ensembles, NW-t table, A0↔S correlations, per-year tables, score autocorrelation, portfolio translation with both τ conventions) and apply the original confirmation criterion clauses (ii)–(iv) with clause (i) replaced by this gate. If (a) passes but (b) fails → verdict = PARKED-NO-SKILL: stop, no adoption, no retuning. No other outcomes exist.

- Budget: `{'sec_per_epoch': 5.378723859786987, 'n_folds': 4, 'n_seeds': 10, 'n_horizons': 2, 'max_epochs': 30, 'gpu_hours': 3.585815906524658, 'gpu_seconds': 12908.93726348877}`
- Folds used: `[2, 9, 15, 17]`
- Horizons: `[7, 10]`
- Shuffle seeds: `[101, 102, 103, 104, 105, 106, 107, 108, 109, 110]`
- Dropped to 2 folds: `False`

**Mechanical E.1b verdict: PARKED-NO-SKILL**

Details: `{'bias_pass': True, 'skill_pass': False, 'n_violate': 1, 'n_cells': 8, 'skill_by_h': {7: {'n_exceed': 2, 'n_folds': 4, 'need': 3, 'pass': False}, 10: {'n_exceed': 2, 'n_folds': 4, 'need': 3, 'pass': False}}, 'verdict': 'PARKED-NO-SKILL', 'need_folds': 3}`

## Null histograms (pit-120 primary)

| h | fold | n | mean | SD | 95th pct | real 3-seed IC |
|---|------|---|------|----|----------|----------------|
| 7 | 2 | 10 | -0.0214 | 0.0218 | 0.0121 | 0.0966 |
| 7 | 9 | 10 | -0.0158 | 0.0283 | 0.0225 | 0.0071 |
| 7 | 15 | 10 | 0.0052 | 0.0467 | 0.0707 | 0.0941 |
| 7 | 17 | 10 | -0.0070 | 0.0267 | 0.0377 | 0.0344 |
| 10 | 2 | 10 | -0.0113 | 0.0344 | 0.0481 | 0.1110 |
| 10 | 9 | 10 | -0.0100 | 0.0434 | 0.0541 | 0.0012 |
| 10 | 15 | 10 | -0.0054 | 0.0656 | 0.0699 | 0.1102 |
| 10 | 17 | 10 | 0.0120 | 0.0309 | 0.0505 | 0.0220 |

Informational top-20:

| h | fold | n | mean | SD | 95th pct | real 3-seed IC |
|---|------|---|------|----|----------|----------------|
| 7 | 2 | 10 | -0.0537 | 0.0647 | 0.0366 | 0.1744 |
| 7 | 9 | 10 | 0.0086 | 0.0448 | 0.0752 | 0.0536 |
| 7 | 15 | 10 | 0.0326 | 0.1065 | 0.1728 | 0.1422 |
| 7 | 17 | 10 | 0.0030 | 0.0552 | 0.0670 | -0.0130 |
| 10 | 2 | 10 | 0.0067 | 0.1020 | 0.1756 | 0.2297 |
| 10 | 9 | 10 | 0.0153 | 0.0753 | 0.0885 | 0.0470 |
| 10 | 15 | 10 | 0.0106 | 0.0779 | 0.1020 | 0.1304 |
| 10 | 17 | 10 | -0.0112 | 0.0621 | 0.0679 | 0.0300 |

## Bias test (primary pit-120)

| h | fold | mean | SD | SE=SD/√R | 2·SE | \|mean\| | pass |
|---|------|------|----|----------|------|---------|------|
| 7 | 2 | -0.0214 | 0.0218 | 0.0069 | 0.0138 | 0.0214 | FAIL |
| 7 | 9 | -0.0158 | 0.0283 | 0.0090 | 0.0179 | 0.0158 | PASS |
| 7 | 15 | 0.0052 | 0.0467 | 0.0148 | 0.0296 | 0.0052 | PASS |
| 7 | 17 | -0.0070 | 0.0267 | 0.0084 | 0.0169 | 0.0070 | PASS |
| 10 | 2 | -0.0113 | 0.0344 | 0.0109 | 0.0217 | 0.0113 | PASS |
| 10 | 9 | -0.0100 | 0.0434 | 0.0137 | 0.0275 | 0.0100 | PASS |
| 10 | 15 | -0.0054 | 0.0656 | 0.0208 | 0.0415 | 0.0054 | PASS |
| 10 | 17 | 0.0120 | 0.0309 | 0.0098 | 0.0195 | 0.0120 | PASS |

## Skill test (real 3-seed vs null 95th, pit-120)

| h | fold | real IC | null 95th | exceeds |
|---|------|---------|-----------|---------|
| 7 | 2 | 0.0966 | 0.0121 | True |
| 7 | 9 | 0.0071 | 0.0225 | False |
| 7 | 15 | 0.0941 | 0.0707 | True |
| 7 | 17 | 0.0344 | 0.0377 | False |
| 10 | 2 | 0.1110 | 0.0481 | True |
| 10 | 9 | 0.0012 | 0.0541 | False |
| 10 | 15 | 0.1102 | 0.0699 | True |
| 10 | 17 | 0.0220 | 0.0505 | False |

Skill-by-horizon: `{7: {'n_exceed': 2, 'n_folds': 4, 'need': 3, 'pass': False}, 10: {'n_exceed': 2, 'n_folds': 4, 'need': 3, 'pass': False}}`

## A0 LightGBM empirical null (informational, no verdict)

| h | fold | n | mean | SD | 95th pct | real A0 IC | exceeds 95th |
|---|------|---|------|----|----------|------------|--------------|
| 7 | 2 | 10 | -0.0118 | 0.0179 | 0.0138 | 0.0425 | True |
| 7 | 9 | 10 | -0.0233 | 0.0142 | -0.0033 | 0.0362 | True |
| 7 | 15 | 10 | -0.0352 | 0.0219 | -0.0095 | 0.0274 | True |
| 7 | 17 | 10 | -0.0193 | 0.0105 | -0.0063 | 0.0474 | True |
| 10 | 2 | 10 | -0.0185 | 0.0140 | 0.0004 | 0.0614 | True |
| 10 | 9 | 10 | -0.0302 | 0.0119 | -0.0105 | 0.0147 | True |
| 10 | 15 | 10 | -0.0369 | 0.0187 | -0.0051 | 0.0234 | True |
| 10 | 17 | 10 | -0.0189 | 0.0169 | 0.0020 | 0.0372 | True |

§2–§4 not resumed.

