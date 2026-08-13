# Phase E.1 — GRU/BLEND verification

**Verification only.** No new ideas, no tuning. Extraordinary first-pass KEEP is guilty until confirmed.

- Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
- Scope: backtest/analysis only; no schedules or live components.

## Pre-registered confirmation criterion (verbatim)

> The GRU BLEND is CONFIRMED only if: (i) all §1 gates pass; (ii) EACH of the three disjoint 3-seed ensembles satisfies the original BLEND KEEP criterion (trailing-18m ΔRankIC ≥ +0.005, full-OOS Δ ≥ 0, ≥60% positive trailing folds) on at least one universe at h=7 or h=10, with the SAME universe/horizon passing for all three ensembles; (iii) the grand-ensemble paired NW-t of trailing-18m ΔIC on that universe/horizon is ≥ 2.0; and (iv) the BLEND portfolio (median-τ, either τ convention) does not lose more than 0.10 net Sharpe vs A0 on the full period while improving or matching on trailing-18m. If CONFIRMED, BLEND is designated candidate baseline A1, pending the D.2 universe decision. Otherwise: NOT CONFIRMED — park, no adoption, no retuning.

**Mechanical verdict: NOT CONFIRMED**

**NOT CONFIRMED — park, no adoption, no retuning.**

Details: `gates_ok=false; gru_label_shuffle FAIL; future_perturbation PASS; fold_isolation PASS (no warm-start); prediction_alignment PASS; §2–§4 not run`

## 1. Leakage gates

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

### future_perturbation: **PASS**

- synthetic_ok=True synthetic_score_ok=True real_ok=True score_ok=True t=2023-04-21
- Sequence windows use only rows with date ≤ t (last 60 bars). Score at t is a fixed-weight GRU of that window. Power check: windows ending after t must change when future rows are perturbed.
- real: `[{'symbol': '1000LUNCUSDT', 't': '2023-04-21', 'window_equal': True, 'score_equal': True, 'n_future_rows': 650}, {'symbol': '1000SHIBUSDT', 't': '2023-04-21', 'window_equal': True, 'score_equal': True, 'n_future_rows': 1197}, {'symbol': '1000XECUSDT', 't': '2023-04-21', 'window_equal': True, 'score_equal': True, 'n_future_rows': 78}, {'symbol': '1INCHUSDT', 't': '2023-04-21', 'window_equal': True, 'score_equal': True, 'n_future_rows': 288}]`

### fold_isolation: **PASS**

- warm_start: no warm-start; metas inspected=108; warm-start files=0
| fold | train_end | max dataloader date | max train slice | n_rows | pass |
|------|-----------|---------------------|-----------------|--------|------|
| 0 | 2022-01-01 | 2021-10-03 | 2022-01-01 | 48391 | True |
| 1 | 2022-04-01 | 2022-01-01 | 2022-04-01 | 58658 | True |
| 2 | 2022-06-30 | 2022-04-01 | 2022-06-30 | 69018 | True |
| 3 | 2022-09-28 | 2022-06-30 | 2022-09-28 | 79684 | True |
| 4 | 2022-12-27 | 2022-09-28 | 2022-12-27 | 90253 | True |
| 5 | 2023-03-27 | 2022-12-27 | 2023-03-27 | 100510 | True |
| 6 | 2023-06-25 | 2023-03-27 | 2023-06-25 | 110673 | True |
| 7 | 2023-09-23 | 2023-06-25 | 2023-09-23 | 120979 | True |
| 8 | 2023-12-22 | 2023-09-23 | 2023-12-22 | 130704 | True |
| 9 | 2024-03-21 | 2023-12-22 | 2024-03-21 | 140456 | True |
| 10 | 2024-06-19 | 2024-03-21 | 2024-06-19 | 150514 | True |
| 11 | 2024-09-17 | 2024-06-19 | 2024-09-17 | 160699 | True |
| 12 | 2024-12-16 | 2024-09-17 | 2024-12-16 | 170328 | True |
| 13 | 2025-03-16 | 2024-12-16 | 2025-03-16 | 179404 | True |
| 14 | 2025-06-14 | 2025-03-16 | 2025-06-14 | 188706 | True |
| 15 | 2025-09-12 | 2025-06-14 | 2025-09-12 | 198000 | True |
| 16 | 2025-12-11 | 2025-09-12 | 2025-12-11 | 206457 | True |
| 17 | 2026-03-11 | 2025-12-11 | 2026-03-11 | 215729 | True |
| 0 | 2021-12-29 | 2021-09-30 | 2021-12-29 | 48038 | True |
| 1 | 2022-03-29 | 2021-12-29 | 2022-03-29 | 58353 | True |
| 2 | 2022-06-27 | 2022-03-29 | 2022-06-27 | 68664 | True |
| 3 | 2022-09-25 | 2022-06-27 | 2022-09-25 | 79339 | True |
| 4 | 2022-12-24 | 2022-09-25 | 2022-12-24 | 89893 | True |
| 5 | 2023-03-24 | 2022-12-24 | 2023-03-24 | 100185 | True |
| 6 | 2023-06-22 | 2023-03-24 | 2023-06-22 | 110325 | True |
| 7 | 2023-09-20 | 2023-06-22 | 2023-09-20 | 120647 | True |
| 8 | 2023-12-19 | 2023-09-20 | 2023-12-19 | 130380 | True |
| 9 | 2024-03-18 | 2023-12-19 | 2024-03-18 | 140116 | True |
| 10 | 2024-06-16 | 2024-03-18 | 2024-06-16 | 150163 | True |
| 11 | 2024-09-14 | 2024-06-16 | 2024-09-14 | 160385 | True |
| 12 | 2024-12-13 | 2024-09-14 | 2024-12-13 | 170002 | True |
| 13 | 2025-03-13 | 2024-12-13 | 2025-03-13 | 179070 | True |
| 14 | 2025-06-11 | 2025-03-13 | 2025-06-11 | 188383 | True |
| 15 | 2025-09-09 | 2025-06-11 | 2025-09-09 | 197703 | True |
| 16 | 2025-12-08 | 2025-09-09 | 2025-12-08 | 206147 | True |
| 17 | 2026-03-08 | 2025-12-08 | 2026-03-08 | 215408 | True |

### prediction_alignment: **PASS**

- 
- h=7 passed=True n_rows=176317 n_bad=0 n_unassigned=0
- h=10 passed=True n_rows=176281 n_bad=0 n_unassigned=0

Gates failed — §2–§4 not run.

