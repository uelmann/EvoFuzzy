# Phase D.2 — top-40 execution, causal τ, micro ablation, hedge decomposition

- Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
- Scope: backtest/analysis only; no schedules or live components. Zero GPU.
- Addendum (written before results): `reports/phaseD2_addendum.md`
- Nominal book for liquidity cap: USD 1,000,000 for a 1.0 gross book.
- Training-window τ on all D.2 portfolio runs (fold_train). Median-τ = house median of {60,70,80,90}.

## 0. Honesty preamble (verbatim, before results)

> The top-40 hypothesis originates from patterns observed in Phase D results (micro features helping on the wide universe while failing on top-20), reinforced by Phases E/E.1b (every surviving signal lives on the wide universe; top-20 IC decays while pit-120 IC holds). This test is therefore not fully independent. Protections: the adoption criterion below is pre-registered before running, and adoption is judged on tradeable net Sharpe with liquidity-tiered costs — a different object from the pit-120 RankIC that surfaced the pattern.

## Pre-registered adoption criterion (verbatim, before results)

> Top-40 execution is ADOPTED if P2 or P4 trailing-18m median-τ net Sharpe ≥ P1 + 0.30 AND its full-period net Sharpe ≥ P1 − 0.20. The micro block is ADOPTED on the chosen universe if the corresponding paired trailing-18m ΔRankIC on that universe ≥ +0.005 AND full-period ΔRankIC on that universe ≥ 0 AND its portfolio (P3 or P4 vs its A counterpart) trailing-18m net Sharpe Δ ≥ 0. Verdicts are mechanical; no post-hoc adjustment.

## Gates

- `label_shuffle`: **PASS** `{'name': 'label_shuffle', 'passed': True, 'mean_ic': 0.0015821390880338343, 'threshold': 0.005, 'n_shuffles': 25, 'null_ic_std': 0.01162563425735333, 'max_abs_shuffle_mean': 0.02612200633170619, 'n_days': 91}`
- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top20`: **PASS** `{'name': 'universe_lookahead_top20', 'passed': True, 'n': 20, 'date': '2023-04-17', 'base_n': 20, 'symmetric_diff': 0}`
- `universe_lookahead_top40`: **PASS** `{'name': 'universe_lookahead_top40', 'passed': True, 'n': 40, 'date': '2023-04-17', 'base_n': 40, 'symmetric_diff': 0}`
- `universe_lookahead_top120`: **PASS** `{'name': 'universe_lookahead_top120', 'passed': True, 'n': 120, 'date': '2023-04-17', 'base_n': 120, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0, 'best_iteration': 270}`

## τ lookahead fix (A0 top-20 tranche h=7, funding on, lag 0)

Same code path except the τ schedule. `pooled` = previous full-OOS |score| percentile (lookahead). `fold_train` = training-window only.

| tau_mode | tau_pct | net Sharpe | gross Sharpe | cost_drag | funding | hedge | avg_npos | %flat |
|----------|---------|------------|--------------|-----------|---------|-------|----------|-------|
| pooled | 60.0 | 1.401 | 1.077 | 0.0755 | -0.1498 | 0.1410 | 9.62 | 0.19 |
| pooled | 70.0 | 1.394 | 1.194 | 0.0791 | -0.1767 | 0.0785 | 8.06 | 0.22 |
| pooled | 80.0 | 1.476 | 1.437 | 0.0769 | -0.3531 | 0.0479 | 5.95 | 0.30 |
| pooled | 90.0 | 0.953 | 0.853 | 0.0606 | -0.4085 | 0.5165 | 3.31 | 0.51 |
| fold_train | 60.0 | 0.757 | 0.389 | 0.0713 | -0.0409 | 0.2884 | 12.20 | 0.09 |
| fold_train | 70.0 | 1.286 | 1.066 | 0.0756 | -0.1396 | -0.0111 | 10.19 | 0.16 |
| fold_train | 80.0 | 1.207 | 1.192 | 0.0754 | -0.2655 | -0.0009 | 6.04 | 0.30 |
| fold_train | 90.0 | 0.763 | 0.589 | 0.0509 | -0.3381 | 0.5439 | 2.63 | 0.55 |

Isolation one-liner: A0 top-20 h=7 τ=60 net Sharpe pooled(full-OOS)=1.401 vs fold_train=0.757 (Δ=-0.644)

## P1–P4 (training-window τ, tranche, funding on, lag 0, median-τ headline, identical days)

| run | h | model | universe | τ | net Sharpe full | trail18m | y2022 | y2023 | y2024 | y2025 | y2026 | gross | cost | funding | hedge | avg_npos | %flat | ann to | avg rank |
|-----|---|-------|----------|---|-----------------|----------|-------|-------|-------|-------|-------|-------|------|---------|-------|----------|-------|--------|----------|
| P1 | 7 | A | top20 | 80.0 | 1.207 | 1.009 | -0.370 | 2.755 | 1.391 | 1.148 | 0.721 | 1.7760 | 0.0754 | -0.2655 | -0.0009 | 6.04 | 0.30 | 20.73 | 10.95 |
| P1 | 10 | A | top20 | 60.0 | 1.131 | 0.145 | 1.707 | 1.986 | 3.178 | 0.318 | 0.007 | 1.9742 | 0.1098 | 0.2637 | 0.0569 | 10.66 | 0.00 | 29.51 | 11.12 |
| P2 | 7 | A | top40 | 70.0 | 1.154 | 0.433 | -0.164 | 1.566 | 2.489 | 1.214 | -1.203 | 1.5558 | 0.1106 | -0.1711 | -0.0367 | 20.28 | 0.15 | 20.81 | 20.39 |
| P2 | 10 | A | top40 | 70.0 | 1.470 | 0.723 | 2.317 | 1.547 | 3.445 | 1.216 | 0.241 | 2.4816 | 0.1677 | 0.2685 | 0.2344 | 16.41 | 0.00 | 32.23 | 21.00 |
| P3 | 7 | A+micro | top20 | 60.0 | 0.912 | 1.295 | 1.516 | -0.700 | 1.510 | 1.593 | 0.248 | 1.1863 | 0.0844 | -0.2041 | 0.0735 | 9.74 | 0.16 | 22.90 | 10.97 |
| P3 | 10 | A+micro | top20 | 60.0 | 0.819 | 0.100 | 2.117 | 0.015 | 2.452 | 0.998 | -1.695 | 1.6309 | 0.0983 | -0.1889 | -0.0737 | 10.41 | 0.00 | 26.42 | 10.94 |
| P4 | 7 | A+micro | top40 | 60.0 | 0.977 | 1.760 | 1.180 | -0.419 | 1.419 | 2.131 | 0.470 | 1.6268 | 0.1249 | -0.2337 | -0.2845 | 19.09 | 0.18 | 23.59 | 20.81 |
| P4 | 10 | A+micro | top40 | 60.0 | 1.178 | 1.239 | 2.241 | -0.384 | 2.500 | 1.864 | -0.110 | 2.6436 | 0.1548 | -0.1223 | -0.5228 | 21.00 | 0.00 | 29.94 | 20.75 |

## Paired ΔRankIC (A+micro vs A)

| h | universe | window | A IC | A+micro IC | ΔIC | n_days |
|---|----------|--------|------|------------|-----|--------|
| 7 | top20 | full | 0.0923 | 0.0441 | -0.0482 | 1620 |
| 7 | top20 | trail18m | 0.0814 | 0.0762 | -0.0052 | 548 |
| 7 | top20 | y2022 | 0.0696 | 0.0377 | -0.0319 | 347 |
| 7 | top20 | y2023 | 0.0946 | -0.0206 | -0.1153 | 365 |
| 7 | top20 | y2024 | 0.1236 | 0.0608 | -0.0628 | 366 |
| 7 | top20 | y2025 | 0.1028 | 0.1023 | -0.0004 | 365 |
| 7 | top20 | y2026 | 0.0454 | 0.0351 | -0.0103 | 177 |
| 10 | top20 | full | 0.1010 | 0.0643 | -0.0366 | 1620 |
| 10 | top20 | trail18m | 0.0656 | 0.0736 | 0.0081 | 548 |
| 10 | top20 | y2022 | 0.1343 | 0.1025 | -0.0318 | 344 |
| 10 | top20 | y2023 | 0.0949 | 0.0164 | -0.0785 | 365 |
| 10 | top20 | y2024 | 0.1304 | 0.0638 | -0.0666 | 366 |
| 10 | top20 | y2025 | 0.0934 | 0.1062 | 0.0127 | 365 |
| 10 | top20 | y2026 | 0.0050 | 0.0047 | -0.0004 | 180 |
| 7 | top40 | full | 0.0792 | 0.0594 | -0.0198 | 1620 |
| 7 | top40 | trail18m | 0.0921 | 0.1186 | 0.0265 | 548 |
| 7 | top40 | y2022 | 0.0590 | 0.0335 | -0.0255 | 347 |
| 7 | top40 | y2023 | 0.0739 | -0.0064 | -0.0803 | 365 |
| 7 | top40 | y2024 | 0.0798 | 0.0601 | -0.0196 | 366 |
| 7 | top40 | y2025 | 0.0964 | 0.1327 | 0.0363 | 365 |
| 7 | top40 | y2026 | 0.0932 | 0.0935 | 0.0004 | 177 |
| 10 | top40 | full | 0.0811 | 0.0663 | -0.0148 | 1620 |
| 10 | top40 | trail18m | 0.0943 | 0.1111 | 0.0168 | 548 |
| 10 | top40 | y2022 | 0.0956 | 0.0705 | -0.0251 | 344 |
| 10 | top40 | y2023 | 0.0598 | 0.0253 | -0.0345 | 365 |
| 10 | top40 | y2024 | 0.0686 | 0.0363 | -0.0323 | 366 |
| 10 | top40 | y2025 | 0.1064 | 0.1224 | 0.0160 | 365 |
| 10 | top40 | y2026 | 0.0707 | 0.0881 | 0.0174 | 180 |

### Paired NW t on daily ΔIC

| h | universe | window | mean ΔIC | NW-t | n_days |
|---|----------|--------|----------|------|--------|
| 7 | top20 | full | -0.0482 | -3.26 | 1620 |
| 7 | top20 | trail18m | -0.0052 | -0.22 | 548 |
| 10 | top20 | full | -0.0366 | -3.65 | 1620 |
| 10 | top20 | trail18m | 0.0081 | 0.54 | 548 |
| 7 | top40 | full | -0.0198 | -1.85 | 1620 |
| 7 | top40 | trail18m | 0.0265 | 1.44 | 548 |
| 10 | top40 | full | -0.0148 | -1.86 | 1620 |
| 10 | top40 | trail18m | 0.0168 | 1.43 | 548 |

## Mechanical verdicts

> Top-40 execution is ADOPTED if P2 or P4 trailing-18m median-τ net Sharpe ≥ P1 + 0.30 AND its full-period net Sharpe ≥ P1 − 0.20. The micro block is ADOPTED on the chosen universe if the corresponding paired trailing-18m ΔRankIC on that universe ≥ +0.005 AND full-period ΔRankIC on that universe ≥ 0 AND its portfolio (P3 or P4 vs its A counterpart) trailing-18m net Sharpe Δ ≥ 0. Verdicts are mechanical; no post-hoc adjustment.

**Universe: ADOPTED**
**Micro on top40: REJECTED**

Operational reading (pre-registered, not a new criterion): top-40 execution is adopted; the micro block is not. The tradeable adopted book is **P2 (model A on top-40)**. P4 may pass the universe Sharpe inequalities without adopting micro, because micro requires ΔIC full ≥ 0 on the chosen universe in addition to the Sharpe delta.

Universe comparisons (identical days):

| candidate | h | trail18m | need (≥ P1+0.30) | full | need (≥ P1−0.20) | pass |
|-----------|---|----------|------------------|------|------------------|------|
| P2 | 7 | 0.433 | 1.309 | 1.154 | 1.007 | False |
| P4 | 7 | 1.760 | 1.309 | 0.977 | 1.007 | False |
| P2 | 10 | 0.723 | 0.445 | 1.470 | 0.931 | True |
| P4 | 10 | 1.239 | 0.445 | 1.178 | 0.931 | True |

Micro comparisons:

| h | universe | ΔIC trail18m | ΔIC full | ΔSharpe trail18m | pass |
|---|----------|--------------|----------|------------------|------|
| 7 | top40 | 0.0265 | -0.0198 | 1.327 | False |
| 10 | top40 | 0.0168 | -0.0148 | 0.515 | False |

## Hedge decomposition (P1, diagnostic only)

Per calendar year, P1 (A, top-20, training-window median-τ, h=7):

| year | gross | hedge | cost | funding | net | net Sharpe |
|------|-------|-------|------|---------|-----|------------|
| 2022 | -0.1366 | 0.0732 | 0.0264 | 0.0098 | -0.0799 | -0.370 |
| 2023 | 0.6539 | 0.1366 | 0.0117 | -0.1232 | 0.6556 | 2.755 |
| 2024 | 0.4700 | 0.0580 | 0.0114 | -0.0171 | 0.4995 | 1.391 |
| 2025 | 0.3619 | -0.0168 | 0.0183 | -0.0447 | 0.2821 | 1.148 |
| 2026 | 0.4267 | -0.2520 | 0.0076 | -0.0903 | 0.0768 | 0.721 |

### Oracle-beta counterfactual (LOOKAHEAD BY DESIGN, diagnostic only)

Hedge ratio replaced with beta estimated on the forward window `[t, t+h]`. Δ(net) = oracle − actual = negative of beta-estimation cost if oracle is better.

| year | actual net | oracle net | Δ(net) oracle−actual |
|------|------------|------------|----------------------|
| 2022 | -0.0799 | 0.0501 | 0.1300 |
| 2023 | 0.6556 | 0.5310 | -0.1246 |
| 2024 | 0.4995 | 0.5008 | 0.0013 |
| 2025 | 0.2821 | 0.2727 | -0.0094 |
| 2026 | 0.0768 | 0.0667 | -0.0102 |

2026 actual net=0.0768 is not a loss; beta-estimation cost (oracle−actual)=-0.0102.

Phase D's printed 2026 net loss (−0.058 at lookahead τ=60) is a different object. Under this task's P1 (training-window median-τ=80, h=7) 2026 net is positive; oracle beta does not improve it.

No production changes from this section.

