# BTC-BEATER Phase 2 — MODEL-V1 winner-tail classifier

**BACKTEST ONLY.** One model. Params and guards frozen a priori. No sweeps beyond the declared 3-point p_enter grid with median convention. CPU only, zero GPU. Frozen COMBO v2.0-combo-final is untouched.

## Pre-registered criteria (verbatim, before results)

> MODEL-V1 is VIABLE if, on the full OOS window at the median p_enter: (a) the relative line (book/BTC) has Sharpe > 0; (b) total return ≥ BTC B&H; (c) MaxDD ≤ BTC B&H MaxDD. MODEL-V1 REPLACES the naive rotation as the project floor if additionally its relative-line Sharpe ≥ naive v3 relative-line Sharpe + 0.15 on the same window. Per-cycle honesty: report 2019–20, 2021, 2022, 2023–24, 2025–26 separately; a verdict is not overridden by any single cycle. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Gates

Gates official: **FAIL**. Results below are official only if all gates pass.

| gate | passed | detail |
|------|--------|--------|
| feature_lookahead | True | `{'max_abs_diff': 0.0, 'id': 8, 'date': '2020-06-16'}` |
| universe_lookahead_top50 | True | `{'n': 50, 'date': '2020-06-16', 'base_n': 50, 'symmetric_diff': 0}` |
| universe_lookahead_top100 | True | `{'n': 100, 'date': '2020-06-16', 'base_n': 100, 'symmetric_diff': 0}` |
| seed_determinism | True | `{'max_score_diff': 0.0, 'best_iteration': 12, 'fold_id': 0, 'horizon': 14}` |

Label-shuffle null (E.1b design, h=14, 2 folds, 25 replicates): **CONTAMINATED** bias_pass=False skill_pass=False. Bias uses |null-mean AUC − 0.5| ≤ 2·(SD/√R), the AUC analogue of E.1b’s RankIC centering.

| fold | n | null mean | SD | 95th | real AUC | bias_ok | exceeds_p95 |
|------|---|-----------|----|------|----------|---------|-------------|
| 0 | 25 | 0.4163 | 0.0182 | 0.4415 | 0.4466 | False | True |
| 9 | 25 | 0.5237 | 0.0153 | 0.5432 | 0.5382 | False | False |

## Walk-forward AUC (calibrated p, OOS)

| h | fold | val_start | val_end | n_valid | auc_oos | auc_raw | best_iter |
|---|------|-----------|---------|---------|---------|---------|-----------|
| 14 | 0 | 2019-10-18 | 2020-01-16 | 8328 | 0.4199 | 0.4466 | 12 |
| 14 | 1 | 2020-01-16 | 2020-04-15 | 8573 | 0.5526 | 0.5737 | 1 |
| 14 | 2 | 2020-04-15 | 2020-07-14 | 8663 | 0.4941 | 0.4914 | 92 |
| 14 | 3 | 2020-07-14 | 2020-10-12 | 8028 | 0.4988 | 0.4774 | 1 |
| 14 | 4 | 2020-10-12 | 2021-01-10 | 7899 | 0.6500 | 0.6621 | 4 |
| 14 | 5 | 2021-01-10 | 2021-04-10 | 8269 | 0.5506 | 0.5432 | 30 |
| 14 | 6 | 2021-04-10 | 2021-07-09 | 8590 | 0.5515 | 0.5193 | 15 |
| 14 | 7 | 2021-07-09 | 2021-10-07 | 8429 | 0.6197 | 0.5450 | 1 |
| 14 | 8 | 2021-10-07 | 2022-01-05 | 8535 | 0.4794 | 0.4803 | 231 |
| 14 | 9 | 2022-01-05 | 2022-04-05 | 8537 | 0.5408 | 0.5382 | 6 |
| 14 | 10 | 2022-04-05 | 2022-07-04 | 8320 | 0.5545 | 0.5054 | 3 |
| 14 | 11 | 2022-07-04 | 2022-10-02 | 8224 | 0.4604 | 0.4643 | 297 |
| 14 | 12 | 2022-10-02 | 2022-12-31 | 8600 | 0.4799 | 0.4770 | 29 |
| 14 | 13 | 2022-12-31 | 2023-03-31 | 8585 | 0.4573 | 0.4583 | 8 |
| 14 | 14 | 2023-03-31 | 2023-06-29 | 8213 | 0.4902 | 0.4672 | 3 |
| 14 | 15 | 2023-06-29 | 2023-09-27 | 8461 | 0.6121 | 0.6017 | 4 |
| 14 | 16 | 2023-09-27 | 2023-12-26 | 8210 | 0.4097 | 0.4026 | 268 |
| 14 | 17 | 2023-12-26 | 2024-03-25 | 7722 | 0.5403 | 0.5490 | 2 |
| 14 | 18 | 2024-03-25 | 2024-06-23 | 7428 | 0.4421 | 0.4187 | 4 |
| 14 | 19 | 2024-06-23 | 2024-09-21 | 7790 | 0.4302 | 0.4543 | 1 |
| 14 | 20 | 2024-09-21 | 2024-12-20 | 7667 | 0.4388 | 0.4483 | 1 |
| 14 | 21 | 2024-12-20 | 2025-03-20 | 7392 | 0.5587 | 0.5462 | 1 |
| 14 | 22 | 2025-03-20 | 2025-06-18 | 7227 | 0.6689 | 0.6711 | 123 |
| 14 | 23 | 2025-06-18 | 2025-09-16 | 7410 | 0.4350 | 0.4038 | 1 |
| 14 | 24 | 2025-09-16 | 2025-12-15 | 6687 | 0.4618 | 0.4587 | 2 |
| 14 | 25 | 2025-12-15 | 2026-03-15 | 6816 | 0.5257 | 0.5224 | 423 |
| 14 | 26 | 2026-03-15 | 2026-06-13 | 6883 | 0.4901 | 0.4902 | 75 |
| 14 | 27 | 2026-06-13 | 2026-08-13 | 4397 | 0.5186 | 0.5062 | 3 |
| 30 | 0 | 2019-11-03 | 2020-02-01 | 8326 | 0.5995 | 0.5969 | 4 |
| 30 | 1 | 2020-02-01 | 2020-05-01 | 8583 | 0.5076 | 0.4834 | 75 |
| 30 | 2 | 2020-05-01 | 2020-07-30 | 8695 | 0.4511 | 0.4470 | 46 |
| 30 | 3 | 2020-07-30 | 2020-10-28 | 7880 | 0.6727 | 0.6989 | 4 |
| 30 | 4 | 2020-10-28 | 2021-01-26 | 8015 | 0.7151 | 0.7243 | 150 |
| 30 | 5 | 2021-01-26 | 2021-04-26 | 8303 | 0.5855 | 0.5934 | 13 |
| 30 | 6 | 2021-04-26 | 2021-07-25 | 8560 | 0.6656 | 0.6657 | 1 |
| 30 | 7 | 2021-07-25 | 2021-10-23 | 8412 | 0.5349 | 0.5350 | 80 |
| 30 | 8 | 2021-10-23 | 2022-01-21 | 8543 | 0.5369 | 0.5408 | 469 |
| 30 | 9 | 2022-01-21 | 2022-04-21 | 8551 | 0.6200 | 0.6701 | 6 |
| 30 | 10 | 2022-04-21 | 2022-07-20 | 8253 | 0.5695 | 0.5796 | 1 |
| 30 | 11 | 2022-07-20 | 2022-10-18 | 8278 | 0.5094 | 0.5126 | 293 |
| 30 | 12 | 2022-10-18 | 2023-01-16 | 8599 | 0.4772 | 0.4792 | 1 |
| 30 | 13 | 2023-01-16 | 2023-04-16 | 8606 | 0.4500 | 0.4475 | 1472 |
| 30 | 14 | 2023-04-16 | 2023-07-15 | 8201 | 0.3315 | 0.3422 | 6 |
| 30 | 15 | 2023-07-15 | 2023-10-13 | 8417 | 0.4999 | 0.5149 | 2 |
| 30 | 16 | 2023-10-13 | 2024-01-11 | 8158 | 0.4492 | 0.4411 | 1 |
| 30 | 17 | 2024-01-11 | 2024-04-10 | 7669 | 0.5104 | 0.4263 | 2 |
| 30 | 18 | 2024-04-10 | 2024-07-09 | 7434 | 0.4696 | 0.4706 | 47 |
| 30 | 19 | 2024-07-09 | 2024-10-07 | 7818 | 0.5269 | 0.4707 | 2 |
| 30 | 20 | 2024-10-07 | 2025-01-05 | 7656 | 0.5202 | 0.4619 | 2 |
| 30 | 21 | 2025-01-05 | 2025-04-05 | 7291 | 0.4999 | 0.5546 | 1 |
| 30 | 22 | 2025-04-05 | 2025-07-04 | 7192 | 0.3960 | 0.3711 | 136 |
| 30 | 23 | 2025-07-04 | 2025-10-02 | 7350 | 0.5393 | 0.6158 | 1 |
| 30 | 24 | 2025-10-02 | 2025-12-31 | 6660 | 0.4981 | 0.4404 | 1 |
| 30 | 25 | 2025-12-31 | 2026-03-31 | 6912 | 0.5382 | 0.5389 | 175 |
| 30 | 26 | 2026-03-31 | 2026-06-29 | 6851 | 0.5920 | 0.5968 | 4 |
| 30 | 27 | 2026-06-29 | 2026-08-13 | 3178 | 0.4697 | 0.4766 | 234 |

## Mechanical verdicts (primary h=14, median p_enter)

- Gates: FAIL
- **MODEL-V1 is NOT VIABLE**
- **REPLACES-FLOOR: False**
- median p_enter = 0.6 (house median of relative-line Sharpe across the grid)
- (a) rel Sharpe = 0.301 > 0 → True
- (b) book total 890.6% vs BTC 714.0% → True
- (c) MaxDD -79.0% vs BTC -76.6% (pass if model drawdown is no worse, i.e. not more negative) → False
- replace-floor need rel Sharpe ≥ 0.739 (naive same-window 0.589 + 0.15)
- OOS window: 2019-10-19 → 2026-08-13 (n=2491)
- Forced exits: n_events=0 n_ids=0
- % time in BTC: 91.4%

A verdict is not overridden by any single cycle.

## Headline book vs naive v3 vs BTC B&H (same OOS window)

| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % in BTC | ann TO | forced |
|------|-------|------|------------|------------|-------|------------|----------|--------|--------|
| MODEL-V1 h=14 p=0.6 | 890.6% | 39.9% | 0.865 | 0.301 | -79.0% | 2.86 | 91.4% | 2.79 | 0 |
| naive rotation v3 (same window) | 9982249.6% | 440.5% | 0.601 | 0.589 | -96.0% | nan | 3.3% | 13.04 | 7 |
| BTC B&H | 712.5% | 35.9% | 0.817 | 0.000 | -76.6% | 0.00 | 100.0% | 0.00 | 0 |
| MODEL-V1 h=30 p=0.6 (robustness) | 502.9% | 30.3% | 0.745 | -0.035 | -84.8% | 7.43 | 82.3% | 2.42 | 0 |

## p_enter grid (h=14)

| p_enter | rel Sharpe | total | MaxDD | % BTC | avg #names |
|---------|------------|-------|-------|-------|------------|
| 0.55 | 0.146 | 773.6% | -77.9% | 83.8% | 5.05 |
| 0.6 ← median | 0.301 | 890.6% | -79.0% | 91.4% | 2.86 |
| 0.65 | 0.307 | 870.9% | -79.1% | 93.8% | 2.18 |

## Per-cycle honesty (headline h=14)

| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC |
|-------|---|----------|---------|------------|------------|-------|------------|-------|
| 2019-20 | 440 | 265.1% | 263.7% | 1.914 | 0.117 | -51.9% | 1.19 | 96.3% |
| 2021 | 365 | 130.1% | 59.7% | 1.428 | 1.496 | -44.7% | 7.73 | 74.5% |
| 2022 | 365 | -64.5% | -64.3% | -1.300 | -0.147 | -67.0% | 3.42 | 92.5% |
| 2023-24 | 731 | 418.3% | 464.6% | 1.942 | -1.092 | -26.2% | 1.00 | 97.1% |
| 2025-26 | 590 | -35.9% | -30.5% | -0.392 | -0.416 | -56.2% | 3.04 | 90.2% |

## h=30 robustness (does not override h=14)

- VIABLE=False REPLACES-FLOOR=False median p_enter=0.6 rel Sharpe=-0.035

| cycle | n | book tot | BTC tot | rel Sharpe | MaxDD | % BTC |
|-------|---|----------|---------|------------|-------|-------|
| 2019-20 | 424 | 157.3% | 214.0% | -1.249 | -54.3% | 87.0% |
| 2021 | 365 | 102.8% | 59.7% | 0.835 | -43.0% | 39.8% |
| 2022 | 365 | -75.1% | -64.3% | -2.012 | -76.2% | 79.0% |
| 2023-24 | 731 | 532.8% | 464.6% | 0.466 | -32.2% | 89.4% |
| 2025-26 | 590 | -26.7% | -30.5% | 1.780 | -50.5% | 98.3% |

## Feature importances (mean gain across h=14 folds, top 15)

| rank | feature | mean gain |
|------|---------|-----------|
| 1 | `ctx_btc_vol` | 33444.29 |
| 2 | `ctx_btc_trend` | 24175.06 |
| 3 | `ctx_alt_btc_trend` | 23958.86 |
| 4 | `ctx_disp` | 21755.33 |
| 5 | `ctx_breadth` | 21600.84 |
| 6 | `ctx_corr` | 17409.22 |
| 7 | `ctx_excess_disp` | 11803.33 |
| 8 | `beta_btc_60` | 3225.08 |
| 9 | `dist_low_90` | 2976.62 |
| 10 | `idio_vol_60` | 2549.01 |
| 11 | `amihud_14` | 2322.42 |
| 12 | `dist_ath` | 2089.83 |
| 13 | `yz_vol_60` | 1747.03 |
| 14 | `mcap_rank` | 1456.67 |
| 15 | `dist_high_90` | 1408.79 |

Elapsed s=428.9. GPU=False. n_features=40. n_train_rows=436534.

COMBO untouched (v2.0-combo-final).

