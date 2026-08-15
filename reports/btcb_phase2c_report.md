# BTC-BEATER Phase 2.c — twin-head spread + repowered skill null

**BACKTEST ONLY.** Cleaned+floored 2.b data reused as-is. Stage T frozen. Context excluded. CPU only, zero GPU. COMBO untouched.

## Pre-registered criteria (verbatim, before results)

> SPREAD has SELECTION SKILL if, at h=14 or h=30: mean per-date RankIC(spread) ≥ +0.01 AND mean per-date AUC ≥ 0.52 AND the §2 gate passes for the spread metric. MODEL-V3 is VIABLE if on the full OOS window at median θ: (a) relative-line Sharpe > 0; (b) total ≥ BTC B&H; (c) MaxDD ≤ BTC B&H. MODEL-V3 is PRODUCT-GRADE if additionally relative-line Sharpe ≥ 0.30 AND average alt allocation ≥ 5% (non-degenerate book). Per-cycle honesty table mandatory; no single cycle overrides. Mechanical, no post-hoc adjustment.

## Repowered skill null (verbatim, before results)

> Bias: every fold's null mean must satisfy the E.1b centering bound (AUC around 0.5, RankIC around 0). Skill passes if, for the judged signal, ≥5 of 6 folds exceed their null 95th percentile OR the Stouffer-combined z across the 6 folds is ≥ 3.0. Symmetric: failure = PARKED, no override, no retest with different folds.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical verdicts

- **SPREAD has SELECTION SKILL: False** (h=14 RankIC=0.1622 AUC=0.4884; h=30 RankIC=0.1911 AUC=0.5116; §2 spread RankIC GREEN 6/6 Stouffer z=11.041)
- **MODEL-V3 is VIABLE**
- **MODEL-V3 is PRODUCT-GRADE**
- median θ=0.2; OOS 2019-10-19 → 2026-08-13 n=2491
- (a) rel Sharpe 0.916 > 0 → True
- (b) book 986.0% vs BTC 714.0% → True
- (c) MaxDD -75.9% vs BTC -76.6% → True
- product-grade need rel ≥ 0.300 and alt ≥ 5.0%; avg alt 5.4%
- % time in BTC 94.6%; avg #names 1.35; gate ON 19.1%; forced=0
- uncertainty↔yz_vol_30 mean per-date RankIC = 0.5498 (lottery diagnostic; not used in trading)

A verdict is not overridden by any single cycle. Operative floor is BTC. Naive v4 (record only): rel Sharpe=-0.670, live_benchmark=False.

## §2 null — p_top per-date AUC (h=14)

Bias pass=False; skill CONTAMINATED; 5/6 exceed p95; Stouffer z=7.190.

| fold | n | null mean | SD | 95th | real | bias_ok | exceeds_p95 |
|------|---|-----------|----|------|------|---------|-------------|
| 0 | 25 | 0.5024 | 0.0212 | 0.5329 | 0.5349 | True | True |
| 5 | 25 | 0.5007 | 0.0285 | 0.5452 | 0.5556 | True | True |
| 9 | 25 | 0.5002 | 0.0220 | 0.5355 | 0.5334 | True | False |
| 15 | 25 | 0.5030 | 0.0211 | 0.5406 | 0.5654 | True | True |
| 21 | 25 | 0.4921 | 0.0299 | 0.5294 | 0.6230 | True | True |
| 24 | 25 | 0.4815 | 0.0409 | 0.5315 | 0.6986 | False | True |

## §2 null — spread per-date RankIC (h=14, judged signal)

Bias pass=True; skill GREEN; 6/6 exceed p95; Stouffer z=11.041. Failure = PARKED, no override, no retest with different folds.

| fold | n | null mean | SD | 95th | real | bias_ok | exceeds_p95 |
|------|---|-----------|----|------|------|---------|-------------|
| 0 | 25 | -0.0110 | 0.0295 | 0.0311 | 0.1314 | True | True |
| 5 | 25 | 0.0089 | 0.0349 | 0.0600 | 0.0831 | True | True |
| 9 | 25 | 0.0027 | 0.0382 | 0.0619 | 0.2540 | True | True |
| 15 | 25 | 0.0068 | 0.0367 | 0.0658 | 0.1827 | True | True |
| 21 | 25 | -0.0047 | 0.0719 | 0.0846 | 0.2522 | True | True |
| 24 | 25 | 0.0035 | 0.0652 | 0.0804 | 0.3384 | True | True |

## Per-fold selection metrics (floored PIT top-100)

| h | fold | RankIC spread | RankIC p_top (control) | AUC spread vs top-q | AUC p_top |
|---|------|---------------|------------------------|---------------------|-----------|
| 14 | 0 | 0.1479 | 0.0319 | 0.4730 | 0.5348 |
| 14 | 1 | 0.1074 | -0.0292 | 0.4819 | 0.5466 |
| 14 | 2 | 0.0555 | -0.0658 | 0.4336 | 0.5265 |
| 14 | 3 | 0.2504 | 0.0372 | 0.5054 | 0.5379 |
| 14 | 4 | 0.0897 | 0.0085 | 0.4392 | 0.5477 |
| 14 | 5 | 0.0805 | -0.0173 | 0.4688 | 0.5552 |
| 14 | 6 | 0.1679 | 0.0332 | 0.5337 | 0.5441 |
| 14 | 7 | 0.1419 | -0.0739 | 0.4603 | 0.5593 |
| 14 | 8 | 0.0913 | -0.0013 | 0.4697 | 0.5710 |
| 14 | 9 | 0.2240 | -0.0568 | 0.5357 | 0.5312 |
| 14 | 10 | 0.1321 | -0.0749 | 0.5233 | 0.5139 |
| 14 | 11 | 0.2508 | 0.0390 | 0.5200 | 0.5651 |
| 14 | 12 | 0.1723 | 0.0392 | 0.4911 | 0.5703 |
| 14 | 13 | 0.1015 | -0.0531 | 0.4578 | 0.5168 |
| 14 | 14 | 0.2214 | -0.0278 | 0.5393 | 0.5687 |
| 14 | 15 | 0.1718 | -0.0313 | 0.4530 | 0.5635 |
| 14 | 16 | 0.0972 | -0.0862 | 0.4376 | 0.5191 |
| 14 | 17 | 0.1153 | -0.0207 | 0.4795 | 0.5446 |
| 14 | 18 | 0.1634 | 0.0503 | 0.5111 | 0.6156 |
| 14 | 19 | 0.0728 | -0.0759 | 0.4394 | 0.5202 |
| 14 | 20 | 0.1115 | -0.0181 | 0.4750 | 0.5513 |
| 14 | 21 | 0.2006 | 0.0061 | 0.4827 | 0.6067 |
| 14 | 22 | 0.2450 | 0.0253 | 0.5471 | 0.6126 |
| 14 | 23 | 0.2608 | -0.0077 | 0.5412 | 0.5865 |
| 14 | 24 | 0.3237 | 0.1567 | 0.5582 | 0.6930 |
| 14 | 25 | 0.1441 | -0.0348 | 0.4445 | 0.5675 |
| 14 | 26 | 0.1892 | 0.0190 | 0.4739 | 0.5506 |
| 14 | 27 | 0.2811 | 0.0202 | 0.5027 | 0.5396 |
| 30 | 0 | 0.1976 | 0.0178 | 0.5094 | 0.5242 |
| 30 | 1 | 0.0775 | -0.0004 | 0.4951 | 0.5313 |
| 30 | 2 | -0.0384 | -0.0805 | 0.3983 | 0.4712 |
| 30 | 3 | 0.3245 | 0.1832 | 0.5560 | 0.5621 |
| 30 | 4 | 0.0735 | -0.0393 | 0.4748 | 0.5055 |
| 30 | 5 | 0.1839 | 0.0747 | 0.5382 | 0.5568 |
| 30 | 6 | 0.1872 | 0.0594 | 0.5328 | 0.5556 |
| 30 | 7 | 0.1335 | 0.0446 | 0.4815 | 0.5347 |
| 30 | 8 | 0.1613 | 0.1414 | 0.4606 | 0.5606 |
| 30 | 9 | 0.2195 | 0.0315 | 0.5577 | 0.5449 |
| 30 | 10 | 0.1975 | -0.0513 | 0.5532 | 0.5061 |
| 30 | 11 | 0.2954 | 0.1930 | 0.5451 | 0.6346 |
| 30 | 12 | 0.1955 | 0.0819 | 0.5345 | 0.5746 |
| 30 | 13 | 0.2574 | 0.1193 | 0.5575 | 0.5730 |
| 30 | 14 | 0.2343 | 0.0083 | 0.5536 | 0.5490 |
| 30 | 15 | 0.1492 | 0.0329 | 0.4524 | 0.5174 |
| 30 | 16 | 0.0892 | 0.0676 | 0.4578 | 0.5153 |
| 30 | 17 | 0.1707 | -0.0097 | 0.5224 | 0.5178 |
| 30 | 18 | 0.1807 | 0.0512 | 0.5223 | 0.5821 |
| 30 | 19 | -0.0302 | -0.1248 | 0.3522 | 0.4015 |
| 30 | 20 | 0.2101 | -0.0364 | 0.5602 | 0.5348 |
| 30 | 21 | 0.2227 | 0.0271 | 0.4693 | 0.5856 |
| 30 | 22 | 0.3236 | 0.0236 | 0.5805 | 0.5862 |
| 30 | 23 | 0.3716 | 0.1559 | 0.6294 | 0.6460 |
| 30 | 24 | 0.3044 | 0.2101 | 0.5260 | 0.6785 |
| 30 | 25 | 0.1960 | 0.0610 | 0.4834 | 0.5631 |
| 30 | 26 | 0.2506 | 0.0993 | 0.5099 | 0.5353 |
| 30 | 27 | 0.3632 | -0.0387 | 0.5418 | 0.3871 |

Aggregate (last-fold-wins OOS): h=14 RankIC(spread)=0.1622 RankIC(p_top)=-0.0083 AUC(spread)=0.4884; h=30 RankIC(spread)=0.1911 RankIC(p_top)=0.0491 AUC(spread)=0.5116.

## MODEL-V3 book vs BTC (same OOS window)

| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC | gate ON | ann TO | forced |
|------|-------|------|------------|------------|-------|------------|-------|---------|--------|--------|
| MODEL-V3 h=14 θ=0.2 | 986.0% | 41.8% | 0.889 | 0.916 | -75.9% | 1.35 | 94.6% | 19.1% | 1.92 | 0 |
| BTC B&H | 714.0% | 36.0% | 0.818 | 0.000 | -76.6% | 0.00 | 100.0% | 0.0% | 0.00 | 0 |

## θ grid (h=14, median convention)

| θ | rel Sharpe | total | MaxDD | % BTC | avg #names | gate ON |
|---|------------|-------|-------|-------|------------|---------|
| 0.1 | 0.993 | 1319.2% | -75.0% | 90.5% | 4.72 | 19.1% |
| 0.15 | 0.884 | 1137.7% | -75.3% | 91.3% | 3.09 | 19.1% |
| 0.2 ← median | 0.916 | 986.0% | -75.9% | 94.6% | 1.35 | 19.1% |

## Per-cycle honesty (headline h=14)

| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | % BTC |
|-------|---|----------|---------|------------|------------|-------|-------|
| 2019-20 | 440 | 310.1% | 263.7% | 2.059 | 1.210 | -47.6% | 92.0% |
| 2021 | 365 | 72.8% | 59.7% | 1.081 | 2.167 | -51.7% | 97.7% |
| 2022 | 365 | -63.2% | -64.3% | -1.251 | 1.627 | -65.9% | 98.3% |
| 2023-24 | 731 | 468.3% | 464.6% | 2.031 | 0.135 | -26.3% | 96.6% |
| 2025-26 | 590 | -26.8% | -30.5% | -0.228 | 0.733 | -52.1% | 89.7% |

## Feature importances (mean gain, h=14)

### Head TOP

| rank | feature | mean gain |
|------|---------|-----------|
| 1 | `dist_high_90` | 8986.71 |
| 2 | `range_pos_28` | 7343.42 |
| 3 | `dist_ath` | 7312.88 |
| 4 | `log_age` | 7149.44 |
| 5 | `log_mcap` | 6934.78 |
| 6 | `beta_btc_60` | 5642.77 |
| 7 | `yz_vol_60` | 4722.74 |
| 8 | `idio_vol_60` | 4423.15 |
| 9 | `skew_60` | 4187.79 |
| 10 | `amihud_14` | 4026.91 |
| 11 | `corr_btc_28` | 3999.79 |
| 12 | `dist_low_90` | 3484.20 |
| 13 | `sma20_sma50` | 3386.45 |
| 14 | `yz_vol_30` | 3374.28 |
| 15 | `mcap_rank` | 3353.54 |

### Head BOTTOM

| rank | feature | mean gain |
|------|---------|-----------|
| 1 | `idio_vol_60` | 62190.15 |
| 2 | `pk_vol_14` | 24091.91 |
| 3 | `log_age` | 20364.04 |
| 4 | `yz_vol_60` | 15220.73 |
| 5 | `corr_btc_28` | 14533.66 |
| 6 | `dist_low_90` | 10554.82 |
| 7 | `close_sma20` | 10466.89 |
| 8 | `yz_vol_30` | 9656.13 |
| 9 | `beta_btc_60` | 9626.70 |
| 10 | `dist_ath` | 8648.92 |
| 11 | `log_mcap` | 8425.78 |
| 12 | `yz_vol_14` | 7910.97 |
| 13 | `turnover` | 7175.00 |
| 14 | `d_rank_90` | 6983.94 |
| 15 | `amihud_14` | 5930.66 |

Elapsed s=7873.0. GPU=False. n_features=33.

COMBO untouched (v2.0-combo-final).

