# BTC-BEATER SPREAD-LS — horizon sweep (h=3 / 7 / 14 / 30)

**BACKTEST ONLY.** Twin heads trained at h=3 and h=7 only. h=14/h=30 caches reused byte-identical. Production U = floored PIT top-100 dollar-volume. β-matched. FUNDING=OFF (3.b has not run). CPU only, zero GPU. COMBO untouched.

## Pre-registered reading (verbatim, before results)

> h=14 is the incumbent. A different horizon becomes production only if its null gate passes AND its trailing-18m net Sharpe ≥ incumbent + 0.15 AND its full-OOS net Sharpe ≥ incumbent − 0.10. Among multiple qualifiers, highest trailing wins. If none qualify, h=14 stays. Mechanical, no post-hoc adjustment.

## Repowered skill null (verbatim)

> Bias: every fold's null mean must satisfy the E.1b centering bound (AUC around 0.5, RankIC around 0). Skill passes if, for the judged signal, ≥5 of 6 folds exceed their null 95th percentile OR the Stouffer-combined z across the 6 folds is ≥ 3.0. Symmetric: failure = PARKED, no override, no retest with different folds.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Funding caveat (verbatim)

> FUNDING = 0. Funding is not available in this dataset; the sign of omitted funding is unknown. This is a material caveat on SPREAD-LS net Sharpe. Shorts on USDT-M perpetuals would have paid or received funding that is not in this book.

## Mechanical choice

- **Chosen production horizon = h=14** (fallback=True; incumbent=h=14)
- incumbent full=1.818 so challengers need ≥ 1.718; incumbent trail-18m=2.458 so challengers need ≥ 2.608
- qualifiers=[]

| h | judged | null passed | full | trail | full_ok | trail_ok | qualifies |
|---|--------|-------------|------|-------|---------|----------|-----------|
| 3 | False | False | 1.789 | 2.065 | True | False | False |
| 7 | True | True | 1.479 | 2.022 | False | False | False |
| 14 | False | True | 1.818 | 2.458 | True | False | False |
| 30 | False | False | 1.394 | 2.093 | False | False | False |

## Per-trade economics (slot-level round-trips, β-matched, funding-off)

| h | avg hold (d) | RT / year | gross edge (bps) | cost / RT (bps) | net edge (bps) | ann cost drag | n RT |
|---|--------------|-----------|------------------|-----------------|----------------|---------------|------|
| 3 | 10.6 | 2224.9 | 190.9 | 18.3 | 172.6 | 0.0620 | 15251 |
| 7 | 21.6 | 2505.3 | 368.6 | 18.3 | 350.3 | 0.0309 | 17146 |
| 14 | 36.3 | 2937.0 | 574.3 | 18.2 | 556.1 | 0.0186 | 20044 |
| 30 | 62.3 | 3521.0 | 691.3 | 18.1 | 673.3 | 0.0112 | 23875 |

Round-trip = one name entering and later leaving a single tranche slot. Gross edge is the signed simple return of that name over the hold, in bps. Cost / RT = 2 × one-way (20 bps long, 16 bps short). Open trades at the end are excluded. Ann cost drag = mean(daily book cost) × 365 (NAV return units).

## Books (β-matched, top-100 DV, funding-off)

| h | net Sharpe | trail-18m | MaxDD | total | #long | #short | shortable | % inc. short | ann TO | squeeze mean | β vs BTC | RankIC |
|---|------------|-----------|-------|-------|-------|--------|-----------|--------------|--------|--------------|----------|--------|
| 3 | 1.789 | 2.065 | -26.2% | 1569.6% | 18.67 | 12.85 | 67.3 | 89.6% | 34.09 | -0.011 | 0.043 | 0.1144 |
| 7 | 1.479 | 2.022 | -30.4% | 1055.5% | 23.48 | 16.01 | 67.4 | 90.6% | 17.10 | -0.014 | 0.037 | 0.1364 |
| 14 | 1.818 | 2.458 | -25.8% | 1706.6% | 26.70 | 18.74 | 67.6 | 89.0% | 10.35 | -0.012 | 0.025 | 0.1622 |
| 30 | 1.394 | 2.093 | -34.6% | 719.4% | 31.25 | 22.75 | 68.0 | 88.6% | 6.38 | -0.009 | 0.018 | 0.1911 |

## Per-cycle net Sharpe

| h | cycle | n | net Sharpe | MaxDD | #long | #short |
|---|-------|---|------------|-------|-------|--------|
| 3 | 2019-20 | 451 | 1.371 | -26.2% | 18.86 | 3.65 |
| 3 | 2021 | 365 | 1.375 | -22.0% | 18.17 | 13.02 |
| 3 | 2022 | 365 | 3.011 | -9.1% | 18.59 | 11.58 |
| 3 | 2023-24 | 731 | 1.620 | -10.0% | 18.82 | 16.25 |
| 3 | 2025-26 | 590 | 2.239 | -13.8% | 18.70 | 16.34 |
| 7 | 2019-20 | 447 | 1.121 | -30.4% | 22.87 | 4.43 |
| 7 | 2021 | 365 | 0.569 | -27.4% | 23.25 | 17.00 |
| 7 | 2022 | 365 | 2.279 | -7.3% | 23.81 | 14.46 |
| 7 | 2023-24 | 731 | 1.673 | -10.8% | 24.36 | 20.00 |
| 7 | 2025-26 | 590 | 2.141 | -19.8% | 22.79 | 20.18 |
| 14 | 2019-20 | 440 | 1.713 | -25.8% | 25.76 | 4.67 |
| 14 | 2021 | 365 | 1.299 | -19.6% | 27.75 | 20.18 |
| 14 | 2022 | 365 | 2.284 | -8.0% | 26.87 | 17.00 |
| 14 | 2023-24 | 731 | 1.402 | -9.6% | 27.41 | 22.81 |
| 14 | 2025-26 | 590 | 2.523 | -21.6% | 25.77 | 24.38 |
| 30 | 2019-20 | 424 | 1.184 | -34.6% | 28.27 | 5.36 |
| 30 | 2021 | 365 | 0.749 | -16.7% | 31.56 | 23.56 |
| 30 | 2022 | 365 | 1.814 | -10.7% | 30.86 | 18.81 |
| 30 | 2023-24 | 731 | 1.240 | -13.6% | 33.29 | 27.78 |
| 30 | 2025-26 | 590 | 2.117 | -14.4% | 30.91 | 30.95 |

## §2 null — spread per-date RankIC (h=3, judged signal)

Bias pass=False; skill CONTAMINATED; 6/6 exceed p95; Stouffer z=11.580. passed=False. Failure = PARKED, no override, no retest with different folds.

| fold | n | null mean | SD | 95th | real | bias_ok | exceeds_p95 |
|------|---|-----------|----|------|------|---------|-------------|
| 0 | 25 | -0.0022 | 0.0170 | 0.0198 | 0.0897 | True | True |
| 5 | 25 | -0.0034 | 0.0243 | 0.0341 | 0.0371 | True | True |
| 9 | 25 | 0.0128 | 0.0269 | 0.0520 | 0.1859 | False | True |
| 15 | 25 | 0.0025 | 0.0216 | 0.0258 | 0.1158 | True | True |
| 21 | 25 | 0.0073 | 0.0297 | 0.0511 | 0.1503 | True | True |
| 24 | 25 | -0.0034 | 0.0374 | 0.0558 | 0.1751 | True | True |

## §2 null — spread per-date RankIC (h=7, judged signal)

Bias pass=True; skill GREEN; 5/6 exceed p95; Stouffer z=10.391. passed=True. Failure = PARKED, no override, no retest with different folds.

| fold | n | null mean | SD | 95th | real | bias_ok | exceeds_p95 |
|------|---|-----------|----|------|------|---------|-------------|
| 0 | 25 | 0.0053 | 0.0284 | 0.0510 | 0.1227 | True | True |
| 5 | 25 | -0.0059 | 0.0250 | 0.0370 | 0.0369 | True | False |
| 9 | 25 | 0.0112 | 0.0361 | 0.0680 | 0.2160 | True | True |
| 15 | 25 | 0.0099 | 0.0302 | 0.0572 | 0.1472 | True | True |
| 21 | 25 | -0.0061 | 0.0484 | 0.0756 | 0.1708 | True | True |
| 24 | 25 | 0.0058 | 0.0432 | 0.0572 | 0.2531 | True | True |

## §2 null — h=14 (reused from Phase 2.c, not re-run)

passed=True; verdict=GREEN; 6/6; Stouffer z=11.041. reused from Phase 2.c; not re-run this freeze

## h=30 null

no horizon-specific null in this freeze; 2.c null was h=14 only. Reported, not judged.

## Cache / reuse

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` n_files=112 (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- new-head pred dir sha256 = `95ff82c784fd61a6c717dd5658e1123d8701afa99363baa619013c53ba846ff8` n_files=112
- BTC in book hits (all runs) = 0
- GPU=False. Elapsed s=261.8.

Charts: `charts/btcb_horizon_equity.png`, `charts/btcb_horizon_rankic.png`.

COMBO untouched (v2.0-combo-final).

