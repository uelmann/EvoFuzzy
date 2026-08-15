# BTC-BEATER Phase 3.c — Binance replay of SPREAD-LS

**BACKTEST ONLY.** Same 2.c positions (β-matched, h=14, floored PIT top-100 DV). Only pricing and native funding change. CPU only, zero GPU. COMBO untouched. No MASTER book. Phase 3.b replaced.

## Addenda (verbatim, frozen before results)

### 1. β-match designation with post-observation disclosure

> The Phase 3 freeze designated dollar-neutral as the judged headline and β-matched as reported-not-judged. After seeing results (DN β=−0.122, β-matched β=0.025), β-matched is designated the production SPREAD-LS book. This is disclosed, not hidden. Phase 3 mechanical verdicts (VIABLE, not SLEEVE-GRADE, not replacement) remain those of the DN headline and are not retroactively rewritten. All subsequent work (funding-on, MASTER) uses the β-matched book.

### 2. House-rule correction (record only)

> The bias clause of future null gates reverts to the original E.1b tolerance: CONTAMINATED requires ≥2 fold-level violations of the 2·SE bound, not 1. The 'every fold' variant has ≈25% false-alarm probability with 6 folds. No past verdicts change.

### 3. MASTER removed from scope

> MASTER (COMBO+SPREAD-LS combination) removed from scope by PI decision; the 0.157 correlation remains on the ledger for allocation purposes.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Pre-registered validation (verbatim, before results)

> PRICES ARE VALIDATED if, on the replayable subset, BOOK-BINANCE-ONLY daily PnL correlation with the same-days CMC-priced book is ≥ 0.95 AND its net Sharpe ≥ (same-days CMC Sharpe − 0.15). If validated, BOOK-HYBRID (funding-on) becomes the OFFICIAL SPREAD-LS record; funding-off CMC numbers are deprecated with a ledger footnote. If NOT validated, the discrepancy is quantified per year and per name-tier, the official record is suspended, and no improvement work proceeds until the pricing gap is understood. Mechanical, no post-hoc adjustment.

## Mechanical verdict

- **PRICES ARE NOT VALIDATED**
- Replayable-subset daily-PnL correlation CMC↔Binance = `0.9838` (need ≥ 0.950)
- BOOK-BINANCE-ONLY net Sharpe = `1.296` vs same-days CMC `1.559` (need ≥ `1.409`; gap `-0.263`)
- n_days = 2491
- **Official SPREAD-LS record is SUSPENDED. No improvement work proceeds until the pricing gap is understood.**

Mechanical, no post-hoc adjustment.

## Position identity

- Position-log sha256 = `f47f7ece40d6cee536b2a07c25961d1d69284f92ddf716447a52b5f57fcc232b`
- BOOK-CMC vs engine max |daily PnL| = `0.000000000000` (need ≤ 1e-6)
- BOOK-CMC Sharpe `1.818` vs 3.x β-matched `1.818` (n_days=2491 vs 2491; start 2019-10-19 end 2026-08-13)
- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` n_files=112 (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- BTC in book hits = 0

## Coverage

**long_spot:** 74.0% (49234/66506 name-days). Never listed on Binance: 95. Listed but no replayable name-days: 19.

| year | name-days | replayable | % |
|------|-----------|------------|---|
| 2019 | 1836 | 895 | 48.7% |
| 2020 | 9498 | 5532 | 58.2% |
| 2021 | 10127 | 8054 | 79.5% |
| 2022 | 9807 | 8345 | 85.1% |
| 2023 | 9534 | 8440 | 88.5% |
| 2024 | 10502 | 8008 | 76.3% |
| 2025 | 9222 | 6380 | 69.2% |
| 2026 | 5980 | 3580 | 59.9% |

**short_perp:** 99.0% (46197/46685 name-days). Never listed on Binance: 0. Listed but no replayable name-days: 2.

| year | name-days | replayable | % |
|------|-----------|------------|---|
| 2020 | 2054 | 2054 | 100.0% |
| 2021 | 7366 | 7366 | 100.0% |
| 2022 | 6206 | 6080 | 98.0% |
| 2023 | 7608 | 7480 | 98.3% |
| 2024 | 9064 | 9025 | 99.6% |
| 2025 | 9176 | 9160 | 99.8% |
| 2026 | 5211 | 5032 | 96.6% |

Hybrid flagged (CMC-priced) share of name-days = `15.7%`.
Spot symbols downloaded this run = 169; reused = 1; attempted = 170.
Funding events applied = 45750; short name-days with missing funding (treated as 0) = 447.

### Never-listed / unmapped names (kept at CMC in hybrid, flagged)

Longs never listed on Binance spot: 95 (plus 19 listed but with no replayable long name-days). Shorts never listed on USDT-M perp: 0 (plus 2 listed but with no replayable short name-days).

Long never-listed sample: None(1229), AE(1700), ETP(1703), GXC(1750), PAY(1758), BTM(1866), KNCL(1982), KCS(2087), BCD(2222), BIX(2307), QC(2319), XIN(2349), RNT(2400), EKT(2453), TRUE(2457), HT(2502), XDC(2634), LBA(2760), SEELE(2830), XMX(2859), YOU(3053), BCV(3066), ZB(3351), ABBC(3437), ZT(3458)

## Three books (identical positions)

| book | Sharpe full | trail-18m | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 | MaxDD | funding PnL | ann TO | forced exits | forced covers |
|------|-------------|-----------|---------|------|------|---------|---------|-------|-------------|--------|--------------|---------------|
| BOOK-CMC (funding=0) | 1.818 | 2.458 | 1.713 | 1.299 | 2.284 | 1.402 | 2.523 | -25.8% | 0.0000 | 10.35 | 14 | 7 |
| BOOK-HYBRID (funding-on) | 1.555 | 1.381 | 1.856 | 1.805 | 1.969 | 1.134 | 1.480 | -24.9% | -0.3490 | 10.35 | 14 | 7 |
| BOOK-BINANCE-ONLY | 1.296 | 1.236 | 1.593 | 1.002 | 2.564 | 0.692 | 1.348 | -24.5% | -0.3490 | 10.35 | 14 | 7 |

Hybrid funding share of |gross| = `-1.6%`; funding total PnL = `-0.3490`.

## Squeeze-days (20 largest EW top-100 up-days)

| date | EW basket | BOOK-CMC | BOOK-HYBRID (funding-in) |
|------|-----------|----------|--------------------------|
| 2021-05-24 | 25.46% | 0.67% | 0.63% |
| 2020-03-19 | 19.91% | 1.36% | 1.63% |
| 2022-11-10 | 18.35% | -0.90% | -1.26% |
| 2021-05-20 | 17.43% | -0.14% | -0.04% |
| 2021-04-26 | 15.47% | -5.45% | -5.18% |
| 2024-11-06 | 14.34% | 0.10% | 0.18% |
| 2021-09-22 | 13.95% | -2.73% | -2.53% |
| 2024-08-08 | 13.62% | 0.53% | 0.39% |
| 2025-11-07 | 13.12% | -2.49% | -2.67% |
| 2025-04-09 | 13.04% | -2.56% | -2.44% |
| 2020-03-13 | 12.97% | -0.35% | -0.87% |
| 2025-05-08 | 12.67% | -1.06% | -1.03% |
| 2025-08-22 | 12.01% | -1.27% | 0.12% |
| 2021-03-01 | 11.85% | -5.93% | -5.79% |
| 2021-05-26 | 11.79% | -1.26% | -1.31% |
| 2021-01-28 | 11.53% | -2.37% | -2.59% |
| 2022-02-28 | 11.44% | -1.07% | -1.00% |
| 2025-10-12 | 11.31% | -0.24% | -0.28% |
| 2021-02-05 | 11.28% | 1.19% | 1.24% |
| 2025-03-02 | 11.24% | 0.40% | 0.69% |

Squeeze-day mean CMC = `-1.18%`; hybrid (funding-in) = `-1.10%`.

## Largest single-day price disagreements (top 10 by |w·Δr|)

| date | id | symbol | side | w | r_cmc | r_bn | Δr | w·Δr |
|------|----|--------|------|---|-------|------|----|------|
| 2026-07-24 | 7326 | DEXE | short | -0.0712 | 1.2318 | 1.5954 | 0.3637 | -0.02590 |
| 2026-06-11 | 36922 | H | short | -0.0135 | -0.0310 | 1.2824 | 1.3134 | -0.01768 |
| 2025-08-22 | 8911 | STRK | short | -0.0059 | 2.3439 | 0.1012 | -2.2427 | 0.01312 |
| 2021-11-25 | 1966 | MANA | short | -0.0485 | 0.2540 | 0.0059 | -0.2481 | 0.01204 |
| 2021-11-12 | 1808 | OMG | short | -0.0411 | -0.2601 | 0.0206 | 0.2807 | -0.01154 |
| 2026-06-16 | 36922 | H | short | -0.0267 | -0.1613 | 0.2702 | 0.4315 | -0.01153 |
| 2021-11-25 | 6210 | SAND | short | -0.0311 | 0.3264 | -0.0385 | -0.3649 | 0.01135 |
| 2021-11-24 | 1966 | MANA | short | -0.0500 | 0.0866 | 0.2832 | 0.1966 | -0.00983 |
| 2022-12-09 | 4195 | FTT | short | -0.0381 | 0.2468 | 0.0000 | -0.2468 | 0.00940 |
| 2022-09-09 | 8891 | BTCST | short | -0.0387 | 0.2411 | 0.0000 | -0.2411 | 0.00933 |

## Discrepancy (NOT validated — per year and name-tier)

| year | n | corr | Sharpe BN | Sharpe CMC | gap | PnL diff sum |
|------|---|------|-----------|------------|-----|--------------|
| 2019 | 74 | 0.9915 | -1.701 | -1.786 | 0.084 | 0.0006 |
| 2020 | 366 | 0.9948 | 1.998 | 1.824 | 0.174 | 0.0561 |
| 2021 | 365 | 0.9691 | 1.002 | 0.460 | 0.542 | 0.1665 |
| 2022 | 365 | 0.9904 | 2.564 | 2.882 | -0.317 | -0.0636 |
| 2023 | 365 | 0.9876 | 0.465 | 1.070 | -0.605 | -0.0934 |
| 2024 | 366 | 0.9894 | 0.906 | 0.848 | 0.058 | 0.0119 |
| 2025 | 365 | 0.9940 | 1.780 | 2.297 | -0.517 | -0.1093 |
| 2026 | 225 | 0.9843 | 1.201 | 2.736 | -1.536 | -0.4131 |

| PIT rank tier | n | corr | Sharpe BN | Sharpe CMC | gap | PnL diff sum |
|---------------|---|------|-----------|------------|-----|--------------|
| 1-10 | 2476 | 0.9934 | 0.948 | 0.954 | -0.006 | 0.0038 |
| 11-50 | 2478 | 0.9875 | 1.007 | 1.094 | -0.087 | -0.0569 |
| 51-100 | 2433 | 0.9950 | 0.745 | 0.765 | -0.020 | -0.0279 |

## Ledger

OFFICIAL SPREAD-LS RECORD SUSPENDED.

- GPU=False. Elapsed s=405.1.
- Charts: `charts/btcb_phase3c_hybrid_equity.png`, `charts/btcb_phase3c_pnl_scatter.png`.
- COMBO untouched (v2.0-combo-final). Frozen BTC-BEATER v1 untouched. No MASTER book.

