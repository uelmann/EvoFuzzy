# BTC-BEATER Phase 4 v2 — TAIL ROUND 1

**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.

Supersedes the cancelled Phase 4 v1 (unlock calendar) prompt.

## Addendum notes (verbatim)

> Catalyst and attention data families (unlocks, listing announcements, search volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume plus derivatives data already retrievable (funding, open interest, basis, taker flows).

> The old project's microstructure KILL applied to the mean-regression label on the old system; this phase judges a positioning block on the NEW system's tail metrics — fresh pre-registration, not a kill-list retest.

## Pre-registered criteria (verbatim, before results)

> TAIL-LOSS EXTRACTS if RANK or blend improves tail-IC(top-half) ≥ +0.010 AND overlap ≥ +0.015 vs baseline with the null passing; BARREN otherwise. POSITIONING LIVE if the positioning block adds tail-IC(top-half) ≥ +0.010 OR overlap ≥ +0.015 on top of the best A-signal, with ≥50% perp coverage of top-100 name-days from 2021. PRICE-ADDITIONS LIVE at the same thresholds. Verdicts mechanical; nothing adopted; any production change requires a fresh pre-registered phase. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Window 2017-08-17 → 2026-07-31 n_dates=2473
- GPU used = `False`
- LambdaRank config = one (truncation 10, ndcg@10, 5-grade labels, h=14); no sweeps
- New Vision downloads (OI/metrics gaps) = 365 symbol jobs, 3256 new rows

## 1 — RANK-head null (E.1b on per-date tail metrics)

Judged = tail-IC(top-half). Bias: original E.1b 2·SE bound; CONTAMINATED requires ≥2 fold violations (house rule). Skill: ≥5/6 exceed p95 or Stouffer z ≥ 3.0.

tail-IC(top-half): verdict=`CONTAMINATED` bias_pass=False skill_pass=False exceed=2/6 violations=4 Stouffer z=`1.348`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0113 | 0.0000 | 0.0225 | True | 0.0987 | 0.0238 | False |
| 5 | 25 | -0.0111 | 0.0000 | 0.0176 | True | 0.0581 | 0.0207 | False |
| 9 | 25 | 0.1410 | 0.0000 | 0.0212 | False | 0.2106 | 0.0817 | False |
| 15 | 25 | 0.0280 | 0.0000 | 0.0209 | False | 0.1165 | 0.1299 | True |
| 21 | 25 | 0.1828 | 0.0000 | 0.0239 | False | 0.2484 | 0.1498 | False |
| 24 | 25 | 0.2313 | 0.0000 | 0.0282 | False | 0.3520 | 0.3779 | True |

Overlap (centre=0.10): verdict=`CONTAMINATED` bias_pass=False skill_pass=False exceed=1/6 violations=5 Stouffer z=`1.498`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0883 | 0.1000 | 0.0115 | False | 0.1282 | 0.0476 | False |
| 5 | 25 | 0.0827 | 0.1000 | 0.0085 | False | 0.1170 | 0.0986 | False |
| 9 | 25 | 0.1139 | 0.1000 | 0.0087 | False | 0.1469 | 0.1115 | False |
| 15 | 25 | 0.0978 | 0.1000 | 0.0082 | True | 0.1212 | 0.1190 | False |
| 21 | 25 | 0.1703 | 0.1000 | 0.0145 | False | 0.2298 | 0.2011 | False |
| 24 | 25 | 0.1436 | 0.1000 | 0.0239 | False | 0.2497 | 0.2967 | True |

## 2 — Positioning coverage

Perp coverage of top-100 name-days from 2021: **85.2%** (n=205008; live threshold 50%).

| slice | n | perp | funding | ΔOI | basis | taker |
|-------|---|------|---------|-----|-------|-------|
| year 2016 | 986 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| year 2017 | 9904 | 69.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| year 2018 | 34618 | 48.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| year 2019 | 34426 | 51.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| year 2020 | 36600 | 61.2% | 32.3% | 0.3% | 31.8% | 0.3% |
| year 2021 | 36500 | 82.7% | 70.4% | 5.1% | 63.7% | 4.8% |
| year 2022 | 36500 | 86.6% | 73.4% | 71.6% | 68.2% | 47.6% |
| year 2023 | 36500 | 90.8% | 82.5% | 77.7% | 70.0% | 77.9% |
| year 2024 | 36600 | 86.1% | 81.5% | 75.5% | 65.0% | 76.0% |
| year 2025 | 36500 | 83.2% | 80.3% | 68.9% | 55.0% | 69.3% |
| year 2026 | 22408 | 79.9% | 74.0% | 64.2% | 53.0% | 63.8% |
| tier 1-10 | 35609 | 96.0% | 63.2% | 45.9% | 62.7% | 42.6% |
| tier 11-50 | 131614 | 76.5% | 55.1% | 41.2% | 49.5% | 38.0% |
| tier 51-100 | 154319 | 67.7% | 48.7% | 34.4% | 34.7% | 32.3% |

First available OI date reported for 438 perp symbols. Sample: 0GUSDT:2025-09-26, 1000BONKUSDT:2023-12-01, 1000LUNCUSDT:2022-09-18, 1000RATSUSDT:2023-12-24, 1000SHIBUSDT:2021-12-01, 1000XECUSDT:2022-04-02, 1000XUSDT:2024-11-22, 1INCHUSDT:2021-12-01, AAVEUSDT:2021-12-01, ACEUSDT:2023-12-27, ACHUSDT:2023-03-03, ACTUSDT:2024-11-20.

## 3 — Tail-metric ablation grid (primary, per-date, floored top-100, Binance-listed)

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster top-3 | RankIC | n |
|--------|-------------|------|-------------|---------|---------------|--------|---|
| frozen spread (baseline) | 0.0637 | 5.01 | 0.0970 | 0.0944 | 0.0815 | 0.1223 | 2473 |
| RANK head | 0.0801 | 5.09 | 0.0543 | 0.1174 | 0.1010 | 0.1047 | 2473 |
| SPREAD+RANK | 0.0932 | 6.17 | 0.0817 | 0.1027 | 0.0857 | 0.1272 | 2473 |
| spread retrained +positioning | 0.0570 | 4.51 | 0.0941 | 0.0918 | 0.0789 | 0.1239 | 2473 |
| +price-additions | 0.0663 | 5.25 | 0.0907 | 0.0963 | 0.0821 | 0.1223 | 2473 |
| full stack | 0.0789 | 5.32 | 0.0841 | 0.1037 | 0.0876 | 0.1218 | 2473 |

Trailing-18m:

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC |
|--------|-------------|------|-------------|---------|---------|--------|
| frozen spread (baseline) | 0.1085 | 3.84 | 0.1209 | 0.1154 | 0.1022 | 0.1766 |
| RANK head | 0.1668 | 4.42 | 0.0550 | 0.1415 | 0.1204 | 0.1644 |
| SPREAD+RANK | 0.1765 | 5.19 | 0.1114 | 0.1255 | 0.1071 | 0.1849 |
| spread retrained +positioning | 0.1002 | 3.68 | 0.1130 | 0.0909 | 0.0779 | 0.1635 |
| +price-additions | 0.1023 | 3.16 | 0.1071 | 0.1158 | 0.0979 | 0.1746 |
| full stack | 0.1314 | 3.52 | 0.1155 | 0.1270 | 0.1052 | 0.1922 |

Overlap by cycle:

| cycle | frozen spread (baseline) | RANK head | SPREAD+RANK | spread retrained +positioning | +price-additions | full stack |
|-------|------|------|------|------|------|------|
| 2019-20 | 0.0529 | 0.0803 | 0.0581 | 0.0731 | 0.0631 | 0.0624 |
| 2021 | 0.0812 | 0.0817 | 0.0761 | 0.0739 | 0.0744 | 0.0780 |
| 2022 | 0.1156 | 0.1382 | 0.1336 | 0.1154 | 0.1204 | 0.1198 |
| 2023-24 | 0.0950 | 0.1240 | 0.1048 | 0.0948 | 0.0981 | 0.1115 |
| 2025-26 | 0.1205 | 0.1472 | 0.1316 | 0.0987 | 0.1184 | 0.1317 |

Tail-IC(top-half) by cycle:

| cycle | frozen spread (baseline) | RANK head | SPREAD+RANK | spread retrained +positioning | +price-additions | full stack |
|-------|------|------|------|------|------|------|
| 2019-20 | 0.0074 | 0.0507 | 0.0362 | 0.0094 | 0.0294 | 0.0117 |
| 2021 | 0.0690 | 0.0075 | 0.0400 | 0.0489 | 0.0772 | 0.0348 |
| 2022 | 0.0960 | 0.1004 | 0.1394 | 0.0854 | 0.1205 | 0.1287 |
| 2023-24 | 0.0445 | 0.0498 | 0.0627 | 0.0393 | 0.0292 | 0.0697 |
| 2025-26 | 0.1078 | 0.1750 | 0.1809 | 0.1032 | 0.1007 | 0.1388 |

## 4 — Secondary: crude 14d book (information check, not adopted)

Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance.

| book | total | CAGR | MaxDD | Sharpe | n |
|------|-------|------|-------|--------|---|
| frozen spread (baseline) | 133.8% | 13.4% | -74.0% | 0.509 | 176 |
| RANK head | 213.9% | 18.5% | -64.0% | 0.621 | 176 |
| SPREAD+RANK | 275.2% | 21.6% | -60.4% | 0.662 | 176 |
| spread retrained +positioning | 190.5% | 17.1% | -67.5% | 0.566 | 176 |
| +price-additions | 220.5% | 18.8% | -67.0% | 0.593 | 176 |
| full stack | 262.3% | 21.0% | -73.0% | 0.633 | 176 |

## 5 — Mechanical verdicts

- **BARREN** (best A = `spread_rank`; Δ tail-IC(top-half) `+0.0295`; Δ overlap `+0.0083`; null pass=False)
- **POSITIONING NOT LIVE** (Δ vs best A: tail-IC `-0.0363`, overlap `-0.0109`; perp coverage from 2021 `85.2%`)
- **PRICE-ADDITIONS NOT LIVE** (Δ vs positioning: tail-IC `+0.0093`, overlap `+0.0046`)

Mechanical, no post-hoc adjustment. Nothing adopted.

## Plain language

RANK alone clears both metric deltas vs frozen spread (Δ tail-IC +0.0164, Δ overlap +0.0230) but the RANK tail-IC null is CONTAMINATED (bias violations=4), so TAIL-LOSS is BARREN. Blend clears IC but not overlap. Positioning hurts the best A-signal; price-additions do not clear the LIVE thresholds. Perp coverage from 2021 is 85.2%. Nothing adopted.

## Notes

- Frozen spread is the 2.c cache (not retrained). RANK uses one LambdaRank config.
- Positioning features are 0 + `pos_missing` for non-perp names. Coverage flags enforced.
- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**
- Metrics gap downloads: 365 jobs, 3256 new rows (Binance Vision only).
- Elapsed s=`5632.5`. GPU=`False`.

COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.

