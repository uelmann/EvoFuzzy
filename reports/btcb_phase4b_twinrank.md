# BTC-BEATER Phase 4.b — TWIN-RANK

**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.

Positioning and price-additions remain **NOT LIVE** (Phase 4 v2, recorded). Not retested.

## Addendum notes (verbatim)

> Catalyst and attention data families (unlocks, listing announcements, search volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume plus derivatives data already retrievable (funding, open interest, basis, taker flows).

## Vol-matched null (NEW HOUSE STANDARD; verbatim, before results)

> For tail metrics (tail-IC top-half, top-decile overlap, monster capture), the empirical null shuffles labels WITHIN vol-quintile buckets per date (yz_vol_30 quintiles), preserving the vol→outcome loading. Folds {0,5,9,15,21,24} × 25 replicates. The null mean per fold becomes the structural reference level; bias check = null mean stability across replicates (2·SE band around the fold's own null mean, E.1b tolerance: ≥2 fold violations for CONTAMINATED). Skill = real metric exceeds the vol-matched null 95th percentile on ≥5/6 folds OR Stouffer z ≥ 3.0. This supersedes the plain within-date shuffle for tail metrics in all future phases; plain-shuffle results remain on the record.

## Pre-registered criteria (verbatim, before results)

> TWIN-RANK EXTRACTS if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread, with the vol-matched null passing. DIR LIVE at the same thresholds with the same null. If BOTH fail, the ledger records: 'PRICE-VOLUME TAIL CEILING — within this data perimeter, tail improvements beyond the frozen spread are not demonstrable under vol-matched nulls; the fork (capital phase | Phase-5 hourly attention | perimeter expansion) passes to the PI.' Verdicts mechanical; nothing adopted; any production change requires a fresh pre-registered phase. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## DIR rationale (verbatim)

> label-distribution reweighting for rare extreme positives (Yang et al. ICML 2021 lineage; logit-adjustment Menon et al. 2021), the cheapest tail-emphasis intervention available.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Window 2017-08-17 → 2026-07-31 n_dates=2473
- GPU used = `False`
- LambdaRank config = one per head (truncation 10, ndcg@10, 5-grade labels, h=14); no sweeps
- DIR = one weight rule `w=1+2·1[top decile]` on the top classifier head; bottom = frozen 2.c
- 4v2 RANK cache reused = `True`

## 1 — Vol-matched null tables

Centre per fold = that fold's own null mean (structural vol-matched reference). Plain-shuffle Phase 4 v2 RANK null remains on the record (CONTAMINATED vs centre 0).

### TWIN-RANK

**tail-IC(top-half)** verdict=`PARKED` bias_pass=True skill_pass=False exceed=1/6 violations=0 Stouffer z=`0.273`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | -0.0154 | -0.0154 | 0.0168 | True | 0.0473 | 0.0508 | True |
| 5 | 25 | 0.0210 | 0.0210 | 0.0146 | True | 0.0718 | 0.0707 | False |
| 9 | 25 | 0.1958 | 0.1958 | 0.0104 | True | 0.2366 | 0.1740 | False |
| 15 | 25 | 0.1123 | 0.1123 | 0.0142 | True | 0.1627 | 0.1383 | False |
| 21 | 25 | 0.1953 | 0.1953 | 0.0111 | True | 0.2334 | 0.1427 | False |
| 24 | 25 | 0.2829 | 0.2829 | 0.0321 | True | 0.3737 | 0.2611 | False |

**overlap** verdict=`PARKED` bias_pass=True skill_pass=False exceed=1/6 violations=0 Stouffer z=`-0.166`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0700 | 0.0700 | 0.0088 | True | 0.0971 | 0.0403 | False |
| 5 | 25 | 0.0759 | 0.0759 | 0.0072 | True | 0.1041 | 0.0895 | False |
| 9 | 25 | 0.0750 | 0.0750 | 0.0043 | True | 0.0964 | 0.1005 | True |
| 15 | 25 | 0.1080 | 0.1080 | 0.0073 | True | 0.1319 | 0.1209 | False |
| 21 | 25 | 0.2036 | 0.2036 | 0.0115 | True | 0.2403 | 0.1443 | False |
| 24 | 25 | 0.1224 | 0.1224 | 0.0130 | True | 0.1701 | 0.0945 | False |

**monster top-3** verdict=`PARKED` bias_pass=True skill_pass=False exceed=0/6 violations=0 Stouffer z=`-0.455`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0708 | 0.0708 | 0.0092 | True | 0.0952 | 0.0403 | False |
| 5 | 25 | 0.0721 | 0.0721 | 0.0110 | True | 0.1121 | 0.0879 | False |
| 9 | 25 | 0.0510 | 0.0510 | 0.0059 | True | 0.0733 | 0.0476 | False |
| 15 | 25 | 0.1143 | 0.1143 | 0.0094 | True | 0.1553 | 0.1355 | False |
| 21 | 25 | 0.2249 | 0.2249 | 0.0149 | True | 0.2703 | 0.1978 | False |
| 24 | 25 | 0.0803 | 0.0803 | 0.0136 | True | 0.1238 | 0.0696 | False |

### Retro Phase-4v2 RANK head (informational)

**tail-IC(top-half)** verdict=`PARKED` bias_pass=True skill_pass=False exceed=0/6 violations=0 Stouffer z=`-0.572`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | -0.0257 | -0.0257 | 0.0211 | True | 0.0586 | 0.0212 | False |
| 5 | 25 | 0.0236 | 0.0236 | 0.0168 | True | 0.0898 | 0.0168 | False |
| 9 | 25 | 0.1873 | 0.1873 | 0.0132 | True | 0.2360 | 0.0817 | False |
| 15 | 25 | 0.1196 | 0.1196 | 0.0162 | True | 0.1771 | 0.1282 | False |
| 21 | 25 | 0.1532 | 0.1532 | 0.0220 | True | 0.2211 | 0.1507 | False |
| 24 | 25 | 0.3105 | 0.3105 | 0.0302 | True | 0.3816 | 0.3785 | False |

**overlap** verdict=`PARKED` bias_pass=True skill_pass=False exceed=2/6 violations=0 Stouffer z=`1.833`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0786 | 0.0786 | 0.0119 | True | 0.1183 | 0.0504 | False |
| 5 | 25 | 0.0750 | 0.0750 | 0.0085 | True | 0.1122 | 0.0950 | False |
| 9 | 25 | 0.0843 | 0.0843 | 0.0048 | True | 0.1024 | 0.1099 | True |
| 15 | 25 | 0.1181 | 0.1181 | 0.0065 | True | 0.1407 | 0.1190 | False |
| 21 | 25 | 0.2145 | 0.2145 | 0.0116 | True | 0.2592 | 0.2033 | False |
| 24 | 25 | 0.2097 | 0.2097 | 0.0134 | True | 0.2752 | 0.2989 | True |

**monster top-3** verdict=`PARKED` bias_pass=True skill_pass=False exceed=1/6 violations=0 Stouffer z=`1.037`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0800 | 0.0800 | 0.0121 | True | 0.1231 | 0.0476 | False |
| 5 | 25 | 0.0678 | 0.0678 | 0.0099 | True | 0.1077 | 0.1062 | False |
| 9 | 25 | 0.0573 | 0.0573 | 0.0049 | True | 0.0799 | 0.0513 | False |
| 15 | 25 | 0.1081 | 0.1081 | 0.0093 | True | 0.1480 | 0.0952 | False |
| 21 | 25 | 0.2360 | 0.2360 | 0.0143 | True | 0.2923 | 0.2454 | False |
| 24 | 25 | 0.1411 | 0.1411 | 0.0152 | True | 0.2081 | 0.2491 | True |

Retro answer: vol-matched RANK verdict=`PARKED` bias_pass=True skill_pass=False. Gain beyond vol: **NO**.

### DIR-spread

**tail-IC(top-half)** verdict=`PARKED` bias_pass=True skill_pass=False exceed=2/6 violations=0 Stouffer z=`-2.211`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0458 | 0.0458 | 0.0137 | True | 0.0961 | -0.0387 | False |
| 5 | 25 | 0.0132 | 0.0132 | 0.0082 | True | 0.0420 | 0.0761 | True |
| 9 | 25 | 0.1802 | 0.1802 | 0.0053 | True | 0.2008 | 0.1092 | False |
| 15 | 25 | 0.0740 | 0.0740 | 0.0054 | True | 0.0931 | 0.0512 | False |
| 21 | 25 | 0.0290 | 0.0290 | 0.0074 | True | 0.0646 | -0.0291 | False |
| 24 | 25 | 0.1457 | 0.1457 | 0.0074 | True | 0.1735 | 0.2222 | True |

**overlap** verdict=`GREEN` bias_pass=True skill_pass=True exceed=3/6 violations=0 Stouffer z=`7.686`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0723 | 0.0723 | 0.0036 | True | 0.0833 | 0.0614 | False |
| 5 | 25 | 0.0594 | 0.0594 | 0.0025 | True | 0.0712 | 0.1170 | True |
| 9 | 25 | 0.0987 | 0.0987 | 0.0021 | True | 0.1049 | 0.1350 | True |
| 15 | 25 | 0.0653 | 0.0653 | 0.0030 | True | 0.0783 | 0.0730 | False |
| 21 | 25 | 0.0754 | 0.0754 | 0.0042 | True | 0.0888 | 0.1150 | True |
| 24 | 25 | 0.0818 | 0.0818 | 0.0031 | True | 0.0923 | 0.0747 | False |

**monster top-3** verdict=`GREEN` bias_pass=True skill_pass=True exceed=4/6 violations=0 Stouffer z=`9.264`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0765 | 0.0765 | 0.0043 | True | 0.0872 | 0.0623 | False |
| 5 | 25 | 0.0526 | 0.0526 | 0.0037 | True | 0.0659 | 0.1245 | True |
| 9 | 25 | 0.0583 | 0.0583 | 0.0026 | True | 0.0689 | 0.1355 | True |
| 15 | 25 | 0.0580 | 0.0580 | 0.0041 | True | 0.0696 | 0.0733 | True |
| 21 | 25 | 0.1032 | 0.1032 | 0.0068 | True | 0.1267 | 0.1538 | True |
| 24 | 25 | 0.0588 | 0.0588 | 0.0044 | True | 0.0769 | 0.0549 | False |

## 2 — Vol-correlation diagnostic (report only)

Mean per-date cross-sectional rank-corr vs `yz_vol_30`. Twin subtraction should collapse the vol tilt.

- RANK top head: `-0.6228`
- RANK bottom head: `0.7763`
- TWIN-RANK: `-0.7979`
- frozen spread: `-0.6765`
- DIR-spread: `-0.6614`

## 3 — Tail-metric judgment grid (primary, per-date, floored top-100, Binance-listed)

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster top-3 | RankIC | vol-corr | n |
|--------|-------------|------|-------------|---------|---------------|--------|----------|---|
| frozen spread (baseline) | 0.0637 | 5.01 | 0.0970 | 0.0944 | 0.0815 | 0.1223 | -0.6765 | 2473 |
| TWIN-RANK | 0.0800 | 5.83 | 0.0846 | 0.0843 | 0.0654 | 0.1311 | -0.7979 | 2473 |
| SPREAD+TWIN-RANK | 0.0855 | 5.96 | 0.0935 | 0.0866 | 0.0670 | 0.1351 | -0.7865 | 2473 |
| DIR-spread | 0.0495 | 4.09 | 0.0957 | 0.0951 | 0.0846 | 0.1174 | -0.6614 | 2473 |
| DIR-spread+TWIN-RANK | 0.0681 | 4.97 | 0.0937 | 0.0855 | 0.0663 | 0.1334 | -0.7826 | 2473 |

Trailing-18m:

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | vol-corr |
|--------|-------------|------|-------------|---------|---------|--------|----------|
| frozen spread (baseline) | 0.1085 | 3.84 | 0.1209 | 0.1154 | 0.1022 | 0.1766 | -0.6151 |
| TWIN-RANK | 0.0953 | 3.94 | 0.1321 | 0.0696 | 0.0523 | 0.1841 | -0.7655 |
| SPREAD+TWIN-RANK | 0.1280 | 3.99 | 0.1257 | 0.0827 | 0.0596 | 0.1921 | -0.7437 |
| DIR-spread | 0.0902 | 3.41 | 0.1159 | 0.1165 | 0.1077 | 0.1661 | -0.5700 |
| DIR-spread+TWIN-RANK | 0.1016 | 3.45 | 0.1350 | 0.0840 | 0.0590 | 0.1876 | -0.7270 |

Overlap by cycle:

| cycle | frozen spread (baseline) | TWIN-RANK | SPREAD+TWIN-RANK | DIR-spread | DIR-spread+TWIN-RANK |
|-------|------|------|------|------|------|
| 2019-20 | 0.0529 | 0.0638 | 0.0569 | 0.0626 | 0.0622 |
| 2021 | 0.0812 | 0.0772 | 0.0744 | 0.0827 | 0.0732 |
| 2022 | 0.1156 | 0.1199 | 0.1187 | 0.1134 | 0.1162 |
| 2023-24 | 0.0950 | 0.0867 | 0.0913 | 0.0904 | 0.0850 |
| 2025-26 | 0.1205 | 0.0788 | 0.0907 | 0.1226 | 0.0923 |

Tail-IC(top-half) by cycle:

| cycle | frozen spread (baseline) | TWIN-RANK | SPREAD+TWIN-RANK | DIR-spread | DIR-spread+TWIN-RANK |
|-------|------|------|------|------|------|
| 2019-20 | 0.0074 | 0.0694 | 0.0545 | -0.0149 | 0.0327 |
| 2021 | 0.0690 | 0.0496 | 0.0548 | 0.0652 | 0.0465 |
| 2022 | 0.0960 | 0.1357 | 0.1356 | 0.0955 | 0.1334 |
| 2023-24 | 0.0445 | 0.0507 | 0.0562 | 0.0267 | 0.0372 |
| 2025-26 | 0.1078 | 0.1096 | 0.1347 | 0.0890 | 0.1071 |

## 4 — Secondary: crude 14d book (information check, not adopted)

Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance.

| book | total | CAGR | MaxDD | Sharpe | n |
|------|-------|------|-------|--------|---|
| frozen spread (baseline) | 133.8% | 13.4% | -74.0% | 0.509 | 176 |
| TWIN-RANK | 360.9% | 25.4% | -67.2% | 0.692 | 176 |
| SPREAD+TWIN-RANK | 395.3% | 26.7% | -62.4% | 0.715 | 176 |
| DIR-spread | 160.5% | 15.2% | -70.0% | 0.536 | 176 |
| DIR-spread+TWIN-RANK | 284.8% | 22.1% | -69.6% | 0.645 | 176 |

## 5 — Mechanical verdicts

- **TWIN-RANK BARREN** (clears deltas=False: ΔIC `+0.0163` / Δov `-0.0101`; vol-matched null pass=False)
- **DIR NOT LIVE** (clears deltas=False: ΔIC `-0.0142` / Δov `+0.0007`; vol-matched null pass=False)
- Retro 4v2 RANK beyond vol: **NO** (`PARKED`)

- Ledger clause: **PRICE-VOLUME TAIL CEILING — within this data perimeter, tail improvements beyond the frozen spread are not demonstrable under vol-matched nulls; the fork (capital phase | Phase-5 hourly attention | perimeter expansion) passes to the PI.**

Mechanical, no post-hoc adjustment. Nothing adopted.

## Plain language

TWIN-RANK vs frozen spread: tail-IC(top-half) 0.06373363864654095 → 0.07999626838827102 (Δ 0.016262629741730067), overlap 0.09440625421215797 → 0.08429226118267769 (Δ -0.010113993029480281). Vol-matched null did not pass. DIR vs frozen: tail-IC 0.049496602414671555 (Δ -0.0142370362318694), overlap 0.09513026399399226 (Δ 0.0007240097818342894); null did not pass. vol-corr collapse: top=-0.6227718211771603 bot=0.7763453495706951 twinrank=-0.7978702645017742. RETRO RANK vol-matched verdict=PARKED skill_pass=False gain_beyond_vol=NO. Verdicts: TWIN-RANK BARREN; DIR NOT LIVE. PRICE-VOLUME TAIL CEILING — within this data perimeter, tail improvements beyond the frozen spread are not demonstrable under vol-matched nulls; the fork (capital phase | Phase-5 hourly attention | perimeter expansion) passes to the PI. Nothing adopted.

## Notes

- Frozen spread is the 2.c cache (not retrained). TWIN-RANK uses one LambdaRank config per head.
- Vol-matched null supersedes plain within-date shuffle for tail metrics going forward.
- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**
- Elapsed s=`8427.0`. GPU=`False`.

COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.

