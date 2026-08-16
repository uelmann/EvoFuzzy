# BTC-BEATER Phase 7.b — FUZZY-STACK

**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.

## Firewall (verbatim, before results)

> The PI's hand formula stays quarantined. Rule features come ONLY from RULE-FORGE/NFN outputs (assert provenance).

## Vol-matched null (house standard; verbatim, before results)

> For tail metrics (tail-IC top-half, top-decile overlap, monster capture), the empirical null shuffles labels WITHIN vol-quintile buckets per date (yz_vol_30 quintiles), preserving the vol→outcome loading. Folds {0,5,9,15,21,24} × 25 replicates. The null mean per fold becomes the structural reference level; bias check = null mean stability across replicates (2·SE band around the fold's own null mean, E.1b tolerance: ≥2 fold violations for CONTAMINATED). Skill = real metric exceeds the vol-matched null 95th percentile on ≥5/6 folds OR Stouffer z ≥ 3.0. This supersedes the plain within-date shuffle for tail metrics in all future phases; plain-shuffle results remain on the record.

## Pre-registered criteria (verbatim, before results)

> An arm EXTRACTS if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread with the vol-matched null passing. COMPOSITION-WINS if additionally the arm beats its rule source's own standalone signal on tail-IC(top) (the stack must add over both parents). Whole-list note: if an arm fails tail thresholds but improves whole-list RankIC by ≥ +0.010 with null passing, it is recorded as a WHOLE-RANKING LEAD for a fresh production-book phase (the 4.c precedent), not adopted here. If all arms fail everything, the ledger gains: 'fuzzy-GBM composition on daily 33-features does not exceed the frozen spread; the daily composition question is closed.' Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Window 2017-08-17 → 2026-07-31 n_dates=2473
- GPU used = `False`
- Library = C(66,2) pairwise CDF products = `2145` plus 33 originals
- Two-stage prune k=150 per head, union n=`252` (one prune, no iteration)
- Firewall = `True`

## 0 — Preconditions (Arm B)

STACK-SKIPPED reasons=['RULE-FORGE bank missing', 'NFN bank missing']

## 1 — Training hygiene

ES floor `200`, patience `100`, cap `3000`. UNDERTRAINED if best_iteration < `250`. **UNDERTRAINED count = `246`** (of `262` LightGBM fits in this phase, excluding cached-null hits).

| tag | fold | head | best_iteration | UNDERTRAINED | status | elapsed s |
|-----|------|------|----------------|--------------|--------|-----------|
| arma_s1_top | 0 | top | 277 | False | ok | 120.5 |
| arma_s1_bot | 0 | bot | 152 | True | ok | 132.1 |
| arma_s1_top | 1 | top | 35 | True | ok | 125.1 |
| arma_s1_bot | 1 | bot | 28 | True | ok | 195.8 |
| arma_s1_top | 2 | top | 25 | True | ok | 129.8 |
| arma_s1_bot | 2 | bot | 31 | True | ok | 151.0 |
| arma_s1_top | 3 | top | 2 | True | ok | 141.0 |
| arma_s1_bot | 3 | bot | 132 | True | ok | 275.1 |
| arma_s1_top | 4 | top | 21 | True | ok | 115.2 |
| arma_s1_bot | 4 | bot | 77 | True | ok | 144.5 |
| arma_s1_top | 5 | top | 48 | True | ok | 166.1 |
| arma_s1_bot | 5 | bot | 123 | True | ok | 150.1 |
| arma_s1_top | 6 | top | 4 | True | ok | 192.0 |
| arma_s1_bot | 6 | bot | 15 | True | ok | 171.4 |
| arma_s1_top | 7 | top | 9 | True | ok | 307.2 |
| arma_s1_bot | 7 | bot | 77 | True | ok | 141.7 |
| arma_s1_top | 8 | top | 4 | True | ok | 150.6 |
| arma_s1_bot | 8 | bot | 107 | True | ok | 279.1 |
| arma_s1_top | 9 | top | 31 | True | ok | 185.3 |
| arma_s1_bot | 9 | bot | 477 | False | ok | 290.1 |
| arma_s1_top | 10 | top | 183 | True | ok | 183.7 |
| arma_s1_bot | 10 | bot | 3 | True | ok | 163.7 |
| arma_s1_top | 11 | top | 7 | True | ok | 155.4 |
| arma_s1_bot | 11 | bot | 106 | True | ok | 174.8 |
| arma_s1_top | 12 | top | 272 | False | ok | 204.5 |
| arma_s1_bot | 12 | bot | 210 | True | ok | 174.6 |
| arma_s1_top | 13 | top | 120 | True | ok | 420.6 |
| arma_s1_bot | 13 | bot | 274 | False | ok | 237.9 |
| arma_s1_top | 14 | top | 62 | True | ok | 161.3 |
| arma_s1_bot | 14 | bot | 350 | False | ok | 505.6 |
| arma_s1_top | 15 | top | 56 | True | ok | 195.2 |
| arma_s1_bot | 15 | bot | 128 | True | ok | 190.7 |
| arma_s1_top | 16 | top | 21 | True | ok | 189.2 |
| arma_s1_bot | 16 | bot | 23 | True | ok | 180.0 |
| arma_s1_top | 17 | top | 84 | True | ok | 216.2 |
| arma_s1_bot | 17 | bot | 82 | True | ok | 226.3 |
| arma_s1_top | 18 | top | 198 | True | ok | 201.9 |
| arma_s1_bot | 18 | bot | 41 | True | ok | 190.0 |
| arma_s1_top | 19 | top | 41 | True | ok | 208.2 |
| arma_s1_bot | 19 | bot | 137 | True | ok | 231.1 |
| arma_s1_top | 20 | top | 44 | True | ok | 183.3 |
| arma_s1_bot | 20 | bot | 11 | True | ok | 525.4 |
| arma_s1_top | 21 | top | 1 | True | ok | 230.9 |
| arma_s1_bot | 21 | bot | 223 | True | ok | 215.6 |
| arma_s1_top | 22 | top | 19 | True | ok | 219.7 |
| arma_s1_bot | 22 | bot | 60 | True | ok | 488.0 |
| arma_s1_top | 23 | top | 294 | False | ok | 242.8 |
| arma_s1_bot | 23 | bot | 11 | True | ok | 170.2 |
| arma_s1_top | 24 | top | 18 | True | ok | 217.1 |
| arma_s1_bot | 24 | bot | 37 | True | ok | 214.2 |
| arma_s1_top | 25 | top | 54 | True | ok | 224.3 |
| arma_s1_bot | 25 | bot | 340 | False | ok | 337.4 |
| arma_s1_top | 26 | top | 86 | True | ok | 240.9 |
| arma_s1_bot | 26 | bot | 20 | True | ok | 216.9 |
| arma_s1_top | 27 | top | 345 | False | ok | 309.9 |
| arma_s1_bot | 27 | bot | 318 | False | ok | 299.9 |
| arma_s2_top | 0 | top | 216 | True | ok | 58.5 |
| arma_s2_bot | 0 | bot | 124 | True | ok | 66.1 |
| arma_s2_top | 1 | top | 139 | True | ok | 81.7 |
| arma_s2_bot | 1 | bot | 25 | True | ok | 68.7 |
| arma_s2_top | 2 | top | 10 | True | ok | 67.7 |
| arma_s2_bot | 2 | bot | 35 | True | ok | 83.8 |
| arma_s2_top | 3 | top | 3 | True | ok | 72.1 |
| arma_s2_bot | 3 | bot | 149 | True | ok | 69.4 |
| arma_s2_top | 4 | top | 31 | True | ok | 57.2 |
| arma_s2_bot | 4 | bot | 109 | True | ok | 71.7 |
| arma_s2_top | 5 | top | 17 | True | ok | 67.0 |
| arma_s2_bot | 5 | bot | 126 | True | ok | 66.3 |
| arma_s2_top | 6 | top | 57 | True | ok | 68.7 |
| arma_s2_bot | 6 | bot | 30 | True | ok | 70.1 |
| arma_s2_top | 7 | top | 8 | True | ok | 104.5 |
| arma_s2_bot | 7 | bot | 55 | True | ok | 109.5 |
| arma_s2_top | 8 | top | 73 | True | ok | 59.4 |
| arma_s2_bot | 8 | bot | 110 | True | ok | 68.8 |
| arma_s2_top | 9 | top | 5 | True | ok | 69.3 |
| arma_s2_bot | 9 | bot | 134 | True | ok | 73.1 |
| arma_s2_top | 10 | top | 50 | True | ok | 109.4 |
| arma_s2_bot | 10 | bot | 85 | True | ok | 72.6 |
| arma_s2_top | 11 | top | 166 | True | ok | 65.9 |
| arma_s2_bot | 11 | bot | 133 | True | ok | 85.0 |
| arma_s2_top | 12 | top | 89 | True | ok | 96.9 |
| arma_s2_bot | 12 | bot | 255 | False | ok | 81.6 |
| arma_s2_top | 13 | top | 6 | True | ok | 74.9 |
| arma_s2_bot | 13 | bot | 155 | True | ok | 89.7 |
| arma_s2_top | 14 | top | 50 | True | ok | 76.4 |
| arma_s2_bot | 14 | bot | 14 | True | ok | 71.3 |
| arma_s2_top | 15 | top | 114 | True | ok | 85.4 |
| arma_s2_bot | 15 | bot | 129 | True | ok | 86.5 |
| arma_s2_top | 16 | top | 15 | True | ok | 71.7 |
| arma_s2_bot | 16 | bot | 82 | True | ok | 77.2 |
| arma_s2_top | 17 | top | 192 | True | ok | 95.7 |
| arma_s2_bot | 17 | bot | 107 | True | ok | 77.8 |
| arma_s2_top | 18 | top | 251 | False | ok | 87.3 |
| arma_s2_bot | 18 | bot | 6 | True | ok | 66.2 |
| arma_s2_top | 19 | top | 15 | True | ok | 109.1 |
| arma_s2_bot | 19 | bot | 579 | False | ok | 159.3 |
| arma_s2_top | 20 | top | 17 | True | ok | 95.2 |
| arma_s2_bot | 20 | bot | 12 | True | ok | 74.4 |
| arma_s2_top | 21 | top | 10 | True | ok | 71.0 |
| arma_s2_bot | 21 | bot | 270 | False | ok | 94.3 |
| arma_s2_top | 22 | top | 17 | True | ok | 101.0 |
| arma_s2_bot | 22 | bot | 43 | True | ok | 79.4 |
| arma_s2_top | 23 | top | 225 | True | ok | 80.1 |
| arma_s2_bot | 23 | bot | 284 | False | ok | 84.0 |
| arma_s2_top | 24 | top | 19 | True | ok | 91.5 |
| arma_s2_bot | 24 | bot | 37 | True | ok | 98.8 |
| arma_s2_top | 25 | top | 78 | True | ok | 87.5 |
| arma_s2_bot | 25 | bot | 333 | False | ok | 113.6 |
| arma_s2_top | 26 | top | 125 | True | ok | 128.3 |
| arma_s2_bot | 26 | bot | 23 | True | ok | 77.8 |
| arma_s2_top | 27 | top | 231 | True | ok | 83.7 |
| arma_s2_bot | 27 | bot | 323 | False | ok | 99.2 |
| null_arm_a | 0 | twin | 5 | True | ok | nan |
| null_arm_a | 0 | twin | 36 | True | ok | nan |
| null_arm_a | 0 | twin | 2 | True | ok | nan |
| null_arm_a | 0 | twin | 273 | True | ok | nan |
| null_arm_a | 0 | twin | 25 | True | ok | nan |
| null_arm_a | 0 | twin | 2 | True | ok | nan |
| null_arm_a | 0 | twin | 80 | True | ok | nan |
| null_arm_a | 0 | twin | 111 | True | ok | nan |
| null_arm_a | 0 | twin | 1 | True | ok | nan |
| null_arm_a | 0 | twin | 11 | True | ok | nan |
| null_arm_a | 0 | twin | 2 | True | ok | nan |
| null_arm_a | 0 | twin | 25 | True | ok | nan |
| null_arm_a | 0 | twin | 343 | True | ok | nan |
| null_arm_a | 0 | twin | 30 | True | ok | nan |
| null_arm_a | 0 | twin | 1 | True | ok | nan |
| null_arm_a | 0 | twin | 4 | True | ok | nan |
| null_arm_a | 0 | twin | 268 | True | ok | nan |
| null_arm_a | 0 | twin | 2 | True | ok | nan |
| null_arm_a | 0 | twin | 8 | True | ok | nan |
| null_arm_a | 0 | twin | 3 | True | ok | nan |
| null_arm_a | 0 | twin | 11 | True | ok | nan |
| null_arm_a | 0 | twin | 7 | True | ok | nan |
| null_arm_a | 0 | twin | 136 | True | ok | nan |
| null_arm_a | 0 | twin | 46 | True | ok | nan |
| null_arm_a | 0 | twin | 34 | True | ok | nan |
| null_arm_a | 5 | twin | 270 | True | ok | nan |
| null_arm_a | 5 | twin | 8 | True | ok | nan |
| null_arm_a | 5 | twin | 372 | True | ok | nan |
| null_arm_a | 5 | twin | 5 | True | ok | nan |
| null_arm_a | 5 | twin | 280 | True | ok | nan |
| null_arm_a | 5 | twin | 115 | True | ok | nan |
| null_arm_a | 5 | twin | 1 | True | ok | nan |
| null_arm_a | 5 | twin | 23 | True | ok | nan |
| null_arm_a | 5 | twin | 389 | True | ok | nan |
| null_arm_a | 5 | twin | 19 | True | ok | nan |
| null_arm_a | 5 | twin | 329 | True | ok | nan |
| null_arm_a | 5 | twin | 35 | True | ok | nan |
| null_arm_a | 5 | twin | 494 | True | ok | nan |
| null_arm_a | 5 | twin | 17 | True | ok | nan |
| null_arm_a | 5 | twin | 41 | True | ok | nan |
| null_arm_a | 5 | twin | 2 | True | ok | nan |
| null_arm_a | 5 | twin | 11 | True | ok | nan |
| null_arm_a | 5 | twin | 42 | True | ok | nan |
| null_arm_a | 5 | twin | 184 | True | ok | nan |
| null_arm_a | 5 | twin | 11 | True | ok | nan |
| null_arm_a | 5 | twin | 3 | True | ok | nan |
| null_arm_a | 5 | twin | 10 | True | ok | nan |
| null_arm_a | 5 | twin | 58 | True | ok | nan |
| null_arm_a | 5 | twin | 388 | True | ok | nan |
| null_arm_a | 5 | twin | 30 | True | ok | nan |
| null_arm_a | 9 | twin | 118 | True | ok | nan |
| null_arm_a | 9 | twin | 3 | True | ok | nan |
| null_arm_a | 9 | twin | 21 | True | ok | nan |
| null_arm_a | 9 | twin | 1 | True | ok | nan |
| null_arm_a | 9 | twin | 6 | True | ok | nan |
| null_arm_a | 9 | twin | 52 | True | ok | nan |
| null_arm_a | 9 | twin | 13 | True | ok | nan |
| null_arm_a | 9 | twin | 1 | True | ok | nan |
| null_arm_a | 9 | twin | 31 | True | ok | nan |
| null_arm_a | 9 | twin | 5 | True | ok | nan |
| null_arm_a | 9 | twin | 1 | True | ok | nan |
| null_arm_a | 9 | twin | 157 | True | ok | nan |
| null_arm_a | 9 | twin | 183 | True | ok | nan |
| null_arm_a | 9 | twin | 29 | True | ok | nan |
| null_arm_a | 9 | twin | 1 | True | ok | nan |
| null_arm_a | 9 | twin | 15 | True | ok | nan |
| null_arm_a | 9 | twin | 1 | True | ok | nan |
| null_arm_a | 9 | twin | 23 | True | ok | nan |
| null_arm_a | 9 | twin | 2 | True | ok | nan |
| null_arm_a | 9 | twin | 45 | True | ok | nan |
| null_arm_a | 9 | twin | 23 | True | ok | nan |
| null_arm_a | 9 | twin | 1 | True | ok | nan |
| null_arm_a | 9 | twin | 38 | True | ok | nan |
| null_arm_a | 9 | twin | 18 | True | ok | nan |
| null_arm_a | 9 | twin | 139 | True | ok | nan |
| null_arm_a | 15 | twin | 124 | True | ok | nan |
| null_arm_a | 15 | twin | 161 | True | ok | nan |
| null_arm_a | 15 | twin | 43 | True | ok | nan |
| null_arm_a | 15 | twin | 76 | True | ok | nan |
| null_arm_a | 15 | twin | 46 | True | ok | nan |
| null_arm_a | 15 | twin | 20 | True | ok | nan |
| null_arm_a | 15 | twin | 40 | True | ok | nan |
| null_arm_a | 15 | twin | 3 | True | ok | nan |
| null_arm_a | 15 | twin | 35 | True | ok | nan |
| null_arm_a | 15 | twin | 16 | True | ok | nan |
| null_arm_a | 15 | twin | 41 | True | ok | nan |
| null_arm_a | 15 | twin | 89 | True | ok | nan |
| null_arm_a | 15 | twin | 118 | True | ok | nan |
| null_arm_a | 15 | twin | 19 | True | ok | nan |
| null_arm_a | 15 | twin | 14 | True | ok | nan |
| null_arm_a | 15 | twin | 267 | True | ok | nan |
| null_arm_a | 15 | twin | 119 | True | ok | nan |
| null_arm_a | 15 | twin | 44 | True | ok | nan |
| null_arm_a | 15 | twin | 13 | True | ok | nan |
| null_arm_a | 15 | twin | 141 | True | ok | nan |
| null_arm_a | 15 | twin | 37 | True | ok | nan |
| null_arm_a | 15 | twin | 68 | True | ok | nan |
| null_arm_a | 15 | twin | 39 | True | ok | nan |
| null_arm_a | 15 | twin | 55 | True | ok | nan |
| null_arm_a | 15 | twin | 188 | True | ok | nan |
| null_arm_a | 21 | twin | 1 | True | ok | nan |
| null_arm_a | 21 | twin | 1 | True | ok | nan |
| null_arm_a | 21 | twin | 11 | True | ok | nan |
| null_arm_a | 21 | twin | 5 | True | ok | nan |
| null_arm_a | 21 | twin | 2 | True | ok | nan |
| null_arm_a | 21 | twin | 38 | True | ok | nan |
| null_arm_a | 21 | twin | 18 | True | ok | nan |
| null_arm_a | 21 | twin | 1 | True | ok | nan |
| null_arm_a | 21 | twin | 16 | True | ok | nan |
| null_arm_a | 21 | twin | 2 | True | ok | nan |
| null_arm_a | 21 | twin | 3 | True | ok | nan |
| null_arm_a | 21 | twin | 11 | True | ok | nan |
| null_arm_a | 21 | twin | 13 | True | ok | nan |
| null_arm_a | 21 | twin | 10 | True | ok | nan |
| null_arm_a | 21 | twin | 82 | True | ok | nan |
| null_arm_a | 21 | twin | 13 | True | ok | nan |
| null_arm_a | 21 | twin | 1 | True | ok | nan |
| null_arm_a | 21 | twin | 3 | True | ok | nan |
| null_arm_a | 21 | twin | 3 | True | ok | nan |
| null_arm_a | 21 | twin | 14 | True | ok | nan |
| null_arm_a | 21 | twin | 2 | True | ok | nan |
| null_arm_a | 21 | twin | 1 | True | ok | nan |
| null_arm_a | 21 | twin | 1 | True | ok | nan |
| null_arm_a | 21 | twin | 19 | True | ok | nan |
| null_arm_a | 21 | twin | 20 | True | ok | nan |
| null_arm_a | 24 | twin | 33 | True | ok | nan |
| null_arm_a | 24 | twin | 25 | True | ok | nan |
| null_arm_a | 24 | twin | 30 | True | ok | nan |
| null_arm_a | 24 | twin | 32 | True | ok | nan |
| null_arm_a | 24 | twin | 15 | True | ok | nan |
| null_arm_a | 24 | twin | 5 | True | ok | nan |
| null_arm_a | 24 | twin | 49 | True | ok | nan |
| null_arm_a | 24 | twin | 14 | True | ok | nan |
| null_arm_a | 24 | twin | 57 | True | ok | nan |
| null_arm_a | 24 | twin | 15 | True | ok | nan |
| null_arm_a | 24 | twin | 2 | True | ok | nan |
| null_arm_a | 24 | twin | 3 | True | ok | nan |
| null_arm_a | 24 | twin | 27 | True | ok | nan |
| null_arm_a | 24 | twin | 58 | True | ok | nan |
| null_arm_a | 24 | twin | 3 | True | ok | nan |
| null_arm_a | 24 | twin | 7 | True | ok | nan |
| null_arm_a | 24 | twin | 6 | True | ok | nan |
| null_arm_a | 24 | twin | 220 | True | ok | nan |
| null_arm_a | 24 | twin | 5 | True | ok | nan |
| null_arm_a | 24 | twin | 5 | True | ok | nan |
| null_arm_a | 24 | twin | 9 | True | ok | nan |
| null_arm_a | 24 | twin | 12 | True | ok | nan |
| null_arm_a | 24 | twin | 160 | True | ok | nan |
| null_arm_a | 24 | twin | 54 | True | ok | nan |
| null_arm_a | 24 | twin | 19 | True | ok | nan |

## 2 — Kept products (Arm A, printed formulas)

Top-150 by total gain on TOP ∪ top-150 on BOTTOM; union n=`252`. Top-30 of the union by (TOP+BOTTOM) total gain:

| rank | feature | formula | gain TOP | gain BOTTOM | gain sum |
|------|---------|---------|----------|-------------|----------|
| 1 | `P__mu_yz_vol_14__mu_idio_vol_60` | μ(yz_vol_14) × μ(idio_vol_60) | 2420.9 | 749873.3 | 752294.1 |
| 2 | `P__mu_pk_vol_14__mu_idio_vol_60` | μ(pk_vol_14) × μ(idio_vol_60) | 1678.5 | 431436.5 | 433115.0 |
| 3 | `P__nmu_idio_vol_60__mu_d_rank_90` | (1−μ(idio_vol_60)) × μ(d_rank_90) | 6653.5 | 355461.0 | 362114.5 |
| 4 | `P__nmu_pk_vol_14__mu_corr_btc_28` | (1−μ(pk_vol_14)) × μ(corr_btc_28) | 1800.6 | 330512.7 | 332313.3 |
| 5 | `P__nmu_yz_vol_30__mu_corr_btc_28` | (1−μ(yz_vol_30)) × μ(corr_btc_28) | 2733.5 | 277090.9 | 279824.4 |
| 6 | `P__nmu_dist_low_90__nmu_pk_vol_14` | (1−μ(dist_low_90)) × (1−μ(pk_vol_14)) | 2859.7 | 268706.0 | 271565.7 |
| 7 | `P__nmu_range_pos_28__nmu_dist_ath` | (1−μ(range_pos_28)) × (1−μ(dist_ath)) | 151995.4 | 6917.5 | 158912.9 |
| 8 | `P__nmu_dist_low_90__nmu_yz_vol_14` | (1−μ(dist_low_90)) × (1−μ(yz_vol_14)) | 3289.4 | 140851.1 | 144140.5 |
| 9 | `P__nmu_yz_vol_30__mu_d_rank_90` | (1−μ(yz_vol_30)) × μ(d_rank_90) | 6175.0 | 130149.0 | 136323.9 |
| 10 | `P__nmu_yz_vol_60__mu_d_rank_90` | (1−μ(yz_vol_60)) × μ(d_rank_90) | 4641.8 | 121697.8 | 126339.6 |
| 11 | `P__mu_corr_btc_28__mu_log_age` | μ(corr_btc_28) × μ(log_age) | 6180.3 | 118292.7 | 124473.0 |
| 12 | `P__nmu_close_sma100__mu_log_age` | (1−μ(close_sma100)) × μ(log_age) | 9348.0 | 96762.0 | 106110.1 |
| 13 | `P__mu_close_sma100__nmu_dist_high_90` | μ(close_sma100) × (1−μ(dist_high_90)) | 81740.6 | 11364.1 | 93104.7 |
| 14 | `P__nmu_close_sma20__mu_log_age` | (1−μ(close_sma20)) × μ(log_age) | 2214.4 | 89288.0 | 91502.4 |
| 15 | `P__nmu_pk_vol_14__nmu_idio_vol_60` | (1−μ(pk_vol_14)) × (1−μ(idio_vol_60)) | 1918.7 | 87000.4 | 88919.1 |
| 16 | `P__nmu_corr_btc_28__nmu_log_age` | (1−μ(corr_btc_28)) × (1−μ(log_age)) | 9609.3 | 76973.9 | 86583.2 |
| 17 | `P__mu_log_age__nmu_log_age` | μ(log_age) × (1−μ(log_age)) | 39456.1 | 37272.4 | 76728.5 |
| 18 | `P__nmu_idio_vol_60__mu_log_age` | (1−μ(idio_vol_60)) × μ(log_age) | 8413.3 | 58549.8 | 66963.1 |
| 19 | `P__nmu_idio_vol_60__mu_d_rank_30` | (1−μ(idio_vol_60)) × μ(d_rank_30) | 7035.6 | 58340.0 | 65375.6 |
| 20 | `P__mu_dist_low_90__mu_yz_vol_60` | μ(dist_low_90) × μ(yz_vol_60) | 3864.1 | 61106.5 | 64970.6 |
| 21 | `P__nmu_dist_high_90__nmu_dist_ath` | (1−μ(dist_high_90)) × (1−μ(dist_ath)) | 54108.0 | 8083.8 | 62191.8 |
| 22 | `P__mu_dist_low_90__mu_yz_vol_14` | μ(dist_low_90) × μ(yz_vol_14) | 1862.1 | 57223.1 | 59085.2 |
| 23 | `P__nmu_yz_vol_14__nmu_idio_vol_60` | (1−μ(yz_vol_14)) × (1−μ(idio_vol_60)) | 3945.6 | 55025.7 | 58971.3 |
| 24 | `P__nmu_idio_vol_60__nmu_turnover` | (1−μ(idio_vol_60)) × (1−μ(turnover)) | 4347.1 | 52504.6 | 56851.6 |
| 25 | `P__nmu_dist_low_90__nmu_idio_vol_60` | (1−μ(dist_low_90)) × (1−μ(idio_vol_60)) | 7125.4 | 46727.8 | 53853.2 |
| 26 | `P__nmu_close_sma20__nmu_yz_vol_60` | (1−μ(close_sma20)) × (1−μ(yz_vol_60)) | 1162.9 | 52558.4 | 53721.2 |
| 27 | `P__nmu_log_mcap__nmu_mcap_rank` | (1−μ(log_mcap)) × (1−μ(mcap_rank)) | 31606.8 | 20184.0 | 51790.8 |
| 28 | `P__nmu_yz_vol_60__mu_idio_vol_60` | (1−μ(yz_vol_60)) × μ(idio_vol_60) | 33057.5 | 17957.6 | 51015.1 |
| 29 | `P__nmu_log_mcap__mu_log_age` | (1−μ(log_mcap)) × μ(log_age) | 28525.6 | 22408.4 | 50934.0 |
| 30 | `P__nmu_beta_btc_60__nmu_amihud_14` | (1−μ(beta_btc_60)) × (1−μ(amihud_14)) | 12633.0 | 36872.5 | 49505.5 |

## 3 — Feature-importance gain share

- originals: `5.4%`
- library products: `94.6%`
- rule features: `0.0%` (RULE-FORGE `0.0%` / NFN `0.0%`)
- total gain (judged arm, both heads, all folds) = `22319573.65805483`

Chart: `charts/btcb_phase7b_gain_share.png`.

## 4 — Vol-matched null (best arm only)

Best arm = `arm_a`. Null design = vol-matched, folds {0,5,9,15,21,24} × 25.

**tail-IC(top-half)** verdict=`PARKED` bias_pass=True skill_pass=False exceed=1/6 violations=0 Stouffer z=`-0.251`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0041 | 0.0041 | 0.0163 | True | 0.0541 | -0.0117 | False |
| 5 | 25 | 0.0186 | 0.0186 | 0.0105 | True | 0.0525 | 0.0395 | False |
| 9 | 25 | 0.1655 | 0.1655 | 0.0138 | True | 0.2232 | 0.1085 | False |
| 15 | 25 | 0.1057 | 0.1057 | 0.0112 | True | 0.1387 | 0.0552 | False |
| 21 | 25 | 0.0805 | 0.0805 | 0.0180 | True | 0.1489 | 0.0781 | False |
| 24 | 25 | 0.1158 | 0.1158 | 0.0351 | True | 0.2477 | 0.3339 | True |

**overlap** verdict=`GREEN` bias_pass=True skill_pass=True exceed=3/6 violations=0 Stouffer z=`3.051`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0786 | 0.0786 | 0.0092 | True | 0.1137 | 0.0540 | False |
| 5 | 25 | 0.0603 | 0.0603 | 0.0034 | True | 0.0761 | 0.0984 | True |
| 9 | 25 | 0.0753 | 0.0753 | 0.0036 | True | 0.0914 | 0.1099 | True |
| 15 | 25 | 0.0968 | 0.0968 | 0.0049 | True | 0.1117 | 0.0808 | False |
| 21 | 25 | 0.1230 | 0.1230 | 0.0145 | True | 0.1731 | 0.1220 | False |
| 24 | 25 | 0.1695 | 0.1695 | 0.0182 | True | 0.2308 | 0.2440 | True |

**monster top-3** verdict=`GREEN` bias_pass=True skill_pass=True exceed=3/6 violations=0 Stouffer z=`3.140`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 0 | 25 | 0.0774 | 0.0774 | 0.0094 | True | 0.1099 | 0.0586 | False |
| 5 | 25 | 0.0497 | 0.0497 | 0.0043 | True | 0.0615 | 0.1099 | True |
| 9 | 25 | 0.0566 | 0.0566 | 0.0042 | True | 0.0725 | 0.0769 | True |
| 15 | 25 | 0.0951 | 0.0951 | 0.0065 | True | 0.1165 | 0.0623 | False |
| 21 | 25 | 0.1313 | 0.1313 | 0.0176 | True | 0.1956 | 0.1429 | False |
| 24 | 25 | 0.1146 | 0.1146 | 0.0139 | True | 0.1795 | 0.2088 | True |

## 5 — Tail-metric judgment grid (primary, per-date, floored top-100, Binance-listed)

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster top-3 | RankIC | vol-corr | n |
|--------|-------------|------|-------------|---------|---------------|--------|----------|---|
| frozen spread (baseline) | 0.0637 | 5.01 | 0.0970 | 0.0944 | 0.0815 | 0.1223 | -0.6765 | 2473 |
| ARM-A (product library) | 0.0614 | 4.72 | 0.0910 | 0.0996 | 0.0868 | 0.1259 | -0.6794 | 2473 |

Trailing-18m:

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | vol-corr |
|--------|-------------|------|-------------|---------|---------|--------|----------|
| frozen spread (baseline) | 0.1085 | 3.84 | 0.1209 | 0.1154 | 0.1022 | 0.1766 | -0.6151 |
| ARM-A (product library) | 0.1118 | 3.59 | 0.1128 | 0.1261 | 0.1071 | 0.1823 | -0.6500 |

Overlap by cycle:

| cycle | frozen spread (baseline) | ARM-A (product library) |
|-------|------|------|
| 2019-20 | 0.0529 | 0.0744 |
| 2021 | 0.0812 | 0.0700 |
| 2022 | 0.1156 | 0.1241 |
| 2023-24 | 0.0950 | 0.0938 |
| 2025-26 | 0.1205 | 0.1298 |

Tail-IC(top-half) by cycle:

| cycle | frozen spread (baseline) | ARM-A (product library) |
|-------|------|------|
| 2019-20 | 0.0074 | 0.0285 |
| 2021 | 0.0690 | 0.0420 |
| 2022 | 0.0960 | 0.1082 |
| 2023-24 | 0.0445 | 0.0233 |
| 2025-26 | 0.1078 | 0.1180 |

## 6 — Secondary: crude 14d book (information check, not adopted)

Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance.

| book | total | CAGR | MaxDD | Sharpe | n |
|------|-------|------|-------|--------|---|
| frozen spread (baseline) | 133.8% | 13.4% | -74.0% | 0.509 | 176 |
| ARM-A (product library) | 500.1% | 30.4% | -64.3% | 0.772 | 176 |

## 7 — Mechanical verdicts

- **ARM-A (product library): FAIL** (ΔIC `-0.0023` / Δov `+0.0052` / ΔRankIC `+0.0036`; clears_deltas=False null_pass=False beats_parents=False)
- Arm B: **STACK-SKIPPED** (STACK-SKIPPED reasons=['RULE-FORGE bank missing', 'NFN bank missing'])

- Ledger clause: **fuzzy-GBM composition on daily 33-features does not exceed the frozen spread; the daily composition question is closed.**

Mechanical, no post-hoc adjustment. Nothing adopted.

## Plain language

Arm A product library (252 kept of 2145) vs frozen spread: tail-IC(top-half) 0.06373363864654095 → 0.06139224332971208 (Δ -0.0023413953168288718), overlap 0.09440625421215797 → 0.09962066508770916 (Δ 0.00521441087555119). Best arm=arm_a. Vol-matched null did not pass. UNDERTRAINED count=246. Gain share originals=0.05438706890460848 products=0.9456129310953915 rules=0.0. Verdicts: ARM-A FAIL; ARM-B STACK-SKIPPED. fuzzy-GBM composition on daily 33-features does not exceed the frozen spread; the daily composition question is closed. Nothing adopted.

## Notes

- Frozen spread is the 2.c cache (not retrained). Arm A is one prune of the CDF product library.
- Vol-matched null is the house standard for tail metrics; run on the best fuzzy arm only.
- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**
- Elapsed s=`4523.8`. GPU=`False`.
- Charts: `charts/btcb_phase7b_tail_ic.png`, `charts/btcb_phase7b_gain_share.png`.

COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.

