# Round F5 — top-40 sleeve stack (pruned + context) and COMBO update

- Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
- Scope: backtest only; zero GPU; causal (training-window) τ house standard.
- Ledger: `reports/numbers_ledger.md`
- Addendum (criteria frozen before results): `reports/roundF5_addendum.md`
- Context features reused from Round F volume cache. P2′ = 32 features (A0−8+7 ctx).

**Mechanical:** selected sleeve **C0** (INCUMBENT); COMBO′ **ADOPTED** → reference **COMBO′**.

## Pre-registered sleeve selection rule (verbatim, before results)

> A candidate replaces the incumbent P2 sleeve only if its trailing-18m net Sharpe ≥ incumbent + 0.15 AND its full-period net Sharpe ≥ incumbent − 0.10 AND (for C3 only) its paired ΔRankIC vs plain A0 on top-40 satisfies the house block criterion at h=10 or h=7 (trail ≥ +0.005, full ≥ 0, ≥60% positive trailing folds). Among qualifying candidates, the sleeve with the highest trailing-18m net Sharpe is selected. If none qualify, the incumbent P2 stays. The +0.15 hurdle exists because four candidates are compared on a 548-day window; no post-hoc adjustment.

## Pre-registered COMBO′ rule (verbatim, before results)

> COMBO′ becomes the reference book only if its trailing-18m net Sharpe ≥ COMBO trailing − 0.05 AND its full-period net Sharpe ≥ COMBO full − 0.05, where COMBO is the Round-F adopted book (full 1.711, trail 0.997). Otherwise the Round-F COMBO stays the reference.

## Gates

- `label_shuffle`: **PASS**
- `feature_lookahead`: **PASS**
- `universe_lookahead_top20`: **PASS**
- `universe_lookahead_top40`: **PASS**
- `universe_lookahead_top120`: **PASS**
- `seed_determinism`: **PASS**

## Four-candidate P2 sleeve table (top-40, h=10, causal median-τ, identical days)

C0 uses ledger τ=70. C1/C2/C3 pick own causal median-τ. Round F published: C0 1.470/0.723, C1 1.257/1.045, C2 1.314/1.120.

| id | model | τ | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | gross | cost | funding | hedge | avg #pos | % flat | ann to |
|----|-------|---|------|-----------|------|------|------|------|------|-------|------|---------|-------|----------|--------|--------|
| C0 | C0 incumbent (A0) | 70.0 | 1.470 | 0.723 | 2.317 | 1.547 | 3.445 | 1.216 | 0.241 | 2.482 | 0.1677 | 0.2685 | 0.2344 | 16.41 | 0.00 | 32.23 |
| C1 | C1 A0+context (F1) | 70.0 | 1.257 | 1.045 | 1.061 | 1.591 | 2.405 | 1.141 | 1.290 | 2.696 | 0.1189 | -0.3933 | 0.2293 | 21.66 | 0.01 | 24.75 |
| C2 | C2 A0 pruned (F4) | 60.0 | 1.314 | 1.120 | 0.608 | 1.707 | 2.529 | 1.988 | 0.297 | 1.730 | 0.1655 | 0.2900 | 0.3386 | 21.26 | 0.00 | 31.29 |
| C3 | C3 P2′ (pruned+context) | 60.0 | 1.123 | 0.878 | 1.110 | 0.500 | 2.839 | 1.640 | 0.416 | 1.886 | 0.1270 | -0.2287 | 0.3212 | 26.27 | 0.01 | 26.29 |

% flat by year:

| id | 2022 | 2023 | 2024 | 2025 | 2026 |
|----|------|------|------|------|------|
| C0 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |
| C1 | 0.003 | 0.000 | 0.005 | 0.000 | 0.033 |
| C2 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |
| C3 | 0.003 | 0.000 | 0.000 | 0.025 | 0.000 |

Avg #positions by year:

| id | 2022 | 2023 | 2024 | 2025 | 2026 |
|----|------|------|------|------|------|
| C0 | 22.63 | 16.37 | 13.48 | 13.60 | 16.32 |
| C1 | 30.30 | 19.66 | 17.17 | 25.92 | 9.73 |
| C2 | 26.57 | 21.25 | 18.64 | 18.62 | 21.84 |
| C3 | 32.37 | 26.73 | 23.65 | 27.42 | 16.67 |

## ΔRankIC (top-40)

| pair | h | window | A IC | B IC | ΔIC | n_days |
|------|---|--------|------|------|-----|--------|
| C3_vs_A0 | 7 | full | 0.0792 | 0.0940 | 0.0148 | 961 |
| C3_vs_A0 | 7 | trail18m | 0.0921 | 0.1433 | 0.0512 | 315 |
| C3_vs_A0 | 7 | y2022 | 0.0590 | 0.0203 | -0.0386 | 243 |
| C3_vs_A0 | 7 | y2023 | 0.0739 | 0.0702 | -0.0037 | 275 |
| C3_vs_A0 | 7 | y2024 | 0.0798 | 0.1531 | 0.0733 | 134 |
| C3_vs_A0 | 7 | y2025 | 0.0964 | 0.1729 | 0.0765 | 193 |
| C3_vs_A0 | 7 | y2026 | 0.0932 | 0.1052 | 0.0120 | 116 |
| C3_vs_A0 | 10 | full | 0.0811 | 0.1062 | 0.0251 | 1620 |
| C3_vs_A0 | 10 | trail18m | 0.0943 | 0.1277 | 0.0334 | 548 |
| C3_vs_A0 | 10 | y2022 | 0.0956 | 0.0564 | -0.0392 | 344 |
| C3_vs_A0 | 10 | y2023 | 0.0598 | 0.1095 | 0.0497 | 365 |
| C3_vs_A0 | 10 | y2024 | 0.0686 | 0.1175 | 0.0489 | 366 |
| C3_vs_A0 | 10 | y2025 | 0.1064 | 0.1539 | 0.0475 | 365 |
| C3_vs_A0 | 10 | y2026 | 0.0707 | 0.0747 | 0.0040 | 180 |
| C3_vs_C1 | 7 | full | 0.1128 | 0.0940 | -0.0188 | 898 |
| C3_vs_C1 | 7 | trail18m | 0.1692 | 0.1433 | -0.0260 | 259 |
| C3_vs_C1 | 7 | y2022 | -0.0475 | 0.0203 | 0.0678 | 104 |
| C3_vs_C1 | 7 | y2023 | 0.0783 | 0.0702 | -0.0081 | 275 |
| C3_vs_C1 | 7 | y2024 | 0.1850 | 0.1531 | -0.0319 | 134 |
| C3_vs_C1 | 7 | y2025 | 0.1776 | 0.1729 | -0.0047 | 178 |
| C3_vs_C1 | 7 | y2026 | 0.1597 | 0.1052 | -0.0545 | 75 |
| C3_vs_C1 | 10 | full | 0.1127 | 0.1062 | -0.0065 | 1620 |
| C3_vs_C1 | 10 | trail18m | 0.1260 | 0.1277 | 0.0017 | 548 |
| C3_vs_C1 | 10 | y2022 | 0.0707 | 0.0564 | -0.0143 | 344 |
| C3_vs_C1 | 10 | y2023 | 0.1140 | 0.1095 | -0.0045 | 365 |
| C3_vs_C1 | 10 | y2024 | 0.1307 | 0.1175 | -0.0132 | 366 |
| C3_vs_C1 | 10 | y2025 | 0.1629 | 0.1539 | -0.0090 | 365 |
| C3_vs_C1 | 10 | y2026 | 0.0517 | 0.0747 | 0.0230 | 180 |
| C3_vs_C2 | 7 | full | 0.0853 | 0.0940 | 0.0087 | 961 |
| C3_vs_C2 | 7 | trail18m | 0.1096 | 0.1433 | 0.0337 | 315 |
| C3_vs_C2 | 7 | y2022 | 0.0624 | 0.0203 | -0.0421 | 243 |
| C3_vs_C2 | 7 | y2023 | 0.0703 | 0.0702 | -0.0001 | 275 |
| C3_vs_C2 | 7 | y2024 | 0.0824 | 0.1531 | 0.0707 | 134 |
| C3_vs_C2 | 7 | y2025 | 0.1228 | 0.1729 | 0.0501 | 193 |
| C3_vs_C2 | 7 | y2026 | 0.0898 | 0.1052 | 0.0154 | 116 |
| C3_vs_C2 | 10 | full | 0.0777 | 0.1062 | 0.0284 | 1620 |
| C3_vs_C2 | 10 | trail18m | 0.1006 | 0.1277 | 0.0271 | 548 |
| C3_vs_C2 | 10 | y2022 | 0.0737 | 0.0564 | -0.0173 | 344 |
| C3_vs_C2 | 10 | y2023 | 0.0603 | 0.1095 | 0.0492 | 365 |
| C3_vs_C2 | 10 | y2024 | 0.0646 | 0.1175 | 0.0529 | 366 |
| C3_vs_C2 | 10 | y2025 | 0.1145 | 0.1539 | 0.0394 | 365 |
| C3_vs_C2 | 10 | y2026 | 0.0732 | 0.0747 | 0.0016 | 180 |

### Paired NW t and fold fraction

| pair | h | window | mean ΔIC | NW-t | n | frac+ trail18m folds |
|------|---|--------|----------|------|---|----------------------|
| C3_vs_A0 | 7 | full | 0.0025 | 0.16 | 961 | nan |
| C3_vs_A0 | 7 | trail18m | 0.0445 | 1.73 | 315 | 0.714 |
| C3_vs_A0 | 10 | full | 0.0251 | 2.30 | 1620 | nan |
| C3_vs_A0 | 10 | trail18m | 0.0334 | 1.83 | 548 | 0.857 |
| C3_vs_C1 | 7 | full | 0.0180 | 1.46 | 700 | nan |
| C3_vs_C1 | 7 | trail18m | 0.0063 | 0.25 | 241 | 0.333 |
| C3_vs_C1 | 10 | full | -0.0065 | -1.05 | 1620 | nan |
| C3_vs_C1 | 10 | trail18m | 0.0017 | 0.15 | 548 | 0.429 |
| C3_vs_C2 | 7 | full | -0.0116 | -0.81 | 961 | nan |
| C3_vs_C2 | 7 | trail18m | 0.0129 | 0.58 | 315 | 0.571 |
| C3_vs_C2 | 10 | full | 0.0284 | 2.68 | 1620 | nan |
| C3_vs_C2 | 10 | trail18m | 0.0271 | 1.42 | 548 | 0.714 |

C3 house-block IC gate (vs A0, top-40):

- h=7: Δtrail=0.0512 Δfull=0.0148 frac+=0.714 pass=True
- h=10: Δtrail=0.0334 Δfull=0.0251 frac+=0.857 pass=True
- **C3 IC gate: PASS**

## Mechanical sleeve selection

> A candidate replaces the incumbent P2 sleeve only if its trailing-18m net Sharpe ≥ incumbent + 0.15 AND its full-period net Sharpe ≥ incumbent − 0.10 AND (for C3 only) its paired ΔRankIC vs plain A0 on top-40 satisfies the house block criterion at h=10 or h=7 (trail ≥ +0.005, full ≥ 0, ≥60% positive trailing folds). Among qualifying candidates, the sleeve with the highest trailing-18m net Sharpe is selected. If none qualify, the incumbent P2 stays. The +0.15 hurdle exists because four candidates are compared on a 548-day window; no post-hoc adjustment.

Incumbent C0 trail=0.723 full=1.470; need trail ≥ 0.873 and full ≥ 1.370.

| id | trail-18m | full | sharpe_ok | ic_ok | qualify |
|----|-----------|------|-----------|-------|---------|
| C1 | 1.045 | 1.257 | False | True | False |
| C2 | 1.120 | 1.314 | False | True | False |
| C3 | 0.878 | 1.123 | False | True | False |

**Selected sleeve: C0** (verdict=INCUMBENT; qualifying=[]).

## Stability diagnostic (selected vs incumbent; information only)

Selected avg #pos=16.41 vs incumbent 16.41; % flat 0.001 vs 0.001.

| year | sel avg #pos | inc avg #pos | sel % flat | inc % flat |
|------|--------------|--------------|------------|------------|
| 2022 | 22.63 | 22.63 | 0.003 | 0.003 |
| 2023 | 16.37 | 16.37 | 0.000 | 0.000 |
| 2024 | 13.48 | 13.48 | 0.000 | 0.000 |
| 2025 | 13.60 | 13.60 | 0.000 | 0.000 |
| 2026 | 16.32 | 16.32 | 0.000 | 0.000 |

Daily |Δ position count|:

| book | mean | median | p90 | max | frac≥10 | frac≥20 |
|------|------|--------|-----|-----|---------|---------|
| selected | 0.97 | 1.00 | 2.00 | 14.0 | 0.002 | 0.000 |
| incumbent | 0.97 | 1.00 | 2.00 | 14.0 | 0.002 | 0.000 |

Information only; no verdict. Pathological = large daily |Δn| swings vs incumbent.

## COMBO′ vs COMBO

> COMBO′ becomes the reference book only if its trailing-18m net Sharpe ≥ COMBO trailing − 0.05 AND its full-period net Sharpe ≥ COMBO full − 0.05, where COMBO is the Round-F adopted book (full 1.711, trail 0.997). Otherwise the Round-F COMBO stays the reference.

**COMBO′ verdict: ADOPTED** → reference book = **COMBO′**

COMBO′ equals the Round-F COMBO because the selected P2 sleeve is still C0; the −0.05 band is satisfied by identity (full 1.711 / trail 0.997).

need trail ≥ 0.947 (COMBO trail 0.997−0.05); need full ≥ 1.661 (COMBO full 1.711−0.05).

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | MaxDD | corr sleeves | ann to |
|------|------|-----------|------|------|------|------|------|-------|--------------|--------|
| COMBO (Round F re-run) | 1.711 | 0.997 | 1.097 | 2.544 | 2.843 | 1.444 | 0.441 | -0.332 | 0.254 | 26.48 |
| COMBO′ | 1.711 | 0.997 | 1.097 | 2.544 | 2.843 | 1.444 | 0.441 | -0.332 | 0.254 | 26.48 |
| P1 | 1.216 | 1.009 | -0.329 | 2.755 | 1.391 | 1.148 | 0.721 |  |  |  |
| selected sleeve | 1.470 | 0.723 | 2.317 | 1.547 | 3.445 | 1.216 | 0.241 |  |  | 32.23 |

Ledger confirmation: all Round F5 portfolios used `tau_mode=fold_train` (causal).

## Ledger diff

```
## Deprecated pooled-τ numbers (do not quote)

These used τ from the **full OOS** `|score|` distribution (lookahead). Causal replacements:

| deprecated (pooled / full-OOS τ) | causal replacement | object |
|----------------------------------|--------------------|--------|
| **1.401** | **0.757** | A0 top-20 h=7, τ=60, net Sharpe |
| **1.476** | **1.207** | A0 top-20 h=7, median-τ (τ=80), net Sharpe |
| **B.1 / Phase D 2026 Sharpe −0.82** | **+0.72** | A0 top-20 h=7, P1 causal median-τ, calendar-2026 net Sharpe |

The 1.401 figure is the Phase D published tranche number and the D.2 isolation re-run (identical). The 1.476 figure is pooled τ=80 (the pooled median-τ pick). The −0.82 figure is Phase D’s 2026 net Sharpe under lookahead τ=60 (−0.815).

## Round F5 append (causal τ)

| row | status | model | universe | h | median-τ | net Sharpe full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | gross | cost | funding | hedge | avg #pos | % flat | ann turnover |
|-----|--------|-------|----------|---|----------|-----------------|-----------|------|------|------|------|------|-------|------|---------|-------|----------|--------|--------------|
| P2-C0 | ADOPTED P2 sleeve (incumbent) | A0 | top-40 | 10 | 70.0 | 1.470 | 0.723 | 2.317 | 1.547 | 3.445 | 1.216 | 0.241 | 2.482 | 0.1677 | 0.2685 | 0.2344 | 16.41 | 0.00 | 32.23 |
| COMBO′ | ADOPTED reference | 50/50 P1+C0 | mixed | 7+10 | causal | 1.711 | 0.997 | 1.097 | 2.544 | 2.843 | 1.444 | 0.441 | nan | nan | nan | nan | nan | nan | 26.48 |

Changelog (2026-08-13): Round F5: P2 sleeve INCUMBENT → C0 (full=1.470 trail=0.723); COMBO′ ADOPTED → reference COMBO′ (full=1.711 trail=0.997).
```

