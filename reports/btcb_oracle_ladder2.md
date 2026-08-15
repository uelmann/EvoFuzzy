# BTC-BEATER ORACLE LADDER 2 — tail-blindness vs translation slack

**ANALYSIS ONLY.** Nothing adopted. No retraining, no product changes. CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.

Splits the Ladder-1 BELOW-CURVE gap. White-noise oracles are tail-aware; our model is bottom-heavy.

## Pre-registered reading (verbatim, before results)

> The gap decomposition is: TAIL-INFORMATION share = the part explained by overlap/tail-IC deficits vs the equal-IC ladder; CONSTRUCTION share = the best of V1–V3 minus the crude base, plus the production-construction delta measured on the ladder signal. No variant is adopted here; any adoption requires a fresh pre-registered phase with the house criteria. If the best translation variant improves CAGR by ≥ +10pp over the crude base at comparable MaxDD, translation work is declared the next priority; otherwise the right-tail information hunt (catalysts/attention data) is declared the next priority. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Window 2019-10-20 → 2026-07-18 n=2464 formations=176
- GPU used = `False`

## Mechanical verdict

- **PRIORITY = TAIL-INFORMATION**
- gap (ladder-0.116 crude − model crude) `104.4pp`
- TAIL-INFORMATION `166.8pp` (overlap-implied CAGR `-48.6%` vs ladder `118.2%`)
- CONSTRUCTION `-51.5pp` (best variant `V3` Δ `0.7pp` + production Δ `-52.2pp`)
- UNEXPLAINED `-10.9pp`
- qualified variant (comparable MaxDD) `V3` Δ `0.7pp` (need ≥ +10pp)
- no V1–V3 variant clears +10pp CAGR at comparable MaxDD — **the right-tail information hunt (catalysts/attention data) is the next priority**.

Mechanical, no post-hoc adjustment. Nothing adopted.

## Plain language

of the 104.4pp gap, ~166.8pp is tail information, ~-51.5pp is construction, ~-10.9pp unexplained.

## 1 — Tail diagnostics

| signal | RankIC | top-decile overlap | tail-IC top half | tail-IC bottom half | bottom−top | monster top-3 |
|--------|--------|--------------------|------------------|---------------------|------------|---------------|
| OUR SPREAD | 0.1160 | 0.0917 | 0.0439 | 0.0956 | 0.0517 | 0.0795 |
| ladder-0.116 | 0.1161 [0.1159, 0.1162] | 0.1596 [0.1485, 0.1693] | 0.0769 [0.0668, 0.0887] | 0.0695 [0.0586, 0.0800] | -0.0075 [-0.0167, 0.0090] | 0.1909 [0.1818, 0.1989] |
| ladder-0.16 | 0.1601 [0.1599, 0.1603] | 0.1888 [0.1787, 0.1989] | 0.1079 [0.0965, 0.1147] | 0.0894 [0.0781, 0.1017] | -0.0185 [-0.0285, -0.0046] | 0.2330 [0.2178, 0.2462] |
| ORACLE | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

Overlap = fraction of the signal's top-decile picks that land in the realized top decile. Tail-IC = Spearman(score, next-14d excess) within the top / bottom half of the signal ranking. Monster = fraction of realized top-3 movers held in the signal's top-decile book.

| cycle | OUR overlap | ladder-0.116 overlap | ladder-0.16 overlap | ORACLE overlap |
|-------|-------------|----------------------|---------------------|----------------|
| 2019-20 | 0.0281 | 0.1835 | 0.2004 | 1.0000 |
| 2021 | 0.0788 | 0.1667 | 0.1971 | 1.0000 |
| 2022 | 0.1154 | 0.1538 | 0.1879 | 1.0000 |
| 2023-24 | 0.1179 | 0.1559 | 0.1825 | 1.0000 |
| 2025-26 | 0.1017 | 0.1445 | 0.1827 | 1.0000 |

## 2 — Construction slack

| book | total | CAGR | MaxDD | Sharpe | ann TO | avg #names | n |
|------|-------|------|-------|--------|--------|------------|---|
| OUR MODEL crude 14d (base) | 140.3% | 13.9% | -72.1% | 0.514 | 8.47 | 5.9 | 176 |
| V1 score-weighted | 106.6% | 11.3% | -80.6% | 0.481 | 10.83 | 5.9 | 176 |
| V2 concentrated top-5 | 118.5% | 12.3% | -87.0% | 0.552 | 14.44 | 5.0 | 176 |
| V3 tail-threshold p95 | 150.3% | 14.6% | -44.5% | 0.674 | 4.88 | 2.9 | 176 |
| ladder-0.116 crude 14d | 2.008e+02 | 118.2% | -53.0% | 1.648 | 14.04 | 5.9 | 176 |
| ladder-0.116 PRODUCTION (no gate) | 3.195e+01 | 66.0% | -78.8% | 1.026 | 17.81 | 17.9 | 2477 |

Production = h=14 tranches, k_enter=10 / k_stay=20, n_hold=10, cap 10%, anti-blowoff, BTC park. Not LONG-TIDE (no Stage-T gate). Production Δ on ladder-0.116 = `-52.2pp`.

## Per-cycle (NET)

| cycle | book | n | total | CAGR | MaxDD |
|-------|------|---|-------|------|-------|
| 2019-20 | BASE | 439 | 51.6% | 41.3% | -31.7% |
| 2021 | BASE | 365 | 212.7% | 212.7% | -40.9% |
| 2022 | BASE | 365 | -60.8% | -60.8% | -61.9% |
| 2023-24 | BASE | 731 | 80.0% | 34.1% | -25.9% |
| 2025-26 | BASE | 564 | -28.2% | -19.3% | -37.3% |
| 2019-20 | V1 | 439 | 53.6% | 42.9% | -44.3% |
| 2021 | V1 | 365 | 269.1% | 269.1% | -48.4% |
| 2022 | V1 | 365 | -72.3% | -72.3% | -73.2% |
| 2023-24 | V1 | 731 | 102.0% | 42.1% | -30.3% |
| 2025-26 | V1 | 564 | -35.0% | -24.3% | -44.7% |
| 2019-20 | V2 | 439 | 54.2% | 43.4% | -66.6% |
| 2021 | V2 | 365 | 414.6% | 414.6% | -58.8% |
| 2022 | V2 | 365 | -78.2% | -78.2% | -79.1% |
| 2023-24 | V2 | 731 | 157.0% | 60.2% | -39.8% |
| 2025-26 | V2 | 564 | -50.9% | -36.9% | -61.7% |
| 2019-20 | V3 | 439 | 47.5% | 38.1% | -8.4% |
| 2021 | V3 | 365 | 106.2% | 106.2% | -26.6% |
| 2022 | V3 | 365 | -37.0% | -37.0% | -40.1% |
| 2023-24 | V3 | 731 | 49.8% | 22.3% | -20.8% |
| 2025-26 | V3 | 564 | -12.7% | -8.4% | -19.1% |
| 2019-20 | LADDER crude | 439 | 229.0% | 169.2% | -29.1% |
| 2021 | LADDER crude | 365 | 688.7% | 688.7% | -41.1% |
| 2022 | LADDER crude | 365 | -40.7% | -40.7% | -52.2% |
| 2023-24 | LADDER crude | 731 | 600.6% | 164.3% | -27.4% |
| 2025-26 | LADDER crude | 564 | 32.1% | 19.7% | -33.4% |
| 2019-20 | LADDER prod | 439 | 220.2% | 163.2% | -64.8% |
| 2021 | LADDER prod | 365 | 1.159e+01 | 1158.6% | -62.2% |
| 2022 | LADDER prod | 365 | -72.0% | -72.0% | -75.4% |
| 2023-24 | LADDER prod | 731 | 433.7% | 130.8% | -55.7% |
| 2025-26 | LADDER prod | 577 | -61.4% | -45.3% | -71.3% |

## Notes

- Crude 14d is the Ladder-1 construction (idle cash). V1–V3 change only the weight rule.
- The brief's 33.6% figure is not a gate; the mechanical base is this run's crude book.
- Ladder diagnostics for 0.116 / 0.16 are 5-seed mean [range].
- Shares need not sum to the gap: overlap→CAGR linear extrapolation at the model's overlap (0.092) lands below zero (`CAGR* = -48.6%`), so the tail term exceeds the gap; production construction on the equal-IC ladder is **negative** (`118.2% → 66.0%`).
- Nothing is adopted. Any adoption needs a fresh pre-registered phase.

Elapsed s=215.5. GPU=False.

COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.

