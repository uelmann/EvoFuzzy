# BTC-BEATER LONG-TIDE — full-size long leg + frozen regime gate

**BACKTEST AND ANALYSIS ONLY.** No schedules, no live components, no retraining, no signal changes. CPU only, zero GPU. Frozen products (SPREAD-LS, COMBO, BTC-BEATER v1) untouched as products. Pricing = Binance (3.e canonical). Master only.

## Precondition (mechanical, checked first)

> EXECUTE ONLY IF Phase 3.e verdict = `SIGNAL-CONFIRMED`. Otherwise print `BLOCKED-BY-SUSPENSION: 3.e verdict is <verdict>` and STOP.

- 3.e verdict = **SIGNAL-CONFIRMED**
- Precondition pass = **True**

## Pre-registered criteria (verbatim, before results)

> LONG-TIDE is VIABLE if: (a) total return ≥ BTC B&H; (b) relative-line (book/BTC) Sharpe > 0; (c) MaxDD ≤ BTC B&H MaxDD. It SUPERSEDES BTC-BEATER v1 as the official long product only if additionally: (d) relative-line Sharpe ≥ v1's + 0.15 on the common window; (e) average alt deployment ≥ 15%; (f) no cycle with relative-line Sharpe < −0.30. If (a–c) pass but (d–f) do not, LONG-TIDE is recorded as a parallel long variant and v1 stays official. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Gate params byte-identical = `True` (breadth=0.5, off_hyst=5)
- Window 2019-10-19 → 2026-07-31 n=2478
- Common window with v1 2019-10-19 → 2026-07-31 n=2478
- GPU used = `False`

## Mechanical verdicts

- **LONG-TIDE is VIABLE**
- **LONG-TIDE SUPERSEDES BTC-BEATER v1**
- status = `SUPERSEDES-V1`
- (a) total 1917.9% ≥ BTC 691.3% → True
- (b) rel-line Sharpe 0.780 > 0 → True
- (c) MaxDD -73.2% ≤ BTC MaxDD -76.6% → True
- (d) rel-line Sharpe ≥ v1 0.302 + 0.15 = 0.452 → True
- (e) avg alt deployment 19.2% ≥ 15% → True
- (f) no cycle rel-line Sharpe < −0.30 (worst=0.155) → True

LONG-TIDE **SUPERSEDES** BTC-BEATER v1 as the official long product. v1 is demoted to record-only. SPREAD-LS (BOOK-HYBRID) is unchanged.

Mechanical, no post-hoc adjustment.

## Four-way comparison (identical window)

| book | total | CAGR | USD Sharpe | trail-18m | rel Sharpe | rel total | MaxDD | alt dep | gate ON | avg #names | ann TO | forced |
|------|-------|------|------------|-----------|------------|-----------|-------|---------|---------|------------|--------|--------|
| LONG-TIDE (spot-filter, gated, BN) | 1917.9% | 55.7% | 1.008 | -0.397 | 0.780 | 155.0% | -73.2% | 19.2% | 19.2% | 5.40 | 5.31 | 0 |
| NAKED LONG LEG (no gate, cash idle) | 615.7% | 33.6% | 0.775 | -0.389 | 0.188 | -9.6% | -81.9% | 99.7% | 100.0% | 21.23 | 6.65 | 4 |
| BTC-BEATER v1 (replayed read-only) | 858.7% | 39.5% | 0.859 | -0.631 | 0.302 | 21.7% | -79.0% | 8.7% | nan | 2.87 | 2.79 | 0 |
| BTC B&H (Binance BTCUSDT) | 691.3% | 35.6% | 0.811 | -0.545 | 0.000 | 0.0% | -76.6% | 0.0% | nan | 0.00 | 0.00 | 0 |
| EW floored top-100 (costless CMC) | -50.6% | -9.9% | 0.282 | -1.113 | -0.610 | -93.8% | -96.1% | 100.0% | nan | 0.00 | 0.00 | 0 |

Unrestricted-CMC reference (not judged):

| book | total | CAGR | USD Sharpe | trail-18m | rel Sharpe | rel total | MaxDD | alt dep | gate ON | avg #names | ann TO | forced |
|------|-------|------|------------|-----------|------------|-----------|-------|---------|---------|------------|--------|--------|
| LONG-TIDE CMC unrestricted (reference) | 2385.4% | 60.3% | 1.069 | -0.140 | 1.003 | 205.3% | -73.2% | 19.1% | 19.1% | 5.81 | 5.62 | 0 |

## Per-cycle honesty

| cycle | book | n | tot | CAGR | USD Sharpe | rel Sharpe | MaxDD | alt dep | #names |
|-------|------|---|-----|------|------------|------------|-------|---------|--------|
| 2019-20 | LONG-TIDE | 440 | 269.5% | 195.7% | 1.787 | 0.176 | -63.1% | 35.9% | 8.90 |
| 2019-20 | NAKED | 440 | 145.9% | 110.9% | 1.339 | -0.472 | -66.6% | 100.0% | 18.64 |
| 2019-20 | v1 | 440 | 265.1% | 192.8% | 1.914 | 0.117 | -51.9% | 3.7% | 1.19 |
| 2019-20 | BTC | 440 | 264.0% | 192.0% | 1.849 | 0.000 | -53.6% | 0.0% | 0.00 |
| 2021 | LONG-TIDE | 365 | 175.9% | 175.9% | 1.552 | 1.689 | -52.1% | 13.7% | 4.18 |
| 2021 | NAKED | 365 | 479.7% | 479.7% | 2.185 | 2.260 | -62.4% | 100.0% | 21.88 |
| 2021 | v1 | 365 | 130.1% | 130.1% | 1.428 | 1.496 | -44.7% | 25.5% | 7.73 |
| 2021 | BTC | 365 | 59.8% | 59.8% | 0.981 | 0.000 | -53.1% | 0.0% | 0.00 |
| 2022 | LONG-TIDE | 365 | -58.9% | -58.9% | -1.049 | 1.898 | -62.0% | 9.9% | 2.49 |
| 2022 | NAKED | 365 | -68.6% | -68.6% | -0.946 | -0.098 | -70.0% | 100.0% | 21.39 |
| 2022 | v1 | 365 | -64.5% | -64.5% | -1.300 | -0.147 | -67.0% | 7.5% | 3.42 |
| 2022 | BTC | 365 | -64.2% | -64.2% | -1.288 | 0.000 | -66.9% | 0.0% | 0.00 |
| 2023-24 | LONG-TIDE | 731 | 603.2% | 164.8% | 2.220 | 1.107 | -27.3% | 11.5% | 4.20 |
| 2023-24 | NAKED | 731 | 202.6% | 73.8% | 1.249 | -0.708 | -45.9% | 100.0% | 22.34 |
| 2023-24 | v1 | 731 | 418.3% | 127.4% | 1.942 | -1.092 | -26.2% | 2.9% | 1.00 |
| 2023-24 | BTC | 731 | 465.7% | 137.6% | 2.017 | 0.000 | -26.2% | 0.0% | 0.00 |
| 2025-26 | LONG-TIDE | 577 | -31.5% | -21.3% | -0.238 | 0.155 | -55.1% | 25.6% | 6.87 |
| 2025-26 | NAKED | 577 | -47.1% | -33.1% | -0.385 | -0.363 | -56.9% | 100.0% | 21.28 |
| 2025-26 | v1 | 577 | -37.9% | -26.0% | -0.443 | -0.420 | -56.2% | 10.0% | 3.11 |
| 2025-26 | BTC | 577 | -32.8% | -22.2% | -0.358 | 0.000 | -53.0% | 0.0% | 0.00 |

## Gate ribbon

| metric | value |
|--------|-------|
| REGIME_BREADTH | 0.5 |
| REGIME_OFF_HYSTERESIS | 5 |
| byte-identical to frozen Stage-T | True |
| % days gate ON | 19.2% |
| avg alt deployment | 19.2% |
| n ON stretches | 21 |
| mean ON length (days) | 22.7 |

## Correlations (daily PnL)

- LONG-TIDE vs BTC-BEATER v1: corr=`0.946` n=2478
- LONG-TIDE vs SPREAD-LS (BOOK-HYBRID): corr=`0.098` n=2478

## Squeeze days (EW floored top-100 vs LONG-TIDE)

| date | EW top-100 | LONG-TIDE |
|------|------------|-----------|
| 2021-05-24 | 25.5% | 22.2% |
| 2020-03-19 | 19.9% | 14.3% |
| 2022-11-10 | 18.3% | 10.5% |
| 2021-05-20 | 17.4% | 17.4% |
| 2021-04-26 | 15.5% | 11.3% |
| 2024-11-06 | 14.3% | 8.9% |
| 2021-09-22 | 14.0% | 6.9% |
| 2024-08-08 | 13.6% | 11.9% |
| 2025-11-07 | 13.1% | 2.0% |
| 2025-04-09 | 13.0% | 8.2% |
| 2020-03-13 | 13.0% | 16.5% |
| 2025-05-08 | 12.7% | 12.0% |
| 2025-08-22 | 12.0% | 7.9% |
| 2021-03-01 | 11.9% | 9.9% |
| 2021-05-26 | 11.8% | 7.3% |
| 2021-01-28 | 11.5% | 9.9% |
| 2022-02-28 | 11.4% | 14.5% |
| 2025-10-12 | 11.3% | 6.1% |
| 2021-02-05 | 11.3% | 3.7% |
| 2025-03-02 | 11.2% | 9.5% |

## Forced exits (death convention)

- n_events=0 n_ids=0 weight_sum=0.0000 cost_drag=0.000000
- symbols=[]

## Gate value (MaxDD vs naked leg)

LONG-TIDE MaxDD `-73.2%` vs NAKED MaxDD `-81.9%` (Δ `8.7%`). Positive Δ (tide less negative) is the gate's drawdown value in one number.

Elapsed s=218.8. GPU=False.

COMBO untouched (v2.0-combo-final). SPREAD-LS BOOK-HYBRID untouched as the official long/short product. BTC-BEATER v1 replayed read-only.

