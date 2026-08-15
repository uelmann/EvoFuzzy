# Official numbers ledger — causal (training-window) τ

**House standard:** every portfolio number in this ledger and all future tasks uses **training-window τ** (`fold_train`). Full-OOS / pooled-τ numbers are deprecated and must not be quoted as current.

Source: Phase D.2 (`reports/phaseD2_report.md`), identical-days median-τ headlines. Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`. Tranche, funding on, lag 0. Top-40 books use liquidity-tiered costs and the 0.5% ADV cap (nominal book USD 1,000,000).

Status: **ADOPTED P2 sleeve** remains **C0** (A0, top-40, h=10, τ=70). **Reference book** = **COMBO** = 50/50 P1+P2 (full 1.711 / trail 0.997, causal τ). P1 is the top-20 sleeve. P4 is a D.2 reference row only — micro was **not** adopted. Round F5 did not replace the P2 sleeve.

| row | status | model | universe | h | median-τ | net Sharpe full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | gross | cost | funding | hedge | avg #pos | % flat | ann turnover |
|-----|--------|-------|----------|---|----------|-----------------|-----------|------|------|------|------|------|-------|------|---------|-------|----------|--------|--------------|
| **P1** | **ADOPTED reference** | A0 | top-20 | 7 | 80 | 1.207 | 1.009 | -0.370 | 2.755 | 1.391 | 1.148 | 0.721 | 1.776 | 0.0754 | -0.266 | -0.0009 | 6.04 | 0.30 | 20.73 |
| P1-h10 | informational | A0 | top-20 | 10 | 60 | 1.131 | 0.145 | 1.707 | 1.986 | 3.178 | 0.318 | 0.007 | 1.974 | 0.110 | 0.264 | 0.057 | 10.66 | 0.00 | 29.51 |
| **P2** | **ADOPTED universe** | A0 | top-40 | 10 | 70 | 1.470 | 0.723 | 2.317 | 1.547 | 3.445 | 1.216 | 0.241 | 2.482 | 0.168 | 0.269 | 0.234 | 16.41 | 0.00 | 32.23 |
| P2-h7 | informational | A0 | top-40 | 7 | 70 | 1.154 | 0.433 | -0.164 | 1.566 | 2.489 | 1.214 | -1.203 | 1.556 | 0.111 | -0.171 | -0.037 | 20.28 | 0.15 | 20.81 |
| P4 ref | not adopted | A0+micro | top-40 | 10 | 60 | 1.178 | 1.239 | 2.241 | -0.384 | 2.500 | 1.864 | -0.110 | 2.644 | 0.155 | -0.122 | -0.523 | 21.00 | 0.00 | 29.94 |

Gross / cost / funding / hedge are full-period sums of daily simple-return units. % flat is the fraction of days with zero alpha positions.

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

Program closed at v2.0-combo-final; reference book = COMBO (Round F, confirmed F5); all future numbers must cite this ledger.

## BTC-BEATER SPREAD-LS (Phase 3.c Binance replay)

Production book config: β-matched, h=14, floored PIT top-100 dollar-volume. Positions from the 2.c spread cache (signals not recomputed). COMBO overlap corr remains 0.157 for allocation. MASTER combination book is out of scope (PI).

**OFFICIAL SPREAD-LS = BOOK-HYBRID (funding-on).** Resumed by Phase 3.e **SIGNAL-CONFIRMED**. Binance-priced numbers are canonical for all future phases.

BOOK-HYBRID Sharpe `1.555` / trail `1.381`. RankIC BN `0.1517` vs same-names CMC `0.1542` (Δ `-0.0025`).

Footnote: funding-off CMC BOOK-CMC Sharpe `1.818` is **deprecated** as of Phase 3.e (signal confirmed on Binance returns; gap is execution/pricing-level).

## BTC-BEATER academic factor (analysis only, not adopted)

Unconstrained D10−D1 on CMC from the frozen 2.c spread. Labels diagnostic. Academic factor not adopted. Official SPREAD-LS is BOOK-HYBRID as of Phase 3.e SIGNAL-CONFIRMED.

**PAPER ALPHA EXISTS.** FACTOR-JT top-100 GROSS Sharpe `1.522` / NW-t `3.97` (need ≥ 1.0 and ≥ 3.0). n=2491, 2019-10-19→2026-08-13.

IMPLEMENTATION TAX (paper GROSS − 3.c hybrid) = `-0.033`. Waterfall Sharpe: `1.522` → naive `1.423` (Δ `-0.100`) → shortability `1.393` (Δ `-0.030`) → real costs `1.399` (Δ `+0.006`) → hybrid `1.555` (Δ `+0.156`).

Paper alpha lives mainly in the short side: universe−short Sharpe `1.637` (mean share 0.76) vs long−universe `0.770` (0.24). Corr vs BOOK-CMC `0.693` / vs hybrid `0.677`.

## BTC-BEATER LONG-TIDE

Full-size long leg + frozen Stage-T gate, BTC parking. Backtest only. Binance-priced (3.e canonical). 2.c spread cache reused. No shorts, no funding. SPREAD-LS BOOK-HYBRID remains the official long/short product. COMBO untouched.

**OFFICIAL long product = LONG-TIDE.** SUPERSEDES BTC-BEATER v1 (v1 demoted to record-only).

LONG-TIDE rel-line Sharpe `0.780` / USD Sharpe `1.008` / trail `-0.397` / total `1917.9%` / MaxDD `-73.2%` / alt deployment `19.2%` / gate ON `19.2%`. status=`SUPERSEDES-V1`. Window 2019-10-19→2026-07-31 n=2478.

Mechanical, no post-hoc adjustment.

## BTC-BEATER ORACLE LADDER

Perfect-foresight ceiling and IC-degraded oracle ladder. Analysis only. Nothing adopted. 14d full-rebalance long construction (not the production tranche books). Binance-priced.

**MODEL EFFICIENCY = BELOW-CURVE.** Binding constraint: **TRANSLATION**. Oracle NET h=14 total `1.546e+14` / CAGR `1.254e+04%` / MaxDD `-30.4%`. OUR MODEL RankIC `0.1160` CAGR `13.9%` (0.11% of oracle CAGR) vs curve `119.8%`.

Mechanical, no post-hoc adjustment. Frozen products untouched.
