# Official numbers ledger — causal (training-window) τ

**House standard:** every portfolio number in this ledger and all future tasks uses **training-window τ** (`fold_train`). Full-OOS / pooled-τ numbers are deprecated and must not be quoted as current.

Source: Phase D.2 (`reports/phaseD2_report.md`), identical-days median-τ headlines. Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`. Tranche, funding on, lag 0. Top-40 books use liquidity-tiered costs and the 0.5% ADV cap (nominal book USD 1,000,000).

Status: **ADOPTED** = P1 (reference book, A0 top-20 h=7) and P2 (adopted execution universe, A0 top-40 h=10). P4 is a D.2 reference row only — micro was **not** adopted.

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
