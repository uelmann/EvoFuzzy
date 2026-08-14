# Long-only variants of the frozen COMBO system

**BACKTEST AND ANALYSIS ONLY.** Portfolio layer only. No schedules, no deployments, no live components. No model retraining, no feature changes, no τ re-optimization.

**Frozen A0 SHA256:** `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
**Prediction files (reused, not recomputed):** h7=`8d48ea5a2f4ba47df986b57977f0be6ece2376a9277723a60606bafe150cf3a1` h10=`74359bff9c68b345a531b96d42876d5b3c492800fab9bbdf9c11cb6f9e51f916`
**Reference book:** COMBO v2.0-combo-final (Sleeve A C0 + Sleeve B P2, causal τ). **UNCHANGED.** This evaluation does not rewrite the ledger or the system card.

Sizing: long half of `_size_book` (`0.5 * tg * iv / sum(iv_longs)`). The unused short-side 50% budget is **not** dumped onto longs. Utilization floats.
Exit convention matches the frozen tranche engine (`_hard_threshold_state`; `exit_hysteresis` is discarded there, as in the live COMBO books).

## Pre-registered viability statements

> LO-H is VIABLE as a standalone mandate only if its full-period net Sharpe ≥ 0.7 AND trailing-18m net Sharpe ≥ 0.3. LO-U is VIABLE as a standalone mandate only if its full-period regression alpha vs BTC B&H is positive with NW-t ≥ 2.0 AND trailing-18m alpha is positive. These are viability labels for a parallel product; no outcome changes the reference book. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment. Product verdicts are on **COMBO-LO-H** and **COMBO-LO-U**. Sleeve-level checks are supplementary.

## Mechanical verdicts

- **LO-H (COMBO-LO-H):** **NOT VIABLE** — full Sharpe=0.153 (need ≥ 0.700, pass=False); trail-18m Sharpe=0.157 (need ≥ 0.300, pass=False).
- **LO-U (COMBO-LO-U):** **NOT VIABLE** — full alpha_ann=-0.123 (need > 0, pass=False); NW-t=-0.957 (need ≥ 2.000, pass=False); trail-18m alpha_ann=-0.029 (need > 0, pass=False).

Sleeve-level (not the product verdict):

- LO-H Sleeve A: NOT VIABLE (full=-0.315, trail=-1.246)
- LO-H Sleeve B: NOT VIABLE (full=0.287, trail=0.271)
- LO-U Sleeve A: NOT VIABLE (α=-0.186, NW-t=-1.843, α18=-0.112)
- LO-U Sleeve B: NOT VIABLE (α=-0.060, NW-t=-0.290, α18=0.054)

## Headline books

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | MaxDD | avg #longs | avg gross (alpha) | % flat | funding PnL | ann TO | top-5 name PnL |
|------|------|-----------|------|------|------|------|------|------|-------|------------|-------------------|--------|-------------|--------|----------------|
| LO-H Sleeve A | -0.315 | -1.246 | -0.765 | 1.169 | -0.111 | -1.445 | -0.743 | -5.4% | -31.9% | 1.62 | 0.110 | 71.2% | 0.0731 | 6.88 | FTMUSDT=-0.2419, MOODENGUSDT=-0.1491, BNBUSDT=0.1186, BTCUSDT=-0.0802, ADAUSDT=-0.0791 |
| LO-H Sleeve B | 0.287 | 0.271 | 0.052 | 0.362 | 0.622 | 0.456 | 0.158 | 3.5% | -49.9% | 4.23 | 0.286 | 11.0% | 0.6279 | 17.21 | RIVERUSDT=-0.3145, XRPUSDT=0.2385, FTMUSDT=-0.1556, REEFUSDT=0.1528, SIRENUSDT=0.1407 |
| COMBO-LO-H | 0.153 | 0.157 | -0.354 | 0.644 | 0.563 | 0.239 | 0.122 | 0.8% | -28.4% | 5.85 | 0.198 | 11.0% | 0.3505 | 12.04 | FTMUSDT=-0.1988, RIVERUSDT=-0.1573, MOODENGUSDT=-0.1302, XRPUSDT=0.1193, BNBUSDT=0.0798 |
| LO-U Sleeve A | -0.589 | -1.447 | -1.695 | 2.538 | 0.259 | -1.621 | -1.088 | -14.9% | -65.9% | 1.62 | 0.110 | 71.2% | 0.0363 | 4.43 | FTMUSDT=-0.2419, MOODENGUSDT=-0.1491, BNBUSDT=0.1186, BTCUSDT=-0.0802, ADAUSDT=-0.0791 |
| LO-U Sleeve B | 0.087 | 0.014 | -1.415 | 1.055 | 1.247 | 0.445 | -0.274 | -5.7% | -59.2% | 4.23 | 0.286 | 11.0% | 0.5293 | 8.81 | RIVERUSDT=-0.3145, XRPUSDT=0.2385, FTMUSDT=-0.1556, REEFUSDT=0.1528, SIRENUSDT=0.1407 |
| COMBO-LO-U | -0.164 | -0.186 | -1.592 | 1.646 | 1.202 | -0.022 | -0.323 | -8.7% | -61.6% | 5.85 | 0.198 | 11.0% | 0.2828 | 6.62 | FTMUSDT=-0.1988, RIVERUSDT=-0.1573, MOODENGUSDT=-0.1302, XRPUSDT=0.1193, BNBUSDT=0.0798 |
| Reference Sleeve A | 1.216 | 1.009 | -0.329 | 2.755 | 1.391 | 1.148 | 0.721 | 33.8% | -22.1% | 1.44 | 0.355 | 29.8% | -0.2656 | 20.73 | REEFUSDT=0.3118, TRBUSDT=0.2545, FTMUSDT=-0.2342, BLZUSDT=0.2221, WIFUSDT=-0.1913 |
| Reference Sleeve B | 1.477 | 0.733 | 2.317 | 1.547 | 3.445 | 1.216 | 0.268 | 72.5% | -53.0% | 3.98 | 0.717 | 0.1% | 0.2706 | 32.23 | RIVERUSDT=-0.2988, REEFUSDT=0.2658, XRPUSDT=0.2492, WLDUSDT=0.2295, PUMPUSDT=0.1822 |
| Reference COMBO | 1.711 | 0.997 | 1.097 | 2.544 | 2.843 | 1.444 | 0.441 | 55.7% | -33.2% | 5.41 | 0.536 | 0.0% | 0.0025 | 26.48 | REEFUSDT=0.2888, FTMUSDT=-0.2004, TRBUSDT=0.1921, WLDUSDT=0.1575, XRPUSDT=0.1468 |

Funding PnL is the sum of daily −w·funding_rate (longs pay when the rate is positive). On long-only books this was **expected** to be negative; the realized sign is in the table (a diagnostic, not a viability input).
Avg gross (alpha) is mean Σ|w_i| over alt names only (ex-hedge). LO books do not renormalize the long half toward 1.0 when shorts are absent, so deployed gross sits near ~0.5 when every slot has ≥1 long, and lower when slots are empty.

## §2 Alpha vs BTC B&H (costless) and correlation with reference COMBO

OLS of the book's daily net return on BTC buy-and-hold simple returns. Newey–West t-stat on the intercept uses Bartlett weights with HAC lag = h (Sleeve A: 7; Sleeve B: 10; COMBO: 10). Alpha is annualized as daily intercept × 365.

| book | α_ann full | β full | NW-t α full | n | α_ann 18m | β 18m | NW-t α 18m | n 18m | corr vs ref COMBO |
|------|------------|--------|-------------|---|-----------|-------|------------|-------|-------------------|
| LO-H Sleeve A | -0.044 | -0.004 | -0.643 | 1616 | -0.065 | 0.002 | -1.454 | 548 | 0.215 |
| LO-H Sleeve B | 0.116 | 0.010 | 0.560 | 1616 | 0.161 | 0.010 | 0.322 | 548 | 0.575 |
| COMBO-LO-H | 0.036 | 0.003 | 0.302 | 1616 | 0.048 | 0.006 | 0.189 | 548 | 0.560 |
| LO-U Sleeve A | -0.186 | 0.206 | -1.843 | 1616 | -0.112 | 0.012 | -1.403 | 548 | 0.127 |
| LO-U Sleeve B | -0.060 | 0.388 | -0.290 | 1616 | 0.054 | 0.195 | 0.107 | 548 | 0.466 |
| COMBO-LO-U | -0.123 | 0.297 | -0.957 | 1616 | -0.029 | 0.103 | -0.110 | 548 | 0.401 |
| Reference Sleeve A | 0.324 | 0.010 | 2.626 | 1616 | 0.247 | 0.030 | 1.335 | 548 | 0.670 |
| Reference Sleeve B | 0.628 | 0.042 | 2.813 | 1616 | 0.463 | 0.061 | 0.846 | 548 | 0.888 |
| Reference COMBO | 0.476 | 0.026 | 3.387 | 1616 | 0.355 | 0.046 | 1.176 | 548 | 1.000 |

### Costless benchmarks (not viability inputs)

- BTC B&H: Sharpe full=0.494, trail-18m=-0.536, CAGR=12.9%, MaxDD=-66.7%, total=71.4%.
- EW PIT top-20 (daily rebalanced, costless): Sharpe full=-0.305, trail-18m=-1.079, CAGR=-44.2%, MaxDD=-95.3%, total=-92.5%.

## §3 Long/short attribution of the frozen reference book

Daily net = long-leg + short-leg + hedge-leg + funding − costs. Legs are sums of simple-return units (the same units as `daily_ret` in the tranche engine). `long_share_of_net` = long-leg / net; `long_share_of_alpha` = long / (long+short).

**One-liner:** the long legs contributed **-19.1%** of frozen reference COMBO total net PnL (long=-0.4082, net=2.1353).

### Full period

| book | long | short | hedge | funding | costs | net | long/net | long/(L+S) | recon gap |
|------|------|-------|-------|---------|-------|-----|----------|------------|-----------|
| Reference Sleeve A | -0.5578 | 2.3594 | -0.0174 | -0.2656 | 0.0753 | 1.4433 | -0.386 | -0.310 | 0.000000 |
| Reference Sleeve B | -0.2586 | 2.7493 | 0.2334 | 0.2706 | 0.1675 | 2.8272 | -0.091 | -0.104 | 0.000000 |
| Reference COMBO | -0.4082 | 2.5544 | 0.1080 | 0.0025 | 0.1214 | 2.1353 | -0.191 | -0.190 | 0.000000 |
| COMBO-LO-H | -0.4607 | 0.0000 | 0.3254 | 0.3505 | 0.0532 | 0.1620 | -2.844 | 1.000 | 0.000000 |
| COMBO-LO-U | -0.4607 | 0.0000 | 0.0000 | 0.2828 | 0.0345 | -0.2124 | 2.169 | 1.000 | -0.000000 |
| LO-H Sleeve A | -0.6137 | 0.0000 | 0.3655 | 0.0731 | 0.0240 | -0.1991 | 3.082 | 1.000 | 0.000000 |
| LO-H Sleeve B | -0.3077 | 0.0000 | 0.2853 | 0.6279 | 0.0823 | 0.5231 | -0.588 | 1.000 | -0.000000 |
| LO-U Sleeve A | -0.6137 | 0.0000 | 0.0000 | 0.0363 | 0.0162 | -0.5936 | 1.034 | 1.000 | 0.000000 |
| LO-U Sleeve B | -0.3077 | 0.0000 | 0.0000 | 0.5293 | 0.0528 | 0.1688 | -1.823 | 1.000 | -0.000000 |

### Per calendar year (reference COMBO and sleeves)

**Reference COMBO**

| year | long | short | hedge | funding | costs | net | long/net | long/(L+S) | recon gap |
|------|------|-------|-------|---------|-------|-----|----------|------------|-----------|
| 2022 | -0.6236 | 0.7566 | 0.0932 | 0.0020 | 0.0327 | 0.1955 | -3.190 | -4.688 | 0.000000 |
| 2023 | 0.3018 | 0.1934 | 0.1994 | -0.0887 | 0.0271 | 0.5788 | 0.521 | 0.610 | -0.000000 |
| 2024 | 0.2294 | 0.5876 | 0.0655 | -0.0152 | 0.0226 | 0.8448 | 0.271 | 0.281 | 0.000000 |
| 2025 | -0.0941 | 0.5863 | -0.0749 | 0.0328 | 0.0273 | 0.4229 | -0.223 | -0.191 | 0.000000 |
| 2026 | -0.2217 | 0.4304 | -0.1752 | 0.0716 | 0.0118 | 0.0934 | -2.375 | -1.062 | 0.000000 |

**Reference Sleeve A**

| year | long | short | hedge | funding | costs | net | long/net | long/(L+S) | recon gap |
|------|------|-------|-------|---------|-------|-----|----------|------------|-----------|
| 2022 | -0.6871 | 0.5762 | 0.0567 | 0.0098 | 0.0263 | -0.0707 | 9.714 | 6.194 | 0.000000 |
| 2023 | 0.2786 | 0.3753 | 0.1366 | -0.1232 | 0.0117 | 0.6556 | 0.425 | 0.426 | -0.000000 |
| 2024 | 0.0213 | 0.4487 | 0.0580 | -0.0171 | 0.0114 | 0.4995 | 0.043 | 0.045 | 0.000000 |
| 2025 | -0.1491 | 0.5111 | -0.0168 | -0.0447 | 0.0183 | 0.2821 | -0.529 | -0.412 | -0.000000 |
| 2026 | -0.0214 | 0.4482 | -0.2520 | -0.0903 | 0.0076 | 0.0768 | -0.279 | -0.050 | 0.000000 |

**Reference Sleeve B**

| year | long | short | hedge | funding | costs | net | long/net | long/(L+S) | recon gap |
|------|------|-------|-------|---------|-------|-----|----------|------------|-----------|
| 2022 | -0.5600 | 0.9370 | 0.1297 | -0.0058 | 0.0392 | 0.4616 | -1.213 | -1.486 | 0.000000 |
| 2023 | 0.3251 | 0.0115 | 0.2621 | -0.0542 | 0.0424 | 0.5020 | 0.648 | 0.966 | 0.000000 |
| 2024 | 0.4374 | 0.7266 | 0.0730 | -0.0133 | 0.0337 | 1.1901 | 0.368 | 0.376 | 0.000000 |
| 2025 | -0.0391 | 0.6616 | -0.1330 | 0.1103 | 0.0362 | 0.5636 | -0.069 | -0.063 | 0.000000 |
| 2026 | -0.4220 | 0.4127 | -0.0984 | 0.2335 | 0.0159 | 0.1099 | -3.839 | 45.537 | 0.000000 |

## Correlation with the reference COMBO

Corr vs reference COMBO: COMBO-LO-H=0.560, COMBO-LO-U=0.401 (Sleeve A LO-H=0.215, LO-U=0.127; Sleeve B LO-H=0.575, LO-U=0.466).

A high correlation means long-only is largely a substitute for the reference book; a low correlation means it diversifies. This is a description, not a keep/kill rule.

## Reference book is unchanged

The frozen COMBO (v2.0-combo-final) is the reference book. LO-H / LO-U are parallel product/mandate evaluations on the same frozen A0 scores, universes, and causal median-τ. No outcome here changes the reference book, the system card, or the numbers ledger.

Elapsed seconds: 123.8. GPU used: false. Scheduled jobs created: false.

