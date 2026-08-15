# LONG-CASH — cash-financed alt-long (parallel product)

**BACKTEST AND ANALYSIS ONLY.** New LightGBM heads on frozen A0 features. No schedules, no live components. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**.

**Frozen A0 SHA256 (features/config):** `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
**Mandate:** long alts or cash. Never BTC. Never a BTC hedge.
**Horizon:** h=10. Execution universe: PIT top-40 excluding BTC. Enter if `er_hat > 0` and `p_up > 0.5`; min 3 / max 10 names; full tranche budget.

## Pre-registered viability statement

> LONG-CASH is VIABLE as a standalone cash-financed alt-long mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed alt gross ≥ 0.15; (e) BTC weight is identically 0 every day; (f) the Head-R label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **LONG-CASH: NOT VIABLE** — full Sharpe=-0.608 (need ≥ 0.500, pass=False); trail-18m Sharpe=-1.143 (need ≥ 0.000, pass=False); total=-73.6% (need > 0, pass=False); avg gross=0.271 (need ≥ 0.150, pass=True); BTC weight ≡ 0 pass=True; null=CONTAMINATED pass=False.

## Headline book

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | MaxDD | total | avg #longs | avg gross | % flat | funding | costs | ann TO | forced | BTC max |w| | top-5 name PnL |
|------|------|-----------|------|------|------|------|------|------|-------|-------|------------|-----------|--------|---------|-------|--------|--------|----------------|----------------|
| LONG-CASH | -0.608 | -1.143 | -0.475 | -0.663 | 0.054 | -1.697 | 2.317 | -26.0% | -80.6% | -73.6% | 5.57 | 0.271 | 66.9% | -0.0483 | 0.0316 | 5.22 | 0 | 0.000000 | ARBUSDT=-0.0796, SOLUSDT=-0.0767, AAVEUSDT=-0.0727, ETHUSDT=-0.0665, BNBUSDT=0.0579 |

Mean vs cash (ann.)=-0.229; NW-t vs cash (lag=10)=-1.504; trail NW-t=-1.791.

## Head-R label-shuffle null

Verdict **CONTAMINATED** (bias_pass=False, skill_pass=True, n_violate=2, n_folds=2).

| fold | n | null mean | SD | p95 | real RankIC | bias_ok | exceeds p95 |
|------|---|-----------|----|-----|-------------|---------|-------------|
| 0 | 10 | 0.0279 | 0.0201 | 0.0526 | 0.1047 | False | True |
| 1 | 10 | -0.0147 | 0.0154 | 0.0076 | 0.0355 | False | True |

## Raw-material snapshot (informational, not a gate)

Top-quintile mean of 10-day simple USDT return on PIT top-40 (BTC dropped).

| signal | n days | % days top>0 | mean top | NW-t |
|--------|--------|--------------|----------|------|
| frozen A0 score | 1620 | 48.5% | -0.0000 | -0.005 |
| LONG-CASH er_hat | 1579 | 46.4% | -0.0059 | -0.674 |

### Costless benchmarks (not viability inputs)

- EW PIT top-40 (daily rebalanced, costless): Sharpe full=-0.137, trail-18m=-0.713, CAGR=-37.5%, MaxDD=-91.2%, total=-87.6%.
- BTC B&H (not held): Sharpe full=0.494, trail-18m=-0.471, CAGR=13.0%, MaxDD=-66.7%, total=71.9%.

## Construction notes

Heads R/C LightGBM on frozen A0 33 features; last-fold-wins; n_folds=18; used_fixed_500=False; seed_determinism=True max_diff=0.0; n_merged=167994; write root=/data/quant/long_cash.

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified by this run. LONG-CASH is a parallel product. No outcome here rewrites the system card.

Elapsed seconds: 274.2. GPU used: false. Scheduled jobs created: false. Head-R fallback 500 trees: False.

