# ALPHAMINE-LO — formulaic features vs A0 (long-only)

**BACKTEST AND ANALYSIS ONLY.** Same LightGBM, same always-in top-10 long-only book. MINE adds fold-selected OHLCV formulas. No schedules, no live components. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**.

**Frozen A0 SHA256 (features/config):** `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
**Mandate:** long alts. Never BTC. Never a BTC hedge. Residual is cash.
**Horizon:** h=10. Execution universe: PIT top-40 excluding BTC. Always long top 10 by score; full tranche budget.

## Pre-registered improvement statement

> ALPHAMINE-LO IMPROVES on A0-LO only if ALL of: (a) pooled OOS RankIC of MINE exceeds A0; (b) MINE top-quintile minus universe 10-day USDT simple return exceeds A0's; (c) MINE long-only net Sharpe exceeds A0-LO net Sharpe; (d) BTC weight is identically 0 every day on both books; (e) the MINE label-shuffle null is GREEN. This is an A/B feature test, not a replacement for COMBO. No post-hoc adjustment.

## Pre-registered viability statement

> ALPHAMINE-LO is VIABLE as a standalone long-only mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed alt gross ≥ 0.15; (e) BTC weight is identically 0 every day; (f) the MINE label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Feature A/B: NO LIFT** — RankIC MINE=0.093 vs A0=0.082 (pass=True); top−universe gap MINE=0.00639 vs A0=0.00479 (pass=True); Sharpe MINE=-0.021 vs A0=0.114 (pass=False); BTC0=True; null=CONTAMINATED pass=False.
- **ALPHAMINE-LO (MINE book): NOT VIABLE** — full Sharpe=-0.021 (need ≥ 0.500, pass=False); trail-18m Sharpe=-0.693 (need ≥ 0.000, pass=False); total=-65.9% (need > 0, pass=False); avg gross=0.957 (need ≥ 0.150, pass=True); BTC weight ≡ 0 pass=True; null=CONTAMINATED pass=False.

## Headline books

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | MaxDD | total | avg #longs | avg gross | % flat | funding | costs | ann TO | forced | BTC max |w| | top-5 name PnL |
|------|------|-----------|------|------|------|------|------|------|-------|-------|------------|-----------|--------|---------|-------|--------|--------|----------------|----------------|
| A0-LO | 0.114 | -0.449 | -0.569 | 1.355 | 0.651 | -0.193 | -0.959 | -13.7% | -68.6% | -47.9% | 21.406 | 0.951 | 0.0% | -0.061 | 0.118 | 19.799 | 0 | 0.000000 | XRPUSDT=0.287, DOGEUSDT=0.153, SOLUSDT=0.128, ADAUSDT=0.126, BNBUSDT=0.107 |
| MINE-LO | -0.021 | -0.693 | -0.662 | 1.072 | 0.785 | -0.325 | -1.458 | -21.5% | -73.6% | -65.9% | 20.251 | 0.957 | 0.0% | -0.077 | 0.102 | 16.989 | 0 | 0.000000 | XRPUSDT=0.217, BNBUSDT=0.189, DOGEUSDT=0.127, FILUSDT=-0.102, FTMUSDT=-0.100 |

## RankIC and long-minus-universe (PIT top-40, BTC dropped)

| arm | RankIC | ICIR | NW-t | n days | top−uni mean | % gap>0 | gap NW-t |
|-----|--------|------|------|--------|--------------|---------|----------|
| A0 | 0.082 | 6.400 | 7.813 | 1620 | 0.00479 | 54.1% | 1.934 |
| MINE | 0.093 | 6.386 | 6.982 | 1620 | 0.00639 | 52.9% | 2.492 |

## MINE label-shuffle null

Verdict **CONTAMINATED** (bias_pass=False, skill_pass=False, n_violate=2, n_folds=2).

| fold | n | null mean | SD | p95 | real RankIC | bias_ok | exceeds p95 |
|------|---|-----------|----|-----|-------------|---------|-------------|
| 0 | 10 | 0.025 | 0.015 | 0.046 | 0.013 | False | False |
| 1 | 10 | 0.052 | 0.015 | 0.077 | 0.044 | False | False |

### Costless benchmarks (not viability inputs)

- EW PIT top-40 (daily rebalanced, costless): Sharpe full=-0.137, trail-18m=-0.713, CAGR=-37.5%, MaxDD=-91.2%, total=-87.6%.
- BTC B&H (not held): Sharpe full=0.494, trail-18m=-0.471, CAGR=13.0%, MaxDD=-66.7%, total=71.9%.

## Kept formulas (examples)

- `fold0: ts_std:20(sub(sign(ret),ret)) (hoIC=0.04361585365037351)`
- `fold0: div(high,sub(abs(high),open)) (hoIC=0.038009947237304775)`
- `fold1: sub(delay:40(dollar_volume),ts_max:20(dollar_volume)) (hoIC=0.05041228962511555)`
- `fold1: ts_sum:20(ts_sum:20(ts_sum:10(ts_min:20(vwap)))) (hoIC=0.022190348702421078)`
- `fold2: mul(div(dollar_volume,low),sub(open,high)) (hoIC=0.08571970019150836)`
- `fold2: ts_std:20(ts_min:20(ret)) (hoIC=0.058071381083050926)`
- `fold3: ts_corr:40(high,ts_corr:40(high,ts_sum:20(low))) (hoIC=0.08845635502630911)`
- `fold3: ts_corr:40(ts_sum:40(low),ts_sum:20(low)) (hoIC=0.07882163436190094)`
- `fold4: abs(div(mul(abs(div(mul(volume,low),abs(dollar_volume))),low),abs(dollar_volume))) (hoIC=0.10220651195066009)`
- `fold4: abs(div(mul(volume,low),abs(dollar_volume))) (hoIC=0.10216390809720519)`
- `fold5: mul(mul(add(neg(volume),delta:5(ret)),ts_max:20(ts_sum:5(vwap))),ts_max:20(ts_sum:5(close))) (hoIC=0.06632979184567553)`
- `fold6: ts_min:5(abs(div(low,vwap))) (hoIC=0.17111234557451988)`

Mean formulas per fold: 4.778. Folds with 0 formulas: 0.

## Construction notes

LightGBM A0 vs A0+GP formulas; last-fold-wins; n_folds=18; used_fixed_a0=False used_fixed_mine=False; seed_determinism=True max_diff=0.0; n_a0=167705 n_mine=167705; write root=/data/quant/alphamine.

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified by this run. ALPHAMINE-LO is a parallel A/B test. No outcome here rewrites the system card.

Elapsed seconds: 352.1. GPU used: false. Scheduled jobs created: false. A0 fallback 500 trees: False. MINE fallback 500 trees: False.
