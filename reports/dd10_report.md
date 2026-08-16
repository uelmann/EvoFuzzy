# DD10-LO — P(top 10% of ret/avgDD) vs frozen A0 (long-only)

**BACKTEST AND ANALYSIS ONLY.** Same 33 A0 features. Frozen A0 Huber scores are **not retrained**. DD10 is a new binary LightGBM on a working copy of the labels. No schedules, no live components. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**.

**Frozen A0 SHA256 (features/config):** `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
**Mandate:** long-only top 10% of PIT top-40 by score. Residual is cash. No hedge.
**Horizon:** h=10. Label = 1 iff h=10 USDT simple return / forward average drawdown is in that date's PIT-120 top decile.

## Pre-registered improvement statement

> DD10-LO IMPROVES on A0-LO10 only if ALL of: (a) pooled OOS RankIC of DD10 predicted probability vs the continuous h=10 USDT-return / forward-average-drawdown ratio exceeds A0's RankIC vs the same ratio; (b) DD10 top-decile minus universe 10-day USDT simple return exceeds A0-LO10's; (c) DD10 long-only net Sharpe exceeds A0-LO10 net Sharpe; (d) the DD10 label-shuffle null is GREEN. This is an A/B target test, not a replacement for COMBO. No post-hoc adjustment.

## Pre-registered viability statement

> DD10-LO is VIABLE as a standalone long-only mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed gross ≥ 0.15; (e) the DD10 label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Target A/B: NO LIFT** — RankIC vs ratio DD10=0.023 vs A0=0.050 (pass=False); top−universe gap DD10=0.00545 vs A0=0.00810 (pass=False); Sharpe DD10=0.282 vs A0=0.263 (pass=True); null=GREEN pass=True.
- **DD10-LO: NOT VIABLE** — full Sharpe=0.282 (need ≥ 0.500, pass=False); trail-18m Sharpe=-0.033 (need ≥ 0.000, pass=False); total=-8.7% (need > 0, pass=False); avg gross=0.958 (need ≥ 0.150, pass=True); null=GREEN pass=True.

## Headline books

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | MaxDD | total | avg #longs | avg gross | % flat | funding | costs | ann TO | forced | BTC max |w| | % days BTC | top-5 name PnL |
|------|------|-----------|------|------|------|------|------|------|-------|-------|------------|-----------|--------|---------|-------|--------|--------|----------------|------------|----------------|
| A0-LO10 | 0.263 | -0.445 | -0.598 | 1.999 | 0.611 | -0.295 | -0.569 | -5.6% | -69.8% | -22.7% | 10.948 | 0.933 | 0.0% | 0.077 | 0.151 | 25.934 | 10 | 0.523046 | 57.2% | XRPUSDT=0.557, FTMUSDT=-0.206, SOLUSDT=0.157, ADAUSDT=0.139, DOGEUSDT=0.131 |
| DD10-LO | 0.282 | -0.033 | -1.068 | 2.104 | 0.406 | 0.006 | 0.076 | -2.0% | -71.5% | -8.7% | 9.256 | 0.958 | 0.0% | 0.339 | 0.122 | 22.605 | 0 | 0.703614 | 69.1% | XRPUSDT=0.407, LTCUSDT=-0.218, SIRENUSDT=0.187, ETHUSDT=-0.179, BNBUSDT=0.170 |

## RankIC vs ratio and vs USDT return (PIT top-40)

| arm | RankIC vs ratio | ICIR | NW-t | n | RankIC vs USDT | top−uni USDT | % gap>0 | gap NW-t |
|-----|-----------------|------|------|---|----------------|--------------|---------|----------|
| A0-LO10 | 0.050 | 4.118 | 5.154 | 1620 | 0.075 | 0.00810 | 50.5% | 1.799 |
| DD10 | 0.023 | 1.617 | 1.903 | 1620 | 0.015 | 0.00545 | 50.9% | 0.975 |

## DD10 label-shuffle null (vs binary y)

Verdict **GREEN** (bias_pass=True, skill_pass=True, n_violate=0, n_folds=2).

| fold | n | null mean | SD | p95 | real RankIC | bias_ok | exceeds p95 |
|------|---|-----------|----|-----|-------------|---------|-------------|
| 0 | 10 | 0.004 | 0.023 | 0.037 | 0.052 | True | True |
| 1 | 10 | 0.007 | 0.016 | 0.030 | 0.035 | True | True |

### Costless benchmarks (not viability inputs)

- EW PIT top-40 (daily rebalanced, costless): Sharpe full=-0.129, trail-18m=-0.713, CAGR=-36.5%, MaxDD=-90.5%, total=-86.7%.
- BTC B&H: Sharpe full=0.494, trail-18m=-0.471, CAGR=13.0%, MaxDD=-66.7%, total=71.9%.

## Construction notes

Frozen A0 Huber scores vs new binary LightGBM on P(top-decile R/ADD); last-fold-wins; n_folds=18; used_fixed=False; seed_determinism=True max_diff=0.0; n_a0=169325 n_dd10=169246; mean_y=0.1012; write root=/data/quant/dd10; frozen features sha256=f01dca8b8647b60f520721bd574bc23aa252aa602f63e52c7f34b25ce7263fb6.

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified by this run. Frozen A0 Huber scores and `features_labeled.parquet` are read-only. DD10-LO is a parallel A/B test. No outcome here rewrites the system card.

Elapsed seconds: 196.0. GPU used: false. Scheduled jobs created: false. DD10 fallback 500 trees: False. Mean label rate: 0.1012.
