# BTC-BEATER Phase 5 — CS-ATTN v0

**BACKTEST AND ANALYSIS ONLY.** No schedules, no live components, nothing adopted.
Frozen GBM / 2.c cache / COMBO / SPREAD-LS / LONG-TIDE untouched (read-only).
One architecture config, zero search. GPU = 12×A10G fold-parallel (H100 tried; per-batch matched A10G).

## Pre-registered criteria (verbatim, before results)

> CS-ATTN is LIVE if ALL of: (a) the 3-seed ensemble improves tail-IC(top-half) ≥ +0.010 AND top-decile overlap ≥ +0.015 vs the frozen GBM baseline on the full OOS; (b) seed dispersion is small: max−min of per-seed full-OOS tail-IC(top-half) ≤ 0.010; (c) the §B null passes. CS-ATTN is PARKED otherwise, and the verdict sentence must state which clause failed. If PARKED on clause (a) with dispersion passing, the conclusion 'the price/volume ceiling is real at this scale' is recorded in the project ledger. Nothing is adopted; any production use requires a fresh pre-registered phase. Mechanical, no post-hoc adjustment.

## §B null (verbatim, before results)

> Adapted E.1b on folds {5, 21} × 10 within-date label-shuffle replicates (train seed 42 only). (a) BIAS: for each fold, the null mean of per-date tail-IC(top-half) must satisfy |mean| ≤ 2·(null SD / √R). CONTAMINATED if ≥2 fold-level violations (original E.1b tolerance). (b) SKILL: seed-42 real fold tail-IC(top-half) must exceed that fold's null 95th percentile on both folds. §B null PASSES iff not CONTAMINATED AND skill passes.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical verdict

- **CS-ATTN = `PARKED`**
- failed clauses = `['a', 'b', 'c']`
- (a) Δ tail-IC(top-half) = `0.0081` (need ≥ `0.01`); Δ overlap = `0.0018` (need ≥ `0.015`); pass=`False`
- (b) seed dispersion max−min = `0.0234` (need ≤ `0.01`); per-seed = `[0.02899211485623889, 0.026772508354997056, 0.005628277445548386]`; pass=`False`
- (c) null = `CONTAMINATED` pass=`False`

Mechanical, no post-hoc adjustment. Nothing adopted.

## Plain language

CS-ATTN is PARKED. failed clauses ['a', 'b', 'c']. GBM tail-IC(top) 0.0445 vs ensemble 0.0527 (Δ 0.0081). overlap GBM 0.0912 vs ensemble 0.0930 (Δ 0.0018). seed dispersion 0.0234. null CONTAMINATED. GPU $67.45 / $80.

## Panel audit summary

- hourly rows=`16177940` ids=`532` span=`2019-01-01 00:00:00+00:00`→`2026-07-31 23:00:00+00:00`
- alignment median |Δ| bps=`13.707777572175006` pass=`False` violations=`526`
- sources=`{'perp': 76, 'spot': 456}` duplicates=`0`
- seq cache n=`202832` nbytes=`2044546560`

## Frozen config dump

```
{'seq_len': 504, 'n_channels': 5, 'channels': ['log_ret', 'hl_range', 'vol_z', 'taker_share', 'vs_btc'], 'tcn_blocks': 4, 'tcn_width': 64, 'tcn_kernel': 7, 'tcn_dilations': [1, 4, 16, 64], 'attn_layers': 2, 'attn_heads': 4, 'attn_width': 64, 'dropout': 0.1, 'lr': 0.0003, 'weight_decay': 0.01, 'max_epochs': 20, 'patience': 3, 'batch_dates': 16, 'set_n': 100, 'top_pos_weight': 3.0, 'vol_z_window': 168, 'min_bars_frac': 0.8, 'seeds': [42, 43, 44], 'null_folds': [5, 21], 'null_shuffle_seeds': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110], 'horizon': 14, 'architecture_search': False}
```

## Tail-metric grid (floored PIT top-100, h=14, Binance-listed)

### Full OOS

| signal | tail-IC top | NW-t | tail-IC bot | NW-t | overlap | monster top-3 | RankIC | n |
|--------|-------------|------|-------------|------|---------|---------------|--------|---|
| frozen GBM spread | 0.0445 | 2.00 | 0.0957 | 7.08 | 0.0912 | 0.0777 | 0.1160 | 176 |
| CS-ATTN seed 42 | 0.0290 | 1.33 | -0.0060 | -0.47 | 0.1034 | 0.1155 | 0.0301 | 165 |
| CS-ATTN seed 43 | 0.0268 | 1.48 | -0.0160 | -1.27 | 0.0938 | 0.0947 | 0.0337 | 167 |
| CS-ATTN seed 44 | 0.0056 | 0.31 | 0.0027 | 0.23 | 0.0948 | 0.0985 | 0.0293 | 164 |
| CS-ATTN 3-seed ensemble | 0.0527 | 2.95 | 0.0093 | 0.65 | 0.0930 | 0.1061 | 0.0435 | 176 |

### Trailing 18m

| signal | tail-IC top | NW-t | tail-IC bot | NW-t | overlap | monster top-3 | RankIC | n |
|--------|-------------|------|-------------|------|---------|---------------|--------|---|
| frozen GBM spread | 0.1011 | 2.50 | 0.0913 | 4.03 | 0.1017 | 0.0833 | 0.1655 | 40 |
| CS-ATTN seed 42 | 0.1133 | 5.42 | 0.0234 | 0.98 | 0.1025 | 0.1083 | 0.0750 | 39 |
| CS-ATTN seed 43 | 0.0776 | 2.94 | -0.0088 | -0.65 | 0.0567 | 0.0500 | 0.0421 | 38 |
| CS-ATTN seed 44 | 0.0267 | 0.57 | -0.0185 | -0.74 | 0.0850 | 0.0833 | 0.0385 | 35 |
| CS-ATTN 3-seed ensemble | 0.0851 | 3.69 | -0.0327 | -1.22 | 0.0692 | 0.0750 | 0.0691 | 40 |

### Per-cycle tail-IC(top-half) mean

| signal | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 |
|--------|---|---|---|---|---|
| frozen GBM spread | -0.0248 | 0.0504 | 0.0669 | 0.0296 | 0.1011 |
| CS-ATTN seed 42 | 0.0274 | 0.0184 | -0.0842 | 0.0285 | 0.1133 |
| CS-ATTN seed 43 | 0.0394 | -0.0055 | -0.0663 | 0.0437 | 0.0776 |
| CS-ATTN seed 44 | -0.0086 | 0.0206 | -0.0914 | 0.0451 | 0.0267 |
| CS-ATTN 3-seed ensemble | 0.0259 | 0.0263 | -0.0199 | 0.0936 | 0.0851 |

### Per-cycle top-decile overlap

| signal | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 |
|--------|---|---|---|---|---|
| frozen GBM spread | 0.0344 | 0.0788 | 0.1099 | 0.1151 | 0.1017 |
| CS-ATTN seed 42 | 0.0948 | 0.1236 | 0.0879 | 0.1069 | 0.1025 |
| CS-ATTN seed 43 | 0.0964 | 0.1090 | 0.0769 | 0.1215 | 0.0567 |
| CS-ATTN seed 44 | 0.0854 | 0.1081 | 0.0769 | 0.1103 | 0.0850 |
| CS-ATTN 3-seed ensemble | 0.0802 | 0.1016 | 0.0769 | 0.1229 | 0.0692 |

## Crude 14d book (information only, not adopted)

| signal | total | CAGR | MaxDD | Sharpe | n |
|--------|-------|------|-------|--------|---|
| frozen GBM spread | 133.8% | 13.4% | -74.0% | 0.509 | 176 |
| CS-ATTN seed 42 | 359.2% | 25.3% | -76.1% | 0.681 | 176 |
| CS-ATTN seed 43 | 342.2% | 24.6% | -78.0% | 0.673 | 176 |
| CS-ATTN seed 44 | 205.4% | 18.0% | -78.2% | 0.578 | 176 |
| CS-ATTN 3-seed ensemble | 209.2% | 18.2% | -81.6% | 0.582 | 176 |

## Null tables (folds {5, 21} × 10 within-date shuffles, seed 42)

Verdict: `CONTAMINATED` bias_pass=`False` skill_pass=`False` n_violate=`2` n_exceed=`1` / need `2`.

| fold | n | null mean | SD | 2·SE | |mean| | bias | real IC | null 95th | exceeds |
|------|---|-----------|----|------|--------|------|---------|-----------|---------|
| 5 | 10 | 0.0364 | 0.0117 | 0.0074 | 0.0364 | FAIL | 0.0178 | 0.0516 | False |
| 21 | 10 | -0.1397 | 0.0274 | 0.0173 | 0.1397 | FAIL | 0.0669 | -0.1067 | True |

## GPU spend log

- gpu used = `True` type=`A10G` parallel=`12×A10G fold-parallel`
- A10G sunk = `17.41` (15.83 h × `1.1`/h)
- GPU hours (this wave) = `45.492` × `1.1`/h = `50.04`
- wall seconds = `220749.07786505873` total GPU-hours=`61.319`
- USD total = `67.45` cap=`80.0` aborted=`False` reason=`None`
- folds completed = `[]` seeds done=`[42, 43, 44]` null jobs=`20`

Elapsed total `46.64666199684143` s. Frozen products untouched.
