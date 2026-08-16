# BTC-BEATER — Phase 5 CS-ATTN v0 freeze addendum

**Status:** FROZEN before results. Hourly panel + cross-sectional attention with tail-weighted twin heads.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes.
Master only. Frozen products and caches untouched (read-only). One architecture config, zero search.
GPU allowed for §B ONLY. HARD BUDGET CAP $80. Original freeze named one A10G. On 2026-08-16 the operator tried 4×H100 after sequential A10G projected ~45 h remaining; per-batch time matched A10G (~2.2 s), so Stage B uses **12×A10G fold-parallel** instead (same list rate, wall-clock via more GPUs). Architecture, seeds, folds, and LIVE/PARKED criteria are unchanged.

This is the last major card inside the quantitative-data perimeter. Three structural upgrades vs the frozen GBM, judged on tail metrics. Seed robustness is inside the criteria from day one (the GRU lesson: Phase E seed-42/43/44 RankIC spread was large enough that a 3-seed ensemble without a dispersion clause is not confirmatory).

## Pre-registered criteria (verbatim)

> CS-ATTN is LIVE if ALL of: (a) the 3-seed ensemble improves tail-IC(top-half) ≥ +0.010 AND top-decile overlap ≥ +0.015 vs the frozen GBM baseline on the full OOS; (b) seed dispersion is small: max−min of per-seed full-OOS tail-IC(top-half) ≤ 0.010; (c) the §B null passes. CS-ATTN is PARKED otherwise, and the verdict sentence must state which clause failed. If PARKED on clause (a) with dispersion passing, the conclusion 'the price/volume ceiling is real at this scale' is recorded in the project ledger. Nothing is adopted; any production use requires a fresh pre-registered phase. Mechanical, no post-hoc adjustment.

## §B null (verbatim, adapted E.1b)

> Adapted E.1b on folds {5, 21} × 10 within-date label-shuffle replicates (train seed 42 only). (a) BIAS: for each fold, the null mean of per-date tail-IC(top-half) must satisfy |mean| ≤ 2·(null SD / √R). CONTAMINATED if ≥2 fold-level violations (original E.1b tolerance). (b) SKILL: seed-42 real fold tail-IC(top-half) must exceed that fold's null 95th percentile on both folds. §B null PASSES iff not CONTAMINATED AND skill passes.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Frozen architecture (one config, no search)

- Input per (date t, coin i): last 21 UTC days of hourly bars (504 steps).
- Channels: log-ret, hi-lo range `log(high/low)`, 168h volume z-score, taker-buy share, vs-BTC log-ret.
- Shared temporal encoder: TCN, 4 residual blocks, width 64, kernel 7, dilations (1, 4, 16, 64), dropout 0.1. Receptive field 511 ≥ 504.
- Coin embedding: 64d (TCN last-step, layer-norm).
- Cross-section: per date, set-attention over the floored PIT top-100 embeddings, 2 layers, 4 heads, width 64, plus a date-level pooled token (mean of valid embeddings, prepended as CLS).
- Heads: twin within-date top-quintile and bottom-quintile logits. Class weight 3:1 on the TOP head for true-top names. BOTTOM head unweighted BCE. Signal = p_top − p_bottom after per-fold isotonic fit on the inner-holdout (train only).
- Training: walk-forward on the SAME fold schedule as 2.c (`make_expanding_folds`, refit every 90d, purge last h train days, embargo h+3). AdamW, LR 3e-4, weight decay 0.01, early stop on inner-holdout per-date tail-IC(top-half), patience 3, max 20 epochs/fold. Seeds {42, 43, 44} — full walk-forward each.
- Universe / labels: floored PIT top-100, h=14 excess-vs-BTC quintiles, identical to 2.c.
- Null: folds {5, 21} × shuffle seeds 101–110, train seed 42 only.

## Judgment grid (tail metrics primary)

Rows: frozen GBM spread (2.c cache, read-only) · CS-ATTN per seed · CS-ATTN 3-seed ensemble (mean p) · MANUEL-SCORE Reading A if that report exists.
Columns: tail-IC top-half / bottom-half (per-date, NW-t lag 14), top-decile overlap, monster top-3 capture, whole-list RankIC (secondary) — full OOS / trailing-18m / per-cycle.
Plus crude-14d book CAGR/MaxDD per signal (information, no adoption).

## Budget / hygiene

- GPU (operator override 2026-08-16): 12×A10G fold-parallel (`PHASE5_GPU_MAX_CONTAINERS`). List price `PHASE5_GPU_USD_PER_HOUR = 1.10`. 4×H100 was tried first; per-batch time matched sequential A10G so H100 was stopped. Prior A10G sequential hours plus the short H100 probes are sunk cost counted toward the same **USD 80** cap. Each worker reloads peer spend from the volume before every fold and aborts with `incomplete_budget` if combined spend would exceed $80. Stage A (CPU download) does not count toward the cap. Architecture / seeds / folds / criteria unchanged.
- Watchdog: kill any stage silent > 20 minutes.
- Frozen 2.c pred cache, CMC panel, PIT floors, COMBO, SPREAD-LS, LONG-TIDE: read-only.
- Writes only under `/data/quant/hourly/`, `/data/quant/raw/hourly_spot/`, `/data/quant/raw/hourly_um/`, `/data/quant/btcb/phase5/`, and `reports/charts` names prefixed `btcb_hourly*` / `btcb_phase5*` / `btcb_p5*`.

## What this freeze does not do

- Does not search architectures, learning rates, or horizons.
- Does not adopt a book or touch frozen products.
- Does not introduce schedules or live components.
