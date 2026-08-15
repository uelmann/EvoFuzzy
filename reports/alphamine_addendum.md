# ALPHAMINE-LO — freeze addendum

**Status:** FROZEN before results. Parallel A/B test: same LightGBM, same long-only book, new formulaic features vs A0-only.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Scope:** backtest and analysis only. New Modal app `quant-alphamine`. Writes only under `/data/quant/alphamine/`. Master only. CPU only. Zero GPU. No live components.

This is **not** a product. This is **not** LONG-CASH (no dual heads, no `er_hat`/`p_up` hurdles, no min-3 cash skip). This is **not** LONG-TIDE (no BTC parking). This is **not** COMBO-LO (scores are retrained; frozen A0 scores are not reused). Kronos / MASTER / VQ / MoE / LLM miners are out of scope.

## Pre-registered improvement statement (verbatim)

> ALPHAMINE-LO IMPROVES on A0-LO only if ALL of: (a) pooled OOS RankIC of MINE exceeds A0; (b) MINE top-quintile minus universe 10-day USDT simple return exceeds A0's; (c) MINE long-only net Sharpe exceeds A0-LO net Sharpe; (d) BTC weight is identically 0 every day on both books; (e) the MINE label-shuffle null is GREEN. This is an A/B feature test, not a replacement for COMBO. No post-hoc adjustment.

## Pre-registered viability statement (verbatim)

> ALPHAMINE-LO is VIABLE as a standalone long-only mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed alt gross ≥ 0.15; (e) BTC weight is identically 0 every day; (f) the MINE label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

## Pre-registered null gate (verbatim)

> Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). Skill passes if the real MINE OOS RankIC exceeds the null 95th percentile on **both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds.

Judged null folds: fold 0 and the fold whose val_start is nearest 2022-01-01. R = 10 within-date shuffles, seeds {101,…,110}. Metric = mean per-date Spearman(raw MINE score, `y_h10`) on that fold's OOS val. Center = 0. Formulas are frozen from the real mine; only LightGBM labels are shuffled.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mandate (frozen)

- **Long-only alts.** BTCUSDT is excluded from training rows, scoring for the book, and every position. No BTC parking. No BTC beta hedge. Residual capital is cash/USDT at 0 return.
- Cash earns 0 (no T-bill).
- Benchmark for viability is cash, not BTC. BTC B&H may be plotted as an informational opportunity-cost line and is **not** a gate.

## Arms (frozen)

Two arms, identical folds, identical book, identical LightGBM hyperparameters (frozen A0: Huber, `num_leaves` 31, `lr` 0.03, `min_data_in_leaf` 200, `feature_fraction` 0.8, bagging 0.8/1, `lambda_l2` 1.0, seed 42, n_estimators 3000, patience 100, early-stop mean daily RankIC vs `y_h10`).

- **A0:** 33 frozen A0 columns (`baseline.features.FEATURE_COLS`) only.
- **MINE:** those 33 columns plus up to 8 fold-selected formulaic features.

Label for both arms: frozen A0 `y_h10` (10-day residual vs BTC, winsorized 1/99 per date). Changing the label is out of scope.

Train universe: PIT top-120 rows present in `features_labeled.parquet`, **excluding BTCUSDT**.
Execution universe: PIT top-40, **excluding BTCUSDT**.
Horizon: **h=10 only** (judged).
Walk-forward: same `make_folds` recipe as A0 (`min_train_days` 730, `val_days` 90, `step_days` 90, purge *h*, embargo *h*+3, inner holdout 90d).
Last-fold-wins on overlapping OOS rows.

## Head tree fallback (verbatim)

> If an arm's median best_iteration ≤ 1, refit that arm with fixed 500 trees (A0 h=10 fallback). The judged book for that arm uses the refit. This rule is frozen before results.

## Miner (frozen)

Small genetic program over OHLC + volume. Not AlphaSAGE, not an LLM.

- Fields: `open`, `high`, `low`, `close`, `volume`, `dollar_volume`, `vwap` (= dollar_volume / volume), `ret` (= log close/close.shift(1)).
- Unary: `abs`, `log`, `neg`, `sign`, `cs_rank`.
- Rolling unary (windows {5,10,20,40}): `delay`, `delta`, `ts_mean`, `ts_std`, `ts_max`, `ts_min`, `ts_sum`, `ts_rank`.
- Binary: `add`, `sub`, `mul`, `div`.
- Rolling binary: `ts_corr`.
- Population 32, generations 6, max depth 3, elite 4, tournament 3, crossover 0.6, mutation 0.3. Time budget 180s per fold.
- Fitness = mean daily RankIC vs `y_h10` on the inner-train split (dates ≤ train_end − 90d), BTC excluded.
- Keep up to 8 formulas with inner-holdout RankIC ≥ 0.01 and mean |Spearman| vs each A0 column and vs already-kept formulas < 0.70 (computed on inner-holdout rows).
- Formulas are mined on that fold's train only. Rolling ops are causal. Cross-sectional z-score (clip 5) is applied per date on the PIT-120 feature rows, matching A0.
- If fewer than 1 formula survives, MINE trains on A0 columns only (`n_formulas=0`).

## Book (frozen, both arms)

- Overlapping *h*=10 tranches, lag-0 close execution, funding accrued (`−w·funding_rate`; longs pay when the rate is positive).
- Tranche *k* re-forms on day *i* when `i % 10 == k`.
- **Always long the top 10** names by score in that day's PIT top-40 (BTC dropped). If fewer than 10 names have a finite score, take all of them. No τ, no second head, no min-3 skip.
- Size: inverse `yz_vol_30` among the tranche names, **full tranche budget** `1/h`. ADV cap 0.5% of a USD 1,000,000 notional; no renormalize after the cap (leftover is cash, never BTC).
- Tiered costs as P2: ranks 1–20 use 5+3 bps; ranks 21–40 use 10+8 bps. Rank universe = PIT top-40.
- Daily PnL identity: `net = Σ w_i · (p_{t+1}/p_t − 1) − costs + funding`. Simple returns. Hedge term is identically 0.

## Diagnostics (not gates, except where listed in the improvement statement)

- Pooled OOS RankIC of each arm vs `y_h10` on PIT top-40 (BTC dropped).
- Top-quintile minus universe mean of 10-day simple USDT return (PIT top-40, BTC dropped).
- EW PIT top-40 daily-rebalanced costless basket: informational.
- BTC B&H: informational opportunity cost, not a viability input.

## What this freeze does not do

- Does not retrain, rescore, or rewrite A0 artifacts, COMBO, SPREAD-LS, or LONG-TIDE.
- Writes under `/data/quant/alphamine/` and copies the report/chart into the volume `reports/` and `charts/` sync paths.
- Does not hold or hedge BTC.
- Does not use GPU.
- Does not introduce schedules or live components.
- Does not adopt a catalyst/attention dataset.
- Does not change the label from residual `y_h10` to USD.
