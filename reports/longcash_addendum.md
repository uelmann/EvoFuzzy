# LONG-CASH — freeze addendum

**Status:** FROZEN before results. Parallel cash-financed alt-long product.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Scope:** backtest and analysis only. New LightGBM heads on frozen A0 *features* (not A0 scores). New Modal app `quant-long-cash`. Writes only under `/data/quant/long_cash/`. Master only. CPU only. Zero GPU. No live components.

This is **not** LONG-TIDE (no BTC parking). This is **not** COMBO-LO (no τ on `|score|`, no leftover short-budget cash drag, no BTC hedge). This is **not** the oracle-ladder-2 catalyst/attention hunt: same 33 daily features, different label and book.

## Pre-registered viability statement (verbatim)

> LONG-CASH is VIABLE as a standalone cash-financed alt-long mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed alt gross ≥ 0.15; (e) BTC weight is identically 0 every day; (f) the Head-R label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

## Pre-registered null gate (verbatim)

> Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). Skill passes if the real Head-R OOS RankIC exceeds the null 95th percentile on **both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds.

Judged null folds: fold 0 and the fold whose val_start is nearest 2022-01-01. R = 10 within-date shuffles, seeds {101,…,110}. Metric = mean per-date Spearman(raw Head-R score, y_usd) on that fold's OOS val. Center = 0.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mandate (frozen)

- **Hold only alts.** BTCUSDT is excluded from training, scoring for the book, and every position. No BTC parking. No BTC beta hedge. Residual capital is cash/USDT at 0 return.
- Cash earns 0 (no T-bill).
- Benchmark for viability is cash, not BTC. BTC B&H may be plotted as an informational opportunity-cost line and is **not** a gate.

## Model (frozen)

- Features: frozen A0 33 columns (`baseline.features.FEATURE_COLS`), read-only from `features_labeled.parquet`. No new features. No Kronos, GRU, context, or complexity block.
- Train universe: PIT top-120 names present in the feature file, **excluding BTCUSDT**.
- Execution universe: PIT top-40, **excluding BTCUSDT**.
- Horizon: **h=10 only** (judged).
- Head R: LightGBM Huber on `y_usd` = 10-day forward log-return in USDT, winsorized 1/99 per date. Early-stop on mean daily RankIC vs `y_usd`. Per-fold isotonic on inner holdout maps raw score → `er_hat` (not clipped to [0,1]).
- Head C: LightGBM binary on `y_up` = 1{10-day simple USDT return > 0}. Early-stop on mean per-date AUC. Per-fold isotonic maps raw score → `p_up` in [0,1].
- Same LightGBM hyperparameters as frozen A0 (`num_leaves` 31, `lr` 0.03, `min_data_in_leaf` 200, `feature_fraction` 0.8, bagging 0.8/1, `lambda_l2` 1.0, seed 42, n_estimators 3000, patience 100).
- Walk-forward: same `make_folds` recipe as A0 (`min_train_days` 730, `val_days` 90, `step_days` 90, purge *h*, embargo *h*+3, inner holdout 90d).
- Last-fold-wins on overlapping OOS rows.
- OOS scores are emitted for every val-window row with finite features; NaN labels on the terminal *h* days are excluded from RankIC/null but kept for the book.

## Head-R tree fallback (verbatim)

> If Head-R median best_iteration ≤ 1, refit Head R with fixed 500 trees (A0 h=10 fallback). The judged book uses the refit. This rule is frozen before results.

## Book (frozen)

- Overlapping *h*=10 tranches, lag-0 close execution, funding accrued (`−w·funding_rate`; longs pay when the rate is positive).
- Tranche *k* re-forms on day *i* when `i % 10 == k`.
- Enter a name if `er_hat > 0` **and** `p_up > 0.5`. Hard threshold (no hysteresis), matching the frozen COMBO tranche engine.
- If fewer than **3** names pass, that tranche is cash. If more than **10** pass, keep the 10 with highest `er_hat`.
- Size: inverse `yz_vol_30` among the tranche names, **full tranche budget** `1/h` (not the COMBO 0.5 long-half). ADV cap 0.5% of a USD 1,000,000 notional; no renormalize after the cap (leftover is cash, never BTC).
- Tiered costs as P2: ranks 1–20 use 5+3 bps; ranks 21–40 use 10+8 bps. Rank universe = PIT top-40.
- Daily PnL identity: `net = Σ w_i · (p_{t+1}/p_t − 1) − costs + funding`. Simple returns (cash-mandate identity). Hedge term is identically 0.
- When a tranche is empty, that `1/h` sits in cash.

## Diagnostics (not gates)

- Frozen A0 score quintiles vs the new `y_usd` on PIT top-40 (BTC dropped): % of days the top bucket's mean 10-day simple USDT return is > 0, and Newey–West t on that daily top-bucket mean. Informational raw-material snapshot. Does not kill or save LONG-CASH.
- Same snapshot on `er_hat` (new Head R). Informational.
- EW PIT top-40 daily-rebalanced costless basket: informational.
- BTC B&H: informational opportunity cost, not a viability input.

## What this freeze does not do

- Does not retrain, rescore, or rewrite A0, COMBO, SPREAD-LS, or LONG-TIDE.
- Writes under `/data/quant/long_cash/` and copies the report/chart into the volume `reports/` and `charts/` sync paths. Does not rewrite A0 predictions, COMBO artifacts, or the numbers ledger.
- Does not hold or hedge BTC.
- Does not use GPU.
- Does not introduce schedules or live components.
- Does not adopt a catalyst/attention dataset (ladder-2 next priority remains open).
