# RETSTD-LO — freeze addendum

**Status:** FROZEN before results. Parallel A/B test: same A0 features and LightGBM hyperparameters, new **binary** target (return / forward path std), long-only top-decile book.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Scope:** backtest and analysis only. New Modal app `quant-retstd`. Writes only under `/data/quant/retstd/`. CPU only. Zero GPU. No live components.

This is **not** a product. This is **not** a retrain of frozen A0 Huber. Frozen scores, `features_labeled.parquet`, COMBO, SPREAD-LS, and LONG-TIDE are **read-only**. This is **not** DD10-LO (denominator is path std, not average drawdown). Kronos / MASTER / VQ / MoE / GP miners are out of scope.

## Pre-registered improvement statement (verbatim)

> RETSTD-LO IMPROVES on A0-LO10 only if ALL of: (a) pooled OOS RankIC of RETSTD predicted probability vs the continuous h=10 USDT-return / forward-path-std ratio exceeds A0's RankIC vs the same ratio; (b) RETSTD top-decile minus universe 10-day USDT simple return exceeds A0-LO10's; (c) RETSTD long-only net Sharpe exceeds A0-LO10 net Sharpe; (d) the RETSTD label-shuffle null is GREEN. This is an A/B target test, not a replacement for COMBO. No post-hoc adjustment.

## Pre-registered viability statement (verbatim)

> RETSTD-LO is VIABLE as a standalone long-only mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed gross ≥ 0.15; (e) the RETSTD label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

## Pre-registered null gate (verbatim)

> Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). Skill passes if the real RETSTD OOS RankIC exceeds the null 95th percentile on **both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds.

Judged null folds: fold 0 and the fold whose val_start is nearest 2022-01-01. R = 10 within-date shuffles, seeds {101,…,110}. Metric = mean per-date Spearman(predicted P, binary y) on that fold's OOS val. Center = 0. Features are frozen; only the binary labels are shuffled.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Head tree fallback (verbatim)

> If RETSTD's median best_iteration ≤ 1, refit with fixed 500 trees (A0 h=10 fallback). The judged book uses the refit. This rule is frozen before results.

## Mandate (frozen)

- **Long-only** on the original A0 execution universe (PIT top-40, **including BTCUSDT** if it is in that file). No short book. No BTC beta hedge. Residual capital is cash/USDT at 0 return.
- Cash earns 0 (no T-bill).
- Benchmark for viability is cash, not BTC. BTC B&H may be plotted as an informational opportunity-cost line and is **not** a gate.

## Arms (frozen)

Identical folds, identical book, identical 33 A0 columns (`baseline.features.FEATURE_COLS`). LightGBM hyperparameters match frozen A0 except the objective (binary vs Huber) and the label.

- **A0-LO10 (control):** frozen A0 Huber scores from `/data/quant/predictions/lgbm_price_only_h10.parquet`. **Not retrained.** Same long-only top-decile book.
- **RETSTD (treatment):** new LightGBM **binary** classifier. Score = predicted P(y=1). Same hparams otherwise: `num_leaves` 31, `lr` 0.03, `min_data_in_leaf` 200, `feature_fraction` 0.8, bagging 0.8/1, `lambda_l2` 1.0, seed 42, n_estimators 3000, patience 100, early-stop mean daily RankIC vs the **binary** label.

## Label (frozen)

Horizon **h=10** only (judged). Working copy of features only; never write back to `features_labeled.parquet`.

For each name and date t, using close prices:

- Simple USDT return: `R = close[t+h] / close[t] − 1`.
- Forward path std: daily simple returns `r_{t+k} = close[t+k] / close[t+k−1] − 1` for k = 1..h. `STD` = sample standard deviation of `{r_{t+1}, …, r_{t+h}}` (ddof=1). NaN if any of the h returns is missing.
- Ratio: `ratio = R / (STD + 1e-4)`.
- Binary y = 1 iff `ratio` is at or above the **90th percentile** of that date's finite ratios among the **PIT top-120** rows present in the working frame; else 0. Dates with fewer than 20 finite ratios get y = NaN (dropped at train).

This is **not** lookback `yz_vol_30`. This is **not** DD10 average drawdown. The frozen residual `y_h10` is renamed `y_resid_h10` on the working copy and is unused for training.

## Universes and CV (frozen, match A0)

- Train: PIT top-120 rows in `features_labeled.parquet`.
- Execution: PIT top-40.
- Walk-forward: same `make_folds` recipe as A0 (`min_train_days` 730, `val_days` 90, `step_days` 90, purge *h*, embargo *h*+3, inner holdout 90d).
- Last-fold-wins on overlapping OOS rows.

## Book (frozen, both arms)

- Overlapping *h*=10 tranches, lag-0 close execution, funding accrued (`−w·funding_rate`; longs pay when the rate is positive).
- Tranche *k* re-forms on day *i* when `i % 10 == k`.
- **Long the top 10%** of that day's PIT top-40 names by score (predicted P for RETSTD; Huber score for A0-LO10). `k = max(1, ceil(0.10 · n_finite))`. No τ, no second head, no min-3 skip.
- Size: inverse `yz_vol_30` among the tranche names, **full tranche budget** `1/h`. ADV cap 0.5% of a USD 1,000,000 notional; no renormalize after the cap (leftover is cash).
- Tiered costs as P2: ranks 1–20 use 5+3 bps; ranks 21–40 use 10+8 bps. Rank universe = PIT top-40.
- Daily PnL identity: `net = Σ w_i · (p_{t+1}/p_t − 1) − costs + funding`. Simple returns. Hedge term is identically 0.

## Diagnostics (not gates, except where listed in the improvement statement)

- Pooled OOS RankIC of each arm vs the continuous ratio, and vs 10-day USDT simple return, on PIT top-40.
- Top-decile minus universe mean of 10-day simple USDT return (PIT top-40).
- EW PIT top-40 daily-rebalanced costless basket: informational.
- BTC B&H: informational opportunity cost, not a viability input.
- Max |BTC weight| and fraction of days BTC is held: reported, not a gate.

## What this freeze does not do

- Does not retrain, rescore, or rewrite A0 artifacts, COMBO, SPREAD-LS, or LONG-TIDE.
- Does not overwrite `features_labeled.parquet` or frozen prediction files.
- Writes under `/data/quant/retstd/` and copies the report/chart into the volume `reports/` and `charts/` sync paths.
- Does not use GPU.
- Does not introduce schedules or live components.
- Does not adopt a catalyst/attention dataset.
- Does not restore the long/short τ book.
