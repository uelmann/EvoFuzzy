# RETSTD-FULL — freeze addendum

**Status:** FROZEN before results. Same RETSTD-LO target and book; **training protocol changed**: 3000 trees, no early stop.
**Reference books:** COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE. **UNTOUCHED.**
**Scope:** backtest and analysis only. Modal app `quant-retstd-full`. Writes only under `/data/quant/retstd_full/`. CPU only. Zero GPU. No live components.

This is **not** a retrain of frozen A0 Huber. Frozen scores, `features_labeled.parquet`, COMBO, SPREAD-LS, and LONG-TIDE are **read-only**. This is **not** the early-stop RETSTD-LO run (that record stays). Kronos / MASTER / VQ / MoE / GP miners are out of scope.

## Pre-registered improvement statement (verbatim)

> RETSTD-FULL IMPROVES on A0-LO10 only if ALL of: (a) pooled OOS RankIC of RETSTD-FULL predicted probability vs the continuous h=10 USDT-return / forward-path-std ratio exceeds A0's RankIC vs the same ratio; (b) RETSTD-FULL top-decile minus universe 10-day USDT simple return exceeds A0-LO10's; (c) RETSTD-FULL long-only net Sharpe exceeds A0-LO10 net Sharpe; (d) the RETSTD-FULL label-shuffle null is GREEN. This is an A/B target test, not a replacement for COMBO. No post-hoc adjustment.

## Pre-registered viability statement (verbatim)

> RETSTD-FULL is VIABLE as a standalone long-only mandate only if ALL of: (a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; (c) full-period total return > 0; (d) average deployed gross ≥ 0.15; (e) the RETSTD-FULL label-shuffle null is GREEN. It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment.

## Pre-registered null gate (verbatim)

> Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). Skill passes if the real RETSTD-FULL OOS RankIC exceeds the null 95th percentile on **both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds.

Judged null folds: fold 0 and the fold whose val_start is nearest 2022-01-01. R = 10 within-date shuffles, seeds {101,…,110}. Metric = mean per-date Spearman(predicted P, binary y) on that fold's OOS val. Center = 0. Features are frozen; only the binary labels are shuffled. Null models also train exactly 3000 trees.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Training rule (verbatim)

> Every fold trains exactly 3000 LightGBM trees. No early stopping. No median-best_iteration fallback. This matches the configured n_estimators cap (lr=0.03). The judged book uses those 3000-tree models.

## Mandate, label, book (frozen, identical to RETSTD-LO)

- Long-only top 10% of PIT top-40 by predicted P. Residual cash. No hedge. BTC allowed if in the PIT file.
- Label: `R / (STD + 1e-4)` with `R = close[t+10]/close[t]−1` and `STD` = sample std (ddof=1) of the 10 forward daily simple returns. y=1 iff top decile on that date's PIT-120.
- Same 33 A0 features. LightGBM hparams match frozen A0 except `objective=binary` and **no early stop**.
- Control **A0-LO10**: frozen Huber scores, not retrained.
- Overlapping h=10, inv-vol, P2 costs, ADV cap, simple-return PnL.

## What this freeze does not do

- Does not retrain, rescore, or rewrite A0 artifacts, COMBO, SPREAD-LS, or LONG-TIDE.
- Does not overwrite `features_labeled.parquet`, frozen prediction files, or `/data/quant/retstd/` (early-stop record).
- Writes under `/data/quant/retstd_full/`.
- Does not use GPU, schedules, or live components.
- Does not restore the long/short τ book.
