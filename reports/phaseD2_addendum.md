# Phase D.2 addendum — pre-registered design (before results)

**Written and frozen before any Phase D.2 portfolio, IC, or oracle-beta number is observed.** Backtest/analysis only. No schedules, deployments, or live components. Frozen A0 hash (feature/model layer): `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`. Zero GPU.

This addendum does not alter the Phase D record. Phase D microstructure KEEP on top-20 remains **KILL**. Phase D.2 tests a different object: tradeable net Sharpe on a PIT top-40 execution universe with liquidity-tiered costs, plus a causal τ fix.

## Honesty preamble (verbatim)

> The top-40 hypothesis originates from patterns observed in Phase D results (micro features helping on the wide universe while failing on top-20), reinforced by Phases E/E.1b (every surviving signal lives on the wide universe; top-20 IC decays while pit-120 IC holds). This test is therefore not fully independent. Protections: the adoption criterion below is pre-registered before running, and adoption is judged on tradeable net Sharpe with liquidity-tiered costs — a different object from the pit-120 RankIC that surfaced the pattern.

## Pre-registered adoption criterion (verbatim)

> Top-40 execution is ADOPTED if P2 or P4 trailing-18m median-τ net Sharpe ≥ P1 + 0.30 AND its full-period net Sharpe ≥ P1 − 0.20. The micro block is ADOPTED on the chosen universe if the corresponding paired trailing-18m ΔRankIC on that universe ≥ +0.005 AND full-period ΔRankIC on that universe ≥ 0 AND its portfolio (P3 or P4 vs its A counterpart) trailing-18m net Sharpe Δ ≥ 0. Verdicts are mechanical; no post-hoc adjustment.

## Operationalization (also pre-registered; does not alter the verbatim text)

- Horizons: the four named runs exist at h=7 and h=10. Universe ADOPTED if the two Sharpe inequalities hold for P2 or P4 versus same-horizon P1 at **either** horizon. Otherwise REJECTED. Chosen universe = top-40 if ADOPTED, else top-20.
- If both P2 and P4 pass at one or both horizons, the headline chosen run is the passing (P, h) with the highest trailing-18m median-τ net Sharpe.
- Micro on the chosen universe: pass if **either** horizon satisfies all three micro inequalities on that universe (ΔIC trail18m / ΔIC full / trail18m Sharpe Δ vs the A counterpart: P3 vs P1 on top-20, P4 vs P2 on top-40).
- Median-τ: run τ percentiles `{60, 70, 80, 90}` on the **training-window** schedule. Headline run = the τ whose full-period net Sharpe is the median of those four (house `median_tau_summary`: closest to median Sharpe). Trailing-18m and per-year numbers for that run use the **same** τ, not a re-picked τ.
- Criterion Sharpes are computed on the **intersection of P1–P4 dates** at that horizon (identical days). Native-date metrics are also reported.
- Oracle-beta is diagnostic only. LOOKAHEAD BY DESIGN. No production change and no verdict input.

## τ lookahead fix (all D.2 portfolio runs)

Previous Phase D / A0 portfolios set τ = the `{60,70,80,90}` percentile of `|score|` on the **full OOS** path (lookahead). D.2 replaces that with **training-window τ, per fold, expanding**:

- Walk-forward folds are expanding (`train_start` fixed, `train_end` advances).
- For dates in fold k’s val window, τ is the percentile of `|score|` on dates **strictly before** fold k’s `val_start` (the expanding history that is in-sample by the time fold k is used). Fold 0 warms up day-by-day from past scores only until ≥5 observations exist; no future OOS scores enter τ.
- Isolation table: frozen A0, top-20, tranche h=7, funding on, lag 0, **same code path except τ schedule** — full-OOS (pooled) vs training-window (fold_train), reported at τ=60 (Phase D’s published setting) and at median-τ.

## PIT top-40, costs, liquidity cap

- `universe/top40_pit.parquet`: at each date t, top 40 by 30-day rolling median dollar volume, data ≤ t only (`build_pit_topn`, same mechanism as top-20). PIT universe-lookahead gate extended to n=40 at current strictness.
- Liquidity-tiered costs, rank on the trade date: ranks 1–20 = 5 bps fee + 3 bps slippage per side; ranks 21–40 = 10 bps fee + 8 bps slippage per side. BTC hedge uses BTC’s PIT rank that day, else rank 1.
- Position liquidity cap (P2/P4 only): per-position weight ≤ min(vol-target weight, `0.005 × 30d median dollar volume / NOMINAL_BOOK`). **Nominal book = USD 1,000,000** for a 1.0 gross book. Cap clips; no renormalization. P1/P3 unchanged (top-20, flat 5+3 bps, no extra cap).

## Core comparison

Model A = frozen A0, 33 price features, existing OOS preds (no retrain). Model A+micro = A0 + 10 microstructure features (Phase D’s 12 minus always-NaN `liq_imb_1`, `liq_imb_7`). Retrain A+micro on the 10-col set (feature_fraction makes dropping dead columns non-equivalent to reuse).

All with training-window τ, tranche execution, funding on, lag 0, median-τ headline:

| id | model | execution |
|----|--------|-----------|
| P1 | A | top-20 (reference; also the τ-fix isolation run) |
| P2 | A | top-40, tiered costs, liquidity cap |
| P3 | A+micro | top-20 |
| P4 | A+micro | top-40, tiered costs, liquidity cap |

Paired ΔRankIC (A+micro vs A) on the top-40 PIT subset is reported for both horizons, full and trailing-18m, with Newey-West t (lag = h). Informational: same ΔIC on top-20.

## Hedge decomposition (P1 only, diagnostic)

- Per calendar year: gross / hedge PnL / cost / funding / net (the hedge column Phase D’s printed table omitted).
- Oracle-beta counterfactual: replace the hedge ratio with beta estimated on the forward window `[t, t+h]` (LOOKAHEAD BY DESIGN). Per-year Δ(net) actual vs oracle = beta-estimation cost. 2026 attribution: fraction of 2026 net loss due to beta error vs alpha failure, using h=7 P1 (primary).
