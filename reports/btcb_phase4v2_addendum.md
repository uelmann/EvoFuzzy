# BTC-BEATER — Phase 4 v2 TAIL ROUND 1 freeze addendum

**Status:** FROZEN before results. Third head = top-k ranking objective (LambdaRank) + derivatives-positioning block + two price-only additions.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Zero GPU.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned panel, floored PIT top-100, and the 2.c spread cache are reused (sha256-checked). Canonical pricing = Binance (3.e). Existing caches reused; **new downloads only** from Binance Vision for missing OI/metrics files, logged.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

## PI decisions (verbatim)

> Catalyst and attention data families (unlocks, listing announcements, search volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume plus derivatives data already retrievable (funding, open interest, basis, taker flows).

> The old project's microstructure KILL applied to the mean-regression label on the old system; this phase judges a positioning block on the NEW system's tail metrics — fresh pre-registration, not a kill-list retest.

## Pre-registered criteria (verbatim, before results)

> TAIL-LOSS EXTRACTS if RANK or blend improves tail-IC(top-half) ≥ +0.010 AND overlap ≥ +0.015 vs baseline with the null passing; BARREN otherwise. POSITIONING LIVE if the positioning block adds tail-IC(top-half) ≥ +0.010 OR overlap ≥ +0.015 on top of the best A-signal, with ≥50% perp coverage of top-100 name-days from 2021. PRICE-ADDITIONS LIVE at the same thresholds. Verdicts mechanical; nothing adopted; any production change requires a fresh pre-registered phase. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## A — RANK head (one config, no sweeps)

LightGBM `objective=lambdarank`. Per-date groups. Label = within-date excess-return rank bucketed to 5 grades (0–4, higher = larger next-14d excess vs BTC). `lambdarank_truncation_level=10`, `ndcg_eval_at=10`. h=14. Same expanding folds / purge / embargo / seed as 2.c. Same 33 Stage-S columns (`STAGE_S_COLS`). No isotonic calibration (raw ranking scores). Early stop on NDCG@10.

Judged A-signals: **RANK** alone; **SPREAD+RANK** = fixed 50/50 average of cross-sectional ranks of frozen 2.c spread and RANK.

## Null (E.1b on per-date tail metrics)

6 folds `{0, 5, 9, 15, 21, 24}` × 25 within-date shuffles of the 5-grade labels on train (`NULL_SHUFFLE_SEEDS` 101–125). Judged metric = per-date tail-IC(top-half); overlap reported with null centre = 1/decile = 0.10. Bias tolerance = original E.1b (`|null mean − centre| ≤ 2·SE`); **CONTAMINATED requires ≥2 fold-level violations**, not 1 (Phase 3.c house rule). Skill: ≥5/6 folds exceed their null 95th percentile **or** Stouffer z ≥ 3.0. Failure = PARKED / CONTAMINATED, no override, no retest with different folds.

## B — Positioning block

Perp-mapped names (CMC id → Binance USDT-M symbol). Non-perp names: features = 0 + `pos_missing` flag.

- `funding_z_7`, `funding_z_30`: daily funding vs own trailing mean/sd
- `funding_level_3d`: trailing 3-day mean funding
- `dOI_7`, `dOI_30`: log change in open interest where OI history exists (first available date per name reported)
- `basis`: perp close / spot close − 1 (names with both)
- `taker_imbalance_7`: taker-buy share of volume, 7d vs own 30d (from klines)

Coverage table per year and liquidity tier (PIT ranks 1–10 / 11–50 / 51–100). POSITIONING LIVE additionally requires ≥50% perp coverage of top-100 name-days from 2021-01-01.

## C — Price-only additions

- `past_alpha_60`: intercept of the trailing 60d daily OLS of coin log-return on BTC log-return
- `trend_composite`: mean of `sign(close/SMA_k − 1)` for k ∈ {20, 50, 100, 200}

## D — Ablation grid (same folds, h=14; tail metrics primary)

Signals, frozen:

1. **frozen_spread** — 2.c cache, not retrained (baseline)
2. **rank** — LambdaRank on STAGE_S
3. **spread_rank** — 50/50 CS-rank blend of (1) and (2)
4. **spread_pos** — twin spread retrained on STAGE_S + positioning
5. **spread_pos_price** — twin spread retrained on STAGE_S + positioning + price-additions
6. **full_stack** — 50/50 CS-rank blend of (5) and LambdaRank trained on STAGE_S + positioning + price-additions

Primary metrics: per-date on the floored PIT top-100 labeled cross-section vs next-14d excess, restricted to Binance-listed names at t (canonical pricing). Tail-IC top-half / bottom-half (NW-t, lag=14), top-decile overlap, monster top-3 capture — full OOS / trailing-18m / per-cycle.

Secondary: whole-list RankIC; crude-14d book CAGR/MaxDD (Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side) as an information check, **no adoption**.

## Mechanical comparison (frozen)

- **Best A-signal** = the member of `{rank, spread_rank}` with higher full-OOS tail-IC(top-half); overlap breaks ties.
- **TAIL-LOSS EXTRACTS** iff that A-signal beats `frozen_spread` by tail-IC(top-half) ≥ +0.010 **and** overlap ≥ +0.015 **and** the RANK tail-IC null passes; else **BARREN**.
- **POSITIONING LIVE** iff `spread_pos` minus best A-signal has tail-IC(top-half) ≥ +0.010 **or** overlap ≥ +0.015, **and** perp coverage of top-100 name-days from 2021 ≥ 50%; else not live.
- **PRICE-ADDITIONS LIVE** iff `spread_pos_price` minus `spread_pos` has tail-IC(top-half) ≥ +0.010 **or** overlap ≥ +0.015; else not live.

Verdicts mechanical. Nothing adopted.

## What this freeze does not do

- No unlock / listing / search-volume ingest
- No product file changes, no schedules, no live components
- No GPU, no architecture sweeps, no extra LambdaRank configs
- No mutation of the 2.c pred cache or the CMC panel
