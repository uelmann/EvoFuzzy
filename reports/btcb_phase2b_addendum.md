# BTC-BEATER — Phase 2.b freeze addendum

**Status:** FROZEN before results. Data hygiene first, then honest naive v4, then two-stage MODEL-V2.
**Scope:** backtest + analysis only. No schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final is **untouched**.

Causal thresholds everywhere. Context features are **excluded** from Stage S (they carry no within-date information). Timing is a separate **fixed** gate, not learned.

## Pre-registered criteria (verbatim)

> STAGE-S has SELECTION SKILL if, at h=14, full-OOS mean per-date AUC ≥ 0.52 with the empirical-null gates passing. MODEL-V2 is VIABLE if, on the full OOS window: (a) relative-line Sharpe > 0; (b) total return ≥ BTC B&H; (c) MaxDD ≤ BTC B&H MaxDD. It REPLACES the naive v4 floor if additionally relative-line Sharpe ≥ naive v4 + 0.15. Per-cycle honesty table mandatory; no single cycle overrides. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Naive v4 label (unchanged)

> NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe (book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its numbers become the floor every ML phase of this project must beat net of costs.

## Hygiene (before any backtest)

- Autopsy the +9.98M% naive v3 book (same-window start 2019-10-19): per-name additive PnL contribution; top 10 with price-path max daily returns.
- Clean |daily ret| > 5 suspects (the original 37 plus any new ids on the 0.c archive). Per breakpoint, **splice** if a round split factor (10^k, 2·10^k, 5·10^k or reciprocal, within 30% log-error) is detected **or** mcap is stable (|Δlog mcap| < 0.5) **and** ≥30 sessions follow on the same id; otherwise **truncate** (drop date ≥ breakpoint). Document per id. BTC never spliced/truncated for jumps.
- Investability floor (fixed, all universes from now on): on date t a coin is eligible iff trailing-30d median dollar volume ≥ $2,000,000 AND close ≥ $0.000001 AND ≥ 60 prior sessions AND no |daily ret| > 200% in the trailing 30 days. Rebuild PIT top-50/100 among eligible names (same trailing-DV rank as 0.c).

## Naive v4

Same frozen rotation as v3 (PIT top-50, 90d excess vs BTC, EW up to 10 with excess>0, 10% cap, rest BTC, h=7, 10/2 bps, death convention) on the **cleaned + floored** universes. Contribution table mandatory; flag if any name > 25% of total additive PnL.

## Stage S (selection, within-date)

- y=1 iff h-day excess-vs-BTC is in the top quintile of that date’s PIT top-100. h ∈ {14, 30}, **primary h=14**.
- Features = per-coin blocks only (25 price/excess + 8 new-data). **No context columns** (asserted).
- LightGBM binary, Phase 2 params; early stopping on inner-holdout **mean per-date AUC**.
- Metrics: mean per-date AUC and mean per-date RankIC of p vs realized excess — never pooled AUC as a gate metric.
- Calibration: isotonic on inner-holdout, train-only.
- Gates: lookahead, PIT lookahead on floored top-50/100, seed determinism, E.1b null on mean per-date AUC (25 replicates × 2 folds at h=14). Null must center at 0.5 (|mean−0.5| ≤ 2·SD/√R); real model must exceed null p95 on both folds.

## Stage T (timing, fixed — no learning)

Alt-exposure budget = 50% when [EW top-50/BTC ratio > its 90d SMA] AND [breadth top-100 > 0.5], with 5-day OFF hysteresis (ON immediately when both true; OFF only after 5 consecutive days of failure); else 0%. Frozen, no sweeps.

## Book

When the gate is ON, fill the 50% budget with the top-K PIT top-50 names by calibrated Stage-S p (equal-weight, 10% name cap, K ≤ 10, fewer if fewer names clear p ≥ p_enter). Anti-blowoff on new entries (7d raw > +50%). Remainder always BTC. h-tranche, 10/2 bps, death convention. p_enter grid {0.55, 0.60, 0.65}, house median convention on relative-line Sharpe. Headline = h=14 median p_enter.

## What this freeze does not do

- Does not learn a timing model or put context features in Stage S.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
- Does not introduce schedules or live components.
