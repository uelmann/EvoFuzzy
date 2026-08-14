# BTC-BEATER — Phase 0.c freeze addendum

**Status:** FROZEN before results. New archive (active + inactive), separate from frozen COMBO v2.0-combo-final (**untouched**).
**Scope:** data + analysis only. No model training. No schedules, no live components. Master only. CPU only. Zero GPU.

The 828-coin KuCoin-filtered archive and any benchmark computed on it or on the 2018 circular window are **discarded unread**.

## Gate v2 (verbatim, from Phase 0.b)

> The dataset is USABLE-FROM-YYYY-MM at the first quarterly CMC historical snapshot D whose true-top-100 coverage is ≥ 85% and remains ≥ 85% at every later snapshot, measured against the external snapshot lists. If that first D is after 2023-01, the project is BLOCKED pending a different data source. Mechanical, no post-hoc adjustment.

Mechanical reading (frozen here):

- External snapshots = CMC historical listings top-500, one date per quarter 2017-Q1 → 2025-Q4 (quarter-end; walk back ≤6 days if the API returns empty). Provenance cached.
- True-top-N coverage at snapshot D = fraction of that snapshot's top-N CMC ids that have a close in the new archive on D (nearest panel session within 2 days). Reported for N ∈ {50, 100, 200}.
- USABLE-FROM is the first snapshot D with top-100 coverage ≥ 0.85 **and** every later snapshot also ≥ 0.85.
- If that D is after 2023-01-01, **BLOCKED**.

## Death-in-position convention (verbatim, all project backtests)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Phase 1 label (unchanged)

> NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe (book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its numbers become the floor every ML phase of this project must beat net of costs.

Parameters frozen a priori (no sweeps): PIT top-50, 90d excess log-return vs BTC, equal-weight up to 10 names with excess > 0, 10% name cap, remainder in BTC, h=7 weekly tranche, spot costs 10 bps/side alts and 2 bps BTC, never cash. Keys = cryptocurrency_id.

## Credit guard (frozen)

Existing CMC script uses the public data-api (`api.coinmarketcap.com/data-api/v3`), which returns `credit_count=0` and has no plan meter. Project HTTP volume before OHLCV. Hard-stop if remaining GETs > 100000; print a reduction proposal (top-300 union, or yearly instead of quarterly snapshots). If a Pro API key is present in env, do **not** switch this job onto paid historical listings/OHLCV.

## What this freeze does not do

- Does not train a model.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
- Does not read Phase 1 numbers from the KuCoin-filtered archive.
