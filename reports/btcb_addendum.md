# BTC-BEATER — Phase 0/1 freeze addendum

**Status:** FROZEN before results. New project, separate from frozen COMBO v2.0-combo-final (**untouched**).
**Scope:** backtest and analysis only. No model training. No schedules, no live components. Master only. CPU only. Zero GPU.

Objective of the project (later phases): a LONG-ONLY spot product, benchmarked in BTC terms, actions = enter / hold / exit, undeployed capital parks in BTC (never cash in v1). This task only asks whether the dataset can be trusted and what the ML will have to beat.

## Phase 0 — pre-registered gate (verbatim)

> The dataset is USABLE-FROM-YYYY-MM if, from that date onward, ≥80% of the historical top-200 sample coins are present with correct terminal histories and a PIT universe is reconstructable. The earliest such date is the project's backtest start. If no date before 2021-01 qualifies, the 2018–2020 era is declared FICTION and excluded; if no date before 2023-01 qualifies, the project is BLOCKED pending a different data source. Mechanical, no post-hoc adjustment.

Mechanical reading (frozen here):

- The **historical top-200 sample** is 30 coins drawn with seed 42 from the union of year-end (2018/2019/2020, nearest available session) mcap top-200 in the dataset itself, excluding stables/wrapped/BTC.
- A sample coin is **present with correct terminal history from date D** if it exists in the panel, has first observation ≤ D + 30 days **or** has any observation in 2018–2020 (it was a contemporaneous large-cap), **and** its last observation is either within 14 days of the dataset end (survivor) **or** at least 30 days before the dataset end (series ended; not a silent recent truncation).
- **PIT reconstructable from D** means ≥ 80% of calendar dates t ≥ D have at least 50 names with a trailing-30d median dollar volume (fallback: mcap) computed on data ≤ t.
- Scan month-starts D from 2018-01-01 through 2023-01-01. The earliest D that satisfies (≥ 80% of the sample OK) **and** PIT reconstructable is `USABLE-FROM-YYYY-MM`. If none before 2021-01-01, mark 2018–2020 **FICTION** and take the earliest D ≥ 2021-01-01. If none before 2023-01-01, **BLOCKED**.

## Phase 1 — pre-registered label (verbatim)

Run only if Phase 0 yields a usable window.

> NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe (book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its numbers become the floor every ML phase of this project must beat net of costs.

Parameters frozen a priori (no sweeps): PIT top-50, 90d excess log-return vs BTC, equal-weight up to 10 names with excess > 0, 10% name cap, remainder in BTC, h=7 weekly tranche, spot costs 10 bps/side alts and 2 bps BTC, never cash.

## What this freeze does not do

- Does not train a model.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
