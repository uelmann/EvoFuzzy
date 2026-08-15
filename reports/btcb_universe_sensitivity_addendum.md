# BTC-BEATER — SPREAD-LS universe sensitivity freeze addendum

**Status:** FROZEN before results. Three tradeable universes, identical SPREAD-LS mechanics.
**Scope:** backtest + analysis only. Portfolio layer only. No retraining, no signal changes, no schedules. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final and BTC-BEATER v1 are **untouched**.

Phase 2.c spread cache is **reused byte-identical** (sha256 verified). Phase 3.b (funding-on) has **not** run; every book in this freeze is **funding-off**, flagged consistently. β-matched leg sizing is the convention for all runs.

## Pre-registered reading (verbatim)

> The production universe is the SMALLEST U whose full-OOS net Sharpe ≥ (best U's Sharpe − 0.15) AND trailing-18m ≥ (best U's trailing − 0.15) — i.e., prefer concentration/tradability only when it costs less than 0.15 Sharpe. Dollar-volume ranking remains the house standard; the mcap table is informational unless mcap beats volume by ≥ 0.20 on both windows for the chosen U. Mechanical, no post-hoc adjustment.

## Books (fixed a priori, no sweeps)

For U ∈ {floored PIT top-30, top-50, top-100} by 30d median dollar volume (house standard):

- Signal: 2.c spread = p_top_cal − p_bottom_cal, h=14 cache, last-fold-wins. No retraining.
- Long = top decile of U by spread (3 / 5 / 10 names on an entry day). Short = bottom decile ∩ perp-shortable. If fewer than 5 shortable names qualify, hold only those.
- β-matched leg sizing; EW within each leg; 10% per-name cap; unfilled budget stays cash — never BTC.
- Quintile-exit hysteresis; h=14 tranches; anti-blowoff on new longs; death-in-position.
- Costs: longs spot 10 bps/side; shorts perp 5 bps + 3 bps slippage/side. **FUNDING = 0** (3.b has not run).
- Secondary informational table: the same three U ranked by trailing-30d median **market cap** instead of dollar volume.

## What this freeze does not do

- Does not retrain, change the spread, or rebuild 2.b hygiene.
- Does not apply funding (not available; 3.b not run).
- Does not sweep costs, hysteresis, or gross.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
