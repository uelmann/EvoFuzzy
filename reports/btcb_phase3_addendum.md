# BTC-BEATER — Phase 3 freeze addendum

**Status:** FROZEN before results. SPREAD-LS challenger: decile long/short on the 2.c twin-head spread, no BTC anywhere.
**Scope:** backtest + analysis only. Portfolio layer only. No schedules, no live components, no retraining, no new features. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final and BTC-BEATER v1 are **untouched**.

Cleaned+floored panel, PIT universes, and cached spread scores from Phase 2.c are **reused byte-identical**. No new hygiene. No data changes.

## Pre-registered criteria (verbatim)

> SPREAD-LS is VIABLE if full-OOS net Sharpe ≥ 0.8 AND trailing-18m net Sharpe ≥ 0.3. It is SLEEVE-GRADE (candidate third sleeve alongside the frozen COMBO) if additionally its daily PnL correlation with the COMBO on the overlapping window is < 0.5 AND its same-window net Sharpe ≥ COMBO − 0.10. It is a REPLACEMENT CANDIDATE only if same-window net Sharpe ≥ COMBO + 0.15. Verdicts mechanical; the dollar-neutral variant is the headline; the beta-matched variant is reported, not judged. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

Shorts are force-covered on the same convention (last available close). Forced-exit and forced-cover counts are reported separately.

## Funding caveat (verbatim)

> FUNDING = 0. Funding is not available in this dataset; the sign of omitted funding is unknown. This is a material caveat on SPREAD-LS net Sharpe. Shorts on USDT-M perpetuals would have paid or received funding that is not in this book.

## Book (fixed a priori, no sweeps)

- Signal: spread = p_top_cal − p_bottom_cal from the 2.c cache. h=14 headline; h=30 secondary. Last-fold-wins OOS. No retraining.
- Daily on the floored PIT top-100: LONG = top decile by spread (10 names); SHORT = bottom decile by spread intersected with perp-shortable names (a Binance USDT perpetual live at date *t*, from the old project's Vision listing/kline tables). If fewer than 5 shortable names qualify, the short leg holds only those.
- Weights: equal-weight within each leg; dollar-neutral gross 0.5 long / 0.5 short. Scale the thinner leg's names up to its 0.5 budget with a 10% per-name cap. Unfilled budget stays uninvested — **never in BTC**.
- Rank hysteresis: enter on decile membership, exit only when the name leaves the top/bottom quintile. h-tranche rotation. Anti-blowoff filter on long entries unchanged (7d raw > +50%).
- Costs: longs spot 10 bps/side; shorts perp 5 bps + 3 bps slippage/side. FUNDING = 0 (caveat above).
- Explicit assertion: BTC (id 1) never appears in either leg.
- Secondary variant: beta-matched leg sizing (short-leg budget scaled so Σw·β matches the long leg; still no BTC positions). Reported, not judged.

## Measurements (fixed)

- Full OOS (2019-10 →): net Sharpe, CAGR, MaxDD, per-cycle table, avg #longs / #shorts / shortable count, % days with incomplete short leg, turnover, forced exits/covers.
- Realized beta vs BTC (daily OLS, full and rolling 90d).
- Squeeze-days: the 20 largest up-days of the EW floored top-100 basket — net PnL of SPREAD-LS on those dates.
- vs frozen COMBO on the overlapping window (2022-01 →): same-window net Sharpe both books, daily PnL correlation. COMBO is replayed from frozen A0 scores; the product is not modified.

## What this freeze does not do

- Does not retrain heads, change features, or rebuild hygiene/PIT.
- Does not park residual budget in BTC or add a BTC hedge position.
- Does not sweep decile size, costs, hysteresis, or gross.
- Does not touch COMBO, the system card, the numbers ledger, or frozen A0 scores.
- Does not introduce schedules or live components.
