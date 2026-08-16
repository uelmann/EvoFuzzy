# BTC-BEATER — MANUEL-2 freeze addendum

**Status:** FROZEN before results. PI-specified strategy: `gauss(ret14)·gauss(ret28)/gauss(std63)`, top-50 mcap, top 5%, long-only.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Zero GPU.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned CMC panel is read-only (sha256 asserted). 2.c / 4.b caches are not mutated. Canonical pricing = Binance-hybrid where listed (3.e).

This phase produces a **record**, not a product. Nothing is adopted. If CONFIRMED, a fresh pre-registered phase evaluates adoption.

## Formula (verbatim, frozen)

> Score = gauss(ret_14d) × gauss(ret_28d) / gauss(std_63d), gauss(x) = Φ(z_cs(x)), z_cs = cross-sectional z-score per date across that day's universe. All inputs use data ≤ t.

- Universe: each day, top-50 by MARKET CAP (cleaned panel, PIT, pegged excluded).
- Inputs per name: ret_14d, ret_28d (simple returns), std_63d (std of daily simple returns, 63d window, pandas ddof=1, min_periods=63, `pct_change(fill_method=None)` so missing prints are NA not zero-vol).
- Selection: top 5% by score (`k = max(1, ceil(0.05 · n_scored))` → 3 names on a full 50). Equal weight, no cap beyond EW.
- Rebalance: DAILY (headline) and WEEKLY Mondays (secondary; first session also rebalances so the book is always invested). Always invested; remaining weights renormalized after death-in-position; costs 10 bps/side on traded notional.
- Informational third variant: same formula on the house floored PIT top-100 DV universe, daily, btc-ex (one row).

## Spec completions (declared a priori, mandatory for implementability)

> STABLECOIN EXCLUSION: pegged assets (stablecoins, wrapped/pegged tokens) are excluded from the universe. Rationale on record: the formula divides by gauss(std63); std≈0 names get near-infinite scores — the literal formula would buy Tether. Exclusion list built from the panel's pegged-asset tags + |90d total return| < 2% heuristic, logged.

Operationalisation (frozen with the registration, not a result):

- **Tags:** house `STABLE_OR_WRAP` ∪ `MANUEL2_EXTRA_STABLES` ∪ USD-suffix tickers (len≤5, except SAND/BAND/BOND/AMP) ∪ name/slug needles (`tether`, `trueusd`, `usd-coin`, `wrapped-bitcoin`, `binance-peg`, `pegged`, `stablecoin`, …). The panel has no CMC `tags` column; these are the project's pegged-asset tags. An id is tagged if any of its panel symbol/name/slug matches. Logged.
- **Heuristic:** at date t, exclude names with `|close[t]/close[t-90] − 1| < 2%` (CMC close, data ≤ t). Names without 90d history are not flagged by the heuristic.
- Exclusion is the **OR** of tags and heuristic.

> BTC VARIANTS: run BOTH btc-in and btc-ex universes (BTC's low std makes the denominator favor it; the PI's thesis is a non-BTC-like curve, so both are shown).

> gauss(x) = Φ(z_cs(x)): normal CDF of the CROSS-SECTIONAL z-score, computed per date across the day's universe. All inputs use data ≤ t.

Cross-sectional z uses ddof=0 on names with a finite input that day. `Φ` = standard normal CDF (`scipy.special.ndtr`). Names with `gauss(std_63d) = 0` are dropped (division undefined). No other clip.

## Four books (disclosed 4-way look)

1. daily, btc-in
2. daily, btc-ex
3. weekly Mondays, btc-in
4. weekly Mondays, btc-ex

> Best book = highest full-window total return among the four pre-declared books (daily/weekly × btc-in/btc-ex). Verdicts apply to that book. All four are reported. This is the disclosed 4-way look. No other selection.

## Benchmarks and claim metrics (identical window, 2019-10-19 →)

- BTC B&H (Binance BTCUSDT, zero cost).
- EW basket of the same mcap top-50 ex-pegged universe (btc-in; the "no selection" line; 10 bps/side, daily, always invested).
- Frozen-spread crude book (Ladder-1, 2.c cache, information/reference only).
- Per book: total, CAGR, USD Sharpe, MaxDD, trailing-18m, per-cycle, avg #names, turnover, forced exits, top-5 name PnL concentration.
- Claim metrics: daily-PnL correlation with BTC (full + rolling 90d); relative line vs BTC (Sharpe + total); score-vs-vol cross-sectional rank-corr (mean per-date Spearman of score vs std_63d).

Pricing: Binance spot close where a print exists, else CMC (hybrid). Scores and PnL use the hybrid close. Universe ranks use cleaned-panel CMC mcap.

> MaxDD is the house signed quantity (negative). STRONG clause 'MaxDD ≤ 1.10 × BTC's' is |book MaxDD| ≤ 1.10 × |BTC MaxDD|, i.e. book_maxdd >= 1.10 * btc_maxdd when both ≤ 0.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Pre-registered verdicts (verbatim, before results)

> MANUEL-2 is CONFIRMED if the best pre-declared book (daily or weekly, btc-in or btc-ex — 4 books, disclosed as a 4-way look) on Binance-hybrid pricing has: total ≥ BTC B&H AND daily correlation with BTC ≤ 0.70. It is STRONG if additionally relative-line Sharpe ≥ 0.50 and MaxDD ≤ 1.10 × BTC's. It is PARTIAL if exactly one of the two claim clauses passes (state which). REFUTED if neither. Per-cycle honesty table mandatory; no single cycle overrides. If CONFIRMED, a fresh pre-registered phase evaluates adoption. Mechanical, no post-hoc adjustment.

Claim clause 1 = total ≥ BTC B&H. Claim clause 2 = daily correlation with BTC ≤ 0.70.

No parameter search. Mechanical, no post-hoc adjustment. Nothing adopted.
