# BTC-BEATER — Phase 4.b TWIN-RANK freeze addendum

**Status:** FROZEN before results. TWIN-RANK (vol-cancelling ranking spread) + vol-matched null (new house standard for tail metrics) + DIR/logit-adjustment reweighting arm.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Zero GPU.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned panel, floored PIT top-100 (Binance-listed convention as in 4v2), canonical pricing = Binance (3.e). Caches sha256-verified. 2.c spread cache reused, not retrained.

Positioning and price-additions are **closed** (Phase 4 v2: NOT LIVE, recorded). Not retested.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

## Why this phase (on the record)

Phase 4 v2 found the single RANK head clears the tail deltas vs the frozen spread but its plain-shuffle null is CONTAMINATED by a VOL CONFOUND: noise-trained rankers score tail-IC 0.14–0.23 on late folds because the realized top decile is vol-loaded and single-head rankers drift vol-tilted. This is the same confound the twin-head subtraction cured in the classifier. This phase applies that cure to the ranking loss, upgrades the null to a vol-matched design, and tests DIR reweighting from the research survey.

## PI decisions (verbatim; still binding)

> Catalyst and attention data families (unlocks, listing announcements, search volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume plus derivatives data already retrievable (funding, open interest, basis, taker flows).

## Vol-matched null (NEW HOUSE STANDARD for tail metrics; verbatim)

> For tail metrics (tail-IC top-half, top-decile overlap, monster capture), the empirical null shuffles labels WITHIN vol-quintile buckets per date (yz_vol_30 quintiles), preserving the vol→outcome loading. Folds {0,5,9,15,21,24} × 25 replicates. The null mean per fold becomes the structural reference level; bias check = null mean stability across replicates (2·SE band around the fold's own null mean, E.1b tolerance: ≥2 fold violations for CONTAMINATED). Skill = real metric exceeds the vol-matched null 95th percentile on ≥5/6 folds OR Stouffer z ≥ 3.0. This supersedes the plain within-date shuffle for tail metrics in all future phases; plain-shuffle results remain on the record.

Operationalisation (frozen with the registration, not a result):

- Shuffle is joint across label columns of a head: within each date, `yz_vol_30` is bucketed into 5 rank-based quintiles (missing vol = its own bucket); labels are permuted **inside** each bucket.
- Fold-level E.1b centre = that fold's own null mean (structural vol-matched reference), not 0 and not 1/decile. A fold violates if the 2·SE band around that mean cannot be formed (n<2 or non-finite SE). CONTAMINATED iff ≥2 fold violations (Phase 3.c house rule).
- Judged skill metric = tail-IC(top-half). Overlap and monster capture are reported on the same null design.
- Applied to: TWIN-RANK (both ranking heads, same seed so the permutation is joint), DIR-spread (DIR-weighted top head; bottom head unchanged = frozen 2.c), and retroactively (informational) to the Phase-4v2 single RANK head.

## Pre-registered criteria (verbatim, before results)

> TWIN-RANK EXTRACTS if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread, with the vol-matched null passing. DIR LIVE at the same thresholds with the same null. If BOTH fail, the ledger records: 'PRICE-VOLUME TAIL CEILING — within this data perimeter, tail improvements beyond the frozen spread are not demonstrable under vol-matched nulls; the fork (capital phase | Phase-5 hourly attention | perimeter expansion) passes to the PI.' Verdicts mechanical; nothing adopted; any production change requires a fresh pre-registered phase. No post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## A — TWIN-RANK (one config per head, no sweeps)

LightGBM `objective=lambdarank`. Identical config to Phase 4 v2: truncation 10, ndcg@10, 5-grade labels, h=14, same expanding folds / purge / embargo / seed, `STAGE_S_COLS` (33). No isotonic.

- **Top ranking head:** within-date excess rank, higher grade = larger next-14d excess vs BTC. Reuses the Phase 4 v2 RANK head cache when present (same config, not retrained).
- **Bottom ranking head:** same LambdaRank config trained on the **inverted** within-date excess rank (worst names = highest grade). Fresh train.
- **Signal:** `S_twinrank = cs_rank(score_top_head) − cs_rank(score_bottom_head)`, per date.

Diagnostic (report only): mean per-date cross-sectional rank-corr of each single head with `yz_vol_30`, and of `S_twinrank` with `yz_vol_30` (the subtraction should collapse it).

## B — DIR / logit-adjustment arm (idea #1 from the research survey)

Retrain the frozen twin-head **classifier** configuration (`per_date_auc` early stop, isotonic per fold, `STAGE_S_COLS`) with sample weights on the **TOP head only**:

`w_i = 1 + 2 · 1[name in realized top decile of its date]`

Fixed, no sweep. BOTTOM head unchanged (frozen 2.c cache). `scale_pos_weight` left at default (the weighting is the intervention). Signal: spread as usual (`p_top_DIR − p_bot_2c`).

Rationale (verbatim): label-distribution reweighting for rare extreme positives (Yang et al. ICML 2021 lineage; logit-adjustment Menon et al. 2021), the cheapest tail-emphasis intervention available.

## C — Judgment grid (tail metrics primary; floored top-100; full OOS / trailing-18m / per-cycle)

Signals, frozen:

1. **frozen_spread** — 2.c cache, not retrained (baseline)
2. **twinrank** — TWIN-RANK signal
3. **spread_twinrank** — fixed 50/50 CS-rank average of (1) and (2)
4. **dir_spread** — DIR-weighted top + frozen 2.c bottom
5. **dir_twinrank** — 50/50 CS-rank average of (4) and (2) (informational)

Columns: tail-IC top/bottom (NW-t, lag=14), top-decile overlap, monster top-3 capture, whole-list RankIC (secondary), vol-rank-corr. Secondary: crude-14d book (Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side) CAGR/MaxDD per signal — information only, no adoption.

Eval universe: floored PIT top-100 labeled CS vs next-14d excess, restricted to Binance-listed names at t.

## Mechanical comparison (frozen)

- **TWIN-RANK EXTRACTS** iff `twinrank` minus `frozen_spread` has tail-IC(top-half) ≥ +0.010 **and** overlap ≥ +0.015 **and** the TWIN-RANK vol-matched tail-IC null passes; else not.
- **DIR LIVE** iff `dir_spread` minus `frozen_spread` clears the same two deltas **and** the DIR vol-matched tail-IC null passes; else not.
- If both fail: record the PRICE-VOLUME TAIL CEILING clause on the ledger.
- Retro RANK (4v2 single head) is an informational null column: was any of its gain real beyond vol? Not a rewrite of the Phase 4 v2 BARREN verdict.

Verdicts mechanical. Nothing adopted.

## What this freeze does not do

- No unlock / listing / search-volume ingest
- No product file changes, no schedules, no live components
- No GPU, no architecture sweeps, no extra LambdaRank or DIR configs
- No mutation of the 2.c pred cache, the Phase 4 v2 pred cache, or the CMC panel
- No retest of positioning / price-additions
