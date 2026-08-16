# PROJECT FORGE — Phase F0 freeze addendum

**Status:** FROZEN before results. Evolutionary STRATEGY miner. Fitness = net log-wealth of concentrated long books, not correlation.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Zero GPU.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned CMC panel is read-only (sha256 asserted). 2.c / 4.b / 4v2 caches are not mutated. Canonical pricing = Binance-hybrid where listed (3.e). Pegged assets excluded from the mcap universe. Death-in-position convention. `PYTHONUNBUFFERED`. Heartbeat 60s. Watchdog **kills** at 20 min silence. Checkpoint every generation (idempotent resume). Budget guard: if wall-clock exceeds 6h, checkpoint and finish with the current generation, flagged.

This phase produces a **record**, not a product. Nothing is adopted.

## Non-contamination clause (verbatim)

> The PI's hand-made formulas are EXCLUDED from this project: not as seeds, not as operator biases, not as fitness shaping. The search space contains generic primitives and operators only. If the miner independently evolves similar structures, they are reported without censorship — that is evidence the method works, not contamination.

No MANUEL-2 (or any other PI) formula is injected as a seed, as a preferred operator, or as a term in fitness.

## Search space (frozen)

Primitives per (name, date), all data ≤ t: close, volume, mcap, ret_k for k ∈ {1,3,7,14,21,28,63,90}, std_k for k ∈ {14,30,63,90}, high/low distances {28,90}, dist_ath, amihud_14, turnover, age, mcap_rank, and the derivatives block where mapped (funding_z_7/30, ΔOI_7/30, basis, taker_imb_7; 0+flag where unmapped).

Operators: {+, −, ×, protected ÷, log1p|x|, abs, neg, min, max, rank_cs, z_cs, Φ_cs (normal CDF of z_cs), ts_mean_k, ts_std_k, ts_max_k, ts_min_k, ts_rank_k, lag_k} with k ∈ {5,14,28,63}. Max formula depth 8, max 25 nodes.

Universe for the book: floored PIT top-100 DV (house standard) AND top-50 mcap ex-pegged (both evaluated; fitness = mean of the two — cross-universe robustness by construction).

Operative fitness is the section below (median over the 12 books in P). Both universes sit inside P; that is the cross-universe robustness. There is no second fitness definition.

## Fitness (verbatim, the core innovation)

> fitness(S) = median over the perturbation set P of [ann. net log-wealth of BOOK(S, p)] − λ_c · nodes(S)/25, computed ONLY on the MINE window. BOOK(S, p): long-only, top-k by S (k per p), equal-weight, daily rebalance, 10 bps/side, always invested, death convention. Perturbation set P (fixed): k ∈ {3, 5, 8} × rebalance ∈ {daily, weekly} × the two universes = 12 books. λ_c = 0.02. A formula whose 12-book spread (max−min ann log-wealth) exceeds 1.0 is discarded regardless of median (knife-edge filter). No correlation metric appears anywhere in the fitness.

No IC, RankIC, or correlation enters fitness. Pairwise book-PnL correlation is used only later, on SELECT, for diversity among already-scored champions, and in JUDGE reporting.

## Nested windows (frozen; the anti-overfit spine)

- **MINE:** 2019-10-20 → 2022-12-31. GP evolves here. Population 2000, 60 generations, tournament 20, crossover 0.7 / subtree-mutation 0.25 / point 0.05. Early stop if best fitness flat 10 generations. gplearn-style. CPU parallel. Concurrency ≤ 50.
- **SELECT:** 2023-01-01 → 2024-12-31, untouched by evolution. From the final population, take the top-50 by MINE fitness, re-score their 12-book median on SELECT only; keep the top 5 DIVERSE champions (pairwise daily book-PnL corr on SELECT < 0.7; greedy selection by SELECT fitness). Headline book for diversity: k=5, daily, top-100 DV.
- **JUDGE:** 2025-01-01 → 2026-08-13 (`PHASE3C_REF_END`). Touched ONCE, by the 5 champions only, after they are frozen and committed with their formulas printed in the report BEFORE judgment.

## Pre-registered judgment (verbatim, before results)

> On the JUDGE window, per champion (headline book = k=5, daily, top-100 DV): FORGE-ALIVE if ≥1 champion has (a) total return ≥ BTC B&H same-window; (b) relative-line Sharpe > 0; (c) MaxDD ≤ 1.15 × BTC's. FORGE-STRONG if ≥2 champions pass, or one passes with relative-line Sharpe ≥ 0.5. FORGE-DEAD otherwise, and the ledger records that GP-mined formulaic strategies do not survive nesting on this data. Champions' formulas, MINE/SELECT/JUDGE numbers, and the full per-cycle honesty tables are printed regardless. Benchmarks on JUDGE: BTC B&H, EW baskets (both universes), frozen-spread crude book. Mechanical, no post-hoc adjustment; no re-mining, no second look at JUDGE.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Spec completions (declared a priori, mandatory for implementability; not a sweep)

These freeze implementability. They are not results and are not tuned after seeing JUDGE.

- **Pricing:** Binance spot close where a print exists, else cleaned CMC (hybrid). Score inputs `close`, `ret_*`, `std_*` use hybrid. PnL uses hybrid. Universe mcap ranks use cleaned-panel CMC mcap. Dollar volume primitive `volume` = panel `dv`.
- **OHLC distances:** `dist_high_k` = CMC close / rolling-k CMC high − 1; `dist_low_k` = CMC close / rolling-k CMC low − 1; k ∈ {28,90}; min_periods = k. Not BTC-denominated.
- **`dist_ath`:** hybrid close / expanding max − 1, data ≤ t.
- **`amihud_14`:** rolling mean of `|r1| / dv` on hybrid r1 and panel dv, window 14, min_periods=5 (house amihud).
- **`turnover`:** `dv / mcap`.
- **`age`:** calendar days since first panel appearance of that id (raw count, not log).
- **`mcap_rank`:** rank of CMC mcap among all non-`STABLE_OR_WRAP` names present that day (`method=first`, 1 = largest), then aligned to the cube. Not CS-z.
- **Returns / std:** simple returns; `pct_change(fill_method=None)` so missing prints are NA, not zero-vol. Rolling std of daily simple returns uses pandas ddof=1, min_periods=window.
- **CS z / Φ:** `z_cs` is per-date mean/std among the **book universe of that evaluation** (DV100 vs MCAP50 differ) on finite values, ddof=0; zero sd → zeros. `Φ_cs(x) = ndtr(z_cs(x))` (the operator applies z then the standard normal CDF). `rank_cs` is average-rank / n among that same universe mask (pandas `rank(method=average) / n`).
- **TS operators:** applied per id on the cube calendar, min_periods=k, data ≤ t. `ts_std` uses ddof=1. `ts_rank_k` is the rolling percentile of the current value among the last k finite observations (nan if fewer than k finite). `lag_k` shifts by k cube rows (no wrap).
- **Protected ÷:** if `|y| < 1e-8` return 1.0; otherwise `x/y`. Inf after any op becomes NaN.
- **Derivatives:** `funding_z_7`, `funding_z_30`, `dOI_7`, `dOI_30`, `basis`, `taker_imb_7` from the frozen 4v2 positioning cache when present; unmapped name-days are 0 with `pos_missing=1`. Cache is read-only. If the cache file is absent, every name-day is unmapped (0+flag). Positioning is **not** recomputed.
- **DV100 universe:** floored PIT top-100 dollar-volume file, as-of / ffilled onto the close calendar. No extra peg filter (house standard).
- **MCAP50 universe:** each date, top-50 by CMC mcap among names that are not tagged-pegged and not flagged by the MANUEL-2 `|90d CMC ret| < 2%` heuristic, with a listed hybrid close. BTC is eligible. PIT (no future peek); last non-empty membership is carried forward on dates with no rank.
- **Book construction:** long-only, top-k by score among the day's universe with finite score, listed close, and finite next-day hybrid return. Equal weight. Always invested: if fewer than k valid names, take all valid; if none, EW the listed universe members with finite next-day return. Weekly = Mondays (`weekday==0`) plus the first session of the window. Weights held between rebalances. After death, renormalize; if the book is empty, EW the universe as above.
- **Costs:** 10 bps/side on traded notional, matching the house two-way convention: `cost = 0.001 × Σ|Δw|` (same as MANUEL-2). Charged whenever weights change, including death exits.
- **Death:** a held name whose hybrid close is missing at t, or whose next-day return is non-finite, is force-exited at the last available close (that day's contribution is 0). Count and weight of forced exits are reported on JUDGE books.
- **Ann. net log-wealth:** `Σ log(1 + r_net) × (365 / n_days)` on the window's daily net returns (formation t earns t→t+1, labeled at t+1). Invalid / empty / NaN book → −∞. Last cube day has no forward return and is not a formation.
- **Knife-edge:** max − min of the 12 finite-or-not ann log-wealths > 1.0 → discard (fitness −∞), regardless of median.
- **Complexity:** `λ_c = 0.02`; penalty `0.02 × nodes(S)/25`. Node count = every primitive and operator node. Depth = 1 + max child depth; a terminal has depth 1.
- **GP (gplearn-style, custom CS/TS ops, not the gplearn package):** ramped half-and-half init, depths 2–6 cycling, 50% full / 50% grow. Function vs terminal at internal grow nodes uses n_term/(n_term+n_fun). Offspring operator is mutually exclusive: crossover 0.70 / subtree-mutation 0.25 / point 0.05. Tournament size 20. 1-elitist carry of the single best (declared here; not a sweep). Seed 42. Early stop if the best fitness does not improve by more than 1e-12 for 10 generations. Trees violating max depth 8 or max 25 nodes are rejected and retried.
- **Parallelism:** one fat Modal container, `ProcessPool` after cubes are loaded (Linux fork, copy-on-write). Worker count = `min(32, cpu_count, 50)`. "Concurrency ≤ 50" is a cap, not a 50-container fan-out (primitive cubes must not be reloaded 50 times). Zero GPU. `CUDA_VISIBLE_DEVICES=""`.
- **Null-mining control (1 arm, half budget):** identical GP on within-vol-quintile shuffled **forward returns** of the MINE window, 30 generations, population 2000, independent RNG stream `SeedSequence([42, 7])`. Vol buckets = per-date quintiles of primitive `std_30` among names with finite vol and finite fwd that day (`vol_bucket_ids`). Shuffle is joint within (date, bucket). SELECT/JUDGE forward returns are **not** shuffled. After null evolution, null formulas are scored on **real** SELECT; the best null-mined SELECT fitness is the noise floor.
- **SELECT diversity:** greedy by SELECT fitness (same 12-book formula, including knife-edge and complexity, on SELECT only). Skip a candidate if its headline daily book-PnL correlation on SELECT vs any already-chosen champion is ≥ 0.7. Headline = k=5, daily, DV100.
- **JUDGE MaxDD:** house signed MaxDD (negative). Clause (c) is `|book MaxDD| ≤ 1.15 × |BTC MaxDD|`.
- **JUDGE protocol:** two-stage. (1) MINE+SELECT+null write champions with formulas verbatim; that report is committed; JUDGE is not touched. (2) A later job loads the committed champion formulas only and evaluates JUDGE once. No re-mining, no second look.
- **Benchmarks on JUDGE only:** BTC B&H (hybrid BTC, zero cost); EW baskets of each universe (daily, 10 bps/side, always invested, death convention); frozen-spread crude book (Ladder-1, 2.c cache, 14d periodic long, information/reference).
- **Warmup:** primitive cubes from 2018-01-01 so 90d returns and TS windows exist at MINE start. Cube ids = union of names that ever appear in DV100 or MCAP50 over the cube, plus BTC.
- **Checkpoints:** every generation, JSON on the volume (`/data/quant/btcb/forge/`). Resume is idempotent. Watchdog kill attempts a checkpoint first.
- **Top-k ties:** first-encountered among equal scores (stable enough; not a sweep).

No parameter search. Mechanical, no post-hoc adjustment. Nothing adopted.
