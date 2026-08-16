# PROJECT FORGE — Phase F0

**BACKTEST AND ANALYSIS ONLY.** Evolutionary strategy miner. Compounding fitness. Nested windows.
Master only. CPU only. Zero GPU. Frozen products untouched. Nothing adopted.

**Stage:** `judge`.

## Non-contamination clause (verbatim)

> The PI's hand-made formulas are EXCLUDED from this project: not as seeds, not as operator biases, not as fitness shaping. The search space contains generic primitives and operators only. If the miner independently evolves similar structures, they are reported without censorship — that is evidence the method works, not contamination.

## Search space and fitness (frozen)

See `reports/forge_f0_addendum.md` for the full freeze, including search-space completions.

> fitness(S) = median over the perturbation set P of [ann. net log-wealth of BOOK(S, p)] − λ_c · nodes(S)/25, computed ONLY on the MINE window. BOOK(S, p): long-only, top-k by S (k per p), equal-weight, daily rebalance, 10 bps/side, always invested, death convention. Perturbation set P (fixed): k ∈ {3, 5, 8} × rebalance ∈ {daily, weekly} × the two universes = 12 books. λ_c = 0.02. A formula whose 12-book spread (max−min ann log-wealth) exceeds 1.0 is discarded regardless of median (knife-edge filter). No correlation metric appears anywhere in the fitness.

## Nested windows

- MINE: 2019-10-20 → 2022-12-31 (pop 2000, gens 60)
- SELECT: 2023-01-01 → 2024-12-31 (untouched by evolution)
- JUDGE: 2025-01-01 → 2026-08-13 (touched once, after champions committed)

## Pre-registered judgment (verbatim, before results)

> On the JUDGE window, per champion (headline book = k=5, daily, top-100 DV): FORGE-ALIVE if ≥1 champion has (a) total return ≥ BTC B&H same-window; (b) relative-line Sharpe > 0; (c) MaxDD ≤ 1.15 × BTC's. FORGE-STRONG if ≥2 champions pass, or one passes with relative-line Sharpe ≥ 0.5. FORGE-DEAD otherwise, and the ledger records that GP-mined formulaic strategies do not survive nesting on this data. Champions' formulas, MINE/SELECT/JUDGE numbers, and the full per-cycle honesty tables are printed regardless. Benchmarks on JUDGE: BTC B&H, EW baskets (both universes), frozen-spread crude book. Mechanical, no post-hoc adjustment; no re-mining, no second look at JUDGE.

## Death-in-position

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Run hygiene

- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert)
- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- gpu_used = `False`
- elapsed_sec = `48.68971633911133`
- budget_flag = `False`
- null_budget_flag = `None`
- n_jobs = `32`
- cube mean DV100 / MCAP50 = `98.7` / `50.0`
- code/freeze sha = `f7fddbb4591ac791a3c07ceaef163ed83d0a5cc0`
- champions commit sha (pre-JUDGE freeze) = `f7fddbb4591ac791a3c07ceaef163ed83d0a5cc0`

## Evolution (MINE)

| gen | best fitness | median finite | n finite | best nodes |
|-----|--------------|---------------|----------|------------|
| 0 | 0.9416 | 0.0424 | 963 | 2 |
| 1 | 1.0349 | 0.3779 | 1519 | 5 |
| 2 | 1.0349 | 0.8119 | 1688 | 5 |
| 3 | 1.1157 | 0.8119 | 1603 | 8 |
| 4 | 1.1710 | 0.8119 | 1554 | 12 |
| 5 | 1.2611 | 0.8755 | 1593 | 13 |
| 6 | 1.2611 | 1.0317 | 1667 | 13 |
| 7 | 1.3236 | 1.0468 | 1564 | 12 |
| 8 | 1.3236 | 1.0333 | 1595 | 12 |
| 9 | 1.4141 | 1.0886 | 1587 | 21 |
| 10 | 1.4355 | 1.0726 | 1653 | 24 |
| 11 | 1.4784 | 1.0502 | 1656 | 18 |
| 12 | 1.4784 | 1.0718 | 1667 | 18 |
| 13 | 1.4942 | 1.1804 | 1698 | 24 |
| 14 | 1.4999 | 1.1743 | 1681 | 18 |
| 15 | 1.4999 | 1.1660 | 1662 | 18 |
| 16 | 1.5400 | 1.0324 | 1635 | 23 |
| 17 | 1.5408 | 1.1743 | 1685 | 22 |
| 18 | 1.5424 | 1.1272 | 1637 | 20 |
| 19 | 1.5681 | 1.0789 | 1607 | 25 |
| 20 | 1.5697 | 1.1434 | 1574 | 23 |
| 21 | 1.5697 | 1.3037 | 1546 | 23 |
| 22 | 1.5713 | 1.1564 | 1532 | 21 |
| 23 | 1.5769 | 1.0561 | 1558 | 24 |
| 24 | 1.5769 | 1.0378 | 1589 | 24 |
| 25 | 1.5769 | 1.0349 | 1609 | 24 |
| 26 | 1.5769 | 1.0253 | 1570 | 24 |
| 27 | 1.5769 | 1.0274 | 1603 | 24 |
| 28 | 1.5769 | 1.0349 | 1635 | 24 |
| 29 | 1.5769 | 1.0307 | 1645 | 24 |
| 30 | 1.5769 | 1.2201 | 1603 | 24 |
| 31 | 1.5774 | 1.2352 | 1614 | 25 |
| 32 | 1.5811 | 1.2886 | 1661 | 24 |
| 33 | 1.5926 | 1.3677 | 1662 | 25 |
| 34 | 1.5959 | 1.3365 | 1621 | 25 |
| 35 | 1.5988 | 1.3204 | 1615 | 24 |
| 36 | 1.5988 | 1.3728 | 1607 | 24 |
| 37 | 1.5988 | 1.3761 | 1590 | 24 |
| 38 | 1.5988 | 1.3664 | 1637 | 24 |
| 39 | 1.6072 | 1.3898 | 1594 | 25 |
| 40 | 1.6210 | 1.3699 | 1574 | 25 |
| 41 | 1.6210 | 1.3699 | 1591 | 25 |
| 42 | 1.6210 | 1.3741 | 1590 | 25 |
| 43 | 1.6210 | 1.3704 | 1611 | 25 |
| 44 | 1.6210 | 1.3824 | 1612 | 25 |
| 45 | 1.6210 | 1.3888 | 1655 | 25 |
| 46 | 1.6210 | 1.3851 | 1689 | 25 |
| 47 | 1.6210 | 1.3247 | 1674 | 25 |
| 48 | 1.6210 | 1.3612 | 1664 | 25 |
| 49 | 1.6210 | 1.3888 | 1675 | 25 |
| 50 | 1.6210 | 1.3019 | 1653 | 25 |

Early stop / last gen: `None`. Best MINE formula: `None`.

## MINE → SELECT decay (top-20 by MINE fitness)

| rank | nodes | MINE fitness | SELECT fitness | formula |
|------|-------|--------------|----------------|---------|
| 1 | 25 | 1.6210 | 0.4754 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_max_28(turnover)), dist_high_28))))` |
| 2 | 25 | 1.6168 | 0.4713 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(lag_28(ts_std_14(turnover)), dist_high_28))))` |
| 3 | 25 | 1.6135 | 0.4783 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` |
| 4 | 24 | 1.6108 | 0.4791 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(amihud_14, mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` |
| 5 | 24 | 1.6108 | 0.4791 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(dist_ath, mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` |
| 6 | 24 | 1.6108 | 0.4791 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(mcap, mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` |
| 7 | 25 | 1.6100 | 0.4783 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(ts_std_14(turnover), mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` |
| 8 | 25 | 1.6100 | 0.4783 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(ts_std_14(close), mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` |
| 9 | 25 | 1.6100 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), pdiv(funding_z_7, pdiv(ts_rank_5(ts_std_14(dOI_7)), dist_high_28))))` |
| 10 | 25 | 1.6100 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), pdiv(funding_z_7, pdiv(ts_rank_5(ts_std_14(basis)), dist_high_28))))` |
| 11 | 25 | 1.6100 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), pdiv(funding_z_7, pdiv(ts_rank_5(ts_std_14(funding_z_30)), dist_high_28))))` |
| 12 | 25 | 1.6077 | 0.4783 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_14(ts_max_28(turnover)), dist_high_28))))` |
| 13 | 25 | 1.6072 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(turnover)), dist_high_28))))` |
| 14 | 25 | 1.6072 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_std_14(ts_rank_5(turnover)), dist_high_28))))` |
| 15 | 25 | 1.6070 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(pos_missing)), dist_high_28))))` |
| 16 | 25 | 1.6070 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(close)), dist_high_28))))` |
| 17 | 25 | 1.6070 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(dOI_30)), dist_high_28))))` |
| 18 | 25 | 1.6070 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(taker_imb_7)), dist_high_28))))` |
| 19 | 25 | 1.6070 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(funding_z_30)), dist_high_28))))` |
| 20 | 25 | 1.6070 | 0.4681 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(z_cs(dist_high_28), mul(funding_z_7, pdiv(ts_rank_5(ts_std_14(funding_z_7)), dist_high_28))))` |

## Champions (formulas verbatim; frozen before JUDGE)

Formulas frozen in commit `f7fddbb4591ac791a3c07ceaef163ed83d0a5cc0` before this JUDGE job. The final MINE population collapsed to one lineage (top-20 share the same skeleton). Greedy SELECT diversity (headline corr < 0.7) keeps **1** champion. This is reported, not patched.

### Champion 1

`min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(amihud_14, mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))`

- nodes = `24`
- MINE fitness = `1.6108`
- SELECT fitness = `0.4791`
- SELECT 12-book median / spread = `0.4983` / `0.9233`
- max pairwise headline corr (SELECT) = `0.0000`

## Null-mining floor

- null gens = `30`
- best null-mined SELECT fitness = `−∞` (all 50 null-mined formulas discarded on real SELECT: knife-edge or invalid book)
- best null expr (by shuffled-MINE fitness, then discarded on SELECT) = `ts_std_28(min(min(min(ts_std_28(mcap_rank), min(abs(ts_mean_5(ret_21)), ts_std_28(mcap_rank))), ts_std_28(abs(ret_21))), ts_std_28(mcap)))`

Champions' SELECT fitness vs null floor:

- C1 SELECT `0.4791` − null `−∞` (discarded) = C1 is above the noise floor on SELECT; JUDGE is a separate window.

## Ingredient census (top-50 by MINE fitness)

- `primitives[dist_high_28×137, turnover×78, dist_ath×54, mcap×54, funding_z_30×53] operators[mul×196, min×150, pdiv×104, ts_std×86, sub×50]`

| primitive | count |
|-----------|-------|
| `dist_high_28` | 137 |
| `turnover` | 78 |
| `dist_ath` | 54 |
| `mcap` | 54 |
| `funding_z_30` | 53 |
| `pos_missing` | 53 |
| `funding_z_7` | 53 |
| `mcap_rank` | 51 |
| `dist_low_28` | 3 |
| `close` | 2 |
| `basis` | 2 |
| `ret_1` | 2 |
| `amihud_14` | 1 |
| `dOI_7` | 1 |
| `dOI_30` | 1 |
| `taker_imb_7` | 1 |
| `ret_3` | 1 |
| `ret_14` | 1 |
| `ret_28` | 1 |
| `ret_21` | 1 |

| operator | count |
|----------|-------|
| `mul` | 196 |
| `min` | 150 |
| `pdiv` | 104 |
| `ts_std` | 86 |
| `sub` | 50 |
| `ts_rank` | 41 |
| `z_cs` | 35 |
| `ts_max` | 18 |
| `ts_mean` | 7 |
| `lag` | 1 |
| `neg` | 1 |
| `abs` | 1 |

## JUDGE

**Verdict: FORGE-DEAD** (n_pass=0 / 1).

Headline book = k=5, daily, floored PIT top-100 DV. Mechanical, no post-hoc adjustment.

| champion | formula | total | BTC | rel Sharpe | MaxDD | BTC MaxDD | a | b | c | pass |
|----------|---------|-------|-----|------------|-------|-----------|---|---|---|------|
| C1 | `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(amihud_14, mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))` | -0.5491 | -0.3308 | -0.1480 | -0.7245 | -0.5297 | False | False | False | False |

### Per-champion honesty (PHASE2 cycles on JUDGE window)

#### C1 `min(mul(mcap_rank, mul(sub(mul(min(dist_ath, pdiv(funding_z_30, dist_high_28)), ts_std_14(turnover)), pos_missing), mcap)), min(amihud_14, mul(funding_z_7, pdiv(ts_mean_28(ts_std_14(turnover)), dist_high_28))))`

| cycle | n | book total | BTC total | rel Sharpe | MaxDD | BTC MaxDD |
|-------|---|------------|-----------|------------|-------|-----------|
| 2025-26 | 584 | -0.5491 | -0.3138 | -0.1480 | -0.7245 | -0.5297 |

### Benchmarks (JUDGE)

| book | total | CAGR | Sharpe | MaxDD | rel Sharpe | corr BTC |
|------|-------|------|--------|-------|------------|----------|
| btc_bh | -0.3308 | -0.2220 | -0.3623 | -0.5297 | 0.0000 | 1.0000 |
| ew_dv100 | 1.3378 | 0.7002 | 0.7055 | -0.8152 | 0.7135 | 0.0771 |
| ew_mcap50 | -0.6550 | -0.4858 | -0.7946 | -0.6887 | -1.1185 | 0.8458 |
| frozen_spread | -0.2747 | -0.1847 | -0.1774 | -0.4898 | 0.2794 | 0.7843 |

If FORGE-DEAD: GP-mined formulaic strategies do not survive nesting on this data.

Mechanical, no post-hoc adjustment. Frozen products untouched. Nothing adopted.
