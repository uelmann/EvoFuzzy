# BTC-BEATER Phase 8 — MODEL-ZOO

**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. Frozen products untouched. Pricing = Binance-hybrid (3.e canonical). Master only. GPU only for Arm B if TabPFN required it.

Independent of Phases 7.c / 7.d. One config per arm. Zero architecture search.

## Firewall (verbatim, before results)

> The PI's hand-made formulas (including MANUEL-2) are quarantined from this phase: not imported, not seeded, not used as features or targets.

## PI data-perimeter (verbatim)

> Catalyst and attention data families (unlocks, listing announcements, search volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume plus derivatives data already retrievable (funding, open interest, basis, taker flows).

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Pre-registered criteria (verbatim, before results)

> An arm is LIVE if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread with the vol-matched null passing. An arm is a WHOLE-RANKING LEAD if instead ΔRankIC ≥ +0.010 with the null passing (recorded for a fresh production-book phase, not adopted here). LINEAR-CEILING (informational, from Arm C): if Arm C's whole-list RankIC ≥ 0.90 × the frozen spread's, the ledger records that nonlinearity contributes less than 10% of the daily signal and future daily modeling effort is unjustified. If any arm's signal has correlation < 0.6 with the frozen spread while reaching RankIC ≥ 0.10, it is recorded as an ORTHOGONAL SIGNAL candidate for a fresh blending phase. Nothing adopted here. Mechanical, no post-hoc adjustment.

## TabPFN caveat (verbatim, before results)

> TabPFN assumes i.i.d.-like structure from its synthetic prior; financial non-stationarity violates it. This arm tests whether that matters in practice.

## Date subsample (verbatim, before results)

> Primary comparison = full OOS for every arm that completes. If TabPFN cannot finish full OOS inside the $20 GPU cap, ALL arms (and the frozen spread) are judged on the pre-declared 1-in-3 OOS date subsample: sorted unique OOS dates, keep i % 3 == 0 (0-indexed). Arms A/C still report full-OOS metrics as informational. The 1-in-3 rule is frozen before results.

## Vol-matched null (verbatim, before results)

> VOL-MATCHED NULL on the single BEST zoo arm by whole-list RankIC: folds {5,15,21,24} × 15 within-vol-quintile shuffles (first 15 of NULL_SHUFFLE_SEEDS), Modal .map fan-out. Skill = real exceeds vol-matched null p95 on ≥3/4 folds OR Stouffer z ≥ 3.0. Bias = 2·SE band around the fold's own null mean; CONTAMINATED iff ≥2 fold violations. LIVE uses the tail-IC(top-half) gate; WHOLE-RANKING LEAD uses the RankIC gate on the same shuffles. Only the best arm is nulled; other arms cannot be LIVE or LEAD. CS-ATTN null cells are cold-start, primary seed 42 only (not 3-init bag; null folds are non-contiguous so warm-start does not apply). Ridge and TabPFN nulls rerun the frozen procedure. RankIC null band is recorded for the chart.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- Window 2016-01-01 → 2026-08-08 n_dates=2473
- Judgment date set = `full_oos` n=2473
- GPU used = `True` type=`A10G` estimated USD=`0.23` cap=$20
- TabPFN subsample flag = `False`
- Best arm by RankIC = `ridge`
- Elapsed sec = `326.3`

## Per-arm config dumps (frozen, one each)

```json
{
  "cs_attn": {
    "n_in": 33,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.0,
    "positional_encoding": false,
    "temporal_encoder": false,
    "lr": 0.0001,
    "lr_min": 1e-05,
    "weight_decay": 0.0001,
    "clip": 1.0,
    "es_floor": 10,
    "patience": 8,
    "cap": 40,
    "swa_window": 3,
    "inner_holdout_dates": 120,
    "seeds": [
      42,
      43,
      44
    ],
    "date_batch": 64,
    "optimizer": "AdamW",
    "scheduler": "cosine",
    "n_params": 69250
  },
  "ridge": {
    "features": "cs_percentile_ranks_of_STAGE_S_COLS",
    "target": "cs_percentile_rank_of_excess_h14",
    "model": "sklearn.linear_model.Ridge",
    "fit_intercept": true,
    "alpha_grid": [
      0.01,
      0.1,
      1.0,
      10.0,
      100.0
    ],
    "alpha_rule": "max inner-ho RankIC; ties \u2192 larger alpha",
    "inner_holdout_dates": 120
  },
  "tabpfn": {
    "context_cap": 10000,
    "n_estimators": 8,
    "features": "STAGE_S_COLS (33)",
    "targets": [
      "y_h14 top-quintile",
      "y_bot_h14 bottom-quintile"
    ],
    "signal": "p_top - p_bot",
    "fine_tune": false,
    "gradient_steps": 0,
    "subsample": "even across train dates, seed 42, one context per fold",
    "inference": "batched per fold (all val rows one predict_proba per head)",
    "device": "cuda",
    "total_pred_sec": 629.5124981403351,
    "total_elapsed_sec": 736.153608083725,
    "n_dates": 2492,
    "mean_pred_sec_per_date": 0.2526133620145807,
    "n_folds_ok": 28
  }
}
```

## Wall-times and budget flags

- Arm A CS-ATTN elapsed_sec = `6254.8` (CPU)
- Arm B TabPFN elapsed_sec = `737.3` pred_sec_total=`629.5` pred_sec_per_date=`0.253` status=`ok`
- Arm C ridge elapsed_sec = `37.9` (CPU)
- GPU estimate USD = `0.23` / $20 cap; flag=`None`

## Mechanical verdicts

- Arm A CS-ATTN-DAILY: **NOT LIVE (null not run)** (Δtail-IC `+0.0010`, Δoverlap `-0.0004`, ΔRankIC `-0.0166`)
- Arm B TabPFN v2: **NOT LIVE (null not run)** (Δtail-IC `+0.0187`, Δoverlap `+0.0062`, ΔRankIC `+0.0123`)
- Arm C RIDGE ON RANKS: **NOT LIVE** (Δtail-IC `+0.0034`, Δoverlap `-0.0119`, ΔRankIC `+0.0145`)
- LINEAR-CEILING YES: Arm C RankIC=0.1767 vs frozen RankIC=0.1622 (ratio=1.089, threshold=0.9) — nonlinearity contributes less than 10% of the daily signal; future daily modeling effort is unjustified.
- ORTHOGONAL SIGNAL: none

## Conclusion

Phase 8 MODEL-ZOO on full_oos: A CS-ATTN NOT LIVE (null not run); B TabPFN NOT LIVE (null not run); C RIDGE NOT LIVE. LINEAR-CEILING YES: Arm C RankIC=0.1767 vs frozen RankIC=0.1622 (ratio=1.089, threshold=0.9) — nonlinearity contributes less than 10% of the daily signal; future daily modeling effort is unjustified. ORTHOGONAL=none. Nothing adopted. CS-ATTN uses the cross-section (mean attention entropy 0.937, collapse_to_self=false, ~69k params) but does not beat the frozen spread. TabPFN finished full OOS (no 1-in-3 subsample) in 737s on A10G (~$0.23 / $20 cap). Ridge is the best whole-list RankIC of the zoo but its vol-matched null is PARKED (0/4 folds). Nothing adopted.

## 1 — Judgment grid (judgment date set)

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | n |
|--------|-------------|------|-------------|---------|---------|--------|---|
| frozen spread | 0.0684 | 6.07 | 0.1358 | 0.0820 | 0.0576 | 0.1622 | 2473 |
| CS-ATTN-DAILY | 0.0694 | 6.91 | 0.1056 | 0.0817 | 0.0650 | 0.1456 | 2473 |
| TabPFN v2 | 0.0871 | 6.28 | 0.1362 | 0.0883 | 0.0620 | 0.1745 | 2473 |
| RIDGE ON RANKS | 0.0718 | 5.25 | 0.1402 | 0.0702 | 0.0445 | 0.1767 | 2473 |

Trailing 18m:

| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC |
|--------|-------------|------|-------------|---------|---------|--------|
| frozen spread | 0.1219 | 4.47 | 0.1812 | 0.0908 | 0.0450 | 0.2309 |
| CS-ATTN-DAILY | 0.0528 | 2.82 | 0.0926 | 0.0691 | 0.0560 | 0.1268 |
| TabPFN v2 | 0.1072 | 3.17 | 0.1937 | 0.0702 | 0.0225 | 0.2352 |
| RIDGE ON RANKS | 0.1131 | 4.02 | 0.1983 | 0.0629 | 0.0335 | 0.2519 |

Per-cycle RankIC:

| cycle | frozen spread | CS-ATTN-DAILY | TabPFN v2 | RIDGE ON RANKS |
|-------|------|------|------|------|
| 2019-20 | 0.1324 | 0.1475 | 0.1482 | 0.1326 |
| 2021 | 0.1135 | 0.0905 | 0.1057 | 0.1213 |
| 2022 | 0.1964 | 0.2024 | 0.2048 | 0.2062 |
| 2023-24 | 0.1330 | 0.1576 | 0.1564 | 0.1522 |
| 2025-26 | 0.2320 | 0.1276 | 0.2426 | 0.2588 |

Per-cycle tail-IC(top-half):

| cycle | frozen spread | CS-ATTN-DAILY | TabPFN v2 | RIDGE ON RANKS |
|-------|------|------|------|------|
| 2019-20 | 0.0261 | 0.0793 | 0.0589 | 0.0180 |
| 2021 | 0.0463 | 0.0331 | 0.0341 | 0.0392 |
| 2022 | 0.1106 | 0.1283 | 0.1502 | 0.1265 |
| 2023-24 | 0.0422 | 0.0641 | 0.0761 | 0.0521 |
| 2025-26 | 0.1217 | 0.0542 | 0.1163 | 0.1243 |

### Arm A seed dispersion (unbagged spread RankIC)

- seeds {42,43,44}: `{"42": {"rankic": 0.11036946279090232, "tail_ic_top": 0.04604096051053735, "overlap": 0.10482095520510404, "n_dates": 2473}, "43": {"rankic": 0.13047809487258893, "tail_ic_top": 0.05233888198106327, "overlap": 0.1029788381183448, "n_dates": 2473}, "44": {"rankic": 0.1095405835246057, "tail_ic_top": 0.043542649852359935, "overlap": 0.0961944556768657, "n_dates": 2473}, "rankic_mean": 0.11679604706269898, "rankic_std": 0.011856246641058454}`

## 2 — Vol-matched null (best arm only)

Best arm = `ridge`. Completions: CS-ATTN null = seed 42 cold-start; ridge/TabPFN = full frozen procedure. Folds {5,15,21,24} × 15.

**tail-IC(top-half) [LIVE gate]** verdict=`PARKED` bias_pass=True skill_pass=False exceed=0/4 violations=0 Stouffer z=`-11.485`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 5 | 15 | -0.0071 | -0.0071 | 0.0058 | True | 0.0082 | 0.0027 | False |
| 15 | 15 | 0.0963 | 0.0963 | 0.0039 | True | 0.1086 | 0.0911 | False |
| 21 | 15 | 0.2768 | 0.2768 | 0.0042 | True | 0.2890 | 0.1751 | False |
| 24 | 15 | 0.3303 | 0.3303 | 0.0052 | True | 0.3413 | 0.2208 | False |

**overlap** verdict=`PARKED` bias_pass=True skill_pass=False exceed=0/4 violations=0 Stouffer z=`-4.736`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 5 | 15 | 0.0803 | 0.0803 | 0.0035 | True | 0.0882 | 0.0794 | False |
| 15 | 15 | 0.0567 | 0.0567 | 0.0036 | True | 0.0658 | 0.0586 | False |
| 21 | 15 | 0.1575 | 0.1575 | 0.0050 | True | 0.1696 | 0.1050 | False |
| 24 | 15 | 0.0866 | 0.0866 | 0.0024 | True | 0.0935 | 0.0672 | False |

**monster top-3** verdict=`PARKED` bias_pass=True skill_pass=False exceed=1/4 violations=0 Stouffer z=`-0.390`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 5 | 15 | 0.0352 | 0.0352 | 0.0050 | True | 0.0484 | 0.0330 | False |
| 15 | 15 | 0.0650 | 0.0650 | 0.0029 | True | 0.0733 | 0.0403 | False |
| 21 | 15 | 0.0672 | 0.0672 | 0.0047 | True | 0.0806 | 0.0586 | False |
| 24 | 15 | 0.0171 | 0.0171 | 0.0009 | True | 0.0183 | 0.0256 | True |

**whole-list RankIC [LEAD gate]** verdict=`PARKED` bias_pass=True skill_pass=False exceed=0/4 violations=0 Stouffer z=`-17.736`.

| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |
|------|---|-----------|--------|------|---------|-----|------|-------------|
| 5 | 15 | 0.0874 | 0.0874 | 0.0016 | True | 0.0914 | 0.0774 | False |
| 15 | 15 | 0.1912 | 0.1912 | 0.0021 | True | 0.1977 | 0.1937 | False |
| 21 | 15 | 0.3306 | 0.3306 | 0.0012 | True | 0.3340 | 0.2962 | False |
| 24 | 15 | 0.3983 | 0.3983 | 0.0016 | True | 0.4019 | 0.3417 | False |

## 3 — Signal correlation (mean per-date Spearman)

n_dates=`2487`

|  | frozen_spread | cs_attn | tabpfn | ridge |
|--|---|---|---|---|
| frozen_spread | 1.0000 | 0.6162 | 0.8032 | 0.7988 |
| cs_attn | 0.6162 | 1.0000 | 0.6259 | 0.6297 |
| tabpfn | 0.8032 | 0.6259 | 1.0000 | 0.8635 |
| ridge | 0.7988 | 0.6297 | 0.8635 | 1.0000 |

## 4 — Arm A attention diagnostics

- mean attention entropy (normalized) = `0.9372`
- mean self-weight of highest-scored coin = `0.0103`
- collapse_to_self (≥0.50 self-weight) = `False`
- n diagnostic dates = `10`

Top-5 attended peers (highest-scored coin, 10 linspace OOS dates, fold model seed 42):

```json
[
  {
    "date": "2019-10-18",
    "query_id": 52,
    "top5": [
      {
        "id": 2348,
        "weight": 0.018373407423496246,
        "is_self": false
      },
      {
        "id": 3053,
        "weight": 0.01769331842660904,
        "is_self": false
      },
      {
        "id": 2830,
        "weight": 0.017657054588198662,
        "is_self": false
      },
      {
        "id": 3890,
        "weight": 0.01708199828863144,
        "is_self": false
      },
      {
        "id": 4079,
        "weight": 0.0166594460606575,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9725527048002998
  },
  {
    "date": "2020-07-21",
    "query_id": 3794,
    "top5": [
      {
        "id": 1934,
        "weight": 0.025845058262348175,
        "is_self": false
      },
      {
        "id": 1727,
        "weight": 0.02263428084552288,
        "is_self": false
      },
      {
        "id": 4846,
        "weight": 0.019362477585673332,
        "is_self": false
      },
      {
        "id": 2010,
        "weight": 0.018794285133481026,
        "is_self": false
      },
      {
        "id": 3783,
        "weight": 0.01833835057914257,
        "is_self": false
      }
    ],
    "mean_entropy": 0.954615604126549
  },
  {
    "date": "2021-04-24",
    "query_id": 3964,
    "top5": [
      {
        "id": 3945,
        "weight": 0.018797852098941803,
        "is_self": false
      },
      {
        "id": 4066,
        "weight": 0.017531130462884903,
        "is_self": false
      },
      {
        "id": 7158,
        "weight": 0.01617630571126938,
        "is_self": false
      },
      {
        "id": 2130,
        "weight": 0.015561823733150959,
        "is_self": false
      },
      {
        "id": 4256,
        "weight": 0.01522504910826683,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9382766221418037
  },
  {
    "date": "2022-01-25",
    "query_id": 2319,
    "top5": [
      {
        "id": 1934,
        "weight": 0.023357421159744263,
        "is_self": false
      },
      {
        "id": 6210,
        "weight": 0.022842060774564743,
        "is_self": false
      },
      {
        "id": 7080,
        "weight": 0.021512459963560104,
        "is_self": false
      },
      {
        "id": 7535,
        "weight": 0.02126886695623398,
        "is_self": false
      },
      {
        "id": 3801,
        "weight": 0.02070353366434574,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9350712007609715
  },
  {
    "date": "2022-10-29",
    "query_id": 8353,
    "top5": [
      {
        "id": 3437,
        "weight": 0.03879536688327789,
        "is_self": false
      },
      {
        "id": 74,
        "weight": 0.026838842779397964,
        "is_self": false
      },
      {
        "id": 7226,
        "weight": 0.026668354868888855,
        "is_self": false
      },
      {
        "id": 4256,
        "weight": 0.020568516105413437,
        "is_self": false
      },
      {
        "id": 6758,
        "weight": 0.01708681508898735,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9324575351926367
  },
  {
    "date": "2023-08-02",
    "query_id": 1839,
    "top5": [
      {
        "id": 7083,
        "weight": 0.01865459606051445,
        "is_self": false
      },
      {
        "id": 11419,
        "weight": 0.018462013453245163,
        "is_self": false
      },
      {
        "id": 9444,
        "weight": 0.018092673271894455,
        "is_self": false
      },
      {
        "id": 4039,
        "weight": 0.018002722412347794,
        "is_self": false
      },
      {
        "id": 7102,
        "weight": 0.01755046658217907,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9316546535750267
  },
  {
    "date": "2024-05-05",
    "query_id": 29520,
    "top5": [
      {
        "id": 28782,
        "weight": 0.019273318350315094,
        "is_self": false
      },
      {
        "id": 30096,
        "weight": 0.018265429884195328,
        "is_self": false
      },
      {
        "id": 28752,
        "weight": 0.01736251264810562,
        "is_self": false
      },
      {
        "id": 7080,
        "weight": 0.015870777890086174,
        "is_self": false
      },
      {
        "id": 20362,
        "weight": 0.01525571383535862,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9313570665044242
  },
  {
    "date": "2025-02-05",
    "query_id": 33788,
    "top5": [
      {
        "id": 16116,
        "weight": 0.024835584685206413,
        "is_self": false
      },
      {
        "id": 8526,
        "weight": 0.022183172404766083,
        "is_self": false
      },
      {
        "id": 6536,
        "weight": 0.021008452400565147,
        "is_self": false
      },
      {
        "id": 22533,
        "weight": 0.02060980722308159,
        "is_self": false
      },
      {
        "id": 7192,
        "weight": 0.017669925466179848,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9323006882861091
  },
  {
    "date": "2025-11-09",
    "query_id": 27075,
    "top5": [
      {
        "id": 328,
        "weight": 0.02038661204278469,
        "is_self": false
      },
      {
        "id": 16116,
        "weight": 0.02031692862510681,
        "is_self": false
      },
      {
        "id": 11092,
        "weight": 0.019334500655531883,
        "is_self": false
      },
      {
        "id": 34387,
        "weight": 0.018033038824796677,
        "is_self": false
      },
      {
        "id": 7961,
        "weight": 0.017707860097289085,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9209503242482036
  },
  {
    "date": "2026-08-13",
    "query_id": 22533,
    "top5": [
      {
        "id": 35881,
        "weight": 0.02546827122569084,
        "is_self": false
      },
      {
        "id": 37745,
        "weight": 0.023892898112535477,
        "is_self": false
      },
      {
        "id": 11294,
        "weight": 0.02107192948460579,
        "is_self": false
      },
      {
        "id": 5309,
        "weight": 0.020839380100369453,
        "is_self": false
      },
      {
        "id": 32994,
        "weight": 0.02058505266904831,
        "is_self": false
      }
    ],
    "mean_entropy": 0.9230069844948918
  }
]
```

## 5 — Crude 14d books (information only; nothing adopted)

| signal | total | CAGR | MaxDD | Sharpe | n_form |
|--------|-------|------|-------|--------|--------|
| frozen spread | 127.4% | 12.9% | -80.7% | 0.515 | 177 |
| CS-ATTN-DAILY | 608.2% | 33.4% | -77.4% | 0.766 | 177 |
| TabPFN v2 | 1279.5% | 47.2% | -75.0% | 0.948 | 177 |
| RIDGE ON RANKS | 430.5% | 27.9% | -79.2% | 0.724 | 177 |

- RankIC by arm with null band: `charts/btcb_phase8_rankic.png`
- Signal correlation heatmap: `charts/btcb_phase8_corr.png`
- Crude equity curves: `charts/btcb_phase8_equity.png`

Mechanical, no post-hoc adjustment. Frozen products untouched. Nothing adopted.
