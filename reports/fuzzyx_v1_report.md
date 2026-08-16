# FuzzyX-v1 report

**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. Does not replace COMBO / A0. Addendum: `reports/fuzzyx_addendum.md`.

**Mode:** `LOCAL-RESTRICTED`
**Verdict:** **CONTAMINATED**
**Params:** 17668

## Keep rule (verbatim)

> See `reports/fuzzyx_addendum.md`. VIABLE only if leakage, shuffle-bias, full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs A0 Sleeve A when A0 preds exist.

## Gates

| clause | result |
|---|---|
| (i) leakage | PASS |
| (ii) shuffle-bias | FAIL |
| (iii) Sharpe ≥ 0 | FAIL (-0.145 weekly) |
| (iv) vs A0 | SKIP |

- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top30`: **PASS** `{'name': 'universe_lookahead_top30', 'passed': True, 'n': 30, 'date': '2023-04-17', 'base_n': 30, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0}`

## Shuffle-bias folds

- `{'name': 'label_shuffle_bias', 'passed': False, 'mean_core': -0.03644843716174364, 'sd': 0.04910606875702977, 'se': 0.015528702420904764, 'threshold': 0.03105740484180953, 'n': 10, 'cores': [0.04429584741592407, -0.0772324651479721, -0.02077685482800007, -0.07638328522443771, 0.054053910076618195, -0.07811803370714188, -0.0614621676504612, -0.04796944558620453, -0.033096928149461746, -0.06779494881629944], 'fold_id': 0}`
- `{'name': 'label_shuffle_bias', 'passed': False, 'mean_core': 0.14243328971788288, 'sd': 0.14841863021742066, 'se': 0.04693409186893409, 'threshold': 0.09386818373786818, 'n': 10, 'cores': [0.11855874210596085, 0.3935721218585968, -0.11569558829069138, 0.1307988166809082, 0.18876999616622925, 0.005652065388858318, 0.20193138718605042, 0.051426440477371216, 0.32869184017181396, 0.12062707543373108], 'fold_id': 17}`

## Book

```json
{
  "net_sharpe_weekly": -0.14501845100988903,
  "hard_loss": {
    "loss": 0.06777591523772364,
    "core": -0.051842537374834775,
    "trend": -0.38318579622991417,
    "maxdd": 0.37223828287491645,
    "ddur": 0.7844827586206896,
    "long_frac": 0.49516524751046326,
    "short_frac": 0.5006494443642662,
    "traded_frac": 0.9958146918747294,
    "turnover": 0.3185649599442703,
    "bias": 0.00010259731350694498,
    "ann_mean": -0.09717678232890203
  },
  "soft_loss": {
    "loss": 0.09843875743268485,
    "core": -0.07431582492953162,
    "trend": -0.7880625122036689,
    "maxdd": 0.5624389914084518,
    "ddur": 0.7844827586206896,
    "long_frac": 0.44061192091210855,
    "short_frac": 0.44826093231346514,
    "traded_frac": 0.8888728532255736,
    "turnover": 0.4805267920632052,
    "bias": 0.0019318579998592947,
    "ann_mean": -0.3917626529771562
  },
  "long_frac": 0.49516524751046326,
  "short_frac": 0.5006494443642662,
  "traded_frac": 0.9958146918747294,
  "n_reb": 232,
  "n_symbols": 40
}
```

## vs A0

```json
{
  "skipped": true,
  "reason": "A0 predictions not on disk"
}
```

## Folds

- fold 0 2022-01-10→2022-04-09 status=ok best_epoch=2 hold_loss=0.0240654107183218 n_reb=13
- fold 1 2022-04-10→2022-07-08 status=ok best_epoch=29 hold_loss=-0.11410641670227051 n_reb=13
- fold 2 2022-07-09→2022-10-06 status=ok best_epoch=25 hold_loss=-0.45836278796195984 n_reb=13
- fold 3 2022-10-07→2023-01-04 status=ok best_epoch=9 hold_loss=0.04498414695262909 n_reb=13
- fold 4 2023-01-05→2023-04-04 status=ok best_epoch=7 hold_loss=-0.057805564254522324 n_reb=12
- fold 5 2023-04-05→2023-07-03 status=ok best_epoch=18 hold_loss=0.02787843532860279 n_reb=13
- fold 6 2023-07-04→2023-10-01 status=ok best_epoch=4 hold_loss=-0.3261030912399292 n_reb=13
- fold 7 2023-10-02→2023-12-30 status=ok best_epoch=3 hold_loss=-0.3681522011756897 n_reb=13
- fold 8 2023-12-31→2024-03-29 status=ok best_epoch=2 hold_loss=0.0273739080876112 n_reb=13
- fold 9 2024-03-30→2024-06-27 status=ok best_epoch=1 hold_loss=-0.19731347262859344 n_reb=13
- fold 10 2024-06-28→2024-09-25 status=ok best_epoch=3 hold_loss=-0.28311651945114136 n_reb=13
- fold 11 2024-09-26→2024-12-24 status=ok best_epoch=3 hold_loss=-0.31481441855430603 n_reb=12
- fold 12 2024-12-25→2025-03-24 status=ok best_epoch=2 hold_loss=-0.0019125520484521985 n_reb=13
- fold 13 2025-03-25→2025-06-22 status=ok best_epoch=36 hold_loss=-0.38420572876930237 n_reb=13
- fold 14 2025-06-23→2025-09-20 status=ok best_epoch=2 hold_loss=0.08485159277915955 n_reb=13
- fold 15 2025-09-21→2025-12-19 status=ok best_epoch=1 hold_loss=0.031041845679283142 n_reb=13
- fold 16 2025-12-20→2026-03-19 status=ok best_epoch=3 hold_loss=-0.37774524092674255 n_reb=13
- fold 17 2026-03-20→2026-06-17 status=ok best_epoch=7 hold_loss=-0.15480343997478485 n_reb=13

## Sample rules (eval argmax literals)

- `R00 SHORT: NOT dv_trend IS HIGH AND ret_14 IS LOW AND NOT skew_60 IS LOW AND dv_z_30 IS LOW`
- `R01 FLAT: dist_low_90 IS LOW AND NOT corr_btc_28 IS LOW AND NOT yz_vol_30 IS HIGH AND ema12_ema26 IS MID`
- `R02 LONG: NOT dist_high_90 IS MID AND NOT dv_trend IS HIGH AND yz_vol_60 IS HIGH AND NOT beta_btc_60 IS LOW`
- `R03 FLAT: NOT dv_trend IS MID AND NOT max_ret_14 IS MID AND range_pos_28 IS MID AND NOT close_sma100 IS MID`
- `R04 LONG: NOT dist_high_90 IS HIGH AND NOT rev_1 IS HIGH AND NOT vol_of_vol_30 IS HIGH AND NOT ret_28 IS LOW`
- `R05 FLAT: NOT min_ret_14 IS MID AND pk_vol_14 IS MID AND amihud_14 IS LOW AND NOT vol_ratio IS LOW`
- `R06 SHORT: NOT ema12_ema26 IS LOW AND NOT beta_btc_60 IS LOW AND dv_trend IS LOW AND yz_vol_14 IS MID`
- `R07 SHORT: NOT dv_trend IS MID AND NOT ret_28 IS MID AND NOT range_pos_28 IS HIGH AND NOT ret_7 IS MID`
- `R08 LONG: dv_z_30 IS HIGH AND sma20_sma50 IS LOW AND vol_of_vol_30 IS HIGH AND NOT mom_28_skip7 IS LOW`
- `R09 SHORT: NOT ret_56 IS MID AND close_sma50 IS HIGH AND NOT ema12_ema26 IS MID AND beta_btc_60 IS LOW`
- `R10 SHORT: NOT vol_of_vol_30 IS MID AND beta_btc_60 IS MID AND sma20_sma50 IS HIGH AND skew_28 IS MID`
- `R11 SHORT: idio_vol_60 IS LOW AND ret_90 IS LOW AND yz_vol_30 IS HIGH AND NOT min_ret_14 IS HIGH`
- `R12 SHORT: pk_vol_14 IS HIGH AND NOT rev_1 IS LOW AND sma20_sma50 IS HIGH AND NOT rev_3 IS MID`
- `R13 SHORT: NOT close_sma20 IS LOW AND NOT corr_btc_28 IS HIGH AND amihud_14 IS HIGH AND NOT max_ret_14 IS HIGH`
- `R14 SHORT: max_ret_14 IS LOW AND ret_28 IS HIGH AND ret_56 IS LOW AND NOT ret_90 IS MID`
- `R15 LONG: NOT rev_1 IS MID AND sma20_sma50 IS MID AND NOT ret_90 IS HIGH AND NOT vol_ratio IS HIGH`

## Notes

- Clause (iv) SKIP: A0 h=7 preds missing.
- Official VIABLE is disabled for LOCAL-RESTRICTED even if Sharpe≥0.
