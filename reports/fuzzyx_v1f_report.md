# FuzzyX-v1f report

**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. Does not replace COMBO / A0. Addendum: `reports/fuzzyx_addendum_v1f.md`.

**Mode:** `LOCAL-RESTRICTED`
**Verdict:** **PARK**
**Params:** 17668

## Keep rule (verbatim)

> See `reports/fuzzyx_addendum_v1f.md`. VIABLE only if leakage, shuffle-bias on mean weekly net PnL (per-fold weights), full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs A0 Sleeve A when A0 preds exist. LOCAL-RESTRICTED cannot be official VIABLE.

## Gates

| clause | result |
|---|---|
| (i) leakage | PASS |
| (ii) shuffle-bias | PASS |
| (iii) Sharpe ≥ 0 | FAIL (-2.000 weekly) |
| (iv) vs A0 | SKIP |

- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top30`: **PASS** `{'name': 'universe_lookahead_top30', 'passed': True, 'n': 30, 'date': '2023-04-17', 'base_n': 30, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0}`

## Shuffle-bias folds

- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': True, 'mean_pnl': 0.0018092569574946538, 'sd': 0.004676299349972743, 'se': 0.0014787756966678717, 'threshold': 0.0029575513933357434, 'n': 10, 'pnls': [0.00923374854028225, 0.002816005377098918, 0.005305763799697161, -0.00034673590562306345, -0.00577034056186676, 0.0013979095965623856, 0.0038558929227292538, 0.006702330429106951, -0.0024383440613746643, -0.0026636605616658926], 'mean_corr_st_r': 0.0018092569574946538, 'corrs': [0.00923374854028225, 0.002816005377098918, 0.005305763799697161, -0.00034673590562306345, -0.00577034056186676, 0.0013979095965623856, 0.0038558929227292538, 0.006702330429106951, -0.0024383440613746643, -0.0026636605616658926], 'fold_id': 0}`
- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': True, 'mean_pnl': -0.0018019019160419702, 'sd': 0.003287144334632888, 'se': 0.0010394863095158633, 'threshold': 0.0020789726190317266, 'n': 10, 'pnls': [-0.00418322067707777, -0.0034317465033382177, -0.00605635903775692, -0.0012220849748700857, -0.0006103491177782416, -0.0013353071408346295, 0.0017788487020879984, -0.005170551594346762, 0.004831917118281126, -0.0026201659347862005], 'mean_corr_st_r': -0.0018019019160419702, 'corrs': [-0.00418322067707777, -0.0034317465033382177, -0.00605635903775692, -0.0012220849748700857, -0.0006103491177782416, -0.0013353071408346295, 0.0017788487020879984, -0.005170551594346762, 0.004831917118281126, -0.0026201659347862005], 'fold_id': 17}`

## Book

```json
{
  "net_sharpe_weekly": -2.000118120824083,
  "mean_weekly_pnl": -0.002397865090659302,
  "hard_loss": {
    "loss": 0.002397865090659302,
    "core": -0.002397865090659302,
    "trend": -0.9042493723389755,
    "trend_equity": -0.9042493723389755,
    "trend_returns": 0.026724129068267516,
    "equity_end": 0.5385931697992795,
    "cumret_last": -0.4614068302007205,
    "maxdd": 0.4614068302007205,
    "ddur": 0.7758620689655172,
    "long_frac": 0.004329629095107519,
    "short_frac": 0.994804445085871,
    "traded_frac": 0.9991340741809785,
    "turnover": 0.17860874757426481,
    "bias": 6.669712674876344e-19,
    "net_expo": 2.6319942394130866e-17,
    "ann_mean": -0.8752207580906453,
    "mean_pnl": -0.002397865090659302
  },
  "soft_loss": {
    "loss": 0.0004829062464950169,
    "core": -0.0004829062464950169,
    "trend": -0.5738787582180082,
    "trend_equity": -0.5738787582180082,
    "trend_returns": -0.0072867298568588396,
    "equity_end": 0.8125973198295839,
    "cumret_last": -0.18740268017041606,
    "maxdd": 0.38307937505614464,
    "ddur": 0.7672413793103449,
    "long_frac": 0.003319382306249098,
    "short_frac": 0.9911964208399481,
    "traded_frac": 0.9945158031461971,
    "turnover": 0.8295548454565832,
    "bias": 1.004195529412436e-18,
    "net_expo": 7.976662314384905e-16,
    "ann_mean": -0.17626077997068118,
    "mean_pnl": -0.0004829062464950169
  },
  "long_frac": 0.004329629095107519,
  "short_frac": 0.994804445085871,
  "traded_frac": 0.9991340741809785,
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

- fold 0 2022-01-10→2022-04-09 status=ok best_epoch=22 hold_loss=-0.035069189965724945 n_reb=13
- fold 1 2022-04-10→2022-07-08 status=ok best_epoch=0 hold_loss=0.0004344169865362346 n_reb=13
- fold 2 2022-07-09→2022-10-06 status=ok best_epoch=65 hold_loss=-0.02979789488017559 n_reb=13
- fold 3 2022-10-07→2023-01-04 status=ok best_epoch=21 hold_loss=-0.0021641014609485865 n_reb=13
- fold 4 2023-01-05→2023-04-04 status=ok best_epoch=7 hold_loss=-0.009727860800921917 n_reb=12
- fold 5 2023-04-05→2023-07-03 status=ok best_epoch=8 hold_loss=-0.003068994265049696 n_reb=13
- fold 6 2023-07-04→2023-10-01 status=ok best_epoch=0 hold_loss=-0.00897239800542593 n_reb=13
- fold 7 2023-10-02→2023-12-30 status=ok best_epoch=0 hold_loss=0.0021860834676772356 n_reb=13
- fold 8 2023-12-31→2024-03-29 status=ok best_epoch=9 hold_loss=-0.020009811967611313 n_reb=13
- fold 9 2024-03-30→2024-06-27 status=ok best_epoch=2 hold_loss=-0.007771125063300133 n_reb=13
- fold 10 2024-06-28→2024-09-25 status=ok best_epoch=1 hold_loss=-0.0041288468055427074 n_reb=13
- fold 11 2024-09-26→2024-12-24 status=ok best_epoch=0 hold_loss=-0.004973031580448151 n_reb=12
- fold 12 2024-12-25→2025-03-24 status=ok best_epoch=12 hold_loss=-0.015370845794677734 n_reb=13
- fold 13 2025-03-25→2025-06-22 status=ok best_epoch=1 hold_loss=-0.01119181327521801 n_reb=13
- fold 14 2025-06-23→2025-09-20 status=ok best_epoch=1 hold_loss=-0.00288237351924181 n_reb=13
- fold 15 2025-09-21→2025-12-19 status=ok best_epoch=0 hold_loss=0.004073717165738344 n_reb=13
- fold 16 2025-12-20→2026-03-19 status=ok best_epoch=0 hold_loss=-0.011977712623775005 n_reb=13
- fold 17 2026-03-20→2026-06-17 status=ok best_epoch=0 hold_loss=0.0003533796698320657 n_reb=13

## Sample rules (eval argmax literals)

- `R00 SHORT: NOT dv_trend IS HIGH AND ret_14 IS LOW AND NOT skew_60 IS LOW AND dv_z_30 IS LOW`
- `R01 FLAT: dist_low_90 IS LOW AND NOT corr_btc_28 IS LOW AND NOT yz_vol_30 IS HIGH AND ema12_ema26 IS MID`
- `R02 LONG: NOT dist_high_90 IS MID AND NOT dv_trend IS HIGH AND yz_vol_60 IS HIGH AND NOT sma20_sma50 IS LOW`
- `R03 FLAT: NOT dv_trend IS MID AND NOT max_ret_14 IS MID AND range_pos_28 IS MID AND NOT close_sma100 IS MID`
- `R04 LONG: NOT dist_high_90 IS HIGH AND NOT rev_1 IS HIGH AND NOT vol_of_vol_30 IS HIGH AND NOT ret_28 IS LOW`
- `R05 FLAT: NOT min_ret_14 IS MID AND pk_vol_14 IS MID AND amihud_14 IS LOW AND NOT vol_ratio IS LOW`
- `R06 SHORT: NOT ema12_ema26 IS LOW AND dv_trend IS LOW AND NOT beta_btc_60 IS LOW AND yz_vol_14 IS MID`
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
