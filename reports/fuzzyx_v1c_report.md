# FuzzyX-v1c report

**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. Does not replace COMBO / A0. Addendum: `reports/fuzzyx_addendum_v1c.md`.

**Mode:** `LOCAL-RESTRICTED`
**Verdict:** **CONTAMINATED**
**Params:** 17668

## Keep rule (verbatim)

> See `reports/fuzzyx_addendum_v1c.md`. VIABLE only if leakage, shuffle-bias on mean weekly net PnL (per-fold weights), full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs A0 Sleeve A when A0 preds exist. LOCAL-RESTRICTED cannot be official VIABLE.

## Gates

| clause | result |
|---|---|
| (i) leakage | PASS |
| (ii) shuffle-bias | FAIL |
| (iii) Sharpe ≥ 0 | PASS (0.391 weekly) |
| (iv) vs A0 | SKIP |

- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top30`: **PASS** `{'name': 'universe_lookahead_top30', 'passed': True, 'n': 30, 'date': '2023-04-17', 'base_n': 30, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0}`

## Shuffle-bias folds

- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.010889003053307533, 'sd': 0.0005549354136149125, 'se': 0.0001754859861310737, 'threshold': 0.0003509719722621474, 'n': 10, 'pnls': [0.010952613316476345, 0.012055509723722935, 0.01075304951518774, 0.011271879076957703, 0.010784948244690895, 0.010981022380292416, 0.010393255390226841, 0.011143282987177372, 0.010563136078417301, 0.009991333819925785], 'mean_corr_st_r': -0.19296290278434752, 'corrs': [-0.18700295686721802, -0.19216300547122955, -0.18991562724113464, -0.1979479342699051, -0.19553476572036743, -0.191957488656044, -0.19421280920505524, -0.19686542451381683, -0.1909216195344925, -0.19310739636421204], 'fold_id': 0}`
- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.010028515290468932, 'sd': 0.0035453314779223025, 'se': 0.0011211322530525441, 'threshold': 0.0022422645061050883, 'n': 10, 'pnls': [0.010362444445490837, 0.015404190868139267, 0.006114198826253414, 0.009021259844303131, 0.011841703206300735, 0.007478637155145407, 0.010946582071483135, 0.005575142335146666, 0.015723004937171936, 0.007817989215254784], 'mean_corr_st_r': 0.12073754891753197, 'corrs': [0.0701679065823555, 0.1604970097541809, 0.10727350413799286, 0.13981635868549347, 0.15774817764759064, 0.16201499104499817, 0.0717281773686409, 0.09735652804374695, 0.1494586318731308, 0.09131420403718948], 'fold_id': 17}`

## Book

```json
{
  "net_sharpe_weekly": 0.3913771226488236,
  "mean_weekly_pnl": 0.001359489061104527,
  "hard_loss": {
    "loss": 0.10874314083882115,
    "core": -0.10874314083882115,
    "trend": -0.10874314083882115,
    "trend_equity": -0.6378917649032052,
    "maxdd": 0.7011188809521749,
    "ddur": 0.7844827586206896,
    "long_frac": 0.34983403088468756,
    "short_frac": 0.6039832587674989,
    "traded_frac": 0.9538172896521865,
    "turnover": 0.45729283830419526,
    "bias": 0.006288741690023279,
    "ann_mean": 0.49621350730315233,
    "mean_pnl": 0.001359489061104527
  },
  "soft_loss": {
    "loss": 0.06366947534433476,
    "core": -0.06366947534433476,
    "trend": -0.06366947534433476,
    "trend_equity": -0.7135162453249682,
    "maxdd": 0.712807099952251,
    "ddur": 0.7844827586206896,
    "long_frac": 0.31216625775725215,
    "short_frac": 0.5703564727954972,
    "traded_frac": 0.8825227305527493,
    "turnover": 0.6202047098153157,
    "bias": 0.00835537668663829,
    "ann_mean": 0.5624742227723993,
    "mean_pnl": 0.001541025267869587
  },
  "long_frac": 0.34983403088468756,
  "short_frac": 0.6039832587674989,
  "traded_frac": 0.9538172896521865,
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

- fold 0 2022-01-10→2022-04-09 status=ok best_epoch=4 hold_loss=-0.5015876889228821 n_reb=13
- fold 1 2022-04-10→2022-07-08 status=ok best_epoch=0 hold_loss=0.365133672952652 n_reb=13
- fold 2 2022-07-09→2022-10-06 status=ok best_epoch=26 hold_loss=-0.15406416356563568 n_reb=13
- fold 3 2022-10-07→2023-01-04 status=ok best_epoch=0 hold_loss=-0.4967387914657593 n_reb=13
- fold 4 2023-01-05→2023-04-04 status=ok best_epoch=3 hold_loss=-0.20433761179447174 n_reb=12
- fold 5 2023-04-05→2023-07-03 status=ok best_epoch=6 hold_loss=-0.38056105375289917 n_reb=13
- fold 6 2023-07-04→2023-10-01 status=ok best_epoch=11 hold_loss=-0.4195961356163025 n_reb=13
- fold 7 2023-10-02→2023-12-30 status=ok best_epoch=5 hold_loss=-0.6197477579116821 n_reb=13
- fold 8 2023-12-31→2024-03-29 status=ok best_epoch=17 hold_loss=-0.15942265093326569 n_reb=13
- fold 9 2024-03-30→2024-06-27 status=ok best_epoch=25 hold_loss=-0.5161237120628357 n_reb=13
- fold 10 2024-06-28→2024-09-25 status=ok best_epoch=10 hold_loss=-0.3383805751800537 n_reb=13
- fold 11 2024-09-26→2024-12-24 status=ok best_epoch=14 hold_loss=-0.15401552617549896 n_reb=12
- fold 12 2024-12-25→2025-03-24 status=ok best_epoch=1 hold_loss=-0.5379757285118103 n_reb=13
- fold 13 2025-03-25→2025-06-22 status=ok best_epoch=8 hold_loss=-0.6033666133880615 n_reb=13
- fold 14 2025-06-23→2025-09-20 status=ok best_epoch=29 hold_loss=-0.05361912027001381 n_reb=13
- fold 15 2025-09-21→2025-12-19 status=ok best_epoch=2 hold_loss=-0.44426247477531433 n_reb=13
- fold 16 2025-12-20→2026-03-19 status=ok best_epoch=6 hold_loss=-0.3448618948459625 n_reb=13
- fold 17 2026-03-20→2026-06-17 status=ok best_epoch=0 hold_loss=-0.022721610963344574 n_reb=13

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

## Reading (not a retune)

The notebook line as the **only** train loss does put the book back in the market (`traded_frac = 0.95`, vs v1b’s 0). Full-OOS hard Sharpe is **+0.391** (weekly, costs on) and `corr(st_r, t)` itself is **−0.109** — the train scalar is not even achieved OOS. Inner-holdout `hold_loss` is often negative (local 12–13 week slope fits); the concatenated path does not.

Shuffle-bias is a clean FAIL: fold 0 null mean weekly PnL **+1.09%** with SD 5.5 bps (threshold 3.5 bps). That is residual **net-long × crypto CS-mean**. Within-date shuffle keeps the date’s average return; `E[w·π(r)] = mean(r)·Σw`. Reported `|mean w| = 0.006` on 30 names is ~18% net long notional, which times a fat weekly CS mean ≈ the +1.1% null. Occupancy looks two-sided (L 0.35 / S 0.60) and still has market beta.

So this shot did **not** recover the notebook as a skillful CS policy. It recovered “a fully invested book with a bit of net long,” which is why a single in-sample window can look like the file “worked.” No retune. v1/v1b untouched. COMBO / A0 not replaced.
