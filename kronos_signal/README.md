# Kronos BTC daily → LONG / HOLD / SHORT

Zero-shot **Kronos-base** on **BTCUSDT daily** bars, converted into a trading stance.

## Defaults (v1)

| Knob | Value |
|------|-------|
| Symbol / interval | BTCUSDT `1d` |
| Model | `NeoQuasar/Kronos-base` |
| Lookback | 400 days |
| Horizon | 5 days |
| Monte Carlo paths | 30 |
| LONG | `p_up ≥ 60%` and `mean_r ≥ +0.5%` |
| SHORT | `p_up ≤ 40%` and `mean_r ≤ -0.5%` |
| HOLD | otherwise |

`p_up` = share of paths with predicted horizon close above the last close.  
`mean_r` = mean horizon return across paths.

## Run on Modal (GPU)

```bash
# live signal
modal run kronos_signal/modal_app.py
modal run kronos_signal/modal_app.py --n-paths 5

# walk-forward backtest (non-overlapping 5d steps)
modal run kronos_signal/modal_app.py --mode backtest --n-paths 10 --max-steps 40
```

Writes `kronos_signal/last_signal.json` or `kronos_signal/last_backtest.json`.

### Backtest design
- Non-overlapping: decide every `pred_len` days, hold stance for that horizon
- LONG PnL = realized return; SHORT = −realized; HOLD = 0
- Metrics: hit-rate (active trades), total return, buy&hold, max drawdown

### Improve pipeline (points 1–3)
```bash
modal run kronos_signal/modal_app.py --mode improve --n-paths 10 --max-steps 150
python -m kronos_signal.plot_compare
```

Compares:
1. Raw Kronos rule (baseline)
2. Walk-forward logistic **meta-model** on Kronos + market features
3. Same after **fine-tuning** Kronos on pre-test BTC only (no leakage)

## Local unit tests (no GPU)

```bash
python -m kronos_signal.test_signals
python -m kronos_signal.test_backtest
python -m kronos_signal.test_meta_model
```
