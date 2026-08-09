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
# from repo root, Modal profile already active
modal run kronos_signal/modal_app.py
# cheaper smoke test
modal run kronos_signal/modal_app.py --n-paths 5
```

Writes `kronos_signal/last_signal.json`.

## Local unit tests (no GPU)

```bash
python -m kronos_signal.test_signals
```
