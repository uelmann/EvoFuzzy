"""
Modal app: Kronos-base daily BTC forecast → LONG / HOLD / SHORT.

Usage (from repo root, with Modal CLI authenticated):

    modal run kronos_signal/modal_app.py
    modal run kronos_signal/modal_app.py --n-paths 10   # cheaper smoke test
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "kronos-btc-daily-signal"
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"

# Cache Hugging Face weights across runs.
hf_cache = modal.Volume.from_name("kronos-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "torch>=2.0.0",
        "einops==0.8.1",
        "huggingface_hub==0.33.1",
        "safetensors==0.6.2",
        "tqdm==4.67.1",
        "requests",
    )
    .run_commands(f"git clone --depth 1 {KRONOS_REPO} /opt/Kronos")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("kronos_signal")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="T4",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    memory=8192,
)
def run_daily_signal(
    n_paths: int | None = None,
    pred_len: int | None = None,
    lookback: int | None = None,
    tau: float | None = None,
    verbose: bool = True,
) -> dict:
    from kronos_signal import config
    from kronos_signal.data import fetch_binance_klines, prepare_windows
    from kronos_signal.forecast import forecast_close_paths, load_predictor
    from kronos_signal.signals import decide_signal, path_returns

    n_paths = config.N_PATHS if n_paths is None else n_paths
    pred_len = config.PRED_LEN if pred_len is None else pred_len
    lookback = config.LOOKBACK if lookback is None else lookback
    tau = config.TAU if tau is None else tau

    df = fetch_binance_klines(limit=max(1000, lookback + 10))
    x_df, x_timestamp, y_timestamp = prepare_windows(df, lookback=lookback, pred_len=pred_len)
    last_close = float(x_df["close"].iloc[-1])
    last_ts = str(x_timestamp.iloc[-1])

    if verbose:
        print(
            f"Loaded {config.SYMBOL} {config.INTERVAL}: "
            f"lookback={lookback}, last_close={last_close:.2f} @ {last_ts}",
            flush=True,
        )

    predictor = load_predictor(kronos_root="/opt/Kronos")
    closes = forecast_close_paths(
        predictor,
        x_df,
        x_timestamp,
        y_timestamp,
        pred_len=pred_len,
        n_paths=n_paths,
        verbose=verbose,
    )
    # Horizon return = last predicted close vs current close
    returns = path_returns(last_close, closes[:, -1])
    result = decide_signal(
        returns,
        last_close=last_close,
        horizon_days=pred_len,
        tau=tau,
    )

    payload = {
        "symbol": config.SYMBOL,
        "interval": config.INTERVAL,
        "model": config.MODEL_ID,
        "asof": last_ts,
        "lookback": lookback,
        "pred_len": pred_len,
        "future_timestamps": [str(ts) for ts in y_timestamp.tolist()],
        "path_horizon_closes": closes[:, -1].tolist(),
        "path_returns": returns.tolist(),
        **result.to_dict(),
    }
    hf_cache.commit()
    if verbose:
        print(json.dumps({k: payload[k] for k in ("signal", "p_up", "mean_return", "reason")}, indent=2))
    return payload


@app.local_entrypoint()
def main(n_paths: int = 30, pred_len: int = 5, lookback: int = 400, tau: float = 0.005):
    result = run_daily_signal.remote(
        n_paths=n_paths,
        pred_len=pred_len,
        lookback=lookback,
        tau=tau,
        verbose=True,
    )
    out = Path("kronos_signal") / "last_signal.json"
    out.write_text(json.dumps(result, indent=2))
    print("\n=== SIGNAL ===")
    print(f"{result['symbol']} {result['interval']} → {result['signal']}")
    print(
        f"p_up={result['p_up']:.2%}  mean_r={result['mean_return']:.2%}  "
        f"std={result['std_return']:.2%}  n_paths={result['n_paths']}"
    )
    print(result["reason"])
    print(f"Wrote {out}")
