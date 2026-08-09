"""
Modal app: Kronos-base daily BTC forecast → LONG / HOLD / SHORT.

Usage (from repo root, with Modal CLI authenticated):

    modal run kronos_signal/modal_app.py
    modal run kronos_signal/modal_app.py --n-paths 10
    modal run kronos_signal/modal_app.py --mode backtest --n-paths 10 --max-steps 150
    modal run kronos_signal/modal_app.py --mode improve --n-paths 10 --max-steps 150
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "kronos-btc-daily-signal"
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"

# Cache Hugging Face weights / fine-tuned checkpoints across runs.
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
        "scikit-learn",
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


@app.function(
    gpu="T4",
    timeout=3 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    memory=8192,
)
def run_walk_forward_backtest(
    n_paths: int = 10,
    pred_len: int = 5,
    lookback: int = 400,
    tau: float = 0.005,
    step: int | None = None,
    max_steps: int = 150,
    min_bars: int = 2000,
    verbose: bool = True,
) -> dict:
    """Non-overlapping walk-forward backtest on BTCUSDT daily."""
    from kronos_signal.backtest import run_walk_forward
    from kronos_signal.data import fetch_binance_klines_history
    from kronos_signal.forecast import forecast_close_paths, load_predictor

    step = pred_len if step is None else step
    need = lookback + max_steps * step + pred_len + 5
    df = fetch_binance_klines_history(min_bars=max(min_bars, need))
    if verbose:
        print(
            f"History bars={len(df)}  range={df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}",
            flush=True,
        )
        print(
            f"Backtest: lookback={lookback} pred_len={pred_len} "
            f"n_paths={n_paths} step={step} max_steps={max_steps}",
            flush=True,
        )

    predictor = load_predictor(kronos_root="/opt/Kronos")

    def forecast_fn(x_df, x_ts, y_ts, pl):
        return forecast_close_paths(
            predictor,
            x_df,
            x_ts,
            y_ts,
            pred_len=pl,
            n_paths=n_paths,
            verbose=False,
        )

    summary = run_walk_forward(
        df,
        forecast_fn,
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
        max_steps=max_steps,
        verbose=verbose,
    )
    hf_cache.commit()
    payload = summary.to_dict()
    if verbose:
        slim = {
            k: payload[k]
            for k in (
                "n_steps",
                "n_long",
                "n_short",
                "n_hold",
                "hit_rate",
                "total_return",
                "buy_hold_return",
                "max_drawdown",
                "start",
                "end",
                "diagnostics",
            )
        }
        print(json.dumps(slim, indent=2), flush=True)
    return payload


def _slim_result(result: dict) -> dict:
    """Drop bulky per-step arrays for console summaries."""
    out = {k: v for k, v in result.items() if k != "steps"}
    if "steps" in result:
        out["n_steps_listed"] = len(result["steps"])
    return out


@app.function(
    gpu="T4",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    memory=8192,
)
def run_improve_pipeline(
    n_paths: int = 10,
    pred_len: int = 5,
    lookback: int = 400,
    tau: float = 0.005,
    step: int | None = None,
    max_steps: int = 150,
    min_train: int = 40,
    finetune_epochs: int = 5,
    verbose: bool = True,
    zeroshot_steps_json: str | None = None,
) -> dict:
    """
    Points 1-3:
      1-2) meta-model on zero-shot Kronos features
      3) fine-tune Kronos on pre-test BTC, re-forecast, raw+meta again
    """
    import pandas as pd

    from kronos_signal.backtest import run_walk_forward
    from kronos_signal.compare import compare_raw_vs_meta
    from kronos_signal.data import fetch_binance_klines_history
    from kronos_signal.finetune_btc import finetune_predictor_on_btc
    from kronos_signal.forecast import forecast_close_paths, load_predictor

    step = pred_len if step is None else step
    need = lookback + max_steps * step + pred_len + 5
    df = fetch_binance_klines_history(min_bars=max(2000, need))

    # --- zero-shot steps (reuse uploaded JSON if provided) ---
    if zeroshot_steps_json:
        zs_steps = json.loads(zeroshot_steps_json)
        if verbose:
            print(f"Reusing {len(zs_steps)} zero-shot steps from client", flush=True)
        zs_summary = {"steps": zs_steps, "source": "uploaded"}
    else:
        if verbose:
            print("Running zero-shot walk-forward...", flush=True)
        predictor = load_predictor(kronos_root="/opt/Kronos")

        def forecast_fn(x_df, x_ts, y_ts, pl):
            return forecast_close_paths(
                predictor, x_df, x_ts, y_ts, pred_len=pl, n_paths=n_paths, verbose=False
            )

        zs_bt = run_walk_forward(
            df,
            forecast_fn,
            lookback=lookback,
            pred_len=pred_len,
            n_paths=n_paths,
            step=step,
            tau=tau,
            max_steps=max_steps,
            verbose=verbose,
        )
        zs_summary = zs_bt.to_dict()
        zs_steps = zs_summary["steps"]

    zs_compare = compare_raw_vs_meta(
        zs_steps, ohlcv=df, min_train=min_train, proba_long=0.55, proba_short=0.45
    )

    # --- fine-tune only on bars before first backtest decision ---
    first_asof = pd.Timestamp(zs_steps[0]["asof"])
    train_df = df[df["timestamps"] < first_asof].copy()
    if len(train_df) < lookback + 50:
        raise RuntimeError(f"Not enough pre-test bars to fine-tune: {len(train_df)}")
    if verbose:
        print(
            f"Fine-tuning on {len(train_df)} bars ending before {first_asof} "
            f"({finetune_epochs} epochs)",
            flush=True,
        )
    ckpt = "/root/.cache/huggingface/kronos-btc-finetuned"
    ft_info = finetune_predictor_on_btc(
        train_df,
        save_dir=ckpt,
        epochs=finetune_epochs,
        batch_size=8,
        n_samples=1200,
        kronos_root="/opt/Kronos",
    )

    if verbose:
        print("Running fine-tuned walk-forward...", flush=True)
    ft_predictor = load_predictor(model_id=ckpt, kronos_root="/opt/Kronos")

    def ft_forecast_fn(x_df, x_ts, y_ts, pl):
        return forecast_close_paths(
            ft_predictor, x_df, x_ts, y_ts, pred_len=pl, n_paths=n_paths, verbose=False
        )

    ft_bt = run_walk_forward(
        df,
        ft_forecast_fn,
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
        max_steps=max_steps,
        verbose=verbose,
    )
    ft_summary = ft_bt.to_dict()
    ft_compare = compare_raw_vs_meta(
        ft_summary["steps"],
        ohlcv=df,
        min_train=min_train,
        proba_long=0.55,
        proba_short=0.45,
    )

    hf_cache.commit()
    payload = {
        "zeroshot": {
            "backtest": {
                k: zs_summary.get(k)
                for k in (
                    "n_steps",
                    "hit_rate",
                    "total_return",
                    "buy_hold_return",
                    "max_drawdown",
                    "n_long",
                    "n_short",
                    "n_hold",
                    "start",
                    "end",
                    "diagnostics",
                )
            },
            "compare": {
                "raw_aligned": zs_compare["raw_aligned"],
                "meta": zs_compare["meta"],
            },
            "steps": zs_steps,
        },
        "finetuned": {
            "train_info": ft_info,
            "backtest": {
                k: ft_summary[k]
                for k in (
                    "n_steps",
                    "hit_rate",
                    "total_return",
                    "buy_hold_return",
                    "max_drawdown",
                    "n_long",
                    "n_short",
                    "n_hold",
                    "start",
                    "end",
                    "diagnostics",
                )
            },
            "compare": {
                "raw_aligned": ft_compare["raw_aligned"],
                "meta": ft_compare["meta"],
            },
            "steps": ft_summary["steps"],
        },
    }
    if verbose:
        print(
            json.dumps(
                {
                    "zeroshot_raw_aligned": _slim_result(zs_compare["raw_aligned"]).get(
                        "total_return"
                    ),
                    "zeroshot_meta": _slim_result(zs_compare["meta"]).get("total_return"),
                    "finetuned_raw_aligned": _slim_result(ft_compare["raw_aligned"]).get(
                        "total_return"
                    ),
                    "finetuned_meta": _slim_result(ft_compare["meta"]).get("total_return"),
                },
                indent=2,
            ),
            flush=True,
        )
    return payload


@app.local_entrypoint()
def main(
    mode: str = "signal",
    n_paths: int = 0,
    pred_len: int = 5,
    lookback: int = 400,
    tau: float = 0.005,
    max_steps: int = 150,
    step: int = 5,
    finetune_epochs: int = 5,
):
    def _print_bt(title: str, result: dict):
        print(f"\n=== {title} ===")
        print(f"period: {result.get('start')} → {result.get('end')}")
        print(
            f"steps={result.get('n_steps')}  LONG={result.get('n_long')}  "
            f"SHORT={result.get('n_short')}  HOLD={result.get('n_hold')}"
        )
        hr = result.get("hit_rate")
        print(f"hit_rate={hr if hr is None else f'{hr:.2%}'}")
        print(
            f"strategy={result.get('total_return', float('nan')):.2%}  "
            f"buy&hold={result.get('buy_hold_return', float('nan')):.2%}  "
            f"maxDD={result.get('max_drawdown', float('nan')):.2%}"
        )

    if mode == "backtest":
        n_paths = 10 if n_paths <= 0 else n_paths
        result = run_walk_forward_backtest.remote(
            n_paths=n_paths,
            pred_len=pred_len,
            lookback=lookback,
            tau=tau,
            step=step,
            max_steps=max_steps,
            verbose=True,
        )
        out = Path("kronos_signal") / "last_backtest.json"
        out.write_text(json.dumps(result, indent=2))
        _print_bt("BACKTEST", result)
        print(f"Wrote {out}")
        return

    if mode == "improve":
        n_paths = 10 if n_paths <= 0 else n_paths
        zs_path = Path("kronos_signal") / "last_backtest.json"
        zs_json = None
        if zs_path.exists():
            zs = json.loads(zs_path.read_text())
            if zs.get("steps"):
                zs_json = json.dumps(zs["steps"])
        result = run_improve_pipeline.remote(
            n_paths=n_paths,
            pred_len=pred_len,
            lookback=lookback,
            tau=tau,
            step=step,
            max_steps=max_steps,
            finetune_epochs=finetune_epochs,
            verbose=True,
            zeroshot_steps_json=zs_json,
        )
        out = Path("kronos_signal") / "last_improve.json"
        out.write_text(json.dumps(result, indent=2))
        print("\n=== IMPROVE SUMMARY ===")
        zraw = result["zeroshot"]["compare"]["raw_aligned"]
        zmeta = result["zeroshot"]["compare"]["meta"]
        fraw = result["finetuned"]["compare"]["raw_aligned"]
        fmeta = result["finetuned"]["compare"]["meta"]
        print(f"zero-shot raw_aligned: {zraw['total_return']:.2%} hit={zraw['hit_rate']}")
        print(f"zero-shot meta:        {zmeta['total_return']:.2%} hit={zmeta['hit_rate']}")
        print(f"finetuned raw_aligned: {fraw['total_return']:.2%} hit={fraw['hit_rate']}")
        print(f"finetuned meta:        {fmeta['total_return']:.2%} hit={fmeta['hit_rate']}")
        print(f"Wrote {out}")
        return

    if mode != "signal":
        raise SystemExit(f"Unknown mode={mode!r}; use 'signal', 'backtest', or 'improve'")

    n_paths = 30 if n_paths <= 0 else n_paths
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
