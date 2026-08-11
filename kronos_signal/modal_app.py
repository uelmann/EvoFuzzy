"""
Modal app: Kronos-base daily BTC forecast → LONG / HOLD / SHORT.

Usage (from repo root, with Modal CLI authenticated):

    modal run kronos_signal/modal_app.py
    modal run kronos_signal/modal_app.py --n-paths 10
    modal run kronos_signal/modal_app.py --mode backtest --n-paths 10 --max-steps 150
    modal run kronos_signal/modal_app.py --mode improve --n-paths 10 --max-steps 150
    modal run kronos_signal/modal_app.py --mode improve_v2
    modal run kronos_signal/modal_app.py --mode long_annual --n-paths 10 --start-asof 2021-01-01
    modal run kronos_signal/modal_app.py --mode long_annual --pred-len 1 --n-paths 10 --start-asof 2021-01-01
    modal run kronos_signal/modal_app.py --mode official --predictor-size small --official-epochs 30
    modal run kronos_signal/modal_app.py --mode official_bt --predictor-size small --signal mean
    modal run kronos_signal/modal_app.py --mode zs_scores --predictor-size small --lookback 90 --pred-len 10
    modal run kronos_signal/modal_app.py --mode download --max-coins 0 --start-date 2016-01-01
    modal run kronos_signal/modal_app.py --mode official --predictor-size base --official-epochs 30 --max-coins 0
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "kronos-btc-daily-signal"
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"

# Cache Hugging Face weights / fine-tuned checkpoints across runs.
hf_cache = modal.Volume.from_name("kronos-hf-cache", create_if_missing=True)
crypto_data = modal.Volume.from_name("kronos-crypto-data", create_if_missing=True)

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
        "lightgbm",
    )
    .run_commands(f"git clone --depth 1 {KRONOS_REPO} /opt/Kronos")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("kronos_signal")
)

# Bundle local CMC panel if present (gitignored; must exist on the machine running modal).
_local_hist = Path(__file__).resolve().parent / "data" / "historical_data.csv"
if _local_hist.exists():
    image = image.add_local_file(
        str(_local_hist),
        remote_path="/root/kronos_signal_data/historical_data.csv",
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
    max_steps: int | None = 150,
    min_bars: int = 2000,
    start_asof: str | None = None,
    end_asof: str | None = None,
    verbose: bool = True,
) -> dict:
    """Non-overlapping walk-forward backtest on BTCUSDT daily (Binance)."""
    from kronos_signal.backtest import run_walk_forward
    from kronos_signal.data import fetch_binance_klines_history
    from kronos_signal.forecast import forecast_close_paths, load_predictor

    step = pred_len if step is None else step
    # Enough history for lookback before start_asof and the full test span.
    need = lookback + (max_steps or 400) * step + pred_len + 5
    if start_asof is not None:
        need = max(need, 3000)
    df = fetch_binance_klines_history(min_bars=max(min_bars, need))
    if verbose:
        print(
            f"Data: Binance {df.shape[0]} daily BTCUSDT bars "
            f"{df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}",
            flush=True,
        )
        print(
            f"Backtest: lookback={lookback} pred_len={pred_len} "
            f"n_paths={n_paths} step={step} max_steps={max_steps} "
            f"start_asof={start_asof} end_asof={end_asof}",
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
        start_asof=start_asof,
        end_asof=end_asof,
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


@app.function(
    gpu="T4",
    timeout=5 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    memory=8192,
)
def run_long_annual_pipeline(
    n_paths: int = 10,
    pred_len: int = 5,
    step: int | None = None,
    lookback: int = 400,
    tau: float = 0.005,
    start_asof: str = "2021-01-01",
    end_asof: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Long Binance daily Kronos feature backtest from start_asof, then
    expanding annual retrain/test of the logistic meta-model.

    Use pred_len=1 and step=1 for next-day prediction + daily rebalancing.
    """
    from kronos_signal.annual_meta import annual_retrain_meta
    from kronos_signal.backtest import run_walk_forward
    from kronos_signal.data import fetch_binance_klines_history
    from kronos_signal.forecast import forecast_close_paths, load_predictor
    from kronos_signal.meta_model import raw_rule_on_frame
    from kronos_signal.features import steps_to_frame

    step = pred_len if step is None else step
    df = fetch_binance_klines_history(min_bars=3200)
    # For dense daily steps, avoid flooding logs.
    step_verbose = bool(verbose and step > 1)
    if verbose:
        print(
            f"Binance daily BTCUSDT: {len(df)} bars "
            f"{df['timestamps'].iloc[0].date()} → {df['timestamps'].iloc[-1].date()}",
            flush=True,
        )
        print(
            f"Kronos-base feature gen from {start_asof} "
            f"(lookback={lookback}, pred_len={pred_len}, step={step}, n_paths={n_paths}, tau={tau})",
            flush=True,
        )

    predictor = load_predictor(kronos_root="/opt/Kronos")

    def forecast_fn(x_df, x_ts, y_ts, pl):
        return forecast_close_paths(
            predictor, x_df, x_ts, y_ts, pred_len=pl, n_paths=n_paths, verbose=False
        )

    kronos_bt = run_walk_forward(
        df,
        forecast_fn,
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
        max_steps=None,
        start_asof=start_asof,
        end_asof=end_asof,
        verbose=step_verbose,
    )
    steps = kronos_bt.to_dict()["steps"]
    if verbose:
        print(
            f"Generated {len(steps)} Kronos steps "
            f"({steps[0]['asof'][:10]} → {steps[-1]['asof'][:10]}); "
            "running expanding annual meta retrain...",
            flush=True,
        )

    # More train rows available with daily sampling.
    min_train = 60 if step == 1 else 30
    annual = annual_retrain_meta(
        steps,
        df,
        test_years=[2022, 2023, 2024, 2025, 2026],
        min_train=min_train,
        model_type="logistic",
    )
    frame = steps_to_frame(steps, df)
    raw = raw_rule_on_frame(frame).to_dict()
    annual["raw_full_period"] = {k: raw[k] for k in raw if k != "steps"}
    annual["kronos_raw_summary"] = {
        k: kronos_bt.to_dict().get(k)
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
    annual["horizon"] = {
        "model": "NeoQuasar/Kronos-base",
        "pred_len": pred_len,
        "step": step,
        "tau": tau,
        "n_paths": n_paths,
        "lookback": lookback,
        "rebalance": "daily" if step == 1 else f"every_{step}_days",
    }
    annual["kronos_steps"] = steps
    hf_cache.commit()
    if verbose:
        print(json.dumps({"horizon": annual["horizon"], "folds": annual["folds"], "overall": annual["overall"]}, indent=2))
    return annual


@app.function(
    gpu="T4",
    timeout=3 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    memory=8192,
)
def run_improve_v2_pipeline(
    zeroshot_steps_json: str,
    *,
    min_train: int = 40,
    supervised_epochs: int = 8,
    lookback_sup: int = 90,
    pred_len: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Next-step package:
      - LightGBM meta + richer features + embargo
      - Supervised direction head on Kronos context (pre-test only)
      - Meta with supervised proba as extra feature
    """
    import pandas as pd

    from kronos_signal.compare import compare_raw_vs_meta
    from kronos_signal.data import fetch_binance_klines_history
    from kronos_signal.supervised_ft import (
        load_supervised_bundle,
        predict_supervised_p_up_loaded,
        supervised_rule_backtest,
        train_supervised_direction,
    )

    zs_steps = json.loads(zeroshot_steps_json)
    df = fetch_binance_klines_history(min_bars=2000)
    first_asof = pd.Timestamp(zs_steps[0]["asof"])
    train_df = df[df["timestamps"] < first_asof].copy()
    if verbose:
        print(
            f"Improve v2: {len(zs_steps)} zeroshot steps, "
            f"supervised train bars={len(train_df)} before {first_asof}",
            flush=True,
        )

    # 1) Meta variants on zero-shot Kronos features
    meta_base = compare_raw_vs_meta(
        zs_steps,
        ohlcv=df,
        min_train=min_train,
        model_type="logistic",
        embargo_steps=0,
        proba_long=0.55,
        proba_short=0.45,
    )
    meta_embargo = compare_raw_vs_meta(
        zs_steps,
        ohlcv=df,
        min_train=min_train,
        model_type="logistic",
        embargo_steps=1,
        proba_long=0.55,
        proba_short=0.45,
    )
    meta_lgbm = compare_raw_vs_meta(
        zs_steps,
        ohlcv=df,
        min_train=min_train,
        model_type="lightgbm",
        embargo_steps=0,
        rich=True,
        proba_long=0.55,
        proba_short=0.45,
    )

    # 2) Supervised direction head
    ckpt = "/root/.cache/huggingface/kronos-btc-supervised"
    sup_info = train_supervised_direction(
        train_df,
        save_dir=ckpt,
        epochs=supervised_epochs,
        batch_size=8,
        lookback=lookback_sup,
        pred_len=pred_len,
        n_samples=2500,
        unfreeze_last_n=2,
        kronos_root="/opt/Kronos",
    )

    tokenizer, model, head, meta, device = load_supervised_bundle(
        ckpt, kronos_root="/opt/Kronos"
    )
    sup_map: dict[str, float] = {}
    import numpy as np

    ts_vals = pd.to_datetime(df["timestamps"])
    for i, s in enumerate(zs_steps):
        asof = pd.Timestamp(s["asof"])
        matches = np.where(ts_vals == asof)[0]
        if len(matches) == 0:
            # try tz-normalize match on date
            matches = np.where(ts_vals.dt.tz_convert("UTC") == asof.tz_convert("UTC"))[0]
        if len(matches) == 0:
            raise RuntimeError(f"asof {asof} not found in OHLCV")
        end_idx = int(matches[0])
        hist = df.iloc[max(0, end_idx - lookback_sup + 1) : end_idx + 1]
        p = predict_supervised_p_up_loaded(
            hist, tokenizer, model, head, meta, device
        )
        sup_map[str(asof)] = p
        if verbose and (i + 1) % 25 == 0:
            print(f"supervised probs {i + 1}/{len(zs_steps)}", flush=True)

    # 3) Meta + supervised feature (baseline logistic config)
    zs_sup_compare = compare_raw_vs_meta(
        zs_steps,
        ohlcv=df,
        min_train=min_train,
        model_type="logistic",
        embargo_steps=0,
        supervised_p_up=sup_map,
        proba_long=0.55,
        proba_short=0.45,
    )
    sup_alone = supervised_rule_backtest(
        zs_steps, sup_map, min_train=min_train, proba_long=0.55, proba_short=0.45
    )

    hf_cache.commit()
    payload = {
        "meta_v2": {
            "raw_aligned": meta_base["raw_aligned"],
            "meta_logistic": meta_base["meta"],
            "meta_logistic_embargo": meta_embargo["meta"],
            "meta_market_only": meta_base["meta_market_only"],
            "meta_lgbm_rich": meta_lgbm["meta"],
        },
        "supervised": {
            "train_info": {
                "save_dir": sup_info.get("save_dir"),
                "n_train_bars": sup_info.get("n_train_bars"),
                "lookback": sup_info.get("lookback"),
                "pred_len": sup_info.get("pred_len"),
                "history": sup_info.get("history"),
                "supervised_epochs": supervised_epochs,
            },
            "head_alone": sup_alone,
            "meta_with_sup": zs_sup_compare["meta"],
            "sup_p_up": sup_map,
        },
    }
    if verbose:
        print(
            json.dumps(
                {
                    "raw_aligned": meta_base["raw_aligned"]["total_return"],
                    "meta_logistic": meta_base["meta"]["total_return"],
                    "meta_logistic_embargo": meta_embargo["meta"]["total_return"],
                    "meta_market_only": meta_base["meta_market_only"]["total_return"],
                    "meta_lgbm_rich": meta_lgbm["meta"]["total_return"],
                    "sup_head_alone": sup_alone["total_return"],
                    "meta_with_sup": zs_sup_compare["meta"]["total_return"],
                },
                indent=2,
            ),
            flush=True,
        )
    return payload


@app.function(
    timeout=60 * 60 * 4,
    volumes={"/data/crypto": crypto_data},
    memory=16384,
    cpu=4,
)
def download_crypto_panel_job(
    max_coins: int = 0,
    start_date: str = "2016-01-01",
    sleep_s: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Download KuCoin/CMC daily history (notebook recipe) onto the crypto volume."""
    from pathlib import Path as P

    from kronos_signal.download_cmc_kucoin import download_universe

    out = P("/data/crypto/historical_data_full.csv")
    if verbose:
        print(
            f"Downloading KuCoin universe max_coins={max_coins or 'ALL'} "
            f"start_date={start_date} → {out}",
            flush=True,
        )
    df = download_universe(
        out_path=out,
        max_coins=None if max_coins == 0 else max_coins,
        skip_stables=True,
        sleep_s=sleep_s,
        save_every=25,
        start_date=start_date,
    )
    # Also refresh the default alias used by FT pipeline.
    alias = P("/data/crypto/historical_data.csv")
    df.to_csv(alias, index=False)
    crypto_data.commit()
    return {
        "out": str(out),
        "alias": str(alias),
        "n_rows": int(len(df)),
        "n_symbols": int(df["currency_symbol"].nunique()),
        "timestamp_min": str(df["timestamp"].min()),
        "timestamp_max": str(df["timestamp"].max()),
        "start_date": start_date,
        "max_coins": max_coins,
    }


@app.function(
    timeout=60 * 30,
    volumes={"/data/crypto": crypto_data},
    memory=32768,
    cpu=8,
)
def prepare_robust_base_ft_job(verbose: bool = True) -> dict:
    """Create leakage-free train/val/test pickles for robust Kronos-base FT."""
    from pathlib import Path as P

    from kronos_signal.prepare_official_pickles import prepare_pickles
    from kronos_signal.robust_finetune import robust_config

    csv_path = P("/data/crypto/historical_data_full.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing full panel: {csv_path}")
    root = P("/data/crypto/official_runs_base_robust")
    cfg = robust_config(root, predictor_size="base")
    meta = prepare_pickles(csv_path, cfg)
    crypto_data.commit()
    result = {
        "root": str(root),
        "csv": str(csv_path),
        "config": cfg.as_dict(),
        "dataset_meta": meta,
    }
    (root / "prepare_summary.json").write_text(
        __import__("json").dumps(result, indent=2, default=str)
    )
    crypto_data.commit()
    if verbose:
        print(__import__("json").dumps(result, indent=2, default=str), flush=True)
    return result


@app.function(
    gpu="H100",
    timeout=60 * 30,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data/crypto": crypto_data,
    },
    memory=65536,
)
def run_robust_base_epoch_job(
    phase: str,
    patience: int = 4,
    min_delta: float = 1e-4,
) -> dict:
    """Run one resumable FT epoch and persist all state before returning."""
    from pathlib import Path as P

    from kronos_signal.robust_finetune import robust_config, train_one_epoch

    if phase not in ("tokenizer", "predictor"):
        raise ValueError("phase must be tokenizer or predictor")
    root = P("/data/crypto/official_runs_base_robust")
    cfg = robust_config(root, predictor_size="base")
    result = train_one_epoch(
        cfg,
        phase=phase,
        device="cuda",
        kronos_root="/opt/Kronos",
        patience=patience,
        min_delta=min_delta,
    )
    crypto_data.commit()
    hf_cache.commit()
    return result


@app.function(
    gpu="H100",
    timeout=60 * 30,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data/crypto": crypto_data,
    },
    memory=65536,
)
def run_robust_base_scores_job(
    lookback_window: int = 90,
    predict_window: int = 10,
    universe_n: int = 30,
    score_stride: int = 10,
    zero_shot: bool = False,
    symbol_chunk_index: int = 0,
    symbol_chunk_count: int = 1,
) -> dict:
    """Generate robust FT scores only for PIT universe/rebalance dates."""
    import pickle
    from pathlib import Path as P

    from kronos_signal.official_infer_bt import generate_ft_scores
    from kronos_signal.robust_finetune import robust_config

    root = P("/data/crypto/official_runs_base_robust")
    cfg = robust_config(root, predictor_size="base")
    cfg.lookback_window = int(lookback_window)
    cfg.predict_window = int(predict_window)
    scores = generate_ft_scores(
        cfg,
        device="cuda",
        kronos_root="/opt/Kronos",
        tokenizer_path=cfg.pretrained_tokenizer_path if zero_shot else None,
        predictor_path=cfg.pretrained_predictor_path if zero_shot else None,
        pit_universe_n=universe_n,
        score_stride=score_stride,
        symbol_chunk_index=symbol_chunk_index,
        symbol_chunk_count=symbol_chunk_count,
    )
    tag = f"lb{lookback_window}_h{predict_window}_pit{universe_n}_s{score_stride}"
    if symbol_chunk_count > 1:
        tag += f"_c{symbol_chunk_index}of{symbol_chunk_count}"
    prefix = "zs" if zero_shot else "ft"
    score_path = root / f"{prefix}_prediction_scores_{tag}.pkl"
    with open(score_path, "wb") as f:
        pickle.dump(scores, f)
    result = {
        "recipe": "robust_base_zero_shot_scores" if zero_shot else "robust_base_ft_scores",
        "root": str(root),
        "score_path": str(score_path),
        "lookback": lookback_window,
        "predict_window": predict_window,
        "universe_n": universe_n,
        "score_stride": score_stride,
        "zero_shot": zero_shot,
        "symbol_chunk_index": symbol_chunk_index,
        "symbol_chunk_count": symbol_chunk_count,
        "n_dates": {k: int(len(v)) for k, v in scores.items()},
        "n_symbols": {k: int(v.shape[1]) for k, v in scores.items()},
    }
    (root / f"last_{prefix}_scores_{tag}.json").write_text(
        __import__("json").dumps(result, indent=2)
    )
    crypto_data.commit()
    hf_cache.commit()
    return result


@app.function(
    gpu="T4",
    timeout=60 * 30,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data/crypto": crypto_data,
    },
    memory=16384,
)
def run_asof_forecast_job(
    symbols: str = "BTC,ETH",
    asof: str = "2026-01-10",
    lookback: int = 90,
    pred_len: int = 60,
    n_paths: int = 20,
    use_finetuned: bool = True,
    T: float = 0.6,
    top_p: float = 0.9,
) -> dict:
    """Monte Carlo forecast for a few symbols from a fixed asof date."""
    from datetime import timedelta
    from pathlib import Path as P

    import numpy as np
    import pandas as pd

    from kronos_signal.forecast import forecast_close_paths, load_predictor
    from kronos_signal.robust_finetune import robust_config

    csv_path = P("/data/crypto/historical_data_full.csv")
    if not csv_path.exists():
        csv_path = P("/data/crypto/historical_data.csv")
    raw = pd.read_csv(csv_path)
    raw["timestamps"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["symbol"] = raw["currency_symbol"].astype(str).str.upper()
    if "amount" not in raw.columns:
        raw["amount"] = raw["volume"].astype(float) * raw[
            ["open", "high", "low", "close"]
        ].mean(axis=1)

    asof_ts = pd.Timestamp(asof, tz="UTC")
    root = P("/data/crypto/official_runs_base_robust")
    cfg = robust_config(root, predictor_size="base")
    if use_finetuned:
        tok = cfg.finetuned_tokenizer_path
        pred = cfg.finetuned_predictor_path
        model_tag = "robust_ft_base"
    else:
        tok = cfg.pretrained_tokenizer_path
        pred = cfg.pretrained_predictor_path
        model_tag = "zero_shot_base"

    predictor = load_predictor(
        model_id=pred,
        tokenizer_id=tok,
        max_context=cfg.max_context,
        device="cuda",
        kronos_root="/opt/Kronos",
    )

    out_symbols: dict[str, dict] = {}
    for sym in [s.strip().upper() for s in symbols.split(",") if s.strip()]:
        sdf = raw.loc[raw["symbol"] == sym].sort_values("timestamps").reset_index(drop=True)
        hist = sdf.loc[sdf["timestamps"] <= asof_ts].copy()
        if len(hist) < lookback:
            raise ValueError(f"{sym}: need {lookback} bars <= {asof}, got {len(hist)}")
        hist = hist.iloc[-lookback:].reset_index(drop=True)
        last_ts = hist["timestamps"].iloc[-1]
        x_df = hist[["open", "high", "low", "close", "volume", "amount"]].copy()
        x_ts = hist["timestamps"]
        y_ts = pd.Series(
            [last_ts + timedelta(days=i) for i in range(1, pred_len + 1)],
            name="timestamps",
        )
        paths = forecast_close_paths(
            predictor,
            x_df,
            x_ts,
            y_ts,
            pred_len=pred_len,
            n_paths=n_paths,
            T=T,
            top_p=top_p,
            verbose=True,
        )
        # realized future closes if available in CSV
        future = sdf.loc[sdf["timestamps"] > last_ts].head(pred_len)
        actual = {
            "timestamps": [t.isoformat() for t in future["timestamps"]],
            "close": future["close"].astype(float).tolist(),
        }
        q10, q50, q90 = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
        mean_path = paths.mean(axis=0)
        last_close = float(hist["close"].iloc[-1])
        out_symbols[sym] = {
            "asof_used": last_ts.isoformat(),
            "last_close": last_close,
            "pred_timestamps": [t.isoformat() for t in y_ts],
            "path_closes": paths.astype(float).tolist(),
            "mean": mean_path.astype(float).tolist(),
            "q10": q10.astype(float).tolist(),
            "q50": q50.astype(float).tolist(),
            "q90": q90.astype(float).tolist(),
            "mean_horizon_return": float(mean_path[-1] / last_close - 1.0),
            "median_horizon_return": float(q50[-1] / last_close - 1.0),
            "actual": actual,
        }
        print(
            f"[{sym}] asof={last_ts.date()} close={last_close:.2f} "
            f"mean_r60={out_symbols[sym]['mean_horizon_return']:+.2%} "
            f"med_r60={out_symbols[sym]['median_horizon_return']:+.2%}",
            flush=True,
        )

    result = {
        "recipe": "asof_forecast",
        "model": model_tag,
        "asof": asof,
        "lookback": lookback,
        "pred_len": pred_len,
        "n_paths": n_paths,
        "T": T,
        "top_p": top_p,
        "symbols": out_symbols,
    }
    out_path = root / f"asof_forecast_{asof}_{model_tag}_h{pred_len}.json"
    out_path.write_text(json.dumps(result, indent=2))
    crypto_data.commit()
    return result


@app.function(
    gpu="H100",
    timeout=60 * 60 * 12,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data/crypto": crypto_data,
    },
    memory=65536,
)
def run_official_ft_pipeline(
    predictor_size: str = "small",
    epochs: int = 30,
    skip_train: bool = False,
    skip_tokenizer: bool = False,
    skip_predictor: bool = False,
    max_coins: int = 60,
    start_date: str = "2016-01-01",
    force_download: bool = False,
    verbose: bool = True,
) -> dict:
    """Official Kronos recipe: pickles → FT tokenizer → FT predictor → TopkDropout."""
    import shutil
    from pathlib import Path as P

    from kronos_signal.download_cmc_kucoin import download_universe

    # Prefer full panel when present (all-KuCoin from 2016); else legacy/top-N CSV.
    csv_candidates = [
        P("/data/crypto/historical_data_full.csv"),
        P("/root/kronos_signal_data/historical_data.csv"),
        P("/data/crypto/historical_data.csv"),
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if force_download or csv_path is None:
        if verbose:
            print(
                f"Downloading panel max_coins={max_coins or 'ALL'} start={start_date}…",
                flush=True,
            )
        csv_path = P("/data/crypto/historical_data_full.csv")
        download_universe(
            csv_path,
            max_coins=None if max_coins == 0 else max_coins,
            skip_stables=True,
            start_date=start_date,
        )
        shutil.copy(csv_path, "/data/crypto/historical_data.csv")
    elif str(csv_path) != "/data/crypto/historical_data.csv":
        P("/data/crypto").mkdir(parents=True, exist_ok=True)
        # Keep a working alias without overwriting a richer full panel.
        if csv_path.name == "historical_data_full.csv":
            shutil.copy(csv_path, "/data/crypto/historical_data.csv")
        else:
            shutil.copy(csv_path, "/data/crypto/historical_data.csv")
            csv_path = P("/data/crypto/historical_data.csv")

    # Separate root for base so small ckpts / pickles are preserved.
    root = (
        P("/data/crypto/official_runs_base")
        if predictor_size == "base"
        else P("/data/crypto/official_runs")
    )
    if verbose:
        print(
            f"Official FT start={predictor_size} epochs={epochs} csv={csv_path} root={root} "
            f"skip_tok={skip_tokenizer} skip_pred={skip_predictor}",
            flush=True,
        )

    # Commit after pickles / tokenizer so a mid-run cancel does not lose progress.
    from kronos_signal.official_config import OfficialConfig
    from kronos_signal.official_pipeline import run_official_pipeline
    from kronos_signal.official_train import run_official_finetune
    from kronos_signal.prepare_official_pickles import prepare_pickles
    from kronos_signal.official_topk_bt import load_full_panel, panels_from_full, roc_scores, topk_dropout_backtest

    cfg = OfficialConfig(root=root, predictor_size=predictor_size, epochs=epochs)
    meta = prepare_pickles(csv_path, cfg)
    crypto_data.commit()
    if verbose:
        print(f"Pickles ready: {meta}", flush=True)

    train_info = None
    if not skip_train:
        if not skip_tokenizer:
            train_info = {"tokenizer": run_official_finetune(
                cfg, device="cuda", kronos_root="/opt/Kronos",
                skip_tokenizer=False, skip_predictor=True,
            ).get("tokenizer")}
            crypto_data.commit()
            hf_cache.commit()
            if verbose:
                print(f"Tokenizer done/committed: {train_info}", flush=True)
        else:
            train_info = {"tokenizer": {"skipped": True, "path": cfg.finetuned_tokenizer_path}}

        if not skip_predictor:
            pred_out = run_official_finetune(
                cfg, device="cuda", kronos_root="/opt/Kronos",
                skip_tokenizer=True, skip_predictor=False,
            )
            train_info = {**(train_info or {}), **pred_out}
            crypto_data.commit()
            hf_cache.commit()
            if verbose:
                print(f"Predictor done/committed: {pred_out}", flush=True)

    data = load_full_panel(cfg)
    panels = panels_from_full(data)
    scores = roc_scores(panels["close"], window=cfg.predict_window)
    bt = topk_dropout_backtest(
        scores, panels["close"], cfg, universe_mcap=panels["marketCap"]
    )
    summary = {
        "recipe": "official_kronos_mirror",
        "notes": {
            "market": "crypto_panel_not_csi300",
            "ft_start": cfg.pretrained_predictor_path,
            "lookback": cfg.lookback_window,
            "predict_window": cfg.predict_window,
            "topk": cfg.backtest_n_symbol_hold,
            "n_drop": cfg.backtest_n_symbol_drop,
            "hold_thresh": cfg.backtest_hold_thresh,
            "long_only": True,
            "not_long_short": True,
            "root": str(root),
        },
        "dataset_meta": meta,
        "train": train_info,
        "topk_roc_baseline": {
            "total_return": bt["total_return"],
            "max_drawdown": bt["max_drawdown"],
            "sharpe": bt["sharpe"],
            "btc_total_return": bt["btc_total_return"],
            "n_days": bt["n_days"],
        },
        "paths": {
            "tokenizer": cfg.finetuned_tokenizer_path,
            "predictor": cfg.finetuned_predictor_path,
            "dataset": cfg.dataset_path,
        },
    }
    import json as _json
    (root / "last_official_summary.json").write_text(_json.dumps(summary, indent=2, default=str))
    crypto_data.commit()
    hf_cache.commit()
    return summary


@app.function(
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data/crypto": crypto_data,
    },
    memory=65536,
)
def run_official_ft_backtest_job(
    predictor_size: str = "small",
    signal: str = "mean",
    lookback_window: int = 90,
    predict_window: int = 10,
    also_pure_topk: bool = True,
    max_infer_symbols: int = 48,
    verbose: bool = True,
) -> dict:
    """Infer with fine-tuned Kronos and run TopkDropout vs ROC/BTC."""
    from pathlib import Path as P

    from kronos_signal.official_infer_bt import run_official_ft_backtest

    root = (
        P("/data/crypto/official_runs_base")
        if predictor_size == "base"
        else P("/data/crypto/official_runs")
    )
    if verbose:
        print(
            f"Official FT backtest root={root} signal={signal} "
            f"lb={lookback_window} pred={predict_window} max_symbols={max_infer_symbols}",
            flush=True,
        )
    out = run_official_ft_backtest(
        root=root,
        device="cuda",
        kronos_root="/opt/Kronos",
        predictor_size=predictor_size,
        signal=signal,
        lookback_window=lookback_window,
        predict_window=predict_window,
        also_pure_topk=also_pure_topk,
        max_infer_symbols=max_infer_symbols,
    )
    crypto_data.commit()
    return out


@app.function(
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data/crypto": crypto_data,
    },
    memory=65536,
)
def run_zero_shot_scores_job(
    predictor_size: str = "small",
    lookback_window: int = 90,
    predict_window: int = 10,
    verbose: bool = True,
) -> dict:
    """Infer with pretrained (non-FT) Kronos and save score matrices for L3/S3."""
    import pickle
    from pathlib import Path as P

    from kronos_signal.official_config import OfficialConfig
    from kronos_signal.official_infer_bt import generate_zero_shot_scores

    root = (
        P("/data/crypto/official_runs_base")
        if predictor_size == "base"
        else P("/data/crypto/official_runs")
    )
    cfg = OfficialConfig(root=root, predictor_size=predictor_size, epochs=30)
    cfg.lookback_window = int(lookback_window)
    cfg.predict_window = int(predict_window)
    if verbose:
        print(
            f"Zero-shot infer root={root} pretrained={cfg.pretrained_predictor_path} "
            f"lb={cfg.lookback_window} pred={cfg.predict_window}",
            flush=True,
        )
    scores = generate_zero_shot_scores(cfg, device="cuda", kronos_root="/opt/Kronos")
    tag = f"lb{cfg.lookback_window}_h{cfg.predict_window}"
    out_path = root / f"zs_prediction_scores_{tag}.pkl"
    alias = root / "zs_prediction_scores.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(scores, f)
    with open(alias, "wb") as f:
        pickle.dump(scores, f)
    meta = {
        "recipe": "zero_shot_scores",
        "predictor": cfg.pretrained_predictor_path,
        "tokenizer": cfg.pretrained_tokenizer_path,
        "lookback": cfg.lookback_window,
        "predict_window": cfg.predict_window,
        "scores_path": str(out_path),
        "n_dates": {k: int(len(v)) for k, v in scores.items()},
        "n_symbols": {k: int(v.shape[1]) for k, v in scores.items()},
    }
    (root / f"last_zs_scores_{tag}.json").write_text(
        __import__("json").dumps(meta, indent=2)
    )
    crypto_data.commit()
    hf_cache.commit()
    return meta


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
    supervised_epochs: int = 8,
    start_asof: str = "2021-01-01",
    end_asof: str = "",
    predictor_size: str = "small",
    official_epochs: int = 30,
    signal: str = "mean",
    max_coins: int = 60,
    start_date: str = "2016-01-01",
    force_download: bool = False,
    skip_tokenizer: bool = False,
    skip_predictor: bool = False,
    robust_phase: str = "",
    robust_max_tokenizer_epochs: int = 20,
    robust_max_predictor_epochs: int = 30,
    robust_patience: int = 4,
    robust_score_stride: int = 10,
    robust_score_chunk_index: int = 0,
    robust_score_chunk_count: int = 1,
    symbols: str = "BTC,ETH",
    asof: str = "",
    use_finetuned: bool = True,
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
            start_asof=start_asof or None,
            end_asof=end_asof or None,
            verbose=True,
        )
        out = Path("kronos_signal") / "last_backtest.json"
        out.write_text(json.dumps(result, indent=2))
        _print_bt("BACKTEST", result)
        print(f"Wrote {out}")
        return

    if mode == "long_annual":
        from kronos_signal import config as cfg

        n_paths = 10 if n_paths <= 0 else n_paths
        # Convenience: pred_len=1 implies daily rebalance + tighter tau unless overridden.
        use_step = step if step > 0 else pred_len
        use_tau = tau
        if pred_len == 1 and abs(tau - 0.005) < 1e-12:
            use_tau = cfg.DAILY_TAU
            use_step = 1
        result = run_long_annual_pipeline.remote(
            n_paths=n_paths,
            pred_len=pred_len,
            step=use_step,
            lookback=lookback,
            tau=use_tau,
            start_asof=start_asof or "2021-01-01",
            end_asof=end_asof or None,
            verbose=True,
        )
        tag = "daily" if pred_len == 1 and use_step == 1 else "h" + str(pred_len)
        out = Path("kronos_signal") / f"last_long_annual_{tag}.json"
        # also keep stable alias for daily / default
        out.write_text(json.dumps(result, indent=2))
        alias = Path("kronos_signal") / "last_long_annual.json"
        alias.write_text(json.dumps(result, indent=2))
        print("\n=== LONG ANNUAL META ===")
        hz = result.get("horizon") or {}
        print(f"data: {result.get('data_source')}")
        print(
            f"horizon: Kronos-base pred_len={hz.get('pred_len')} step={hz.get('step')} "
            f"rebalance={hz.get('rebalance')} tau={hz.get('tau')} n_paths={hz.get('n_paths')}"
        )
        print(
            f"overall: {result['overall']['start']} → {result['overall']['end']}  "
            f"steps={result['overall']['n_steps']}  "
            f"ret={result['overall']['total_return']:.2%}  "
            f"B&H={result['overall']['buy_hold_return']:.2%}  "
            f"hit={result['overall']['hit_rate']}"
        )
        for y in result["by_year"]:
            print(
                f"  TRAIN {y.get('train_start')}→{y.get('train_end')}  "
                f"TEST {y.get('test_start')}→{y.get('test_end')}  "
                f"ret={y['total_return']:+.1%}  B&H={y['buy_hold_return']:+.1%}  "
                f"hit={y['hit_rate']}  L/S/H={y['n_long']}/{y['n_short']}/{y['n_hold']}"
            )
        raw = result.get("kronos_raw_summary") or {}
        print(
            f"raw Kronos full window: ret={raw.get('total_return')} hit={raw.get('hit_rate')}"
        )
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

    if mode == "improve_v2":
        zs_path = Path("kronos_signal") / "last_backtest.json"
        if not zs_path.exists():
            # fall back to zeroshot steps embedded in last_improve.json
            alt = Path("kronos_signal") / "last_improve.json"
            if not alt.exists():
                raise SystemExit("Need kronos_signal/last_backtest.json (or last_improve.json)")
            zs_steps = json.loads(alt.read_text())["zeroshot"]["steps"]
        else:
            zs_steps = json.loads(zs_path.read_text())["steps"]
        result = run_improve_v2_pipeline.remote(
            json.dumps(zs_steps),
            min_train=40,
            supervised_epochs=supervised_epochs,
            lookback_sup=90,
            pred_len=pred_len,
            verbose=True,
        )
        out = Path("kronos_signal") / "last_improve_v2.json"
        out.write_text(json.dumps(result, indent=2))
        print("\n=== IMPROVE V2 SUMMARY ===")
        mv = result["meta_v2"]
        for key in (
            "raw_aligned",
            "meta_logistic",
            "meta_logistic_embargo",
            "meta_market_only",
            "meta_lgbm_rich",
        ):
            r = mv[key]
            print(
                f"{key:24} {r['total_return']:.2%}  hit={r['hit_rate']}  "
                f"L/S/H={r['n_long']}/{r['n_short']}/{r['n_hold']}"
            )
        for key in ("head_alone", "meta_with_sup"):
            r = result["supervised"][key]
            print(
                f"{key:24} {r['total_return']:.2%}  hit={r['hit_rate']}  "
                f"L/S/H={r['n_long']}/{r['n_short']}/{r['n_hold']}"
            )
        print(f"Wrote {out}")
        return

    if mode == "download":
        result = download_crypto_panel_job.remote(
            max_coins=max_coins,
            start_date=start_date,
            verbose=True,
        )
        out = Path("kronos_signal") / "last_download.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print("\n=== CRYPTO PANEL DOWNLOAD ===")
        print(json.dumps(result, indent=2, default=str))
        print(f"Wrote {out}")
        return

    if mode == "robust_ft":
        prep = prepare_robust_base_ft_job.remote(verbose=True)
        phases = (
            [robust_phase]
            if robust_phase in ("tokenizer", "predictor")
            else ["tokenizer", "predictor"]
        )
        phase_results = {}
        for phase in phases:
            max_epochs = (
                robust_max_tokenizer_epochs
                if phase == "tokenizer"
                else robust_max_predictor_epochs
            )
            while True:
                result = run_robust_base_epoch_job.remote(
                    phase=phase,
                    patience=robust_patience,
                    min_delta=1e-5 if phase == "tokenizer" else 1e-3,
                )
                phase_results[phase] = result
                print(
                    f"[robust loop] {phase} epoch={result.get('completed_epochs')} "
                    f"best={result.get('best_val_loss')} "
                    f"bad={result.get('bad_epochs')} stopped={result.get('stopped_early')}",
                    flush=True,
                )
                if result.get("stopped_early"):
                    break
                if int(result.get("completed_epochs", 0)) >= max_epochs:
                    break
        summary = {
            "recipe": "robust_resumable_base_ft",
            "data": prep,
            "phases": phase_results,
            "train_range": ["2016-01-01", "2022-12-31"],
            "val_range": ["2023-01-01", "2024-06-30"],
            "test_range": ["2024-07-01", "2026-08-08"],
        }
        out = Path("kronos_signal") / "last_robust_base_ft.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, indent=2, default=str))
        print(f"Wrote {out}")
        return

    if mode in ("robust_scores", "robust_zs_scores"):
        result = run_robust_base_scores_job.remote(
            lookback_window=90 if lookback == 400 else lookback,
            predict_window=10 if (lookback == 400 and pred_len == 5) else pred_len,
            universe_n=30,
            score_stride=robust_score_stride,
            zero_shot=(mode == "robust_zs_scores"),
            symbol_chunk_index=robust_score_chunk_index,
            symbol_chunk_count=robust_score_chunk_count,
        )
        out = Path("kronos_signal") / (
            "last_robust_base_zs_scores.json"
            if mode == "robust_zs_scores"
            else "last_robust_base_scores.json"
        )
        out.write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        print(f"Wrote {out}")
        return

    if mode == "official":
        result = run_official_ft_pipeline.remote(
            predictor_size=predictor_size,
            epochs=official_epochs,
            skip_train=False,
            skip_tokenizer=skip_tokenizer,
            skip_predictor=skip_predictor,
            max_coins=max_coins,
            start_date=start_date,
            force_download=force_download,
            verbose=True,
        )
        tag = f"{predictor_size}"
        out = Path("kronos_signal") / f"last_official_{tag}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        (Path("kronos_signal") / "last_official.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        print("\n=== OFFICIAL KRONOS RECIPE ===")
        print(json.dumps(result.get("notes", {}), indent=2))
        print(json.dumps(result.get("dataset_meta", {}), indent=2))
        if result.get("train"):
            print("train:", json.dumps(result["train"], indent=2, default=str))
        print("topk ROC baseline:", json.dumps(result.get("topk_roc_baseline", {}), indent=2))
        print(f"Wrote {out}")
        return

    if mode == "official_bt":
        # Inference windows (FT ckpt still trained at 90/10). Pass explicitly, e.g.:
        #   --lookback 90 --pred-len 10   (official)
        #   --lookback 30 --pred-len 3    (short horizon experiment)
        lb = 90 if lookback == 400 else lookback
        ph = 10 if (lookback == 400 and pred_len == 5) else pred_len
        result = run_official_ft_backtest_job.remote(
            predictor_size=predictor_size,
            signal=signal,
            lookback_window=lb,
            predict_window=ph,
            also_pure_topk=True,
            verbose=True,
        )
        tag = f"lb{lb}_h{ph}"
        out = Path("kronos_signal") / f"last_official_ft_bt_{tag}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        (Path("kronos_signal") / "last_official_ft_bt.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        print("\n=== OFFICIAL FT TOPK BACKTEST ===")
        print("config:", json.dumps(result.get("config"), indent=2))
        print("primary (dropout):", json.dumps(result.get("primary"), indent=2))
        print("primary (pure):", json.dumps(result.get("primary_pure"), indent=2))
        print("all signals:", json.dumps(result.get("all_signals"), indent=2))
        print(f"Wrote {out}")
        return

    if mode == "zs_scores":
        lb = 90 if lookback == 400 else lookback
        ph = 10 if (lookback == 400 and pred_len == 5) else pred_len
        result = run_zero_shot_scores_job.remote(
            predictor_size=predictor_size,
            lookback_window=lb,
            predict_window=ph,
            verbose=True,
        )
        out = Path("kronos_signal") / f"last_zs_scores_lb{lb}_h{ph}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print("\n=== ZERO-SHOT KRONOS SCORES ===")
        print(json.dumps(result, indent=2, default=str))
        print(f"Wrote {out}")
        return

    if mode == "forecast_asof":
        lb = 90 if lookback == 400 else lookback
        pl = 60 if pred_len == 5 else pred_len
        n_paths = 20 if n_paths <= 0 else n_paths
        result = run_asof_forecast_job.remote(
            symbols=symbols or "BTC,ETH",
            asof=asof or "2026-01-10",
            lookback=lb,
            pred_len=pl,
            n_paths=n_paths,
            use_finetuned=use_finetuned,
        )
        tag = result.get("model", "model")
        out = Path("kronos_signal") / f"last_asof_forecast_{result['asof']}_{tag}_h{pl}.json"
        # Drop bulky path matrix from local copy summary? keep full for plotting.
        out.write_text(json.dumps(result, indent=2))
        print("\n=== ASOF FORECAST ===")
        print(
            f"model={tag} asof={result['asof']} lookback={result['lookback']} "
            f"pred_len={result['pred_len']} n_paths={result['n_paths']}"
        )
        for sym, payload in result["symbols"].items():
            print(
                f"  {sym}: close={payload['last_close']:.4g} "
                f"mean_r={payload['mean_horizon_return']:+.2%} "
                f"med_r={payload['median_horizon_return']:+.2%} "
                f"asof_used={payload['asof_used'][:10]}"
            )
        print(f"Wrote {out}")
        return

    if mode != "signal":
        raise SystemExit(
            f"Unknown mode={mode!r}; use 'signal', 'backtest', 'long_annual', "
            f"'improve', 'improve_v2', 'download', 'official', 'official_bt', "
            f"'zs_scores', or 'forecast_asof'"
        )

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
