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
    verbose: bool = True,
) -> dict:
    """Official Kronos recipe: pickles → FT tokenizer → FT predictor → TopkDropout."""
    import shutil
    from pathlib import Path as P

    from kronos_signal.download_cmc_kucoin import download_universe
    from kronos_signal.official_pipeline import run_official_pipeline

    csv_candidates = [
        P("/root/kronos_signal_data/historical_data.csv"),
        P("/data/crypto/historical_data.csv"),
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        if verbose:
            print("No panel CSV found — downloading top-60 KuCoin/CMC…", flush=True)
        csv_path = P("/data/crypto/historical_data.csv")
        download_universe(csv_path, max_coins=60, skip_stables=True)
    elif str(csv_path) != "/data/crypto/historical_data.csv":
        P("/data/crypto").mkdir(parents=True, exist_ok=True)
        shutil.copy(csv_path, "/data/crypto/historical_data.csv")
        csv_path = P("/data/crypto/historical_data.csv")

    root = P("/data/crypto/official_runs")
    if verbose:
        print(
            f"Official FT start={predictor_size} epochs={epochs} csv={csv_path} root={root}",
            flush=True,
        )
    summary = run_official_pipeline(
        csv_path=csv_path,
        root=root,
        predictor_size=predictor_size,
        epochs=epochs,
        device="cuda",
        kronos_root="/opt/Kronos",
        skip_train=skip_train,
        skip_tokenizer=skip_tokenizer,
        skip_predictor=skip_predictor,
    )
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
    verbose: bool = True,
) -> dict:
    """Infer with fine-tuned Kronos and run TopkDropout vs ROC/BTC."""
    from pathlib import Path as P

    from kronos_signal.official_infer_bt import run_official_ft_backtest

    root = P("/data/crypto/official_runs")
    if verbose:
        print(
            f"Official FT backtest root={root} signal={signal} "
            f"lb={lookback_window} pred={predict_window}",
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
    )
    crypto_data.commit()
    return out


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

    if mode == "official":
        result = run_official_ft_pipeline.remote(
            predictor_size=predictor_size,
            epochs=official_epochs,
            skip_train=False,
            verbose=True,
        )
        out = Path("kronos_signal") / "last_official.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print("\n=== OFFICIAL KRONOS RECIPE ===")
        print(json.dumps(result.get("notes", {}), indent=2))
        print(json.dumps(result.get("dataset_meta", {}), indent=2))
        if result.get("train"):
            print("train:", json.dumps(result["train"], indent=2, default=str))
        print("topk ROC baseline:", json.dumps(result.get("topk_roc_baseline", {}), indent=2))
        print(f"Wrote {out}")
        return

    if mode == "official_bt":
        # Reuse CLI lookback/pred_len for inference windows (defaults 400/5 are BTC;
        # for official recipe pass --lookback 90 --pred-len 10 or 30/3).
        lb = lookback if lookback not in (0, 400) else 90
        # If user explicitly set lookback via flag it's fine; for 30/3 they pass both.
        # Prefer explicit: when pred_len is 3 or lookback is 30, use as-is.
        if lookback == 400 and pred_len == 5:
            lb, ph = 90, 10
        else:
            lb, ph = lookback, pred_len
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

    if mode != "signal":
        raise SystemExit(
            f"Unknown mode={mode!r}; use 'signal', 'backtest', 'long_annual', "
            f"'improve', 'improve_v2', 'official', or 'official_bt'"
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
