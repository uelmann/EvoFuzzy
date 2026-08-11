"""Best-effort Kronos-ft score export into canonical prediction format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def try_export_kronos_ft(
    out_path: Path,
    horizon: int = 10,
    search_paths: list[Path] | None = None,
) -> dict:
    """
    Look for existing Kronos FT score pickles and convert to
    date, symbol, score, horizon, model_name parquet.
    Non-blocking: returns status dict; never raises to caller pipeline.
    """
    search_paths = search_paths or [
        Path("/data/quant/kronos_import"),
        Path("/opt/cursor/artifacts/crypto_data"),
        Path("kronos_signal"),
        Path("/data/crypto"),
    ]
    candidates = []
    names = [
        "robust_base_daily_ft_scores.pkl",
        "robust_base_ft_scores.pkl",
        "ft_prediction_scores.pkl",
    ]
    for root in search_paths:
        for name in names:
            p = root / name
            if p.exists():
                candidates.append(p)
    if not candidates:
        return {"exported": False, "reason": "no kronos score pickle found"}

    path = candidates[0]
    try:
        obj = pd.read_pickle(path)
        # expected dict with keys last/mean/...
        if isinstance(obj, dict):
            score_df = obj.get("last") or obj.get("mean") or next(iter(obj.values()))
        else:
            score_df = obj
        if not isinstance(score_df, pd.DataFrame):
            return {"exported": False, "reason": f"unsupported object in {path}"}
        try:
            long = score_df.stack().reset_index()
        except TypeError:
            long = score_df.stack(future_stack=True).reset_index()
        long.columns = ["date", "symbol", "score"]
        long["date"] = pd.to_datetime(long["date"], utc=True)
        # map bare symbols to *USDT if needed
        long["symbol"] = long["symbol"].astype(str).str.upper()
        long.loc[~long["symbol"].str.endswith("USDT"), "symbol"] = long["symbol"] + "USDT"
        long["horizon"] = horizon
        long["model_name"] = "kronos_ft"
        long = long.dropna(subset=["score"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(out_path, index=False)
        return {
            "exported": True,
            "source": str(path),
            "n_rows": int(len(long)),
            "out": str(out_path),
        }
    except Exception as e:
        return {"exported": False, "reason": str(e), "source": str(path)}
