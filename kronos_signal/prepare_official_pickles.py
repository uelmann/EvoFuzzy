"""Build Kronos-official train/val/test pickles from historical_data.csv."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .official_config import OfficialConfig
from .panel_data import DEFAULT_CSV, EXCLUDE_SYMBOLS, load_historical_long


def _symbol_frame(long_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    g = long_df[long_df["currency_symbol"] == symbol].copy()
    g = g.sort_values("date")
    g = g.drop_duplicates("date", keep="last").set_index("date")
    out = pd.DataFrame(
        {
            "open": g["open"].astype(float),
            "high": g["high"].astype(float),
            "low": g["low"].astype(float),
            "close": g["close"].astype(float),
            "vol": g["volume"].astype(float),
        }
    )
    # Same amount proxy as upstream Qlib preprocess (avg price * vol)
    out["amt"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0 * out["vol"]
    out["marketCap"] = g["marketCap"].astype(float)
    out = out.dropna()
    return out


def prepare_pickles(
    csv_path: Path = DEFAULT_CSV,
    cfg: OfficialConfig | None = None,
    min_bars: int | None = None,
) -> dict:
    cfg = cfg or OfficialConfig()
    min_bars = min_bars or (cfg.lookback_window + cfg.predict_window + 1)
    long_df = load_historical_long(csv_path)
    begin = pd.Timestamp(cfg.dataset_begin_time, tz="UTC")
    long_df = long_df[long_df["date"] >= begin].copy()
    symbols = sorted(s for s in long_df["currency_symbol"].unique() if s not in EXCLUDE_SYMBOLS)

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = _symbol_frame(long_df, sym)
        if len(sdf) < min_bars:
            continue
        data[sym] = sdf[cfg.feature_list + ["marketCap"]]

    def slice_range(start: str, end: str) -> dict[str, pd.DataFrame]:
        out = {}
        for sym, sdf in data.items():
            part = sdf.loc[(sdf.index >= pd.Timestamp(start, tz="UTC")) & (sdf.index <= pd.Timestamp(end, tz="UTC"))]
            # Drop marketCap from training features pickle (keep only model cols)
            part = part[cfg.feature_list].dropna()
            if len(part) >= min_bars:
                out[sym] = part
        return out

    train = slice_range(*cfg.train_time_range)
    val = slice_range(*cfg.val_time_range)
    test = slice_range(*cfg.test_time_range)

    # Full panel for backtest / universe (includes marketCap)
    full_path = Path(cfg.dataset_path)
    full_path.mkdir(parents=True, exist_ok=True)
    with open(full_path / "train_data.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(full_path / "val_data.pkl", "wb") as f:
        pickle.dump(val, f)
    with open(full_path / "test_data.pkl", "wb") as f:
        pickle.dump(test, f)
    with open(full_path / "full_panel.pkl", "wb") as f:
        pickle.dump(data, f)

    meta = {
        "n_train_symbols": len(train),
        "n_val_symbols": len(val),
        "n_test_symbols": len(test),
        "n_full_symbols": len(data),
        "train_range": cfg.train_time_range,
        "val_range": cfg.val_time_range,
        "test_range": cfg.test_time_range,
        "feature_list": cfg.feature_list,
    }
    (full_path / "meta.json").write_text(__import__("json").dumps(meta, indent=2))
    print(meta)
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--predictor-size", choices=["small", "base"], default="small")
    args = p.parse_args()
    cfg = OfficialConfig(root=args.root, predictor_size=args.predictor_size)
    prepare_pickles(args.csv, cfg)


if __name__ == "__main__":
    main()
