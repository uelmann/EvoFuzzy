"""
FuzzyX-v1d one shot — loss = −corr(cumprod(1+st_r), arange(T)).

BACKTEST ONLY. Addendum frozen: reports/fuzzyx_addendum_v1d.md
Does not replace COMBO / A0. Does not overwrite v1 / v1b / v1c reports.

    python3 fuzzyx_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baseline.data import build_pit_topn, download_symbol_months, load_panel, month_range
from baseline.features import build_feature_panel
from baseline.seedutil import seed_everything
from fuzzyx.constants import (
    EXEC_DV_WINDOW,
    FEATURE_COLS,
    INNER_HOLDOUT_DAYS,
    MIN_TRAIN_DAYS,
    REBALANCE_DAYS,
    SEED,
    STEP_DAYS,
    UNIVERSE_N,
    VAL_DAYS,
)
from fuzzyx.eval import attach_returns, book_from_pred, positions_to_frame, verdict
from fuzzyx.gates_fx import gate_shuffle_bias, run_leakage_gates
from fuzzyx.pack import pack_weekly, slice_packed
from fuzzyx.report import write_report
from fuzzyx.torch_model import FuzzyXNet
from fuzzyx.train import predict_packed, train_fold

LOCAL_SEED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "TIAUSDT", "FILUSDT", "AAVEUSDT",
    "LDOUSDT", "INJUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT", "TRXUSDT", "FETUSDT",
    "WLDUSDT", "SEIUSDT", "STXUSDT", "IMXUSDT", "ALGOUSDT", "VETUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "GRTUSDT", "EOSUSDT", "XTZUSDT",
]


def _cfg() -> dict:
    p = Path("config_fuzzyx.yaml")
    if p.exists():
        return yaml.safe_load(p.read_text())
    return {}


def _find_panel(root: Path) -> tuple[Path | None, str | None]:
    for p in (
        root / "panel" / "panel.parquet",
        root / "btcb" / "full" / "panel.parquet",
    ):
        if p.exists() and p.stat().st_size > 1000:
            return p, "VOLUME-PANEL"
    local = Path("artifacts/fuzzyx/panel.parquet")
    if local.exists() and local.stat().st_size > 1000:
        return local, "LOCAL-RESTRICTED"
    return None, None


def _find_a0_pred() -> Path | None:
    for p in (
        Path("/data/quant/predictions/lgbm_price_only_h7.parquet"),
        Path("artifacts/predictions/lgbm_price_only_h7.parquet"),
    ):
        if p.exists():
            return p
    return None


def _ensure_local_panel(raw_dir: Path, start_month: str = "2019-09") -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    raw_dir.mkdir(parents=True, exist_ok=True)
    months = month_range(start_month)

    def _one(sym: str) -> str:
        download_symbol_months(sym, months, raw_dir, interval="1d", kind="um")
        return sym

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, s) for s in LOCAL_SEED_SYMBOLS]
        for i, fut in enumerate(as_completed(futs), 1):
            sym = fut.result()
            print(f"[data] {i}/{len(LOCAL_SEED_SYMBOLS)} {sym}", flush=True)
    panel = load_panel(raw_dir, LOCAL_SEED_SYMBOLS)
    counts = panel.groupby("symbol").size()
    keep = counts[counts >= 100].index.tolist()
    if "BTCUSDT" not in keep:
        raise RuntimeError("BTCUSDT missing after local download")
    return panel[panel["symbol"].isin(keep)].copy()


def _oos_concat(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    dates = np.concatenate([np.asarray(p["dates"]) for p in parts])
    hard = np.concatenate([p["hard_pos"] for p in parts], axis=0)
    soft = np.concatenate([p["soft_pos"] for p in parts], axis=0)
    mask = np.concatenate([p["mask"] for p in parts], axis=0)
    ret = np.concatenate([p["ret_h7"] for p in parts], axis=0)
    sl = parts[0]["symbols"]
    return {
        "dates": pd.DatetimeIndex(dates),
        "symbols": sl,
        "hard_pos": hard,
        "soft_pos": soft,
        "mask": mask,
        "ret_h7": ret,
        "hard_loss": {},
        "soft_loss": {},
    }


def run_fuzzyx(root: Path | None = None) -> dict:
    from fuzzyx.loss import path_loss as np_path_loss

    cfg = _cfg()
    seed_everything(SEED)
    t0 = time.time()
    root = root or Path(cfg.get("paths", {}).get("volume_root") or "/data/quant")
    art = Path("artifacts/fuzzyx")
    art.mkdir(parents=True, exist_ok=True)
    notes = []

    panel_path, found_mode = _find_panel(root)
    if panel_path is not None:
        mode = found_mode or "VOLUME-PANEL"
        print(f"[fuzzyx] panel {panel_path} mode={mode}", flush=True)
        panel = pd.read_parquet(panel_path)
        panel["date"] = pd.to_datetime(panel["date"], utc=True)
        if "dollar_volume" not in panel.columns:
            if "quote_volume" in panel.columns:
                panel["dollar_volume"] = panel["quote_volume"].astype(float)
            elif "dv" in panel.columns:
                panel["dollar_volume"] = panel["dv"].astype(float)
            else:
                panel["dollar_volume"] = panel["close"].astype(float) * panel["volume"].astype(float)
        if "symbol" not in panel.columns and "id" in panel.columns:
            raise RuntimeError("CMC id-panel needs symbol map; use Vision panel for this shot")
        # Vision symbols vs BTC
        if "BTCUSDT" not in set(panel["symbol"].astype(str)) and "BTC" in set(panel["symbol"].astype(str)):
            notes.append("panel uses BTC not BTCUSDT; mapping BTC→BTCUSDT")
            panel["symbol"] = panel["symbol"].replace({"BTC": "BTCUSDT"})
    else:
        mode = "LOCAL-RESTRICTED"
        notes.append("No /data/quant panel; downloaded LOCAL_SEED_SYMBOLS from Binance Vision.")
        notes.append("LOCAL-RESTRICTED cannot produce a VIABLE vs-A0 verdict.")
        print("[fuzzyx] no volume panel; Vision seed download", flush=True)
        panel = _ensure_local_panel(art / "raw" / "klines")
        panel.to_parquet(art / "panel.parquet", index=False)

    if "symbol" in panel.columns:
        panel["symbol"] = panel["symbol"].astype(str)

    print(f"[fuzzyx] panel rows={len(panel)} names={panel['symbol'].nunique()}", flush=True)
    feat = build_feature_panel(panel, clip=5.0, zscore=True)
    feat = feat.drop_duplicates(["date", "symbol"], keep="last")
    feat_path = art / "feat.parquet"
    feat.to_parquet(feat_path, index=False)

    uni30 = build_pit_topn(panel, n=UNIVERSE_N, window=EXEC_DV_WINDOW)
    uni30.to_parquet(art / "top30_pit.parquet", index=False)
    packed = pack_weekly(feat, uni30, n=UNIVERSE_N, every=REBALANCE_DAYS)
    print(
        f"[fuzzyx] packed T={packed.X.shape[0]} S={packed.X.shape[1]} "
        f"mask_frac={packed.mask.mean():.2f}",
        flush=True,
    )
    np.savez_compressed(
        art / "packed.npz",
        X=packed.X,
        mask=packed.mask,
        ret_h7=packed.ret_h7,
        symbols=np.array(packed.symbols),
        dates=packed.reb_dates.asi8,
    )

    from baseline.model import make_folds

    dates = pd.DatetimeIndex(sorted(feat["date"].unique()))
    folds = make_folds(
        dates,
        horizon=7,
        min_train_days=MIN_TRAIN_DAYS,
        val_days=VAL_DAYS,
        step_days=STEP_DAYS,
    )
    print(f"[fuzzyx] folds={len(folds)}", flush=True)
    if not folds:
        notes.append("INSUFFICIENT_DATA: make_folds returned empty (need 730d+).")

    fold_rows = []
    oos_parts = []
    fold_states: dict[int, dict] = {}
    last_state = None
    n_params = 0
    for fold in folds:
        print(f"[fuzzyx] train fold {fold.fold_id} {fold.val_start.date()}→{fold.val_end.date()}", flush=True)
        res = train_fold(
            packed,
            fold.train_start,
            fold.train_end,
            fold.val_start,
            fold.val_end,
            seed=SEED,
            inner_holdout_days=INNER_HOLDOUT_DAYS,
        )
        n_params = res.n_params
        last_state = res.model_state
        rec = {
            "fold_id": fold.fold_id,
            "train_start": str(fold.train_start.date()),
            "train_end": str(fold.train_end.date()),
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "status": res.status,
            "best_epoch": res.best_epoch,
            "best_val": res.best_val,
            "elapsed": res.elapsed,
            "n_reb": int(slice_packed(packed, fold.val_start, fold.val_end).X.shape[0]),
        }
        fold_rows.append(rec)
        if res.status != "ok" or not res.model_state:
            continue
        fold_states[int(fold.fold_id)] = res.model_state
        model = FuzzyXNet(seed=SEED)
        model.load_state_dict(res.model_state)
        va = slice_packed(packed, fold.val_start, fold.val_end)
        pred = predict_packed(model, va)
        pred = attach_returns(pred, va)
        oos_parts.append(pred)
        rec["val_hard_loss"] = pred["hard_loss"]
        (art / f"fold{fold.fold_id}_meta.json").write_text(json.dumps(rec, indent=2, default=str))

    oos = _oos_concat(oos_parts)
    if oos is not None:
        oos["hard_loss"] = np_path_loss(oos["hard_pos"], oos["ret_h7"], mask=oos["mask"])
        oos["soft_loss"] = np_path_loss(oos["soft_pos"], oos["ret_h7"], mask=oos["mask"])
        book = book_from_pred(oos)
        pos_df = positions_to_frame(oos["dates"], oos["symbols"], oos["hard_pos"], oos["mask"])
        pos_df.to_parquet(art / "oos_positions.parquet", index=False)
    else:
        book = {"error": "no OOS predictions"}
        oos = None

    gates = run_leakage_gates(panel, packed)
    bias_folds = []
    if folds:
        for fold in (folds[0], folds[-1]):
            st = fold_states.get(int(fold.fold_id))
            if not st:
                bias_folds.append(
                    {
                        "name": "label_shuffle_bias",
                        "passed": False,
                        "fold_id": fold.fold_id,
                        "reason": "missing fold weights",
                    }
                )
                continue
            model = FuzzyXNet(seed=SEED)
            model.load_state_dict(st)
            b = gate_shuffle_bias(model, packed, fold.val_start, fold.val_end)
            b["fold_id"] = fold.fold_id
            bias_folds.append(b)
            print(f"[gates] shuffle fold {fold.fold_id}: {'PASS' if b['passed'] else 'FAIL'} {b}", flush=True)
    bias_ok = {"passed": bool(bias_folds) and all(b.get("passed") for b in bias_folds)}

    a0_path = _find_a0_pred()
    if a0_path is None:
        a0_delta = {"skipped": True, "reason": "A0 predictions not on disk"}
        notes.append("Clause (iv) SKIP: A0 h=7 preds missing.")
    else:
        a0_delta = {"skipped": True, "reason": "identical-days top-20 compare not wired this shot"}
        notes.append("Clause (iv) SKIP: A0 file present but identical-days compare not wired.")

    if mode == "LOCAL-RESTRICTED":
        notes.append("Official VIABLE is disabled for LOCAL-RESTRICTED even if Sharpe≥0.")

    v = verdict(gates, bias_ok, book if oos else {"net_sharpe_weekly": float("nan")}, a0_delta)
    if mode == "LOCAL-RESTRICTED" and v["verdict"] == "VIABLE candidate":
        v["verdict"] = "LOCAL-RESTRICTED (not official VIABLE)"

    rules = []
    if last_state:
        model = FuzzyXNet(seed=SEED)
        model.load_state_dict(last_state)
        rules = model.rule_sheet(FEATURE_COLS)

    write_report(
        Path("reports/fuzzyx_v1d_report.md"),
        mode=mode,
        gates=gates,
        bias_folds=bias_folds,
        book=book,
        a0_delta=a0_delta,
        verdict=v,
        folds=fold_rows,
        rules=rules,
        n_params=n_params or FuzzyXNet(seed=SEED).n_params(),
        notes=notes,
        title="FuzzyX-v1d report",
        addendum="reports/fuzzyx_addendum_v1d.md",
    )
    summary = {
        "mode": mode,
        "verdict": v,
        "book": book,
        "n_folds": len(folds),
        "n_params": n_params,
        "elapsed": time.time() - t0,
        "notes": notes,
    }
    (art / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[fuzzyx] DONE verdict={v.get('verdict')} elapsed={summary['elapsed']:.0f}s", flush=True)
    return summary


def main() -> None:
    run_fuzzyx()


if __name__ == "__main__":
    main()
