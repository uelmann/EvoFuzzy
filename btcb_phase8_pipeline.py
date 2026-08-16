"""
BTC-BEATER Phase 8 — MODEL-ZOO (CS-ATTN / TabPFN v2 / ridge-on-ranks).

BACKTEST / ANALYSIS ONLY. CPU for A+C. GPU allowed ONLY for Arm B (TabPFN), cap $20.
Frozen products untouched. Master only.
Usage:
  modal run btcb_phase8_pipeline.py
  modal run btcb_phase8_pipeline.py --mode judge   # resume: skip prepare/train
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p8-modelzoo"
VOL_Q = "quant-baseline"
quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

_cpu_pkgs = (
    "numpy",
    "pandas==2.2.2",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "httpx",
    "pyyaml",
    "lightgbm",
)

image_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_cpu_pkgs)
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/btcb_phase8_addendum.md", remote_path="/root/btcb_phase8_addendum.md")
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

image_gpu = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_cpu_pkgs)
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("tabpfn>=2.0.1,<3")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/btcb_phase8_addendum.md", remote_path="/root/btcb_phase8_addendum.md")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME)
CMC_PANEL = Path("/data/quant/btcb/full/panel.parquet")
WORK = Path("/data/quant/btcb/phase8")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {
        "daily_ret",
        "equity",
        "id_to_sym",
        "btc_ret",
        "equity_btc",
        "rel_equity",
        "w_btc",
        "n_names",
        "gate_on",
        "alt_gross",
        "aucs",
        "fold_states",
        "seed_preds",
        "state",
    }
    if isinstance(x, dict):
        return {str(k): _jsonable(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, drop) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


class KillHeartbeat:
    def __init__(self, stage: str, ckpt_fn=None, silence_sec: float = 20 * 60):
        self.stage = stage
        self.ckpt_fn = ckpt_fn
        self.silence_sec = float(silence_sec)
        self.t0 = time.time()
        self.last = time.time()
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        print(f"[HB] STAGE START {stage}", flush=True)

    def ping(self, msg: str = "") -> None:
        self.last = time.time()
        extra = f" {msg}" if msg else ""
        print(f"[HB] {self.stage}{extra} elapsed={time.time() - self.t0:.0f}s", flush=True)

    def _run(self) -> None:
        while not self.stop.wait(60.0):
            now = time.time()
            silent = now - self.last
            print(
                f"[HB] {self.stage} heartbeat elapsed={now - self.t0:.0f}s silent={silent:.0f}s",
                flush=True,
            )
            if silent >= self.silence_sec:
                print(f"[KILL] STAGE {self.stage} silent={silent:.0f}s", flush=True)
                if self.ckpt_fn is not None:
                    try:
                        self.ckpt_fn()
                    except Exception as e:
                        print(f"[KILL] checkpoint failed: {e}", flush=True)
                os._exit(2)

    def close(self) -> None:
        self.stop.set()
        print(f"[HB] STAGE END {self.stage} elapsed={time.time() - self.t0:.0f}s", flush=True)


def _load_close_long(raw_dir: Path, symbols: list[str]):
    import pandas as pd

    parts = []
    for sym in symbols:
        pq = raw_dir / f"{sym}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if df is None or df.empty or "close" not in df.columns:
            continue
        if "symbol" not in df.columns:
            df["symbol"] = sym
        parts.append(df[["date", "close", "symbol"]])
    if not parts:
        return pd.DataFrame(columns=["date", "close", "symbol"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "close"])


def _assert_addendum():
    from btcb.constants import (
        DEATH_CONVENTION,
        PHASE4V2_PI_SCOPE,
        PHASE8_CRITERION,
        PHASE8_DATE_SUBSAMPLE,
        PHASE8_FIREWALL,
        PHASE8_NULL_REGISTRATION,
        PHASE8_TABPFN_CAVEAT,
    )

    addendum = Path("/root/btcb_phase8_addendum.md").read_text()
    for txt in (
        PHASE8_CRITERION,
        PHASE8_FIREWALL,
        PHASE8_TABPFN_CAVEAT,
        PHASE8_DATE_SUBSAMPLE,
        PHASE8_NULL_REGISTRATION,
        DEATH_CONVENTION,
        PHASE4V2_PI_SCOPE,
    ):
        if txt not in addendum:
            raise RuntimeError(f"phase8 addendum missing freeze text: {txt[:80]}")
    return addendum


def _prepare_frames():
    import numpy as np
    import pandas as pd

    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        CMC_PANEL_SHA256,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE8_H,
        STAGE_S_COLS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.gates import assert_no_context
    from btcb.hygiene import clean_panel
    from btcb.labels import add_twin_quintile_labels
    from btcb.manuel2 import cmc_close_wide, hybrid_close_wide
    from btcb.model import make_expanding_folds
    from btcb.oracle_ladder import _as_utc, _utc_idx
    from btcb.phase4b import vol_col_name
    from btcb.phase8 import fold_to_dict, prepare_feature_frame
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    panel = pd.read_parquet(CMC_PANEL)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)

    pit = None
    for p in (
        Path("/data/quant/btcb/universe/btcb_top100_floor.parquet"),
        Path("/data/quant/universe/btcb_top100_floor.parquet"),
        Path("/root/btcb_top100_floor.parquet"),
    ):
        if p.exists():
            pit = pd.read_parquet(p)
            pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            pit["id"] = pit["id"].astype(int)
            pit = pit[pit["date"] <= end].copy()
            print(f"[HB] pit from {p} rows={len(pit)}", flush=True)
            break
    if pit is None:
        raise RuntimeError("missing floored PIT top-100")

    print("[HB] re-applying frozen 2.b cleaner (no CMC writes)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    twin = load_twin_from_cache(pred_dir_2c, int(PHASE8_H))
    twin = twin[twin["date"] <= end].copy()

    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if not feat_path.exists():
        raise RuntimeError("missing feat_s")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)
    from btcb.constants import CTX_COLS

    leaked = [c for c in CTX_COLS if c in feat.columns]
    if leaked:
        raise RuntimeError(f"context leaked into feat_s: {leaked}")

    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(PHASE8_H,))
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    labeled = prepare_feature_frame(labeled, list(STAGE_S_COLS))
    volc = vol_col_name(labeled)
    feats_s = list(STAGE_S_COLS)
    assert_no_context(feats_s)
    feat_rank = [f"rk_{c}" for c in feats_s]

    print("[HB] loading Binance-hybrid close...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet")) if spot_dir.exists() else []
    all_ids = sorted(set(int(i) for i in pit["id"].unique()) | {int(btc_id)})
    id_to_spot = build_id_symbol_map(all_ids, cleaned, set(spot_syms), set(spot_syms))
    id_to_spot[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in set(spot_syms) else id_to_spot.get(int(btc_id))
    spot_needed = sorted({s for s in id_to_spot.values() if s})
    spot_long = _load_close_long(spot_dir, spot_needed)
    spot_wide = close_wide_from_panel(spot_long.rename(columns={"close": "close"}), id_to_spot)
    cmc_px = cmc_close_wide(cleaned)
    close = hybrid_close_wide(cmc_px, spot_wide)
    if int(btc_id) not in close.columns:
        raise RuntimeError("BTC missing from hybrid close")
    close = close[close.index <= end].sort_index()
    close.index = _utc_idx(close.index)
    btc_ok = close[int(btc_id)].astype(float)
    close = close.loc[np.isfinite(btc_ok) & (btc_ok > 0)].copy()

    folds = make_expanding_folds(pd.DatetimeIndex(labeled["date"].unique()), horizon=PHASE8_H)
    print(
        f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} folds={len(folds)} "
        f"close={close.shape} vol={volc}",
        flush=True,
    )
    return {
        "labeled": labeled,
        "close": close,
        "pit": pit,
        "twin": twin,
        "folds": folds,
        "btc_id": int(btc_id),
        "volc": volc,
        "feats_s": feats_s,
        "feat_rank": feat_rank,
        "pred_hash": pred_hash,
        "cmc_sha": cmc_panel_sha0,
        "cleaned": cleaned,
    }


def _dump_prepared(blob: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    blob["labeled"].to_parquet(WORK / "labeled.parquet", index=False)
    blob["close"].to_parquet(WORK / "close.parquet")
    blob["pit"].to_parquet(WORK / "pit.parquet", index=False)
    twin = blob["twin"].copy()
    twin.to_parquet(WORK / "twin.parquet", index=False)
    folds = [fold_to_dict_safe(f) for f in blob["folds"]]
    (WORK / "folds.json").write_text(json.dumps(folds, indent=2, default=str))
    meta = {
        "btc_id": blob["btc_id"],
        "volc": blob["volc"],
        "feats_s": blob["feats_s"],
        "feat_rank": blob["feat_rank"],
        "pred_hash": blob["pred_hash"],
        "cmc_sha": blob["cmc_sha"],
    }
    (WORK / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def fold_to_dict_safe(f) -> dict:
    from btcb.phase8 import fold_to_dict

    return fold_to_dict(f)


def _load_prepared(need_close=True, slim=False):
    import pandas as pd

    from btcb.phase8 import fold_from_dict

    labeled = pd.read_parquet(WORK / "labeled.parquet")
    labeled["date"] = pd.to_datetime(labeled["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = pd.read_parquet(WORK / "close.parquet") if need_close else None
    if close is not None:
        close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True)).tz_convert("UTC").normalize()
        close.columns = [int(c) for c in close.columns]
    pit = twin = None
    if not slim:
        pit = pd.read_parquet(WORK / "pit.parquet")
        pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        twin = pd.read_parquet(WORK / "twin.parquet")
        twin["date"] = pd.to_datetime(twin["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    folds = [fold_from_dict(x) for x in json.loads((WORK / "folds.json").read_text())]
    meta = json.loads((WORK / "meta.json").read_text())
    return labeled, close, pit, twin, folds, meta


@app.function(
    image=image_cpu,
    timeout=60 * 60 * 3,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=65536,
)
def prepare_phase8() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    addendum = _assert_addendum()
    print("[HB] PHASE 8 MODEL-ZOO ANALYSIS ONLY; GPU only for TabPFN; nothing adopted", flush=True)
    hb = KillHeartbeat("prepare")
    try:
        blob = _prepare_frames()
        _dump_prepared(blob)
        quant_vol.commit()
        hb.ping("prepared")
        return {
            "n_labeled": int(len(blob["labeled"])),
            "n_folds": int(len(blob["folds"])),
            "btc_id": blob["btc_id"],
            "cmc_sha": blob["cmc_sha"],
            "pred_sha": blob["pred_hash"]["sha256"],
            "addendum_ok": True,
            "n_addendum_chars": len(addendum),
        }
    finally:
        hb.close()


@app.function(
    image=image_cpu,
    timeout=60 * 60 * 12,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_arms_ac() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    import torch

    torch.set_num_threads(8)
    from btcb.phase4v2 import collapse_fold_preds
    from btcb.phase8 import linspace_dates, train_cs_attn_all_folds, train_ridge_all_folds, _date_groups, attention_diagnostics

    hb = KillHeartbeat("arms-ac")
    t0 = time.time()
    try:
        labeled, close, pit, twin, folds, meta = _load_prepared()
        feats_s = list(meta["feats_s"])
        feat_rank = list(meta["feat_rank"])
        hb.ping(f"loaded labeled={len(labeled)} folds={len(folds)}")

        t_r = time.time()
        ridge_preds, ridge_extra = train_ridge_all_folds(labeled, folds, feat_rank)
        ridge_sec = time.time() - t_r
        ridge_preds.to_parquet(WORK / "preds_ridge.parquet", index=False)
        (WORK / "ridge_meta.json").write_text(json.dumps(_jsonable(ridge_extra), indent=2, default=str))
        quant_vol.commit()
        hb.ping(f"ridge done sec={ridge_sec:.0f} rows={len(ridge_preds)}")

        t_a = time.time()
        attn_preds, attn_extra = train_cs_attn_all_folds(labeled, folds, feats_s, ping=hb.ping)
        attn_sec = time.time() - t_a
        attn_preds.to_parquet(WORK / "preds_cs_attn.parquet", index=False)
        seed_preds = attn_extra.get("seed_preds")
        if seed_preds is not None and not seed_preds.empty:
            seed_preds.to_parquet(WORK / "preds_cs_attn_seeds.parquet", index=False)
        states = attn_extra.get("fold_states") or {}
        if states:
            import torch as _t

            _t.save(states, WORK / "csattn_states.pt")
        (WORK / "cs_attn_meta.json").write_text(
            json.dumps(_jsonable({k: v for k, v in attn_extra.items() if k != "fold_states"}), indent=2, default=str)
        )
        quant_vol.commit()
        hb.ping(f"cs-attn done sec={attn_sec:.0f} rows={len(attn_preds)}")

        attn_diag = {}
        try:
            collapsed = collapse_fold_preds(attn_preds, "signal") if not attn_preds.empty else attn_preds
            sample = linspace_dates(collapsed["date"] if not collapsed.empty else [], 10)
            groups = _date_groups(labeled)
            attn_diag = attention_diagnostics(labeled, groups, folds, feats_s, states, sample, collapsed)
        except Exception as e:
            attn_diag = {"error": str(e)}
            print(f"[HB] attn diag failed: {e}", flush=True)
        (WORK / "attn_diag.json").write_text(json.dumps(_jsonable(attn_diag), indent=2, default=str))
        quant_vol.commit()
        return {
            "ridge_sec": ridge_sec,
            "cs_attn_sec": attn_sec,
            "ridge_rows": int(len(ridge_preds)),
            "cs_attn_rows": int(len(attn_preds)),
            "n_params": attn_extra.get("n_params"),
            "elapsed": time.time() - t0,
            "attn_diag_n": (attn_diag or {}).get("n_dates"),
        }
    finally:
        hb.close()


@app.function(
    image=image_gpu,
    timeout=60 * 60 * 6,
    retries=0,
    volumes={"/data/quant": quant_vol},
    gpu="A10G",
    memory=32768,
)
def run_arm_b() -> dict:
    os.environ["PYTHONUNBUFFERED"] = "1"
    import torch

    from btcb.constants import PHASE8_GPU_USD_CAP, PHASE8_GPU_USD_PER_HOUR_A10G
    from btcb.phase8 import subsample_oos_dates, train_tabpfn_all_folds

    hb = KillHeartbeat("tabpfn")
    t0 = time.time()
    gpu_ok = bool(torch.cuda.is_available())
    device = "cuda" if gpu_ok else "cpu"
    budget_flag = None
    subsample = False
    try:
        labeled, close, pit, twin, folds, meta = _load_prepared(need_close=False)
        feats_s = list(meta["feats_s"])
        hb.ping(f"tabpfn device={device} cuda={gpu_ok} folds={len(folds)}")
        query_dates = None
        preds, extra = train_tabpfn_all_folds(
            labeled, folds, feats_s, device=device, query_dates=query_dates, ping=hb.ping
        )
        wall = extra.get("wall") or {}
        elapsed = time.time() - t0
        usd = elapsed / 3600.0 * float(PHASE8_GPU_USD_PER_HOUR_A10G)
        if usd > float(PHASE8_GPU_USD_CAP):
            budget_flag = "exceeded_cap_after_run"
        statuses = [m.get("status") for m in (extra.get("meta") or [])]
        if any(s not in ("ok",) for s in statuses) and (preds is None or preds.empty):
            budget_flag = budget_flag or "tabpfn_failed"
        if not preds.empty:
            preds.to_parquet(WORK / "preds_tabpfn.parquet", index=False)
        (WORK / "tabpfn_meta.json").write_text(json.dumps(_jsonable(extra), indent=2, default=str))
        rec = {
            "device": device,
            "gpu_ok": gpu_ok,
            "elapsed": elapsed,
            "gpu_usd_est": usd,
            "budget_flag": budget_flag,
            "subsample": subsample,
            "n_rows": int(len(preds)) if preds is not None else 0,
            "n_dates": int(preds["date"].nunique()) if preds is not None and not preds.empty else 0,
            "pred_sec_total": wall.get("total_pred_sec"),
            "pred_sec_per_date": wall.get("mean_pred_sec_per_date"),
            "status": "ok" if preds is not None and not preds.empty else "unavailable",
            "n_folds_ok": wall.get("n_folds_ok"),
        }
        (WORK / "tabpfn_wall.json").write_text(json.dumps(rec, indent=2, default=str))
        quant_vol.commit()
        hb.ping(f"tabpfn done status={rec['status']} usd={usd:.2f}")
        return rec
    except Exception as e:
        rec = {
            "device": device,
            "gpu_ok": gpu_ok,
            "elapsed": time.time() - t0,
            "gpu_usd_est": (time.time() - t0) / 3600.0 * float(PHASE8_GPU_USD_PER_HOUR_A10G),
            "budget_flag": "exception",
            "status": "unavailable",
            "error": str(e),
        }
        (WORK / "tabpfn_wall.json").write_text(json.dumps(rec, indent=2, default=str))
        quant_vol.commit()
        print(f"[HB] tabpfn EXCEPTION {e}", flush=True)
        return rec
    finally:
        hb.close()


def _null_payload_run(payload: dict, device: str) -> dict:
    from btcb.phase8 import fold_from_dict, run_one_null_cell

    labeled, close, _pit, _twin, _folds, meta = _load_prepared(slim=True)
    fold = fold_from_dict(payload["fold"])
    return run_one_null_cell(
        labeled,
        fold,
        payload["arm"],
        int(payload["shuffle_seed"]),
        list(meta["feats_s"]),
        list(meta["feat_rank"]),
        labeled,
        close,
        int(meta["btc_id"]),
        device=device,
    )


def _coerce_null_recs(payloads: list, recs: list) -> list:
    clean = []
    for p, r in zip(payloads, recs):
        if isinstance(r, BaseException):
            clean.append(
                {
                    "status": f"exc:{type(r).__name__}:{r}",
                    "tail_ic_top": None,
                    "overlap": None,
                    "monster": None,
                    "rankic": None,
                    "fold_id": int(p["fold"]["fold_id"]),
                    "shuffle_seed": int(p["shuffle_seed"]),
                    "arm": p["arm"],
                }
            )
        elif isinstance(r, dict):
            clean.append(r)
        else:
            clean.append(
                {
                    "status": f"bad:{type(r).__name__}",
                    "tail_ic_top": None,
                    "overlap": None,
                    "monster": None,
                    "rankic": None,
                    "fold_id": int(p["fold"]["fold_id"]),
                    "shuffle_seed": int(p["shuffle_seed"]),
                    "arm": p["arm"],
                }
            )
    return clean


@app.function(
    image=image_cpu,
    timeout=60 * 60 * 3,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def null_cell_cpu(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    import torch

    torch.set_num_threads(4)
    return _null_payload_run(payload, device="cpu")


@app.function(
    image=image_gpu,
    timeout=60 * 60 * 2,
    retries=0,
    volumes={"/data/quant": quant_vol},
    gpu="A10G",
    memory=32768,
)
def null_cell_gpu(payload: dict) -> dict:
    os.environ["PYTHONUNBUFFERED"] = "1"
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _null_payload_run(payload, device=device)


@app.function(
    image=image_cpu,
    timeout=60 * 60 * 4,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=65536,
)
def null_seq_cpu(payloads: list) -> list:
    """Sequential fallback if .map fan-out fails. Same frozen ridge/attn/tabpfn procedure."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    import torch

    torch.set_num_threads(8)
    hb = KillHeartbeat("null-seq")
    out = []
    try:
        for i, p in enumerate(payloads):
            rec = _null_payload_run(p, device="cpu")
            out.append(rec)
            hb.ping(f"cell {i+1}/{len(payloads)} fold={p.get('fold', {}).get('fold_id')} seed={p.get('shuffle_seed')}")
        return out
    finally:
        hb.close()


@app.function(
    image=image_cpu,
    timeout=60 * 60 * 4,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=65536,
)
def judge_metrics() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    import pandas as pd

    from btcb.constants import (
        ALT_BPS,
        PHASE3C_REF_START,
        PHASE8_CRITERION,
        PHASE8_H,
        PHASE8_NULL_FOLD_IDS,
    )
    from btcb.gates import pick_folds_by_id
    from btcb.oracle_ladder import ffill_members, formation_dates, run_periodic_long
    from btcb.phase4v2 import collapse_fold_preds, preds_to_score_at
    from btcb.phase8 import (
        fold_to_dict,
        null_shuffle_seeds,
        pick_best_arm,
        real_fold_metrics,
        score_signal,
        seed_dispersion,
        signal_corr_matrix,
        subsample_oos_dates,
    )

    hb = KillHeartbeat("judge-metrics")
    t0 = time.time()
    try:
        labeled, close, pit, twin, folds, meta = _load_prepared()
        hb.ping("loaded prepared")
        btc_id = int(meta["btc_id"])
        frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})

        def _load_pred(name):
            p = WORK / f"preds_{name}.parquet"
            if not p.exists():
                return pd.DataFrame()
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            return df

        attn = _load_pred("cs_attn")
        ridge = _load_pred("ridge")
        tab = _load_pred("tabpfn")
        if not attn.empty:
            attn = collapse_fold_preds(attn, "signal")
        if not ridge.empty:
            ridge = collapse_fold_preds(ridge, "signal")
        if not tab.empty:
            tab = collapse_fold_preds(tab, "signal")
        hb.ping(f"preds attn={len(attn)} ridge={len(ridge)} tab={len(tab)}")

        tab_wall = {}
        if (WORK / "tabpfn_wall.json").exists():
            tab_wall = json.loads((WORK / "tabpfn_wall.json").read_text())
        tab_ok = bool(not tab.empty and tab_wall.get("status") == "ok")
        judgment_set = "full_oos"
        judge_dates = None
        if not tab_ok:
            judgment_set = "full_oos"
        elif tab_wall.get("subsample"):
            judgment_set = "1-in-3"
            oos = sorted(frozen["date"].unique())
            judge_dates = subsample_oos_dates(oos)
        print(f"[HB] {PHASE8_CRITERION}", flush=True)
        print(f"[HB] judgment_set={judgment_set} tab_ok={tab_ok}", flush=True)

        frames = {
            "frozen_spread": (frozen, "spread"),
            "cs_attn": (attn, "signal"),
            "tabpfn": (tab, "signal"),
            "ridge": (ridge, "signal"),
        }
        grid, grid_full = {}, {}
        for name, (df, col) in frames.items():
            grid_full[name] = score_signal(df, col, labeled, close, btc_id, None)
            grid[name] = score_signal(df, col, labeled, close, btc_id, judge_dates)
            hb.ping(
                f"{name} rankic={grid[name].get('rankic')} tailIC={grid[name].get('tail_ic_top')} "
                f"n={grid[name].get('n_dates')}"
            )

        seed_path = WORK / "preds_cs_attn_seeds.parquet"
        disp = {}
        if seed_path.exists():
            sp = pd.read_parquet(seed_path)
            disp = seed_dispersion(sp, labeled, close, btc_id, judge_dates)
        hb.ping("seed dispersion")

        corr = signal_corr_matrix(frames, labeled, close, btc_id, judge_dates)
        best = pick_best_arm(grid)
        hb.ping(f"best arm by RankIC={best}")

        from btcb.academic_factor import pit_members as _pm

        dates = list(close.index)
        members = ffill_members(_pm(pit, btc_id), dates)
        start = pd.Timestamp(PHASE3C_REF_START, tz="UTC")
        oos = [d for d in dates if d >= start]
        pairs14 = formation_dates(oos, int(PHASE8_H))
        books = {}
        for name, (df, col) in frames.items():
            if df is None or df.empty:
                continue
            scores = preds_to_score_at(df, col, [t for t, _, _ in pairs14])
            books[name] = run_periodic_long(
                close, members, btc_id, scores, pairs14, cost_bps=float(ALT_BPS), label=name
            )
            hb.ping(f"book {name} CAGR={books[name].get('cagr')} RankIC={books[name].get('rankic')}")

        attn_diag = {}
        if (WORK / "attn_diag.json").exists():
            attn_diag = json.loads((WORK / "attn_diag.json").read_text())

        configs = {}
        if (WORK / "cs_attn_meta.json").exists():
            configs["cs_attn"] = (json.loads((WORK / "cs_attn_meta.json").read_text()) or {}).get("config")
        if (WORK / "ridge_meta.json").exists():
            configs["ridge"] = (json.loads((WORK / "ridge_meta.json").read_text()) or {}).get("config")
        if (WORK / "tabpfn_meta.json").exists():
            configs["tabpfn"] = (json.loads((WORK / "tabpfn_meta.json").read_text()) or {}).get("config")

        ridge_sec = None
        cs_sec = None
        if (WORK / "ridge_meta.json").exists():
            rm = json.loads((WORK / "ridge_meta.json").read_text())
            ridge_sec = sum((m.get("elapsed") or 0) for m in (rm.get("meta") or []))
        if (WORK / "cs_attn_meta.json").exists():
            am = json.loads((WORK / "cs_attn_meta.json").read_text())
            cs_sec = sum((m.get("elapsed") or 0) for m in (am.get("meta") or []))

        payloads = []
        real = {}
        if best:
            null_folds = pick_folds_by_id(folds, PHASE8_NULL_FOLD_IDS)
            raw_fold = _load_pred(best)
            if raw_fold.empty:
                raw_fold = {"cs_attn": attn, "tabpfn": tab, "ridge": ridge}.get(best)
            real = real_fold_metrics(raw_fold, null_folds, labeled, close, btc_id, "signal")
            for f in null_folds:
                for ss in null_shuffle_seeds():
                    payloads.append({"arm": best, "fold": fold_to_dict(f), "shuffle_seed": int(ss)})
        hb.ping(f"null payloads arm={best} cells={len(payloads)}")

        blob = {
            "best_arm": best,
            "grid": grid,
            "grid_full": grid_full,
            "corr": corr,
            "books": books,
            "disp": disp,
            "attn_diag": attn_diag,
            "configs": configs,
            "tab_wall": tab_wall,
            "judgment_set": judgment_set,
            "tab_ok": tab_ok,
            "real": real,
            "meta": meta,
            "ridge_sec": ridge_sec,
            "cs_sec": cs_sec,
            "metrics_sec": time.time() - t0,
        }
        with open(WORK / "judge_partial.pkl", "wb") as f:
            pickle.dump(blob, f, protocol=4)
        (WORK / "null_payloads.json").write_text(json.dumps(payloads, indent=2, default=str))
        (WORK / "judge_metrics.json").write_text(
            json.dumps(
                {
                    "best_arm": best,
                    "judgment_set": judgment_set,
                    "tab_ok": tab_ok,
                    "grid": {k: _jsonable(v) for k, v in grid.items()},
                    "corr": _jsonable(corr),
                    "n_payloads": len(payloads),
                },
                indent=2,
                default=str,
            )
        )
        quant_vol.commit()
        hb.ping("pickled judge_partial")
        return {
            "best_arm": best,
            "n_dates": int((grid.get("frozen_spread") or {}).get("n_dates") or 0),
            "judgment_set": str(judgment_set),
            "tab_ok": bool(tab_ok),
            "n_payloads": int(len(payloads)),
            "payloads": payloads,
        }
    finally:
        hb.close()


@app.function(
    image=image_cpu,
    timeout=60 * 60 * 2,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=4,
    memory=32768,
)
def write_reports(null_recs: list | None = None, null_gpu_sec: float = 0.0, null_map_sec: float = 0.0) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUNBUFFERED"] = "1"
    import pandas as pd

    from btcb.constants import PHASE8_CRITERION, PHASE8_GPU_USD_PER_HOUR_A10G, PHASE8_NULL_FOLD_IDS
    from btcb.gates import pick_folds_by_id
    from btcb.phase8 import finish_phase8_null, fold_cell, mechanical_verdicts
    from btcb.phase8_report import plot_corr, plot_equity, plot_rankic, update_ledger_phase8, write_phase8

    hb = KillHeartbeat("write-reports")
    t0 = time.time()
    addendum = Path("/root/btcb_phase8_addendum.md").read_text()
    try:
        quant_vol.reload()
        hb.ping("volume reloaded")
        with open(WORK / "judge_partial.pkl", "rb") as f:
            blob = pickle.load(f)
        labeled, close, _pit, _twin, folds, meta = _load_prepared(slim=True)
        best = blob["best_arm"]
        grid = blob["grid"]
        grid_full = blob["grid_full"]
        corr = blob["corr"]
        books = blob["books"]
        disp = blob.get("disp") or {}
        attn_diag = blob.get("attn_diag") or {}
        configs = blob.get("configs") or {}
        tab_wall = blob.get("tab_wall") or {}
        judgment_set = blob.get("judgment_set") or "full_oos"
        real = blob.get("real") or {}
        hb.ping(f"loaded pickle best={best}")

        clean = null_recs
        if clean is None:
            if (WORK / "null_cells.json").exists():
                clean = json.loads((WORK / "null_cells.json").read_text())
            else:
                clean = []
        (WORK / "null_cells.json").write_text(json.dumps(_jsonable(clean), indent=2, default=str))

        null = None
        if best and clean:
            null_folds = pick_folds_by_id(folds, PHASE8_NULL_FOLD_IDS)
            cells = {"tail_ic_top": [], "overlap": [], "monster": [], "rankic": []}
            for f in null_folds:
                sl = [c for c in clean if c.get("fold_id") == int(f.fold_id)]
                cells["tail_ic_top"].append(fold_cell(f, [c.get("tail_ic_top") for c in sl], real, "tail_ic_top"))
                cells["overlap"].append(fold_cell(f, [c.get("overlap") for c in sl], real, "overlap"))
                cells["monster"].append(fold_cell(f, [c.get("monster") for c in sl], real, "monster"))
                cells["rankic"].append(fold_cell(f, [c.get("rankic") for c in sl], real, "rankic"))
            null = finish_phase8_null(f"{best}_vol_matched_null", cells)
            (WORK / "null.json").write_text(json.dumps(_jsonable(null), indent=2, default=str))
            hb.ping(f"null passed={null.get('passed')} verdict={(null.get('tail_ic_top') or {}).get('verdict')}")

        verdict = mechanical_verdicts(grid, null, best, corr)
        gpu_usd = float(tab_wall.get("gpu_usd_est") or 0) + (float(null_gpu_sec) / 3600.0) * float(
            PHASE8_GPU_USD_PER_HOUR_A10G
        )
        extra = {
            "pred_sha256": (meta.get("pred_hash") or {}).get("sha256"),
            "cmc_panel_sha256": meta.get("cmc_sha"),
            "cmc_readonly_ok": True,
            "start": str(close.index.min().date()) if len(close) else None,
            "end": str(close.index.max().date()) if len(close) else None,
            "n_eval_dates": (grid.get("frozen_spread") or {}).get("n_dates"),
            "n_judgment_dates": (grid.get("frozen_spread") or {}).get("n_dates"),
            "judgment_set": judgment_set,
            "gpu_used": bool(tab_wall.get("gpu_ok")),
            "gpu_type": "A10G" if tab_wall.get("gpu_ok") else None,
            "gpu_usd_est": gpu_usd,
            "budget_flag": tab_wall.get("budget_flag"),
            "tabpfn_subsample_flag": bool(tab_wall.get("subsample")),
            "tabpfn_status": tab_wall.get("status"),
            "tabpfn_sec": tab_wall.get("elapsed"),
            "tabpfn_pred_sec": tab_wall.get("pred_sec_total"),
            "tabpfn_pred_per_date": tab_wall.get("pred_sec_per_date"),
            "cs_attn_sec": blob.get("cs_sec"),
            "ridge_sec": blob.get("ridge_sec"),
            "null_map_sec": float(null_map_sec),
            "elapsed_sec": float(blob.get("metrics_sec") or 0) + float(null_map_sec) + (time.time() - t0),
            "seed_dispersion": disp,
            "plain": None,
        }
        va = verdict.get("arms") or {}
        lc = (verdict.get("linear_ceiling_text") or "LINEAR-CEILING n/a").rstrip(".")
        extra["plain"] = (
            f"Phase 8 MODEL-ZOO on {judgment_set}: "
            f"A CS-ATTN {(va.get('cs_attn') or {}).get('verdict')}; "
            f"B TabPFN {(va.get('tabpfn') or {}).get('verdict')}; "
            f"C RIDGE {(va.get('ridge') or {}).get('verdict')}. "
            f"{lc}. "
            f"ORTHOGONAL={verdict.get('orthogonal') or 'none'}. Nothing adopted."
        )

        ledger_path = Path("/root/numbers_ledger.md")
        update_ledger_phase8(ledger_path, verdict=verdict, extra=extra)

        rep_dir = Path("/data/quant/reports")
        chart_dir = Path("/data/quant/charts")
        for d in (rep_dir, chart_dir):
            d.mkdir(parents=True, exist_ok=True)
        write_phase8(
            rep_dir / "btcb_phase8_modelzoo.md",
            grid=grid,
            grid_full=grid_full if judgment_set == "1-in-3" else None,
            books={k: _jsonable(v) for k, v in books.items()},
            null=null,
            corr=corr,
            verdict=verdict,
            attn=attn_diag,
            configs=configs,
            extra=extra,
        )
        plot_rankic(grid, null, best, chart_dir / "btcb_phase8_rankic.png")
        plot_corr(corr, chart_dir / "btcb_phase8_corr.png")
        plot_equity(books, chart_dir / "btcb_phase8_equity.png")
        (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
        payload = {
            "criterion": PHASE8_CRITERION,
            "verdict": _jsonable(verdict),
            "grid": {k: _jsonable(v) for k, v in grid.items()},
            "grid_full": {k: _jsonable(v) for k, v in grid_full.items()},
            "books": {k: _jsonable(v) for k, v in books.items()},
            "null": _jsonable(null),
            "corr": _jsonable(corr),
            "attn": _jsonable(attn_diag),
            "configs": _jsonable(configs),
            "extra": extra,
            "tab_wall": tab_wall,
        }
        (rep_dir / "btcb_phase8_modelzoo.json").write_text(json.dumps(payload, indent=2, default=str))
        (rep_dir / "btcb_phase8_addendum.md").write_text(addendum)
        quant_vol.commit()
        hb.ping("wrote reports")

        print(f"ARM A CS-ATTN-DAILY: {(va.get('cs_attn') or {}).get('verdict')}", flush=True)
        print(f"ARM B TabPFN: {(va.get('tabpfn') or {}).get('verdict')}", flush=True)
        print(f"ARM C RIDGE: {(va.get('ridge') or {}).get('verdict')}", flush=True)
        print(verdict.get("linear_ceiling_text") or "LINEAR-CEILING n/a", flush=True)
        orth = verdict.get("orthogonal") or []
        if orth:
            for o in orth:
                print(f"ORTHOGONAL SIGNAL: {o.get('arm')} corr={o.get('corr')} RankIC={o.get('rankic')}", flush=True)
        else:
            print("ORTHOGONAL SIGNAL: none", flush=True)
        print(
            f"TabPFN wall-time total={tab_wall.get('elapsed')}s per_date={tab_wall.get('pred_sec_per_date')} "
            f"status={tab_wall.get('status')} gpu_usd_est={gpu_usd:.2f}",
            flush=True,
        )
        print(f"[HB] DONE elapsed={time.time()-t0:.1f}s nothing_adopted=true", flush=True)
        return {
            "verdict": _jsonable(verdict),
            "extra": extra,
            "best_arm": best,
            "elapsed_sec": time.time() - t0,
        }
    finally:
        hb.close()


def _pull_artifacts():
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase8_modelzoo.md", "reports"),
        ("reports/btcb_phase8_modelzoo.json", "reports"),
        ("reports/btcb_phase8_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase8_rankic.png", "charts"),
        ("charts/btcb_phase8_corr.png", "charts"),
        ("charts/btcb_phase8_equity.png", "charts"),
    ]
    for remote, kind in pulls:
        name = Path(remote).name
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOL_Q, remote, str(dest), "--force"], check=False)
        candidate = dest if dest.is_file() else dest / name
        if candidate.exists() and candidate.is_file():
            Path(kind).mkdir(exist_ok=True)
            shutil.copy2(candidate, Path(kind) / name)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "charts").glob("btcb_phase8*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase8*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())


def _run_null_map(best: str | None, payloads: list) -> tuple[list, float]:
    if not payloads:
        return [], 0.0
    fn = null_cell_gpu if best == "tabpfn" else null_cell_cpu
    print(f"[local] null .map arm={best} cells={len(payloads)} (local entrypoint, not nested)", flush=True)
    t_n = time.time()
    try:
        recs = list(fn.map(payloads, return_exceptions=True, order_outputs=True))
    except Exception as e:
        print(f"[local] null map failed ({e}); sequential fallback", flush=True)
        recs = null_seq_cpu.remote(payloads)
        return _coerce_null_recs(payloads, recs), time.time() - t_n
    clean = _coerce_null_recs(payloads, recs)
    failed_p, failed_i = [], []
    for i, (p, c) in enumerate(zip(payloads, clean)):
        st = str(c.get("status") or "")
        if st.startswith("exc") or st.startswith("error") or st.startswith("bad"):
            failed_p.append(p)
            failed_i.append(i)
    if failed_p:
        print(f"[local] retrying {len(failed_p)} failed null cells sequentially", flush=True)
        retry = null_seq_cpu.remote(failed_p)
        for i, rec in zip(failed_i, retry):
            clean[i] = rec
    return clean, time.time() - t_n


@app.local_entrypoint()
def main(mode: str = "full"):
    print(f"[local] Phase 8 MODEL-ZOO mode={mode}", flush=True)
    if mode not in ("full", "judge"):
        raise RuntimeError(f"unknown mode {mode}; use full|judge")
    if mode == "full":
        prep = prepare_phase8.remote()
        print(json.dumps(_jsonable(prep), indent=2, default=str), flush=True)
        ac = run_arms_ac.spawn()
        b = run_arm_b.spawn()
        try:
            ac_s = ac.get()
            print("AC", json.dumps(_jsonable(ac_s), indent=2, default=str), flush=True)
        except Exception as e:
            print(f"[local] AC FAILED: {e}", flush=True)
        try:
            b_s = b.get()
            print("B", json.dumps(_jsonable(b_s), indent=2, default=str), flush=True)
        except Exception as e:
            print(f"[local] B FAILED: {e}", flush=True)
    jm = judge_metrics.remote()
    print("METRICS", json.dumps(_jsonable({k: v for k, v in jm.items() if k != "payloads"}), indent=2, default=str), flush=True)
    payloads = jm.get("payloads") or []
    best = jm.get("best_arm")
    clean, null_map_sec = _run_null_map(best, payloads)
    null_gpu_sec = null_map_sec if best == "tabpfn" else 0.0
    print(f"[local] null done cells={len(clean)} sec={null_map_sec:.0f}", flush=True)
    summary = write_reports.remote(clean, null_gpu_sec, null_map_sec)
    _pull_artifacts()
    print(json.dumps(_jsonable(summary), indent=2, default=str))
    print("[local] Phase 8 complete.", flush=True)
