"""
BTC-BEATER Phase 7.c — NFN v1 (same architecture, corrected training craft).

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Frozen products untouched. Master only.
Usage: modal run btcb_phase7c_nfn_v1_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p7c-nfn-v1"
VOL_Q = "quant-baseline"
quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "lightgbm",
        "matplotlib",
        "httpx",
        "pyyaml",
        "scikit-learn",
    )
    .pip_install("torch==2.4.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"})
    .add_local_python_source("baseline", "btcb", "nfn")
    .add_local_file(
        "reports/btcb_phase7c_nfn_v1_addendum.md",
        remote_path="/root/btcb_phase7c_nfn_v1_addendum.md",
    )
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("reports/btcb_phase7_nfn.json", remote_path="/root/btcb_phase7_nfn.json")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME, image=image)
CMC_PANEL = Path("/data/quant/btcb/full/panel.parquet")


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
        "exponents",
        "membership",
        "train_curve",
        "holdout_curve",
        "film",
        "aucs",
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


def _as_utc(ts):
    import pandas as pd

    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC").normalize()
    return t.tz_convert("UTC").normalize()


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
    return pd.concat(parts, ignore_index=True)


def _cmc_close_wide(panel):
    import pandas as pd

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    wide = df.pivot_table(index="date", columns="id", values="close", aggfunc="last").sort_index()
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index, utc=True)).tz_convert("UTC").normalize()
    return wide


def _hybrid_close(cmc_wide, spot_wide):
    import pandas as pd

    idx = cmc_wide.index.union(spot_wide.index).sort_values()
    cmc = cmc_wide.reindex(idx)
    spot = spot_wide.reindex(idx)
    cols = sorted(set(int(c) for c in cmc.columns) | set(int(c) for c in spot.columns))
    out = {}
    for c in cols:
        s = spot[c] if c in spot.columns else None
        m = cmc[c] if c in cmc.columns else None
        if s is not None and m is not None:
            out[c] = s.astype(float).combine_first(m.astype(float))
        elif s is not None:
            out[c] = s.astype(float)
        else:
            out[c] = m.astype(float)
    wide = pd.DataFrame(out).sort_index()
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index, utc=True)).tz_convert("UTC").normalize()
    return wide


def _fold_to_dict(f) -> dict:
    return {
        "fold_id": int(f.fold_id),
        "train_start": str(f.train_start),
        "train_end": str(f.train_end),
        "purge_end": str(f.purge_end),
        "embargo_end": str(f.embargo_end),
        "val_start": str(f.val_start),
        "val_end": str(f.val_end),
        "horizon": int(f.horizon),
    }


def _fold_from_dict(d):
    from btcb.model import FoldSpec

    return FoldSpec(
        fold_id=int(d["fold_id"]),
        train_start=_as_utc(d["train_start"]),
        train_end=_as_utc(d["train_end"]),
        purge_end=_as_utc(d["purge_end"]),
        embargo_end=_as_utc(d["embargo_end"]),
        val_start=_as_utc(d["val_start"]),
        val_end=_as_utc(d["val_end"]),
        horizon=int(d["horizon"]),
    )


def _mean_spread(frames: list, col: str = "spread"):
    import pandas as pd

    acc = None
    for i, df in enumerate(frames):
        sl = df[["date", "id", col]].copy()
        sl = sl.rename(columns={col: f"s{i}"})
        acc = sl if acc is None else acc.merge(sl, on=["date", "id"], how="outer")
    if acc is None:
        return pd.DataFrame(columns=["date", "id", "spread"])
    cols = [c for c in acc.columns if c.startswith("s")]
    acc["spread"] = acc[cols].mean(axis=1)
    return acc[["date", "id", "spread"]]


def _v0_from_json(path: Path) -> dict:
    if not path.exists():
        return {"epochs": [], "seed_ics": [], "dispersion": None, "mean_best_epoch": None}
    blob = json.loads(path.read_text())
    hyg = blob.get("hygiene") or []
    epochs = [h.get("best_epoch") for h in hyg if h.get("best_epoch") is not None]
    seed_ics = []
    for _k, v in (blob.get("seed_metrics") or {}).items():
        if isinstance(v, dict) and v.get("tail_ic_top") is not None:
            seed_ics.append(float(v["tail_ic_top"]))
    disp = (blob.get("verdict") or {}).get("seed_dispersion_tail_ic")
    mean_ep = float(sum(epochs) / len(epochs)) if epochs else None
    return {"epochs": epochs, "seed_ics": seed_ics, "dispersion": disp, "mean_best_epoch": mean_ep}


@app.function(
    timeout=60 * 60 * 3,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=2,
    memory=8192,
    max_containers=40,
)
def nfn_v1_null_job(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import json
    import pandas as pd

    from nfn.data import load_pack
    from nfn.nulls_v1 import run_null_cell_v1

    cache = Path(payload.get("null_cache_dir") or "/data/quant/btcb/phase7c/null")
    cache.mkdir(parents=True, exist_ok=True)
    outp = cache / f"fold{payload['fold']['fold_id']}_seed{payload['shuffle_seed']}.json"
    if outp.exists():
        try:
            rec = json.loads(outp.read_text())
            if rec.get("status") == "ok":
                print(
                    f"[HB] null CACHE HIT fold={payload['fold']['fold_id']} shuffle={payload['shuffle_seed']}",
                    flush=True,
                )
                return rec
        except Exception:
            pass
    pack = load_pack(payload["pack_path"])
    fold = _fold_from_dict(payload["fold"])
    labeled = pd.read_parquet(payload["labeled_path"])
    close = pd.read_parquet(payload["close_path"])
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True)).tz_convert("UTC").normalize()
    close.columns = [int(c) for c in close.columns]
    print(f"[HB] null-v1 fold={fold.fold_id} shuffle={payload['shuffle_seed']}", flush=True)
    rec = run_null_cell_v1(
        pack,
        fold,
        int(payload["shuffle_seed"]),
        labeled,
        close,
        int(payload["btc_id"]),
    )
    outp.write_text(json.dumps(rec, default=str))
    try:
        quant_vol.commit()
    except Exception:
        pass
    return rec


@app.function(
    timeout=60 * 60 * 10,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=4,
    memory=16384,
    max_containers=16,
)
def nfn_v1_chain_job(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(4)
    from nfn.data import load_pack
    from nfn.train_v1 import train_chain_v1

    pack = load_pack(payload["pack_path"])
    folds = [_fold_from_dict(d) for d in payload["folds"]]
    seed = int(payload["seed"])
    bag = int(payload["bag"])
    cache_dir = Path(payload["cache_dir"])
    print(f"[HB] chain seed={seed} bag={bag} n_folds={len(folds)}", flush=True)
    pred, metas = train_chain_v1(
        pack,
        folds,
        seed=seed,
        bag=bag,
        device="cpu",
        cache_dir=cache_dir,
        cache_ver=payload.get("cache_ver") or "p7cv1",
        commit_fn=quant_vol.commit,
    )
    n_ok = sum(1 for m in metas if m.get("status") == "ok")
    print(f"[HB] chain seed={seed} bag={bag} folds_ok={n_ok}/{len(metas)} rows={len(pred)}", flush=True)
    return {
        "seed": seed,
        "bag": bag,
        "n_ok": n_ok,
        "n_rows": int(len(pred)),
        "metas": [_jsonable(m) for m in metas],
    }


@app.function(
    timeout=60 * 60 * 2,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=4,
    memory=16384,
    max_containers=16,
)
def nfn_v1_cold_job(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import json

    import torch

    torch.set_num_threads(4)
    from nfn.data import load_pack
    from nfn.train_v1 import train_one_fold_v1

    cache = Path(payload["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    seed = int(payload["seed"])
    bag = int(payload["bag"])
    fold = _fold_from_dict(payload["fold"])
    js = cache / f"cold_meta_seed{seed}_bag{bag}_fold{fold.fold_id}.json"
    if js.exists():
        try:
            meta = json.loads(js.read_text())
            if meta.get("cache_ver") == payload.get("cache_ver") and meta.get("status") == "ok":
                meta["cache_hit"] = True
                print(f"[HB] cold CACHE HIT fold={fold.fold_id} bag={bag}", flush=True)
                return meta
        except Exception:
            pass
    pack = load_pack(payload["pack_path"])
    print(f"[HB] cold-start fold={fold.fold_id} seed={seed} bag={bag}", flush=True)
    _pred, meta, _swa = train_one_fold_v1(
        pack, fold, seed=seed, bag=bag, prev_state=None, device="cpu"
    )
    meta["cache_ver"] = payload.get("cache_ver") or "p7cv1"
    meta["cache_hit"] = False
    meta["cold"] = True
    js.write_text(json.dumps(_jsonable(meta), default=str))
    try:
        quant_vol.commit()
    except Exception:
        pass
    return _jsonable(meta)


@app.function(
    timeout=60 * 60 * 16,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def run_btcb_p7c_nfn_v1() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(8)

    from baseline.seedutil import seed_everything
    from btcb.academic_factor import pit_members
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        ALT_BPS,
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        NULL_SHUFFLE_SEEDS,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
        STAGE_S_COLS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.gates import assert_no_context, pick_folds_by_id
    from btcb.hygiene import clean_panel
    from btcb.labels import add_twin_quintile_labels
    from btcb.model import make_expanding_folds
    from btcb.oracle_ladder import ffill_members, formation_dates, run_periodic_long
    from btcb.phase4b import real_fold_metrics, vol_col_name
    from btcb.phase4v2 import collapse_fold_preds, per_date_tail_metrics, preds_to_score_at, restrict_eval_frame
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache
    from nfn.constants import FIREWALL, FIREWALL_PHASE7, HORIZON, NULL_FOLD_IDS, NULL_MAP_CONCURRENCY, NULL_REPLICATES, SEEDS
    from nfn.constants_v1 import (
        CACHE_VER,
        COLD_CONTROL_FOLDS,
        N_BAG,
        N_PARAMS_EXPECTED,
        PHASE7C_CRITERION,
        PHASE7C_NULL_REGISTRATION,
        V0_SEED_DISPERSION,
        assert_architecture_unchanged,
    )
    from nfn.data import pack_labeled, save_pack
    from nfn.firewall import assert_firewall
    from nfn.interpret import membership_movement, top_rules
    from nfn.model import build_nfn
    from nfn.nulls_v1 import cells_from_replicates
    from nfn.regime import regime_frame
    from nfn.report import plot_film_ribbon, plot_membership
    from nfn.report_v1 import (
        aggregate_seed_fold,
        plot_dispersion,
        plot_epoch_hist,
        plot_tail_ic_null,
        update_ledger_v1,
        write_phase7c,
    )
    from nfn.verdict_v1 import mechanical_verdict_v1

    t0 = time.time()
    seed_everything(int(SEEDS[0]))
    fw = assert_firewall()
    arch = assert_architecture_unchanged()
    addendum = Path("/root/btcb_phase7c_nfn_v1_addendum.md").read_text()
    for txt in (FIREWALL, FIREWALL_PHASE7, PHASE7C_CRITERION, PHASE7C_NULL_REGISTRATION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"phase7c addendum missing freeze text: {txt[:80]}")
    print("[HB] PHASE 7.c NFN v1 ANALYSIS ONLY; zero GPU; architecture frozen; nothing adopted", flush=True)
    print(f"[HB] {PHASE7C_CRITERION}", flush=True)
    print(f"[HB] architecture assert {arch}", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    start = pd.Timestamp(PHASE3C_REF_START, tz="UTC")
    panel = pd.read_parquet(CMC_PANEL)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id} rows={len(panel)}", flush=True)

    def _load_pit(name: str):
        for p in (
            Path(f"/data/quant/btcb/universe/{name}"),
            Path(f"/data/quant/universe/{name}"),
        ):
            if p.exists():
                pit = pd.read_parquet(p)
                pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                pit["id"] = pit["id"].astype(int)
                pit = pit[pit["date"] <= end].copy()
                print(f"[HB] pit from {p} rows={len(pit)}", flush=True)
                return pit
        return None

    pit100 = _load_pit("btcb_top100_floor.parquet")
    pit50 = _load_pit("btcb_top50_floor.parquet")
    if pit100 is None:
        raise RuntimeError("missing floored PIT top-100")
    if pit50 is None:
        raise RuntimeError("missing floored PIT top-50")

    print("[HB] re-applying frozen 2.b cleaner (no CMC writes)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash['sha256']}")
    twin = load_twin_from_cache(pred_dir_2c, int(PHASE3C_REF_H))
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
    assert_no_context(list(STAGE_S_COLS))

    print("[HB] labels h=14 twin quintiles...", flush=True)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(HORIZON,))
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    labeled["date"] = pd.to_datetime(labeled["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    labeled["id"] = labeled["id"].astype(int)
    volc = vol_col_name(labeled)
    print(f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} vol_col={volc}", flush=True)

    print("[HB] regime vector m_t...", flush=True)
    regime = regime_frame(cleaned, pit50, pit100, btc_id)

    print("[HB] canonical hybrid close...", flush=True)
    spot_dir = Path("/data/quant/raw/spot_klines")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet")) if spot_dir.exists() else []
    all_ids = sorted(set(int(i) for i in pit100["id"].unique()) | {int(btc_id)})
    id_to_spot = build_id_symbol_map(all_ids, cleaned, set(spot_syms), set(spot_syms))
    id_to_spot[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in set(spot_syms) else id_to_spot.get(int(btc_id))
    spot_needed = sorted({s for s in id_to_spot.values() if s})
    spot_long = _load_close_long(spot_dir, spot_needed)
    spot_wide = close_wide_from_panel(spot_long.rename(columns={"close": "close"}), id_to_spot)
    cmc_wide = _cmc_close_wide(cleaned)
    close = _hybrid_close(cmc_wide, spot_wide)
    close = close[close.index <= end].sort_index()
    if int(btc_id) not in close.columns:
        raise RuntimeError("BTC missing from hybrid close")
    print(f"[HB] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    pack = pack_labeled(labeled, regime)
    work = Path("/data/quant/btcb/phase7c")
    work.mkdir(parents=True, exist_ok=True)
    pack_path = work / "pack.npz"
    labeled_path = work / "labeled.parquet"
    close_path = work / "close.parquet"
    save_pack(pack, pack_path)
    labeled.to_parquet(labeled_path, index=False)
    close.to_parquet(close_path)
    quant_vol.commit()

    folds_all = make_expanding_folds(pd.DatetimeIndex(labeled["date"].unique()), horizon=HORIZON)
    print(f"[HB] n_folds={len(folds_all)} bags={N_BAG} seeds={list(SEEDS)}", flush=True)
    n_params = int(build_nfn(42).n_params())
    if int(n_params) != int(N_PARAMS_EXPECTED):
        raise RuntimeError(f"n_params {n_params} != {N_PARAMS_EXPECTED}")
    print(f"[HB] n_params={n_params} gpu=false", flush=True)

    cache_dir = work / "preds"
    fold_dicts = [_fold_to_dict(f) for f in folds_all]
    chain_payloads = []
    for seed in SEEDS:
        for bag in range(int(N_BAG)):
            chain_payloads.append(
                {
                    "pack_path": str(pack_path),
                    "folds": fold_dicts,
                    "seed": int(seed),
                    "bag": int(bag),
                    "cache_dir": str(cache_dir),
                    "cache_ver": CACHE_VER,
                }
            )
    print(f"[HB] map {len(chain_payloads)} walk-forward chains", flush=True)
    chain_recs = list(nfn_v1_chain_job.map(chain_payloads, order_outputs=True))
    try:
        quant_vol.reload()
    except Exception:
        pass

    hygiene = []
    bag_frames: dict[tuple[int, int], pd.DataFrame] = {}
    last_meta_42_0 = None
    for rec in chain_recs:
        seed, bag = int(rec["seed"]), int(rec["bag"])
        for m in rec.get("metas") or []:
            hygiene.append(m)
        parts = []
        for fold in folds_all:
            pq = cache_dir / f"preds_seed{seed}_bag{bag}_fold{fold.fold_id}.parquet"
            if pq.exists():
                parts.append(pd.read_parquet(pq))
        if parts:
            bag_frames[(seed, bag)] = pd.concat(parts, ignore_index=True)

    last_js = cache_dir / f"meta_seed42_bag0_fold{folds_all[-1].fold_id}.json"
    if last_js.exists():
        last_meta_42_0 = json.loads(last_js.read_text())

    seed_raw = {}
    seed_collapsed = {}
    for seed in SEEDS:
        frames = []
        for b in range(int(N_BAG)):
            df = bag_frames.get((int(seed), b))
            if df is not None and not df.empty:
                frames.append(df)
        raw = _mean_spread(frames) if frames else pd.DataFrame(columns=["date", "id", "spread"])
        if not raw.empty:
            b0 = bag_frames.get((int(seed), 0))
            if b0 is not None and not b0.empty and "fold_id" in b0.columns:
                raw = raw.merge(b0[["date", "id", "fold_id"]].drop_duplicates(["date", "id"]), on=["date", "id"], how="left")
        seed_raw[int(seed)] = raw
        seed_collapsed[int(seed)] = collapse_fold_preds(raw, "spread") if raw is not None and not raw.empty else pd.DataFrame()
        print(f"[HB] seed={seed} bagged rows={len(raw)}", flush=True)

    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
    ensemble = _mean_spread([seed_collapsed[s] for s in SEEDS if s in seed_collapsed and not seed_collapsed[s].empty])

    def _eval(df, col, name):
        if df is None or df.empty:
            return {"label": name, "tail_ic_top": float("nan"), "overlap": float("nan"), "rankic": float("nan")}
        ev = restrict_eval_frame(df, labeled, close, btc_id, col)
        met = per_date_tail_metrics(ev, col)
        met["label"] = name
        print(
            f"[HB] {name} tailIC_top={met.get('tail_ic_top')} overlap={met.get('overlap')} "
            f"rankic={met.get('rankic')} n={met.get('n_dates')}",
            flush=True,
        )
        return met

    grid = {
        "frozen_spread": _eval(frozen, "spread", "frozen_spread"),
        "nfn_ensemble": _eval(ensemble, "spread", "nfn_ensemble"),
    }
    seed_metrics = {}
    for s, df in seed_collapsed.items():
        seed_metrics[int(s)] = _eval(df, "spread", f"nfn_v1_seed_{s}")

    # cold-start control
    null_folds = pick_folds_by_id(folds_all, NULL_FOLD_IDS)
    cold_payloads = []
    id_to_fold = {int(f.fold_id): f for f in folds_all}
    for fid in COLD_CONTROL_FOLDS:
        if int(fid) not in id_to_fold:
            continue
        for bag in range(int(N_BAG)):
            cold_payloads.append(
                {
                    "pack_path": str(pack_path),
                    "fold": _fold_to_dict(id_to_fold[int(fid)]),
                    "seed": 42,
                    "bag": int(bag),
                    "cache_dir": str(work / "cold"),
                    "cache_ver": CACHE_VER,
                }
            )
    print(f"[HB] cold-start control n={len(cold_payloads)} folds={COLD_CONTROL_FOLDS}", flush=True)
    cold_metas = list(nfn_v1_cold_job.map(cold_payloads, order_outputs=True)) if cold_payloads else []
    warm_lookup = {}
    for h in hygiene:
        if int(h.get("seed", -1)) == 42:
            warm_lookup[(int(h.get("fold_id", -1)), int(h.get("bag", -1)))] = h
    cold_control = []
    deltas = []
    for cm in cold_metas:
        key = (int(cm.get("fold_id", -1)), int(cm.get("bag", -1)))
        wm = warm_lookup.get(key) or {}
        d_tr = None
        d_ic = None
        try:
            d_tr = float(wm.get("best_holdout_trail3")) - float(cm.get("best_holdout_trail3"))
        except (TypeError, ValueError):
            d_tr = float("nan")
        try:
            d_ic = float(wm.get("best_holdout_tail_ic")) - float(cm.get("best_holdout_tail_ic"))
        except (TypeError, ValueError):
            d_ic = float("nan")
        if np.isfinite(d_tr):
            deltas.append(d_tr)
        cold_control.append(
            {
                "fold_id": key[0],
                "bag": key[1],
                "warm_epoch": wm.get("selected_epoch"),
                "cold_epoch": cm.get("selected_epoch"),
                "delta_trail3": d_tr,
                "delta_ic": d_ic,
            }
        )
    cold_mean_delta = float(np.mean(deltas)) if deltas else float("nan")

    real = {}
    raw42 = seed_raw.get(42)
    if raw42 is not None and not raw42.empty:
        real = real_fold_metrics(raw42, null_folds, labeled, close, btc_id, "spread")
    print(f"[HB] real fold metrics { {k: (v or {}).get('tail_ic_top') for k, v in real.items()} }", flush=True)

    use_seeds = list(NULL_SHUFFLE_SEEDS)[: int(NULL_REPLICATES)]
    payloads = []
    for fold in null_folds:
        for ss in use_seeds:
            payloads.append(
                {
                    "pack_path": str(pack_path),
                    "labeled_path": str(labeled_path),
                    "close_path": str(close_path),
                    "btc_id": int(btc_id),
                    "fold": _fold_to_dict(fold),
                    "shuffle_seed": int(ss),
                    "null_cache_dir": str(work / "null"),
                }
            )
    print(f"[HB] vol-matched null map n={len(payloads)} concurrency={NULL_MAP_CONCURRENCY}", flush=True)
    recs = list(nfn_v1_null_job.map(payloads, order_outputs=True))
    null = cells_from_replicates(null_folds, recs, real)
    print(
        f"[HB] null passed={null.get('passed')} verdict={(null.get('tail_ic_top') or {}).get('verdict')}",
        flush=True,
    )

    dates = list(close.index)
    members = ffill_members(pit_members(pit100, btc_id), dates)
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(HORIZON))
    books = {}
    for name, df in (("frozen_spread", frozen), ("nfn_ensemble", ensemble)):
        if df is None or df.empty:
            continue
        scores = preds_to_score_at(df, "spread", [t for t, _, _ in pairs14])
        packed = run_periodic_long(close, members, btc_id, scores, pairs14, cost_bps=float(ALT_BPS), label=name)
        books[name] = packed
        print(f"[HB] book {name} total={packed.get('total')} sharpe={packed.get('sharpe')}", flush=True)

    hygiene_sf = aggregate_seed_fold(hygiene)
    verdict = mechanical_verdict_v1(grid, null, seed_metrics, hygiene)

    e = np.asarray((last_meta_42_0 or {}).get("exponents") or np.zeros((24, 198)))
    w = np.asarray((last_meta_42_0 or {}).get("rule_w") or np.zeros(24))
    rules = top_rules(e, w, n_rules=8)
    mem = (last_meta_42_0 or {}).get("membership") or {}
    movement = membership_movement(
        mem.get("c") or np.zeros((33, 3)),
        mem.get("s") or np.ones((33, 3)),
        mem.get("c_init") or np.zeros((33, 3)),
        mem.get("s_init") or np.ones((33, 3)),
        mem.get("feat_names") or list(STAGE_S_COLS),
        top_n=20,
    )
    v0 = _v0_from_json(Path("/root/btcb_phase7_nfn.json"))
    v0_disp = float(v0.get("dispersion") or V0_SEED_DISPERSION)
    failed = verdict.get("failed_clauses") or []
    clause = "none" if not failed else ",".join(failed)
    v1_ics = [float((seed_metrics[s] or {}).get("tail_ic_top")) for s in SEEDS if s in seed_metrics]
    plain = (
        f"{verdict.get('label')} (failed clauses={clause}). "
        f"CRAFT-CONFIRMED={verdict.get('craft_confirmed')}. "
        f"Ensemble vs frozen: Δtail-IC={verdict.get('delta_tail_ic')} Δoverlap={verdict.get('delta_overlap')}. "
        f"Seed dispersion v0→v1: {v0_disp} → {verdict.get('seed_dispersion_tail_ic')}. "
        f"Mean selected epoch={verdict.get('mean_selected_epoch')}. "
        f"Vol-matched null passed={verdict.get('null_pass')}. Nothing adopted."
    )
    extra = {
        "pred_sha256": pred_hash["sha256"],
        "cmc_panel_sha256": cmc_panel_sha0,
        "cmc_readonly_ok": cmc_panel_sha0 == CMC_PANEL_SHA256,
        "start": str(close.index.min().date()) if len(close) else None,
        "end": str(close.index.max().date()) if len(close) else None,
        "gpu_used": False,
        "elapsed_sec": time.time() - t0,
        "plain": plain,
        "n_params": n_params,
        "n_folds": int(len(folds_all)),
        "n_bag": int(N_BAG),
        "seeds": list(SEEDS),
        "firewall_passed": bool(fw.get("passed")),
        "v0_dispersion": v0_disp,
        "v0_mean_best_epoch": v0.get("mean_best_epoch"),
        "cold_mean_delta_trail3": cold_mean_delta,
        "vol_col": volc,
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger_v1(ledger_path, verdict, extra)

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase7c(
        rep_dir / "btcb_phase7c_nfn_v1.md",
        grid=grid,
        books={k: _jsonable(v) for k, v in books.items()},
        null=null,
        hygiene=hygiene,
        hygiene_seed_fold=hygiene_sf,
        seed_metrics=seed_metrics,
        verdict=verdict,
        rules=rules,
        movement=movement,
        extra=extra,
        cold_control=cold_control,
        arch=arch,
    )
    plot_tail_ic_null(grid, null, chart_dir / "btcb_phase7c_nfn_v1_tail_ic.png")
    plot_dispersion(v0.get("seed_ics") or [], [x for x in v1_ics if np.isfinite(x)], chart_dir / "btcb_phase7c_nfn_v1_dispersion.png")
    v1_epochs = [h.get("selected_epoch") for h in hygiene if h.get("selected_epoch") is not None]
    plot_epoch_hist(
        [float(x) for x in (v0.get("epochs") or []) if x is not None],
        [float(x) for x in v1_epochs],
        chart_dir / "btcb_phase7c_nfn_v1_epochs.png",
    )
    plot_membership(movement, chart_dir / "btcb_phase7c_nfn_v1_membership.png")
    plot_film_ribbon([last_meta_42_0] if last_meta_42_0 else hygiene, chart_dir / "btcb_phase7c_nfn_v1_film.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    (rep_dir / "btcb_phase7c_nfn_v1_addendum.md").write_text(addendum)
    payload = {
        "criterion": PHASE7C_CRITERION,
        "null_registration": PHASE7C_NULL_REGISTRATION,
        "firewall": FIREWALL,
        "verdict": _jsonable(verdict),
        "grid": {k: _jsonable(v) for k, v in grid.items()},
        "seed_metrics": {str(k): _jsonable(v) for k, v in seed_metrics.items()},
        "books": {k: _jsonable(v) for k, v in books.items()},
        "null": _jsonable(null),
        "hygiene": [_jsonable(h) for h in hygiene],
        "hygiene_seed_fold": hygiene_sf,
        "cold_control": cold_control,
        "rules": rules[:8],
        "movement": movement,
        "extra": extra,
        "architecture": arch,
        "craft_only": True,
    }
    (rep_dir / "btcb_phase7c_nfn_v1.json").write_text(json.dumps(payload, indent=2, default=str))
    quant_vol.commit()

    print(f"{verdict.get('label')}; failed={clause}", flush=True)
    print(f"CRAFT-CONFIRMED={verdict.get('craft_confirmed')} mean_selected_epoch={verdict.get('mean_selected_epoch')}", flush=True)
    print(
        f"dispersion v0→v1 {v0_disp} → {verdict.get('seed_dispersion_tail_ic')} | "
        f"Δtail-IC={verdict.get('delta_tail_ic')} Δoverlap={verdict.get('delta_overlap')}",
        flush=True,
    )
    print(
        f"ensemble tail-IC={grid['nfn_ensemble'].get('tail_ic_top')} "
        f"overlap={grid['nfn_ensemble'].get('overlap')} | "
        f"frozen tail-IC={grid['frozen_spread'].get('tail_ic_top')} "
        f"overlap={grid['frozen_spread'].get('overlap')}",
        flush=True,
    )
    for rec in rules[:3]:
        print(f"TOP-RULE {rec.get('formula')} w={rec.get('weight')}", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {
        "label": verdict.get("label"),
        "failed_clauses": verdict.get("failed_clauses"),
        "craft_confirmed": verdict.get("craft_confirmed"),
        "delta_tail_ic": verdict.get("delta_tail_ic"),
        "delta_overlap": verdict.get("delta_overlap"),
        "dispersion": verdict.get("seed_dispersion_tail_ic"),
        "dispersion_v0": v0_disp,
        "mean_selected_epoch": verdict.get("mean_selected_epoch"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] Phase 7.c NFN v1 (detach-safe)...", flush=True)
    fc = run_btcb_p7c_nfn_v1.spawn()
    print("[local] spawned", flush=True)
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase7c_nfn_v1.md", "reports"),
        ("reports/btcb_phase7c_nfn_v1.json", "reports"),
        ("reports/btcb_phase7c_nfn_v1_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase7c_nfn_v1_tail_ic.png", "charts"),
        ("charts/btcb_phase7c_nfn_v1_dispersion.png", "charts"),
        ("charts/btcb_phase7c_nfn_v1_epochs.png", "charts"),
        ("charts/btcb_phase7c_nfn_v1_membership.png", "charts"),
        ("charts/btcb_phase7c_nfn_v1_film.png", "charts"),
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
        for src in (art / "charts").glob("btcb_phase7c_nfn_v1*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase7c*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(summary, flush=True)
