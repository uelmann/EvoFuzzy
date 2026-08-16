"""
BTC-BEATER Phase 7.d — NFN VARIANT A (magnitude labels + ranking loss).

BACKTEST / ANALYSIS ONLY. CPU only. Zero GPU. One shot.
Frozen products untouched.
Usage: modal run --detach --timestamps btcb_phase7d_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p7d-nfn-va"
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
    .add_local_python_source("baseline", "btcb", "nfn_va")
    .add_local_file(
        "reports/btcb_phase7d_addendum.md",
        remote_path="/root/btcb_phase7d_addendum.md",
    )
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
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
        "per_date",
        "inits",
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
    import numpy as np
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
    return wide.replace([np.inf, -np.inf], np.nan)


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


def _mean_score(frames: list, col: str = "score"):
    import pandas as pd

    acc = None
    for i, df in enumerate(frames):
        sl = df[["date", "id", col]].copy()
        sl = sl.rename(columns={col: f"s{i}"})
        acc = sl if acc is None else acc.merge(sl, on=["date", "id"], how="outer")
    if acc is None:
        return pd.DataFrame(columns=["date", "id", col])
    cols = [c for c in acc.columns if c.startswith("s")]
    acc[col] = acc[cols].mean(axis=1)
    return acc[["date", "id", col]]


def _load_nfn_v0() -> dict:
    p = Path("/data/quant/reports/btcb_phase7_nfn.json")
    from nfn_va.constants import NFN_V0_READONLY

    if p.exists():
        try:
            blob = json.loads(p.read_text())
            grid = blob.get("grid") or {}
            ens = grid.get("nfn_ensemble") or {}
            if ens.get("tail_ic_top") is not None:
                return {
                    "source": "volume:/data/quant/reports/btcb_phase7_nfn.json",
                    "label": "nfn_v0",
                    "tail_ic_top": ens.get("tail_ic_top"),
                    "tail_ic_bot": ens.get("tail_ic_bot"),
                    "overlap": ens.get("overlap"),
                    "monster": ens.get("monster"),
                    "rankic": ens.get("rankic"),
                    "n_dates": ens.get("n_dates"),
                    "tail_ic_top_nw_t": ens.get("tail_ic_top_nw_t"),
                    "tail_ic_top_trail": ens.get("tail_ic_top_trail"),
                    "overlap_cycles": ens.get("overlap_cycles"),
                    "tail_ic_top_cycles": ens.get("tail_ic_top_cycles"),
                }
        except Exception as exc:
            print(f"[p7d] nfn v0 volume unreadable: {exc}", flush=True)
    return dict(NFN_V0_READONLY)


@app.function(
    timeout=60 * 90,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=4,
    memory=16384,
    max_containers=40,
)
def nfn_va_null_job(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import json as json_lib

    import pandas as pd
    import torch

    torch.set_num_threads(4)
    if torch.cuda.is_available():
        raise RuntimeError("GPU forbidden in Phase 7.d")

    from nfn_va.data import load_pack
    from nfn_va.nulls import run_null_cell

    cache = Path(payload.get("null_cache_dir") or "/data/quant/btcb/phase7d/null")
    cache.mkdir(parents=True, exist_ok=True)
    outp = cache / f"fold{payload['fold']['fold_id']}_seed{payload['shuffle_seed']}.json"
    if outp.exists():
        try:
            rec = json_lib.loads(outp.read_text())
            if rec.get("status") == "ok":
                print(
                    f"[p7d] null CACHE HIT fold={payload['fold']['fold_id']} shuffle={payload['shuffle_seed']}",
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
    print(f"[p7d] null fold={fold.fold_id} shuffle={payload['shuffle_seed']}", flush=True)
    rec = run_null_cell(
        pack,
        fold,
        int(payload["shuffle_seed"]),
        labeled,
        close,
        int(payload["btc_id"]),
    )
    outp.write_text(json_lib.dumps(rec, default=str))
    try:
        quant_vol.commit()
    except Exception:
        pass
    return rec


@app.function(
    timeout=60 * 60 * 12,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=32768,
)
def nfn_va_train_seed(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_available():
        raise RuntimeError("GPU forbidden in Phase 7.d")

    from nfn_va.constants import CACHE_VER
    from nfn_va.data import load_pack
    from nfn_va.train import train_walkforward
    from nfn_va.warmstart import find_ruleforge_bank

    seed = int(payload["seed"])
    pack = load_pack(payload["pack_path"])
    folds = [_fold_from_dict(d) for d in payload["folds"]]
    blob, ws_meta = find_ruleforge_bank()
    warm_blob = blob if ws_meta.get("viable") else None
    cache_dir = Path(payload["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[p7d] train seed={seed} n_folds={len(folds)} warmstart_viable={ws_meta.get('viable')}",
        flush=True,
    )
    raw, metas = train_walkforward(
        pack,
        folds,
        seed=seed,
        warm_blob=warm_blob,
        device="cpu",
        cache_dir=cache_dir,
        cache_ver=CACHE_VER,
        commit_fn=quant_vol.commit,
    )
    pred_path = cache_dir / f"seed{seed}_all.parquet"
    if raw is not None and not raw.empty:
        raw.to_parquet(pred_path, index=False)
    slim = []
    for m in metas:
        row = {
            k: v
            for k, v in m.items()
            if k
            not in (
                "exponents",
                "membership",
                "film",
                "train_curve",
                "holdout_curve",
                "inits",
                "_states",
            )
        }
        row["seed"] = seed
        slim.append(row)
    meta_path = cache_dir / f"seed{seed}_hygiene.json"
    meta_path.write_text(json.dumps(slim, default=str))
    quant_vol.commit()
    n_ok = sum(1 for m in metas if m.get("status") == "ok")
    print(f"[p7d] seed={seed} folds_ok={n_ok}/{len(metas)} rows={0 if raw is None else len(raw)}", flush=True)
    return {
        "seed": seed,
        "n_rows": 0 if raw is None else int(len(raw)),
        "n_ok": int(n_ok),
        "pred_path": str(pred_path),
        "hygiene_path": str(meta_path),
        "warmstart": ws_meta,
    }


@app.function(
    timeout=60 * 60 * 16,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def run_btcb_p7d_va() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_available():
        raise RuntimeError("GPU forbidden in Phase 7.d")

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
    from nfn_va.constants import (
        CACHE_VER,
        FIREWALL,
        FIREWALL_PHASE7,
        HORIZON,
        NULL_FOLD_IDS,
        NULL_MAP_CONCURRENCY,
        NULL_REPLICATES,
        PHASE7D_CRITERION,
        PHASE7_NULL_REGISTRATION,
        SEEDS,
        VARIANT_A_N_PARAMS,
    )
    from nfn_va.data import pack_labeled, save_pack
    from nfn_va.firewall import assert_firewall
    from nfn_va.interpret import membership_movement, top_rules
    from nfn_va.magdiag import attach_excess, decile_mean_curve, topk_mean_excess, yearly_score_stability
    from nfn_va.model import assert_arch_equal_phase7, build_nfn
    from nfn_va.nulls import cells_from_replicates
    from nfn_va.regime import regime_frame
    from nfn_va.report import plot_decile_curve, plot_tail_ic_bars, update_ledger, write_phase7d
    from nfn_va.verdict import magnitude_gain, mechanical_verdict
    from nfn_va.warmstart import find_ruleforge_bank

    t0 = time.time()
    seed_everything(int(SEEDS[0]))
    fw = assert_firewall()
    arch = assert_arch_equal_phase7()
    n_probe = build_nfn(42).n_params()
    if int(n_probe) != int(VARIANT_A_N_PARAMS):
        raise RuntimeError(f"n_params {n_probe} != {VARIANT_A_N_PARAMS}")
    addendum = Path("/root/btcb_phase7d_addendum.md").read_text()
    for txt in (FIREWALL, FIREWALL_PHASE7, PHASE7D_CRITERION, PHASE7_NULL_REGISTRATION, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"phase7d addendum missing freeze text: {txt[:80]}")
    print("[p7d] PHASE 7.d VARIANT A ANALYSIS ONLY; zero GPU; nothing adopted", flush=True)
    print(f"[p7d] {PHASE7D_CRITERION}", flush=True)

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    print(f"[p7d] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0}", flush=True)
    if cmc_panel_sha0 != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_panel_sha0}")

    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    start = pd.Timestamp(PHASE3C_REF_START, tz="UTC")
    panel = pd.read_parquet(CMC_PANEL)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    print(f"[p7d] btc_id={btc_id} rows={len(panel)}", flush=True)

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
                print(f"[p7d] pit from {p} rows={len(pit)}", flush=True)
                return pit
        return None

    pit100 = _load_pit("btcb_top100_floor.parquet")
    pit50 = _load_pit("btcb_top50_floor.parquet")
    if pit100 is None:
        raise RuntimeError("missing floored PIT top-100")
    if pit50 is None:
        raise RuntimeError("missing floored PIT top-50")

    print("[p7d] re-applying frozen 2.b cleaner (no CMC writes)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[p7d] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
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

    print("[p7d] labels h=14 magnitude (excess from twin helper)...", flush=True)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(HORIZON,))
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    labeled["date"] = pd.to_datetime(labeled["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    labeled["id"] = labeled["id"].astype(int)
    volc = vol_col_name(labeled)
    print(f"[p7d] labeled rows={len(labeled)} dates={labeled['date'].nunique()} vol_col={volc}", flush=True)

    print("[p7d] regime vector m_t...", flush=True)
    regime = regime_frame(cleaned, pit50, pit100, btc_id)

    print("[p7d] canonical hybrid close...", flush=True)
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
    print(f"[p7d] close {close.shape} {close.index.min().date()}→{close.index.max().date()}", flush=True)

    pack = pack_labeled(labeled, regime)
    work = Path("/data/quant/btcb/phase7d")
    work.mkdir(parents=True, exist_ok=True)
    pack_path = work / "pack.npz"
    labeled_path = work / "labeled.parquet"
    close_path = work / "close.parquet"
    save_pack(pack, pack_path)
    labeled.to_parquet(labeled_path, index=False)
    close.to_parquet(close_path)
    quant_vol.commit()

    blob, ws_meta = find_ruleforge_bank()
    warm_blob = blob if ws_meta.get("viable") else None
    print(
        f"[p7d] warm-start path={ws_meta.get('path')} viable={ws_meta.get('viable')} "
        f"reason={ws_meta.get('reason')}",
        flush=True,
    )

    folds_all = make_expanding_folds(pd.DatetimeIndex(labeled["date"].unique()), horizon=HORIZON)
    print(f"[p7d] n_folds={len(folds_all)}", flush=True)
    null_folds = pick_folds_by_id(folds_all, NULL_FOLD_IDS)
    fold_payloads = [_fold_to_dict(f) for f in folds_all]

    seed_payloads = [
        {
            "seed": int(s),
            "pack_path": str(pack_path),
            "folds": fold_payloads,
            "cache_dir": str(work / f"preds_s{s}"),
        }
        for s in SEEDS
    ]
    print(f"[p7d] fan-out {len(seed_payloads)} seeds", flush=True)
    seed_rets = list(nfn_va_train_seed.map(seed_payloads, order_outputs=True))
    try:
        quant_vol.reload()
    except Exception:
        pass

    seed_raw = {}
    seed_collapsed = {}
    hygiene = []
    n_params = int(VARIANT_A_N_PARAMS)
    last_meta_42 = None
    for rec in seed_rets:
        seed = int(rec["seed"])
        pq = Path(rec["pred_path"])
        raw = pd.read_parquet(pq) if pq.exists() else pd.DataFrame()
        hy = json.loads(Path(rec["hygiene_path"]).read_text()) if Path(rec["hygiene_path"]).exists() else []
        for m in hy:
            hygiene.append(m)
            if int(seed) == 42:
                last_meta_42 = m
        seed_raw[seed] = raw
        seed_collapsed[seed] = collapse_fold_preds(raw, "score") if raw is not None and not raw.empty else pd.DataFrame()
        print(f"[p7d] loaded seed={seed} rows={len(raw)}", flush=True)

    # interpretability needs exponents from last fold; reload seed-42 last-fold meta if cached
    def _fold_num(p: Path) -> int:
        try:
            return int(p.stem.split("fold")[-1])
        except Exception:
            return -1

    last_js = sorted((work / "preds_s42").glob("meta_seed42_fold*.json"), key=_fold_num)
    if last_js:
        try:
            last_meta_42 = json.loads(last_js[-1].read_text())
            last_meta_42["seed"] = 42
        except Exception:
            pass

    frozen = collapse_fold_preds(twin.rename(columns={"spread": "p"}), "p").rename(columns={"p": "spread"})
    ensemble = _mean_score(
        [seed_collapsed[s] for s in SEEDS if s in seed_collapsed and not seed_collapsed[s].empty],
        "score",
    )

    def _eval(df, col, name):
        if df is None or df.empty:
            return {"label": name, "tail_ic_top": float("nan"), "overlap": float("nan"), "rankic": float("nan")}
        ev = restrict_eval_frame(df, labeled, close, btc_id, col)
        met = per_date_tail_metrics(ev, col)
        met["label"] = name
        print(
            f"[p7d] {name} tailIC_top={met.get('tail_ic_top')} overlap={met.get('overlap')} "
            f"rankic={met.get('rankic')} n={met.get('n_dates')}",
            flush=True,
        )
        return met

    nfn_v0 = _load_nfn_v0()
    grid = {
        "frozen_spread": _eval(frozen, "spread", "frozen_spread"),
        "nfn_v0": nfn_v0,
        "variant_a_ensemble": _eval(ensemble, "score", "variant_a_ensemble"),
    }
    seed_metrics = {}
    for s, df in seed_collapsed.items():
        seed_metrics[int(s)] = _eval(df, "score", f"va_seed_{s}")

    real = {}
    raw42 = seed_raw.get(42)
    if raw42 is not None and not raw42.empty:
        real = real_fold_metrics(raw42, null_folds, labeled, close, btc_id, "score")
    print(f"[p7d] real fold metrics { {k: (v or {}).get('tail_ic_top') for k, v in real.items()} }", flush=True)

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
    print(f"[p7d] vol-matched null map n={len(payloads)} concurrency={NULL_MAP_CONCURRENCY}", flush=True)
    recs = list(nfn_va_null_job.map(payloads, order_outputs=True))
    null = cells_from_replicates(null_folds, recs, real)
    print(
        f"[p7d] null passed={null.get('passed')} verdict={(null.get('tail_ic_top') or {}).get('verdict')}",
        flush=True,
    )

    dates = list(close.index)
    members = ffill_members(pit_members(pit100, btc_id), dates)
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(HORIZON))
    books = {}
    book_inputs = (("frozen_spread", frozen, "spread"), ("variant_a_ensemble", ensemble, "score"))
    for name, df, col in book_inputs:
        if df is None or df.empty:
            continue
        scores = preds_to_score_at(df, col, [t for t, _, _ in pairs14])
        packed = run_periodic_long(close, members, btc_id, scores, pairs14, cost_bps=float(ALT_BPS), label=name)
        books[name] = packed
        print(f"[p7d] book {name} total={packed.get('total')} sharpe={packed.get('sharpe')}", flush=True)

    excess_col = f"excess_h{HORIZON}"
    print(
        f"[p7d] magdiag excess_col={excess_col} labeled_has={excess_col in labeled.columns} "
        f"frozen_cols={list(frozen.columns)} ens_cols={list(ensemble.columns)}",
        flush=True,
    )
    va_ex = attach_excess(ensemble, labeled, "score", excess_col)
    fr_ex = attach_excess(frozen, labeled, "spread", excess_col)
    va_top10 = topk_mean_excess(va_ex, "score", excess_col)
    fr_top10 = topk_mean_excess(fr_ex, "spread", excess_col)
    mag = magnitude_gain(va_top10.get("mean"), fr_top10.get("mean"))
    deciles = {
        "variant_a": decile_mean_curve(va_ex, "score", excess_col),
        "frozen": decile_mean_curve(fr_ex, "spread", excess_col),
    }
    yearly = yearly_score_stability(ensemble, "score")
    print(
        f"[p7d] top10 mean excess VA={mag.get('va')} frozen={mag.get('frozen')} MAGNITUDE-GAIN={mag.get('yes')}",
        flush=True,
    )

    verdict = mechanical_verdict(grid, null, seed_metrics, mag)

    e = np.asarray((last_meta_42 or {}).get("exponents") or np.zeros((24, 198)))
    w = np.asarray((last_meta_42 or {}).get("rule_w") or np.zeros(24))
    rules = top_rules(e, w, n_rules=8)
    mem = (last_meta_42 or {}).get("membership") or {}
    movement = membership_movement(
        mem.get("c") or np.zeros((33, 3)),
        mem.get("s") or np.ones((33, 3)),
        mem.get("c_init") or np.zeros((33, 3)),
        mem.get("s_init") or np.ones((33, 3)),
        mem.get("feat_names") or list(STAGE_S_COLS),
        top_n=20,
    )
    n_under = int(sum(1 for h in hygiene if h.get("undertrained")))
    failed = verdict.get("failed_clauses") or []
    clause = "none" if not failed else ",".join(failed)
    ens = grid.get("variant_a_ensemble") or {}
    plain = (
        f"{verdict.get('label')} (failed clauses={clause}). "
        f"Ensemble vs frozen: Δtail-IC={verdict.get('delta_tail_ic')} Δoverlap={verdict.get('delta_overlap')}. "
        f"Δ vs NFN v0 tail-IC={float(ens.get('tail_ic_top') or np.nan) - float(nfn_v0.get('tail_ic_top') or np.nan)} "
        f"overlap={float(ens.get('overlap') or np.nan) - float(nfn_v0.get('overlap') or np.nan)}. "
        f"Seed dispersion tail-IC(top)={verdict.get('seed_dispersion_tail_ic')}. "
        f"Vol-matched null passed={verdict.get('null_pass')}. "
        f"MAGNITUDE-GAIN={'yes' if mag.get('yes') else 'no'} "
        f"(VA {mag.get('va')} vs frozen {mag.get('frozen')}). "
        f"UNDERTRAINED count={n_under}. Warm-start={'RULE-FORGE' if warm_blob else 'random'}. "
        f"Nothing adopted."
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
        "config_equal": bool(arch.get("passed")),
        "n_folds": int(len(folds_all)),
        "seeds": list(SEEDS),
        "firewall_passed": bool(fw.get("passed")),
        "warmstart_path": ws_meta.get("path") or "random_init",
        "warmstart_viable": bool(ws_meta.get("viable")),
        "warmstart_rules": int(
            sum(h.get("warmstart_rules") or 0 for h in hygiene if h.get("seed") == 42)
            / max(1, sum(1 for h in hygiene if h.get("seed") == 42))
        ),
        "n_undertrained": n_under,
        "vol_col": volc,
        "cache_ver": CACHE_VER,
        "nfn_v0_source": nfn_v0.get("source"),
    }

    ledger_path = Path("/root/numbers_ledger.md")
    update_ledger(ledger_path, verdict, extra)

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase7d(
        rep_dir / "btcb_phase7d_variantA.md",
        grid=grid,
        books={k: _jsonable(v) for k, v in books.items()},
        null=null,
        hygiene=[{k: v for k, v in h.items() if k not in ("exponents", "membership")} for h in hygiene],
        seed_metrics=seed_metrics,
        verdict=verdict,
        rules=rules,
        movement=movement,
        mag=mag,
        deciles=deciles,
        yearly=yearly,
        extra=extra,
    )
    plot_tail_ic_bars(grid, null, chart_dir / "btcb_phase7d_tail_ic.png")
    plot_decile_curve(deciles, chart_dir / "btcb_phase7d_decile.png")
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())
    (rep_dir / "btcb_phase7d_addendum.md").write_text(addendum)
    stdout = (
        f"{verdict.get('label')} failed={clause} "
        f"MAGNITUDE-GAIN={'yes' if mag.get('yes') else 'no'} "
        f"va_top10={mag.get('va')} frozen_top10={mag.get('frozen')} "
        f"disp={verdict.get('seed_dispersion_tail_ic')} "
        f"d_tail_frozen={verdict.get('delta_tail_ic')} d_ov_frozen={verdict.get('delta_overlap')} "
        f"d_tail_nfnv0={float(ens.get('tail_ic_top') or np.nan) - float(nfn_v0.get('tail_ic_top') or np.nan)}"
    )
    payload = {
        "criterion": PHASE7D_CRITERION,
        "null_registration": PHASE7_NULL_REGISTRATION,
        "firewall": FIREWALL,
        "verdict": _jsonable(verdict),
        "grid": {k: _jsonable(v) for k, v in grid.items()},
        "seed_metrics": {str(k): _jsonable(v) for k, v in seed_metrics.items()},
        "books": {k: _jsonable(v) for k, v in books.items()},
        "null": _jsonable(null),
        "hygiene": [_jsonable(h) for h in hygiene],
        "magnitude": {"top10": mag, "variant_a": va_top10, "frozen": fr_top10},
        "deciles": deciles,
        "yearly_score": yearly,
        "rules": rules[:8],
        "movement": movement,
        "extra": extra,
        "warmstart": ws_meta,
        "stdout": stdout,
    }
    (rep_dir / "btcb_phase7d_variantA.json").write_text(json.dumps(payload, indent=2, default=str))
    quant_vol.commit()

    print(stdout, flush=True)
    print(f"[p7d] DONE elapsed={time.time()-t0:.1f}s gpu=false nothing_adopted=true", flush=True)
    return {
        "label": verdict.get("label"),
        "failed_clauses": verdict.get("failed_clauses"),
        "delta_tail_ic": verdict.get("delta_tail_ic"),
        "delta_overlap": verdict.get("delta_overlap"),
        "dispersion": verdict.get("seed_dispersion_tail_ic"),
        "magnitude_gain": bool(mag.get("yes")),
        "va_top10": mag.get("va"),
        "frozen_top10": mag.get("frozen"),
        "n_undertrained": n_under,
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "stdout": stdout,
    }


@app.local_entrypoint()
def main():
    print("[local] Phase 7.d Variant A (detach-safe)...", flush=True)
    fc = run_btcb_p7d_va.spawn()
    print("[local] spawned", flush=True)
    summary = fc.get()
    import shutil
    import subprocess

    art = Path("artifacts")
    pulls = [
        ("reports/btcb_phase7d_variantA.md", "reports"),
        ("reports/btcb_phase7d_variantA.json", "reports"),
        ("reports/btcb_phase7d_addendum.md", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase7d_tail_ic.png", "charts"),
        ("charts/btcb_phase7d_decile.png", "charts"),
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
        for src in (art / "charts").glob("btcb_phase7d*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
        for src in (art / "reports").glob("btcb_phase7d*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
    print(summary, flush=True)
    print((summary or {}).get("stdout", ""), flush=True)
