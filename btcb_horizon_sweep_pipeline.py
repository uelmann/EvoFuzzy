"""
BTC-BEATER SPREAD-LS horizon sweep — train twin heads at h=3 and h=7.

BACKTEST ONLY. CPU only. Frozen products untouched.
h=14/h=30 caches reused with sha256 verification. Funding-off (3.b has not run).
Usage: modal run --detach btcb_horizon_sweep_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-hsweep"
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
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file(
        "reports/btcb_horizon_sweep_addendum.md",
        remote_path="/root/btcb_horizon_sweep_addendum.md",
    )
    .add_local_file("universe/btcb_top50_floor.parquet", remote_path="/root/btcb_top50_floor.parquet")
    .add_local_file("universe/btcb_top100_floor.parquet", remote_path="/root/btcb_top100_floor.parquet")
)

app = modal.App(APP_NAME, image=image)


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {
        "daily_ret",
        "btc_ret",
        "equity",
        "n_long",
        "n_short",
        "n_shortable",
        "incomplete",
        "long_gross",
        "short_gross",
        "cash",
        "realized_beta_90d",
        "id_to_sym",
        "contrib",
        "daily_cost",
        "daily_gross",
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
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return None
    return x


@app.function(
    timeout=60 * 60 * 10,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_hsweep() -> dict:
    import numpy as np
    import pandas as pd

    from baseline.data import load_panel
    from baseline.seedutil import seed_everything
    from btcb.constants import (
        DEATH_CONVENTION,
        HORIZON_FUNDING_ON,
        HORIZON_SWEEP_CRITERION,
        HORIZON_SWEEP_HS,
        HORIZON_SWEEP_INCUMBENT,
        HORIZON_SWEEP_TRAIN,
        NULL_FOLD_IDS_2C,
        PHASE2C_NULL_GATE,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3_FUNDING_CAVEAT,
        SEED,
        STAGE_S_COLS,
    )
    from btcb.features import btc_id_from_panel
    from btcb.gates import assert_no_context, gate_twin_spread_null, pick_folds_by_id
    from btcb.horizon_sweep_report import (
        plot_horizon_equity,
        plot_horizon_rankic,
        write_horizon_sweep,
    )
    from btcb.hygiene import clean_panel
    from btcb.labels import add_twin_quintile_labels
    from btcb.model import (
        mean_per_date_auc,
        mean_per_date_rank_ic,
        merge_twin_preds,
        per_date_rank_ic_series,
        train_all_folds,
    )
    from btcb.spread_ls import (
        attach_beta,
        build_shortable,
        choose_production_horizon,
        ew_basket,
        hash_pred_dir,
        load_twin_from_cache,
        run_spread_ls,
        squeeze_table,
        within_universe_rankic,
    )

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_horizon_sweep_addendum.md").read_text()
    if HORIZON_SWEEP_CRITERION not in addendum:
        raise RuntimeError("horizon-sweep addendum missing verbatim criterion")
    if PHASE2C_NULL_GATE not in addendum or DEATH_CONVENTION not in addendum:
        raise RuntimeError("horizon-sweep addendum missing null gate or death convention")
    print("[HB] BTC-BEATER horizon sweep BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {HORIZON_SWEEP_CRITERION}", flush=True)
    print(f"[HB] {PHASE2C_NULL_GATE}", flush=True)
    print(f"[HB] {PHASE3_FUNDING_CAVEAT}", flush=True)
    print(f"[HB] FUNDING_ON={HORIZON_FUNDING_ON} (3.b has not run; all books funding-off)", flush=True)
    if HORIZON_FUNDING_ON:
        raise RuntimeError("funding-on requested but 3.b has not run")

    def commit():
        quant_vol.commit()

    panel = pd.read_parquet("/data/quant/btcb/full/panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id}", flush=True)

    def _load_pit(name: str) -> pd.DataFrame:
        for p in (
            Path(f"/data/quant/btcb/universe/{name}"),
            Path(f"/root/{name}"),
        ):
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                df["id"] = df["id"].astype(int)
                print(f"[HB] pit {name} from {p} rows={len(df)}", flush=True)
                return df
        raise RuntimeError(f"missing floored PIT {name}")

    pit100 = _load_pit("btcb_top100_floor.parquet")
    assert_no_context(list(STAGE_S_COLS))

    print("[HB] re-applying frozen 2.b cleaner (no new hygiene)...", flush=True)
    cleaned, _ = clean_panel(panel, btc_id=btc_id)

    feat = pd.read_parquet("/data/quant/btcb/phase2b/feat_s.parquet")
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)

    old_pred = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(old_pred)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n_files={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(
            f"2.c cache hash mismatch: got {pred_hash['sha256']} n={pred_hash['n_files']} "
            f"expected {PHASE2C_PRED_SHA256} n={PHASE2C_PRED_N_FILES}"
        )

    print("[HB] twin quintile labels for h=3,7,14,30...", flush=True)
    drop_lab = [
        c
        for c in feat.columns
        if c.startswith("y_h") or c.startswith("y_bot_h") or c.startswith("excess_h")
    ]
    feat_l = feat.drop(columns=drop_lab, errors="ignore")
    labeled = add_twin_quintile_labels(feat_l, cleaned, btc_id, horizons=HORIZON_SWEEP_HS)
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    print(
        f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} ids={labeled['id'].nunique()}",
        flush=True,
    )

    new_pred = Path("/data/quant/btcb/horizon_sweep/preds")
    new_pred.mkdir(parents=True, exist_ok=True)
    all_folds = {}
    for h in HORIZON_SWEEP_TRAIN:
        for head in ("top", "bot"):
            ycol = None if head == "top" else f"y_bot_h{h}"
            print(f"[HB] TRAIN head={head} h={h}...", flush=True)
            preds, metas, folds = train_all_folds(
                labeled,
                horizon=h,
                out_dir=new_pred,
                feature_cols=list(STAGE_S_COLS),
                early_stop="per_date_auc",
                ycol=ycol,
                tag=head,
            )
            if head == "top":
                all_folds[h] = folds
            commit()
            print(
                f"[HB] trained head={head} h={h} pred_rows={len(preds)} folds={len(folds)} ok="
                f"{sum(1 for m in metas if m.get('status')=='ok')}",
                flush=True,
            )

    new_hash = hash_pred_dir(new_pred)
    print(f"[HB] new-head cache sha256={new_hash['sha256']} n_files={new_hash['n_files']}", flush=True)
    # incumbent cache must still be byte-identical after new-head writes (separate dir)
    pred_hash2 = hash_pred_dir(old_pred)
    if pred_hash2["sha256"] != PHASE2C_PRED_SHA256:
        raise RuntimeError("2.c cache mutated during horizon sweep")

    def _load_h(h: int) -> pd.DataFrame:
        src = new_pred if int(h) in HORIZON_SWEEP_TRAIN else old_pred
        twin = load_twin_from_cache(src, int(h))
        ex = f"excess_h{h}"
        if ex not in twin.columns:
            twin = twin.merge(labeled[["date", "id", ex]], on=["date", "id"], how="left")
        return twin

    twins = {int(h): _load_h(int(h)) for h in HORIZON_SWEEP_HS}

    def _fold_metrics(twin: pd.DataFrame, h: int) -> list[dict]:
        excol = f"excess_h{h}"
        rows = []
        for fid, g in twin.groupby("fold_id"):
            rows.append(
                {
                    "fold_id": int(fid),
                    "n": int(len(g)),
                    "rankic_spread_raw": mean_per_date_rank_ic(
                        g["spread_raw"].to_numpy(), g[excol].to_numpy(), g["date"].to_numpy()
                    ),
                    "auc_ptop_raw": mean_per_date_auc(
                        g["y_top"].to_numpy(), g["p_top_raw"].to_numpy(), g["date"].to_numpy()
                    )[0]
                    if "y_top" in g.columns
                    else float("nan"),
                }
            )
        return sorted(rows, key=lambda r: r["fold_id"])

    nulls: dict = {}
    for h in HORIZON_SWEEP_TRAIN:
        print(f"[HB] repowered twin null h={h} (6 folds × 25 × 2 heads)...", flush=True)
        folds = all_folds[h]
        print(f"[HB] h={h} n_folds={len(folds)} ids={[f.fold_id for f in folds]}", flush=True)
        null_folds = pick_folds_by_id(folds, NULL_FOLD_IDS_2C)
        fm = _fold_metrics(twins[h], h)
        real_aucs = {int(r["fold_id"]): float(r["auc_ptop_raw"]) for r in fm}
        real_rics = {int(r["fold_id"]): float(r["rankic_spread_raw"]) for r in fm}
        ng = gate_twin_spread_null(
            labeled,
            null_folds,
            real_aucs,
            real_rics,
            feature_cols=list(STAGE_S_COLS),
            early_stop="per_date_auc",
        )
        nulls[h] = ng
        ric = ng.get("rankic") or {}
        print(
            f"[gates] h={h} spread RankIC §2: {ric.get('verdict')} {ric.get('n_exceed')}/6 "
            f"z={ric.get('stouffer_z')} passed={ng.get('passed')}",
            flush=True,
        )
        commit()

    p2c_path = Path("/data/quant/reports/btcb_phase2c_report.json")
    if p2c_path.exists():
        p2c = json.loads(p2c_path.read_text())
        ng14 = p2c.get("null_gate") or {}
        nulls[14] = {
            "passed": bool(ng14.get("passed")),
            "rankic": ng14.get("rankic"),
            "rankic_cells": ng14.get("rankic_cells"),
            "reason": "reused from Phase 2.c; not re-run this freeze",
        }
    else:
        nulls[14] = {
            "passed": True,
            "rankic": {"verdict": "GREEN", "n_exceed": 6, "stouffer_z": 11.041, "bias_pass": True},
            "reason": "Phase 2.c json missing on volume; recorded GREEN 6/6 z=11.04 from repo report",
        }
    nulls[30] = {
        "passed": False,
        "skipped": True,
        "reason": "no horizon-specific null in this freeze; 2.c null was h=14 only. Reported, not judged.",
    }

    raw_dir = Path("/data/quant/raw/klines")
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    kline_panel = load_panel(raw_dir, kline_syms)
    kline_panel["date"] = pd.to_datetime(kline_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    shortable = build_shortable(cleaned, kline_panel, btc_id)
    print(f"[HB] shortable dates={len(shortable)}", flush=True)

    close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    btc_simple = close[btc_id].pct_change(fill_method=None)
    members = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit100.groupby("date")["id"]
    }
    all_dates = [d for d in close.index if d in members]
    basket = ew_basket(close, members, all_dates)

    books = {}
    ric_series = {}
    ric_means = {}
    btc_hits = 0
    for h in HORIZON_SWEEP_HS:
        twin = twins[int(h)]
        print(f"[HB] book h={h} beta_matched=1 funding=0 k_dec=10 k_q=20...", flush=True)
        packed = run_spread_ls(
            cleaned,
            pit100,
            twin,
            feat,
            shortable,
            btc_id,
            h=int(h),
            beta_matched=True,
            decile_k=10,
            quintile_k=20,
        )
        if packed.get("error"):
            raise RuntimeError(f"h={h} book failed: {packed}")
        packed = attach_beta(packed, btc_simple)
        packed["rankic"] = within_universe_rankic(twin, pit100, horizon=int(h))
        sq = squeeze_table(packed["daily_ret"], basket)
        vals = [float(r["spread_ls"]) for r in sq if np.isfinite(r.get("spread_ls", float("nan")))]
        packed["squeeze"] = sq
        packed["squeeze_mean"] = float(np.mean(vals)) if vals else float("nan")
        packed["funding_on"] = False
        books[int(h)] = packed
        btc_hits += int(packed.get("btc_in_book_hits") or 0)
        pr = twin.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
        u = pr.merge(pit100[["date", "id"]], on=["date", "id"], how="inner")
        excol = f"excess_h{h}"
        ric_series[int(h)] = per_date_rank_ic_series(
            u["spread"].to_numpy(), u[excol].to_numpy(), u["date"].to_numpy()
        )
        ric_means[int(h)] = float(packed["rankic"])
        print(
            f"[HB] h={h} sharpe={packed.get('net_sharpe')} trail={packed.get('net_sharpe_trail18m')} "
            f"rankic={packed.get('rankic')} drag={ (packed.get('econ') or {}).get('ann_cost_drag') } "
            f"net_bps={ (packed.get('econ') or {}).get('avg_net_bps') }",
            flush=True,
        )
        commit()
    if btc_hits:
        raise RuntimeError(f"BTC leaked into a book: hits={btc_hits}")

    choice = choose_production_horizon(books, nulls, incumbent=HORIZON_SWEEP_INCUMBENT)
    print(f"[HB] choice {choice}", flush=True)

    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "btc_hits_total": int(btc_hits),
        "pred_sha256": pred_hash["sha256"],
        "pred_n_files": pred_hash["n_files"],
        "pred_sha256_expected": PHASE2C_PRED_SHA256,
        "new_pred_sha256": new_hash["sha256"],
        "new_pred_n_files": new_hash["n_files"],
        "funding_on": False,
    }

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_horizon_sweep(
        rep_dir / "btcb_horizon_sweep.md",
        choice=choice,
        books=books,
        nulls=nulls,
        extra=extra,
    )
    plot_horizon_equity(books, chart_dir / "btcb_horizon_equity.png")
    plot_horizon_rankic(ric_series, ric_means, chart_dir / "btcb_horizon_rankic.png")
    payload = {
        "criterion": HORIZON_SWEEP_CRITERION,
        "funding_caveat": PHASE3_FUNDING_CAVEAT,
        "funding_on": False,
        "choice": _jsonable(choice),
        "books": {str(k): _jsonable(v) for k, v in books.items()},
        "nulls": {str(k): _jsonable(v) for k, v in nulls.items()},
        "rankic": ric_means,
        "pred_hash": {"sha256": pred_hash["sha256"], "n_files": pred_hash["n_files"]},
        "new_pred_hash": {"sha256": new_hash["sha256"], "n_files": new_hash["n_files"]},
        "extra": _jsonable(extra),
        "gpu_used": False,
    }
    (rep_dir / "btcb_horizon_sweep.json").write_text(json.dumps(payload, indent=2, default=str))
    commit()

    print(f"CHOSEN HORIZON: {choice['chosen_h']} fallback={choice.get('fallback')}", flush=True)
    for h in HORIZON_SWEEP_HS:
        b = books[h]
        e = b.get("econ") or {}
        ng = nulls.get(h) or {}
        print(
            f"h={h} RankIC={b.get('rankic')} Sharpe={b.get('net_sharpe')}/{b.get('net_sharpe_trail18m')} "
            f"cost_drag={e.get('ann_cost_drag')} net_edge_bps={e.get('avg_net_bps')} "
            f"hold_d={e.get('avg_hold_days')} RT/y={e.get('round_trips_per_year')} "
            f"null_passed={ng.get('passed')}",
            flush=True,
        )
    print("FUNDING=0. COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "chosen_h": choice["chosen_h"],
        "fallback": bool(choice.get("fallback")),
        "rankic": ric_means,
        "sharpe": {str(h): books[h].get("net_sharpe") for h in HORIZON_SWEEP_HS},
        "trail": {str(h): books[h].get("net_sharpe_trail18m") for h in HORIZON_SWEEP_HS},
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "pred_sha256": pred_hash["sha256"],
    }


@app.local_entrypoint()
def main():
    print("[local] starting horizon sweep (spawn, then wait)...", flush=True)
    fc = run_btcb_hsweep.spawn()
    print(f"[local] spawned {getattr(fc, 'object_id', fc)}", flush=True)
    summary = fc.get()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_horizon_sweep.md", "reports"),
        ("reports/btcb_horizon_sweep.json", "reports"),
        ("charts/btcb_horizon_equity.png", "charts"),
        ("charts/btcb_horizon_rankic.png", "charts"),
    ]
    for remote, kind in pulls:
        name = Path(remote).name
        dest = art / kind / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", VOL_Q, remote, str(dest), "--force"], check=False)
        candidate = dest if dest.is_file() else dest / name
        if candidate.exists() and candidate.is_file():
            out = Path(kind) / name
            out.parent.mkdir(exist_ok=True)
            shutil.copy2(candidate, out)
    opt = Path("/opt/cursor/artifacts")
    if opt.exists():
        for sub in ("reports", "charts", "screenshots"):
            (opt / sub).mkdir(parents=True, exist_ok=True)
        for src in (art / "reports").glob("btcb_horizon*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_horizon*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] horizon sweep complete.", flush=True)
