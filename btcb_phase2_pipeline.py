"""
BTC-BEATER Phase 2 — winner-tail classifier (MODEL-V1).

BACKTEST ONLY. CPU only. Frozen COMBO untouched.
Usage: modal run btcb_phase2_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p2"
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
        "pyyaml",
        "scikit-learn",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/btcb_phase2_addendum.md", remote_path="/root/btcb_phase2_addendum.md")
    .add_local_file("universe/btcb_top50_pit.parquet", remote_path="/root/btcb_top50_pit.parquet")
    .add_local_file("universe/btcb_top100_pit.parquet", remote_path="/root/btcb_top100_pit.parquet")
)

app = modal.App(APP_NAME, image=image)


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {"daily_ret", "btc_ret", "equity", "equity_btc", "rel_equity", "w_btc", "n_names"}
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


def _equity_line(book: dict) -> list[dict]:
    import pandas as pd

    eq = book.get("equity")
    if not isinstance(eq, pd.Series):
        return []
    eqb = book.get("equity_btc")
    rel = book.get("rel_equity")
    wbtc = book.get("w_btc")
    nn = book.get("n_names")
    rows = []
    for d in eq.index:
        rows.append(
            {
                "date": str(pd.Timestamp(d).date()),
                "book": float(eq.loc[d]),
                "btc": float(eqb.loc[d]) if eqb is not None and d in eqb.index else float("nan"),
                "rel": float(rel.loc[d]) if rel is not None and d in rel.index else float("nan"),
                "w_btc": float(wbtc.loc[d]) if wbtc is not None and d in wbtc.index else float("nan"),
                "n_names": float(nn.loc[d]) if nn is not None and d in nn.index else float("nan"),
            }
        )
    return rows


@app.function(
    timeout=60 * 60 * 6,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p2() -> dict:
    import numpy as np
    import pandas as pd

    from btcb.benchmark import naive_rotation_v3
    from btcb.book import mechanical_verdicts, pick_median_p_enter, run_hysteresis_book
    from btcb.constants import (
        DEATH_CONVENTION,
        FEATURE_COLS_V1,
        P_ENTER_GRID,
        PHASE2_CRITERION,
        PHASE2_HORIZONS,
        PHASE2_PRIMARY_H,
        SEED,
        USABLE_FROM,
    )
    from btcb.features import assemble_feature_table, btc_id_from_panel
    from btcb.gates import gate_label_shuffle_null, gate_seed_determinism, run_cheap_gates
    from btcb.labels import add_binary_excess_labels
    from btcb.model import mean_gain, pick_null_folds, train_all_folds
    from btcb.phase2_report import plot_calibration, plot_equity, write_phase2
    from baseline.seedutil import seed_everything

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase2_addendum.md").read_text()
    if PHASE2_CRITERION not in addendum or DEATH_CONVENTION not in addendum:
        raise RuntimeError("Phase 2 addendum missing verbatim criterion/convention")
    print("[HB] BTC-BEATER P2 BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {PHASE2_CRITERION}", flush=True)
    print(f"[HB] {DEATH_CONVENTION}", flush=True)

    def commit():
        quant_vol.commit()

    panel_path = Path("/data/quant/btcb/full/panel.parquet")
    if not panel_path.exists():
        raise RuntimeError(f"missing panel {panel_path}")
    print(f"[HB] loading panel {panel_path}", flush=True)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] panel rows={len(panel)} ids={panel['id'].nunique()} btc_id={btc_id}", flush=True)

    def _load_pit(name: str) -> pd.DataFrame:
        cands = [
            Path(f"/data/quant/btcb/universe/{name}"),
            Path(f"/data/quant/universe/{name}"),
            Path(f"/root/{name}"),
        ]
        for p in cands:
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                df["id"] = df["id"].astype(int)
                print(f"[HB] pit {name} from {p} rows={len(df)}", flush=True)
                return df
        raise RuntimeError(f"missing PIT file {name}")

    pit50 = _load_pit("btcb_top50_pit.parquet")
    pit100 = _load_pit("btcb_top100_pit.parquet")

    cache = Path("/data/quant/btcb/phase2")
    cache.mkdir(parents=True, exist_ok=True)
    feat_path = cache / "feat_v1.parquet"

    print("[HB] cheap gates...", flush=True)
    cheap = run_cheap_gates(panel, btc_id=btc_id)

    if feat_path.exists():
        print(f"[HB] reuse features {feat_path}", flush=True)
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    else:
        feat = assemble_feature_table(panel, pit100, pit50, btc_id)
        feat.to_parquet(feat_path, index=False)
        commit()
        print(f"[HB] wrote {feat_path}", flush=True)

    print("[HB] labels...", flush=True)
    labeled = add_binary_excess_labels(feat, panel, btc_id)
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    print(
        f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} ids={labeled['id'].nunique()}",
        flush=True,
    )

    pred_dir = cache / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    all_preds: dict[int, pd.DataFrame] = {}
    all_metas: dict[int, list] = {}
    all_folds: dict[int, list] = {}
    for h in PHASE2_HORIZONS:
        preds, metas, folds = train_all_folds(labeled, horizon=h, out_dir=pred_dir)
        all_preds[h] = preds
        all_metas[h] = metas
        all_folds[h] = folds
        commit()
        print(f"[HB] trained h={h} pred_rows={len(preds)} folds={len(folds)}", flush=True)

    print("[HB] seed determinism...", flush=True)
    folds14 = all_folds[PHASE2_PRIMARY_H]
    seed_gate = gate_seed_determinism(labeled, folds14[0], seed=SEED)
    print(f"[gates] {seed_gate['name']}: {'PASS' if seed_gate.get('passed') else 'FAIL'} {seed_gate}", flush=True)

    print("[HB] label-shuffle null (2 folds × 25)...", flush=True)
    null_folds = pick_null_folds(folds14)
    real_aucs = {}
    for m in all_metas[PHASE2_PRIMARY_H]:
        if m.get("fold_id") in {f.fold_id for f in null_folds}:
            real_aucs[int(m["fold_id"])] = float(m.get("auc_oos_raw") if m.get("auc_oos_raw") is not None else np.nan)
    null_gate = gate_label_shuffle_null(labeled, null_folds, real_aucs)
    print(
        f"[gates] label_shuffle_null: {'PASS' if null_gate.get('passed') else 'FAIL'} {null_gate.get('verdict')}",
        flush=True,
    )

    gates = list(cheap) + [seed_gate, {k: v for k, v in null_gate.items() if k != "cells"}]
    gates[-1]["name"] = "label_shuffle_null"
    gates_ok = all(bool(g.get("passed")) for g in cheap + [seed_gate]) and bool(null_gate.get("passed"))
    print(f"[HB] GATES_OK={gates_ok}", flush=True)

    print("[HB] books...", flush=True)
    books = {h: [] for h in PHASE2_HORIZONS}
    for h in PHASE2_HORIZONS:
        preds = all_preds[h]
        if preds is None or preds.empty:
            continue
        for p_enter in P_ENTER_GRID:
            packed = run_hysteresis_book(
                panel, pit50, preds, feat, btc_id, p_enter=float(p_enter), h=int(h)
            )
            books[h].append(packed)
            print(
                f"[HB] book h={h} p={p_enter} rel={packed.get('rel_sharpe')} "
                f"tot={packed.get('book_total')} wbtc={packed.get('avg_w_btc')}",
                flush=True,
            )

    head14 = pick_median_p_enter(books.get(PHASE2_PRIMARY_H) or [])
    head30 = pick_median_p_enter(books.get(30) or []) if books.get(30) else {}
    if not isinstance(head14, dict) or head14.get("error") or not head14.get("start"):
        raise RuntimeError(f"headline book failed: {head14}")
    oos_start = pd.Timestamp(head14["start"], tz="UTC")
    print(f"[HB] naive v3 same-window start={oos_start.date()}", flush=True)
    naive = naive_rotation_v3(panel, pit50, oos_start)
    btc_ref = {
        "book_total": naive.get("btc_total"),
        "book_cagr": naive.get("btc_cagr"),
        "book_sharpe": naive.get("btc_sharpe"),
        "rel_sharpe": 0.0,
        "maxdd": naive.get("btc_maxdd"),
        "avg_n_names": 0.0,
        "avg_w_btc": 1.0,
        "ann_turnover": 0.0,
        "forced_exits": {"n_events": 0, "n_ids": 0},
        "start": naive.get("start"),
        "end": naive.get("end"),
        "n_days": naive.get("n_days"),
    }
    v14 = mechanical_verdicts(head14, naive)
    v30 = mechanical_verdicts(head30, naive) if head30 and "error" not in head30 else {}
    if not gates_ok:
        v14 = dict(v14)
        v14["viable"] = False
        v14["replaces_floor"] = False
        v14["gates_blocked"] = True
        if v30:
            v30 = dict(v30)
            v30["viable"] = False
            v30["replaces_floor"] = False

    imps = mean_gain(all_metas.get(PHASE2_PRIMARY_H) or [], top_n=15)
    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "gates_ok": gates_ok,
        "n_features": len(FEATURE_COLS_V1),
        "n_train_rows": int(len(labeled)),
        "usable_from": USABLE_FROM,
        "btc_id": int(btc_id),
        "primary_h": PHASE2_PRIMARY_H,
        "feature_cols": list(FEATURE_COLS_V1),
    }

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_phase2(
        rep_dir / "btcb_phase2_model.md",
        gates=cheap + [seed_gate],
        null_gate=null_gate,
        headline=head14,
        grid=books.get(PHASE2_PRIMARY_H) or [],
        naive=naive,
        btc_ref=btc_ref,
        verdicts=v14,
        metas={str(h): all_metas[h] for h in all_metas},
        importances=imps,
        extra=extra,
        h30_headline=head30 if head30 and "error" not in head30 else None,
        h30_verdicts=v30 or None,
    )
    plot_equity(head14, naive, chart_dir / "btcb_model_equity.png")
    plot_calibration(all_metas.get(PHASE2_PRIMARY_H) or [], chart_dir / "btcb_model_calibration.png")

    payload = {
        "criterion": PHASE2_CRITERION,
        "death_convention": DEATH_CONVENTION,
        "gates_ok": gates_ok,
        "gates": _jsonable(cheap + [seed_gate]),
        "null_gate": _jsonable(null_gate),
        "verdicts_h14": _jsonable(v14),
        "verdicts_h30": _jsonable(v30),
        "headline_h14": _jsonable(head14),
        "headline_h30": _jsonable(head30),
        "grid_h14": [_jsonable(x) for x in (books.get(PHASE2_PRIMARY_H) or [])],
        "grid_h30": [_jsonable(x) for x in (books.get(30) or [])],
        "naive_same_window": _jsonable(naive),
        "fold_meta": {str(h): _jsonable(all_metas[h]) for h in all_metas},
        "importances": [{"feature": a, "mean_gain": b} for a, b in imps],
        "extra": extra,
        "gpu_used": False,
    }
    if isinstance(head14.get("equity"), pd.Series):
        payload["equity_line_h14"] = _equity_line(head14)
    if isinstance(naive.get("equity"), pd.Series):
        payload["equity_line_naive"] = _equity_line(naive)
    (rep_dir / "btcb_phase2_model.json").write_text(json.dumps(payload, indent=2, default=str))
    commit()

    rels = {float(r.get("p_enter")): r.get("rel_sharpe") for r in (books.get(PHASE2_PRIMARY_H) or [])}
    fe = (head14.get("forced_exits") or {}) if isinstance(head14, dict) else {}
    viable_s = "VIABLE" if (v14.get("viable") and gates_ok) else "NOT VIABLE"
    repl_s = "REPLACES-FLOOR" if (v14.get("replaces_floor") and gates_ok) else "DOES-NOT-REPLACE-FLOOR"
    print(f"VERDICT: {viable_s}", flush=True)
    print(f"VERDICT: {repl_s}", flush=True)
    print(
        f"rel-Sharpe p_enter=0.55/0.60/0.65: "
        f"{rels.get(0.55)} / {rels.get(0.60)} / {rels.get(0.65)}",
        flush=True,
    )
    print(f"% time in BTC: {head14.get('avg_w_btc')}", flush=True)
    print(f"forced-exit count: {fe.get('n_events')}", flush=True)
    print("COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "viable": bool(v14.get("viable") and gates_ok),
        "replaces_floor": bool(v14.get("replaces_floor") and gates_ok),
        "gates_ok": gates_ok,
        "p_enter": head14.get("p_enter"),
        "rel_sharpe": head14.get("rel_sharpe"),
        "rel_grid": rels,
        "avg_w_btc": head14.get("avg_w_btc"),
        "forced_n": fe.get("n_events"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] starting BTC-BEATER P2...", flush=True)
    summary = run_btcb_p2.remote()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase2_model.md", "reports"),
        ("reports/btcb_phase2_model.json", "reports"),
        ("charts/btcb_model_equity.png", "charts"),
        ("charts/btcb_model_calibration.png", "charts"),
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
        for src in (art / "reports").glob("btcb_phase2*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_model*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] BTC-BEATER P2 complete.", flush=True)
