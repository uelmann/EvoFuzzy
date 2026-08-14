"""
BTC-BEATER Phase 2.c — twin-head spread + repowered skill null.

BACKTEST ONLY. CPU only. Frozen COMBO untouched.
Reuses 2.b cleaned+floored universes. No new hygiene.
Usage: modal run --detach btcb_phase2c_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p2c"
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
    .add_local_file("reports/btcb_phase2c_addendum.md", remote_path="/root/btcb_phase2c_addendum.md")
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
        "equity_btc",
        "rel_equity",
        "w_btc",
        "n_names",
        "gate_on",
        "contrib",
        "id_to_sym",
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


def _btc_ref(book: dict) -> dict:
    return {
        "book_total": book.get("btc_total"),
        "book_cagr": book.get("btc_cagr"),
        "book_sharpe": book.get("btc_sharpe"),
        "rel_sharpe": 0.0,
        "maxdd": book.get("btc_maxdd"),
        "avg_n_names": 0.0,
        "avg_w_btc": 1.0,
        "ann_turnover": 0.0,
        "gate_on_frac": 0.0,
        "forced_exits": {"n_events": 0, "n_ids": 0},
        "start": book.get("start"),
        "end": book.get("end"),
        "n_days": book.get("n_days"),
    }


@app.function(
    timeout=60 * 60 * 6,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p2c() -> dict:
    import numpy as np
    import pandas as pd

    from btcb.book import mechanical_verdicts_v3, pick_median_p_enter, run_gated_book
    from btcb.constants import (
        CTX_COLS,
        DEATH_CONVENTION,
        NULL_FOLD_IDS_2C,
        PHASE2C_CRITERION,
        PHASE2C_NULL_GATE,
        PHASE2_HORIZONS,
        PHASE2_PRIMARY_H,
        REGIME_BUDGET,
        SEED,
        SPREAD_RANKIC_SKILL,
        STAGE_S_AUC_SKILL,
        STAGE_S_COLS,
        THETA_GRID,
    )
    from btcb.features import assemble_stage_s_features, btc_id_from_panel
    from btcb.gates import (
        assert_no_context,
        gate_seed_determinism,
        gate_twin_spread_null,
        pick_folds_by_id,
    )
    from btcb.hygiene import clean_panel
    from btcb.labels import add_twin_quintile_labels
    from btcb.model import (
        mean_gain,
        mean_per_date_auc,
        mean_per_date_rank_ic,
        merge_twin_preds,
        per_date_rank_ic_series,
        train_all_folds,
    )
    from btcb.phase2c_report import (
        plot_calibration_pair,
        plot_equity_gate,
        plot_rankic_series,
        write_phase2c,
    )
    from btcb.timing import breadth_top100, ew_top50_btc_ratio, regime_on_off
    from baseline.seedutil import seed_everything

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase2c_addendum.md").read_text()
    if PHASE2C_CRITERION not in addendum or PHASE2C_NULL_GATE not in addendum or DEATH_CONVENTION not in addendum:
        raise RuntimeError("Phase 2.c addendum missing verbatim criterion/gate/convention")
    print("[HB] BTC-BEATER P2c BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {PHASE2C_CRITERION}", flush=True)
    print(f"[HB] {PHASE2C_NULL_GATE}", flush=True)
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
        raise RuntimeError(f"missing floored PIT {name} (must reuse 2.b, do not rebuild)")

    pit50 = _load_pit("btcb_top50_floor.parquet")
    pit100 = _load_pit("btcb_top100_floor.parquet")

    print("[HB] re-applying frozen 2.b cleaner (no new hygiene)...", flush=True)
    cleaned, _clog = clean_panel(panel, btc_id=btc_id)

    ctx_gate = assert_no_context(list(STAGE_S_COLS))

    cache = Path("/data/quant/btcb/phase2b")
    feat_path = cache / "feat_s.parquet"
    if feat_path.exists():
        print(f"[HB] reuse 2.b Stage-S features {feat_path}", flush=True)
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        leaked = [c for c in CTX_COLS if c in feat.columns]
        if leaked:
            raise RuntimeError(f"context leaked into reused Stage-S features: {leaked}")
    else:
        print("[HB] feat_s missing; assembling Stage-S (same 2.b recipe, no context)", flush=True)
        feat = assemble_stage_s_features(cleaned, pit100, btc_id)
        cache.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(feat_path, index=False)
        commit()

    print("[HB] twin quintile labels...", flush=True)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id)
    labeled = labeled[labeled["id"] != int(btc_id)].copy()
    print(
        f"[HB] labeled rows={len(labeled)} dates={labeled['date'].nunique()} ids={labeled['id'].nunique()}",
        flush=True,
    )

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    pred_dir.mkdir(parents=True, exist_ok=True)
    all_preds = {"top": {}, "bot": {}}
    all_metas = {"top": {}, "bot": {}}
    all_folds = {}
    for h in PHASE2_HORIZONS:
        for head, spec in (("top", {"ycol": None, "tag": "top"}), ("bot", {"tag": "bot"})):
            ycol = None if head == "top" else f"y_bot_h{h}"
            preds, metas, folds = train_all_folds(
                labeled,
                horizon=h,
                out_dir=pred_dir,
                feature_cols=list(STAGE_S_COLS),
                early_stop="per_date_auc",
                ycol=ycol,
                tag=head,
            )
            all_preds[head][h] = preds
            all_metas[head][h] = metas
            if head == "top":
                all_folds[h] = folds
            commit()
            print(f"[HB] trained head={head} h={h} pred_rows={len(preds)} folds={len(folds)}", flush=True)

    print("[HB] seed determinism (top head fold 0)...", flush=True)
    seed_gate = gate_seed_determinism(
        labeled,
        all_folds[PHASE2_PRIMARY_H][0],
        seed=SEED,
        feature_cols=list(STAGE_S_COLS),
        early_stop="per_date_auc",
        ycol=f"y_h{PHASE2_PRIMARY_H}",
    )
    print(f"[gates] {seed_gate['name']}: {'PASS' if seed_gate.get('passed') else 'FAIL'} {seed_gate}", flush=True)

    twins = {}
    fold_metrics = {}
    skill_agg = {}
    ric_series = {}
    for h in PHASE2_HORIZONS:
        twin = merge_twin_preds(all_preds["top"][h], all_preds["bot"][h], h)
        twins[h] = twin
        excol = f"excess_h{h}"
        rows = []
        for fid, g in twin.groupby("fold_id"):
            rows.append(
                {
                    "fold_id": int(fid),
                    "n": int(len(g)),
                    "rankic_spread": mean_per_date_rank_ic(g["spread"].to_numpy(), g[excol].to_numpy(), g["date"].to_numpy()),
                    "rankic_spread_raw": mean_per_date_rank_ic(
                        g["spread_raw"].to_numpy(), g[excol].to_numpy(), g["date"].to_numpy()
                    ),
                    "rankic_ptop": mean_per_date_rank_ic(g["p_top"].to_numpy(), g[excol].to_numpy(), g["date"].to_numpy()),
                    "auc_spread": mean_per_date_auc(g["y_top"].to_numpy(), g["spread"].to_numpy(), g["date"].to_numpy())[0],
                    "auc_ptop": mean_per_date_auc(g["y_top"].to_numpy(), g["p_top"].to_numpy(), g["date"].to_numpy())[0],
                    "auc_ptop_raw": mean_per_date_auc(
                        g["y_top"].to_numpy(), g["p_top_raw"].to_numpy(), g["date"].to_numpy()
                    )[0],
                }
            )
        fold_metrics[str(h)] = sorted(rows, key=lambda r: r["fold_id"])
        pr = twin.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
        ric_s = mean_per_date_rank_ic(pr["spread"].to_numpy(), pr[excol].to_numpy(), pr["date"].to_numpy())
        ric_p = mean_per_date_rank_ic(pr["p_top"].to_numpy(), pr[excol].to_numpy(), pr["date"].to_numpy())
        auc_s, _ = mean_per_date_auc(pr["y_top"].to_numpy(), pr["spread"].to_numpy(), pr["date"].to_numpy())
        skill_agg[h] = {"rankic_spread": ric_s, "rankic_ptop": ric_p, "auc_spread": auc_s}
        ric_series[h] = {
            "spread": per_date_rank_ic_series(pr["spread"].to_numpy(), pr[excol].to_numpy(), pr["date"].to_numpy()),
            "ptop": per_date_rank_ic_series(pr["p_top"].to_numpy(), pr[excol].to_numpy(), pr["date"].to_numpy()),
        }
        print(
            f"[HB] aggregate h={h} RankIC(spread)={ric_s:.4f} RankIC(p_top)={ric_p:.4f} AUC(spread)={auc_s:.4f}",
            flush=True,
        )

    volcol = "yz_vol_30_raw" if "yz_vol_30_raw" in feat.columns else "yz_vol_30"
    tw14 = twins[PHASE2_PRIMARY_H]
    u = tw14.merge(feat[["date", "id", volcol]], on=["date", "id"], how="left")
    uncert_vol = mean_per_date_rank_ic(u["uncertainty"].to_numpy(), u[volcol].to_numpy(), u["date"].to_numpy())
    print(f"[HB] uncertainty↔{volcol} mean per-date RankIC={uncert_vol}", flush=True)

    print("[HB] repowered twin null (6 folds × 25 × 2 heads)...", flush=True)
    null_folds = pick_folds_by_id(all_folds[PHASE2_PRIMARY_H], NULL_FOLD_IDS_2C)
    real_aucs = {int(r["fold_id"]): float(r["auc_ptop_raw"]) for r in fold_metrics[str(PHASE2_PRIMARY_H)]}
    real_rics = {int(r["fold_id"]): float(r["rankic_spread_raw"]) for r in fold_metrics[str(PHASE2_PRIMARY_H)]}
    null_gate = gate_twin_spread_null(
        labeled,
        null_folds,
        real_aucs,
        real_rics,
        feature_cols=list(STAGE_S_COLS),
        early_stop="per_date_auc",
    )
    ric_null = null_gate.get("rankic") or {}
    print(
        f"[gates] spread RankIC §2: {ric_null.get('verdict')} {ric_null.get('n_exceed')}/6 "
        f"z={ric_null.get('stouffer_z')} passed={null_gate.get('passed')}",
        flush=True,
    )

    def _h_pass(h: int) -> bool:
        a = skill_agg[h]
        return bool(
            np.isfinite(a["rankic_spread"])
            and a["rankic_spread"] >= float(SPREAD_RANKIC_SKILL)
            and np.isfinite(a["auc_spread"])
            and a["auc_spread"] >= float(STAGE_S_AUC_SKILL)
        )

    has_skill = bool(null_gate.get("passed") and (_h_pass(14) or _h_pass(30)))
    skill = {
        "has_skill": has_skill,
        "rankic_h14": skill_agg[14]["rankic_spread"],
        "rankic_ptop_h14": skill_agg[14]["rankic_ptop"],
        "auc_h14": skill_agg[14]["auc_spread"],
        "rankic_h30": skill_agg[30]["rankic_spread"],
        "rankic_ptop_h30": skill_agg[30]["rankic_ptop"],
        "auc_h30": skill_agg[30]["auc_spread"],
        "h14_pass": _h_pass(14),
        "h30_pass": _h_pass(30),
        "null_passed": bool(null_gate.get("passed")),
        "null_verdict": ric_null.get("verdict"),
        "n_exceed": ric_null.get("n_exceed"),
        "stouffer_z": ric_null.get("stouffer_z"),
    }
    print(f"[HB] SPREAD skill={has_skill} {skill}", flush=True)

    gates_ok = bool(ctx_gate.get("passed") and seed_gate.get("passed") and null_gate.get("passed"))
    print(f"[HB] GATES_OK={gates_ok}", flush=True)

    print("[HB] Stage T regime gate (frozen 2.b)...", flush=True)
    ratio = ew_top50_btc_ratio(cleaned, pit50, btc_id)
    breadth = breadth_top100(cleaned, pit100)
    regime = regime_on_off(ratio, breadth)
    gate_on = regime["gate_on"]

    print("[HB] MODEL-V3 spread books (θ grid, h=14)...", flush=True)
    twin14 = twins[PHASE2_PRIMARY_H].copy()
    book_preds = twin14.rename(columns={"spread": "p"})
    books = []
    for theta in THETA_GRID:
        packed = run_gated_book(
            cleaned,
            pit50,
            book_preds,
            feat,
            btc_id,
            p_enter=float(theta),
            h=int(PHASE2_PRIMARY_H),
            gate_on=gate_on,
            budget=float(REGIME_BUDGET),
        )
        packed["theta"] = float(theta)
        books.append(packed)
        print(
            f"[HB] book θ={theta} rel={packed.get('rel_sharpe')} tot={packed.get('book_total')} "
            f"wbtc={packed.get('avg_w_btc')} names={packed.get('avg_n_names')} on={packed.get('gate_on_frac')}",
            flush=True,
        )

    head = pick_median_p_enter(books)
    if not isinstance(head, dict) or head.get("error") or not head.get("start"):
        raise RuntimeError(f"headline book failed: {head}")
    head["theta"] = float(head.get("theta", head.get("p_enter")))
    btc_ref = _btc_ref(head)
    v3 = mechanical_verdicts_v3(head)
    if not gates_ok:
        v3 = dict(v3)
        v3["viable"] = False
        v3["product_grade"] = False
        v3["gates_blocked"] = True

    imps_top = mean_gain(all_metas["top"].get(PHASE2_PRIMARY_H) or [], top_n=15)
    imps_bot = mean_gain(all_metas["bot"].get(PHASE2_PRIMARY_H) or [], top_n=15)
    for name, imps in (("top", imps_top), ("bot", imps_bot)):
        leaked = [a for a, _ in imps if a in set(CTX_COLS)]
        if leaked:
            raise RuntimeError(f"context in {name} importances: {leaked}")

    naive_note = {"rel_sharpe": None, "live_benchmark": False}
    p2b_json = Path("/data/quant/reports/btcb_phase2b_report.json")
    if p2b_json.exists():
        try:
            prev = json.loads(p2b_json.read_text())
            nv = prev.get("naive_v4_full") or {}
            naive_note = {
                "rel_sharpe": nv.get("rel_sharpe"),
                "live_benchmark": nv.get("live_benchmark"),
                "book_total": nv.get("book_total"),
            }
        except Exception:
            pass

    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "gates_ok": gates_ok,
        "n_features": len(STAGE_S_COLS),
        "n_train_rows": int(len(labeled)),
        "btc_id": int(btc_id),
        "feature_cols": list(STAGE_S_COLS),
        "uncert_vol_rankic": uncert_vol,
        "volcol": volcol,
        "theta_grid": list(THETA_GRID),
        "null_fold_ids": list(NULL_FOLD_IDS_2C),
    }

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_phase2c(
        rep_dir / "btcb_phase2c_report.md",
        skill=skill,
        null_gate=null_gate,
        fold_metrics=fold_metrics,
        headline=head,
        grid=books,
        btc_ref=btc_ref,
        naive_note=naive_note,
        verdicts=v3,
        importances_top=imps_top,
        importances_bot=imps_bot,
        extra=extra,
    )
    plot_equity_gate(head, gate_on, chart_dir / "btcb_p2c_equity.png")
    plot_rankic_series(
        ric_series[14]["spread"], ric_series[14]["ptop"], chart_dir / "btcb_p2c_rankic.png"
    )
    plot_calibration_pair(
        all_metas["top"].get(PHASE2_PRIMARY_H) or [],
        all_metas["bot"].get(PHASE2_PRIMARY_H) or [],
        chart_dir / "btcb_p2c_calibration.png",
    )

    payload = {
        "criterion": PHASE2C_CRITERION,
        "null_gate_text": PHASE2C_NULL_GATE,
        "death_convention": DEATH_CONVENTION,
        "gates_ok": gates_ok,
        "skill": _jsonable(skill),
        "null_gate": _jsonable(null_gate),
        "fold_metrics": _jsonable(fold_metrics),
        "verdicts": _jsonable(v3),
        "headline": _jsonable(head),
        "grid": [_jsonable(x) for x in books],
        "naive_v4_record": _jsonable(naive_note),
        "importances_top": [{"feature": a, "mean_gain": b} for a, b in imps_top],
        "importances_bot": [{"feature": a, "mean_gain": b} for a, b in imps_bot],
        "fold_meta_top": {str(h): _jsonable(all_metas["top"][h]) for h in all_metas["top"]},
        "fold_meta_bot": {str(h): _jsonable(all_metas["bot"][h]) for h in all_metas["bot"]},
        "extra": _jsonable(extra),
        "gpu_used": False,
    }
    (rep_dir / "btcb_phase2c_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "btcb_phase2c_done.txt").write_text(
        json.dumps({"elapsed_sec": time.time() - t0, "gpu_used": False}, indent=2)
    )
    commit()

    skill_s = "HAS-SELECTION-SKILL" if has_skill else "NO-SELECTION-SKILL"
    viable_s = "VIABLE" if (v3.get("viable") and gates_ok) else "NOT VIABLE"
    prod_s = "PRODUCT-GRADE" if (v3.get("product_grade") and gates_ok) else "NOT PRODUCT-GRADE"
    print(
        f"STAGE-S SPREAD: {skill_s} RankIC={skill.get('rankic_h14')} AUC={skill.get('auc_h14')}",
        flush=True,
    )
    print(
        f"NULL: {ric_null.get('verdict')} {ric_null.get('n_exceed')}/6 Stouffer z={ric_null.get('stouffer_z')}",
        flush=True,
    )
    print(f"VERDICT: MODEL-V3 {viable_s}", flush=True)
    print(f"VERDICT: {prod_s}", flush=True)
    print(f"% time in BTC: {head.get('avg_w_btc')}", flush=True)
    print(f"avg #names: {head.get('avg_n_names')}", flush=True)
    print(f"uncertainty↔vol RankIC: {uncert_vol}", flush=True)
    print("COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "stage_s_skill": has_skill,
        "rankic_spread_h14": skill.get("rankic_h14"),
        "auc_spread_h14": skill.get("auc_h14"),
        "null_verdict": ric_null.get("verdict"),
        "n_exceed": ric_null.get("n_exceed"),
        "stouffer_z": ric_null.get("stouffer_z"),
        "viable": bool(v3.get("viable") and gates_ok),
        "product_grade": bool(v3.get("product_grade") and gates_ok),
        "avg_w_btc": head.get("avg_w_btc"),
        "avg_n_names": head.get("avg_n_names"),
        "uncert_vol_rankic": uncert_vol,
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] starting BTC-BEATER P2c (spawn, then wait)...", flush=True)
    fc = run_btcb_p2c.spawn()
    print(f"[local] spawned {getattr(fc, 'object_id', fc)}", flush=True)
    summary = fc.get()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase2c_report.md", "reports"),
        ("reports/btcb_phase2c_report.json", "reports"),
        ("charts/btcb_p2c_equity.png", "charts"),
        ("charts/btcb_p2c_rankic.png", "charts"),
        ("charts/btcb_p2c_calibration.png", "charts"),
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
        for src in (art / "reports").glob("btcb_phase2c*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_p2c*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] BTC-BEATER P2c complete.", flush=True)
