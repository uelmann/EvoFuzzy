"""
BTC-BEATER Phase 2.b — hygiene, naive v4, two-stage MODEL-V2.

BACKTEST ONLY. CPU only. Frozen COMBO untouched.
Usage: modal run --detach btcb_phase2b_pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p2b"
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
    .add_local_file("reports/btcb_phase2b_addendum.md", remote_path="/root/btcb_phase2b_addendum.md")
    .add_local_file("universe/btcb_top50_pit.parquet", remote_path="/root/btcb_top50_pit.parquet")
    .add_local_file("universe/btcb_top100_pit.parquet", remote_path="/root/btcb_top100_pit.parquet")
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


def _pit_size(pit) -> dict:
    import pandas as pd

    if pit is None or len(pit) == 0:
        return {"n_ids": 0, "n_rows": 0, "median_per_date": 0.0, "min_per_date": 0, "n_dates": 0}
    by = pit.groupby("date").size()
    return {
        "n_ids": int(pit["id"].nunique()),
        "n_rows": int(len(pit)),
        "median_per_date": float(by.median()) if len(by) else 0.0,
        "min_per_date": int(by.min()) if len(by) else 0,
        "n_dates": int(by.shape[0]),
    }


def _btc_ref(naive: dict) -> dict:
    return {
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


@app.function(
    timeout=60 * 60 * 6,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p2b() -> dict:
    import numpy as np
    import pandas as pd

    from btcb.benchmark import naive_rotation_v3
    from btcb.book import mechanical_verdicts, pick_median_p_enter, run_gated_book
    from btcb.constants import (
        AUTOPSY_START,
        CONTRIB_SHARE_FLAG,
        CTX_COLS,
        DEATH_CONVENTION,
        P_ENTER_GRID,
        PHASE2B_CRITERION,
        PHASE2_HORIZONS,
        PHASE2_PRIMARY_H,
        REGIME_BUDGET,
        SEED,
        STAGE_S_AUC_SKILL,
        STAGE_S_COLS,
        USABLE_FROM,
    )
    from btcb.features import assemble_stage_s_features, btc_id_from_panel
    from btcb.gates import (
        assert_no_context,
        gate_label_shuffle_null,
        gate_seed_determinism,
        run_cheap_gates,
    )
    from btcb.hygiene import (
        build_floored_pit,
        clean_panel,
        contribution_table,
        find_jump_days,
    )
    from btcb.labels import add_quintile_excess_labels
    from btcb.model import (
        mean_gain,
        mean_per_date_auc,
        mean_per_date_rank_ic,
        per_date_auc_series,
        pick_null_folds,
        train_all_folds,
    )
    from btcb.phase2b_report import (
        plot_calibration,
        plot_equity_gate,
        plot_pdauc_series,
        write_phase2b,
    )
    from btcb.timing import breadth_top100, ew_top50_btc_ratio, regime_on_off
    from baseline.seedutil import seed_everything

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase2b_addendum.md").read_text()
    if PHASE2B_CRITERION not in addendum or DEATH_CONVENTION not in addendum:
        raise RuntimeError("Phase 2.b addendum missing verbatim criterion/convention")
    print("[HB] BTC-BEATER P2b BACKTEST ONLY; zero GPU; COMBO untouched", flush=True)
    print(f"[HB] {PHASE2B_CRITERION}", flush=True)
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

    old50 = _load_pit("btcb_top50_pit.parquet")
    old100 = _load_pit("btcb_top100_pit.parquet")
    old50_sz = _pit_size(old50)
    old100_sz = _pit_size(old100)

    # ------------------------------------------------------------------
    # 1. Hygiene — autopsy of the dirty +9.98M% naive v3 book FIRST
    # ------------------------------------------------------------------
    print("[HB] autopsy naive v3 on UNCLEANED panel + unfloored PIT...", flush=True)
    autopsy_start = pd.Timestamp(AUTOPSY_START, tz="UTC")
    dirty = naive_rotation_v3(panel, old50, autopsy_start)
    dirty_tab = contribution_table(
        dirty.get("contrib") or {}, panel, dirty.get("id_to_sym") or {}, btc_id, top_n=10
    )
    autopsy = {
        "start": dirty.get("start"),
        "end": dirty.get("end"),
        "book_total": dirty.get("book_total"),
        "rel_sharpe": dirty.get("rel_sharpe"),
        "book_sharpe": dirty.get("book_sharpe"),
        "maxdd": dirty.get("maxdd"),
        "contrib_table": dirty_tab,
    }
    print(
        f"[HB] autopsy total={dirty.get('book_total')} rel={dirty.get('rel_sharpe')} "
        f"top_alt_share={dirty_tab.get('top_alt_share')} flag25={dirty_tab.get('flag_single_name_gt_25pct')}",
        flush=True,
    )
    for i, r in enumerate(dirty_tab.get("top") or [], start=1):
        print(
            f"[HB] autopsy#{i} {r.get('symbol')} id={r.get('id')} share={r.get('share'):.4f} "
            f"max_dret={r.get('max_daily_ret')} max_abs={r.get('max_abs_daily_ret')}",
            flush=True,
        )

    jumps_before = find_jump_days(panel)
    print(
        f"[HB] jump suspects |ret|>5: rows={len(jumps_before)} ids={jumps_before['id'].nunique() if len(jumps_before) else 0}",
        flush=True,
    )

    print("[HB] redenom/split clean...", flush=True)
    cleaned, clog = clean_panel(panel, btc_id=btc_id)
    jumps_after = find_jump_days(cleaned)
    n_splice = sum(1 for r in clog for a in r["actions"] if a.get("action") == "splice")
    n_trunc = sum(1 for r in clog for a in r["actions"] if a.get("action") == "truncate")
    clean_summary = {
        "n_ids_touched": int(len(clog)),
        "n_splice": int(n_splice),
        "n_truncate": int(n_trunc),
        "n_rows_before": int(len(panel)),
        "n_rows_after": int(len(cleaned)),
        "n_jumps_before_ids": int(jumps_before["id"].nunique()) if len(jumps_before) else 0,
        "n_jumps_before_rows": int(len(jumps_before)),
        "n_jumps_after": int(len(jumps_after)),
        "n_jumps_after_ids": int(jumps_after["id"].nunique()) if len(jumps_after) else 0,
        "log_head": clog,
        "log": clog,
    }
    print(
        f"[HB] clean done splice={n_splice} trunc={n_trunc} jumps_after={len(jumps_after)}",
        flush=True,
    )

    print("[HB] investability floor + floored PIT rebuild...", flush=True)
    pit50, floor50 = build_floored_pit(cleaned, n=50)
    pit100, floor100 = build_floored_pit(cleaned, n=100)
    uni_dir = Path("/data/quant/btcb/universe")
    uni_dir.mkdir(parents=True, exist_ok=True)
    pit50.to_parquet(uni_dir / "btcb_top50_floor.parquet", index=False)
    pit100.to_parquet(uni_dir / "btcb_top100_floor.parquet", index=False)
    commit()
    print(
        f"[HB] floored PIT wrote top50 rows={len(pit50)} top100 rows={len(pit100)} "
        f"(old50 rows={old50_sz['n_rows']} old100 rows={old100_sz['n_rows']})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Cheap gates on CLEANED panel, floored PIT
    # ------------------------------------------------------------------
    print("[HB] cheap gates (cleaned, floored)...", flush=True)
    cheap = run_cheap_gates(cleaned, btc_id=btc_id, floored=True)
    ctx_gate = assert_no_context(list(STAGE_S_COLS))

    # ------------------------------------------------------------------
    # 2. Naive v4 on cleaned + floored (full usable window = project floor)
    # ------------------------------------------------------------------
    print("[HB] naive v4 full window...", flush=True)
    naive_full = naive_rotation_v3(cleaned, pit50, pd.Timestamp(USABLE_FROM, tz="UTC"))
    naive_full_tab = contribution_table(
        naive_full.get("contrib") or {},
        cleaned,
        naive_full.get("id_to_sym") or {},
        btc_id,
        top_n=10,
    )
    if naive_full_tab.get("flag_single_name_gt_25pct"):
        print(
            f"[HB] FLAG: naive v4 single-name share {naive_full_tab.get('top_alt_share')} "
            f"> {CONTRIB_SHARE_FLAG} of additive PnL",
            flush=True,
        )
    print(
        f"[HB] naive v4 FULL rel={naive_full.get('rel_sharpe')} tot={naive_full.get('book_total')} "
        f"btc={naive_full.get('btc_total')} live={naive_full.get('live_benchmark')} "
        f"top_alt_share={naive_full_tab.get('top_alt_share')}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Stage S features + quintile labels (no context)
    # ------------------------------------------------------------------
    cache = Path("/data/quant/btcb/phase2b")
    cache.mkdir(parents=True, exist_ok=True)
    feat_path = cache / "feat_s.parquet"
    stamp = {
        "n_rows": int(len(cleaned)),
        "n_ids": int(cleaned["id"].nunique()),
        "n_pit100": int(len(pit100)),
        "n_feat": len(STAGE_S_COLS),
    }
    stamp_path = cache / "feat_s_stamp.json"
    reuse = False
    if feat_path.exists() and stamp_path.exists():
        try:
            prev = json.loads(stamp_path.read_text())
            reuse = prev == stamp
        except Exception:
            reuse = False
    if reuse:
        print(f"[HB] reuse Stage-S features {feat_path}", flush=True)
        feat = pd.read_parquet(feat_path)
        feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    else:
        feat = assemble_stage_s_features(cleaned, pit100, btc_id)
        leaked = [c for c in CTX_COLS if c in feat.columns]
        if leaked:
            raise RuntimeError(f"context leaked into Stage S features: {leaked}")
        feat.to_parquet(feat_path, index=False)
        stamp_path.write_text(json.dumps(stamp))
        commit()
        print(f"[HB] wrote {feat_path} rows={len(feat)}", flush=True)

    print("[HB] quintile excess labels (within-date, PIT top-100)...", flush=True)
    labeled = add_quintile_excess_labels(feat, cleaned, btc_id)
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
        preds, metas, folds = train_all_folds(
            labeled,
            horizon=h,
            out_dir=pred_dir,
            feature_cols=list(STAGE_S_COLS),
            early_stop="per_date_auc",
        )
        all_preds[h] = preds
        all_metas[h] = metas
        all_folds[h] = folds
        commit()
        print(f"[HB] trained h={h} pred_rows={len(preds)} folds={len(folds)}", flush=True)

    print("[HB] seed determinism...", flush=True)
    folds14 = all_folds[PHASE2_PRIMARY_H]
    seed_gate = gate_seed_determinism(
        labeled,
        folds14[0],
        seed=SEED,
        feature_cols=list(STAGE_S_COLS),
        early_stop="per_date_auc",
    )
    print(f"[gates] {seed_gate['name']}: {'PASS' if seed_gate.get('passed') else 'FAIL'} {seed_gate}", flush=True)

    print("[HB] label-shuffle null (mean per-date AUC, 2 folds × 25)...", flush=True)
    null_folds = pick_null_folds(folds14)
    real_aucs = {}
    for m in all_metas[PHASE2_PRIMARY_H]:
        if m.get("fold_id") in {f.fold_id for f in null_folds}:
            real_aucs[int(m["fold_id"])] = float(
                m.get("pdauc_oos_raw") if m.get("pdauc_oos_raw") is not None else np.nan
            )
    null_gate = gate_label_shuffle_null(
        labeled,
        null_folds,
        real_aucs,
        feature_cols=list(STAGE_S_COLS),
        early_stop="per_date_auc",
    )
    print(
        f"[gates] label_shuffle_null: {'PASS' if null_gate.get('passed') else 'FAIL'} {null_gate.get('verdict')}",
        flush=True,
    )

    gates = list(cheap) + [ctx_gate, seed_gate, {k: v for k, v in null_gate.items() if k != "cells"}]
    gates[-1]["name"] = "label_shuffle_null"
    gates_ok = all(bool(g.get("passed")) for g in cheap + [ctx_gate, seed_gate]) and bool(null_gate.get("passed"))
    print(f"[HB] GATES_OK={gates_ok}", flush=True)

    preds14 = all_preds.get(PHASE2_PRIMARY_H)
    ycol = f"y_h{PHASE2_PRIMARY_H}"
    excol = f"excess_h{PHASE2_PRIMARY_H}"
    pdauc_series = pd.Series(dtype=float)
    mean_pdauc = float("nan")
    mean_rankic = float("nan")
    if preds14 is not None and not preds14.empty:
        pr = preds14.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
        mean_pdauc, _ = mean_per_date_auc(pr[ycol].to_numpy(), pr["p"].to_numpy(), pr["date"].to_numpy())
        if excol in pr.columns:
            mean_rankic = mean_per_date_rank_ic(pr["p"].to_numpy(), pr[excol].to_numpy(), pr["date"].to_numpy())
        pdauc_series = per_date_auc_series(pr[ycol].to_numpy(), pr["p"].to_numpy(), pr["date"].to_numpy())
    has_skill = bool(
        np.isfinite(mean_pdauc)
        and mean_pdauc >= float(STAGE_S_AUC_SKILL)
        and bool(null_gate.get("passed"))
    )
    skill = {
        "has_skill": has_skill,
        "mean_pdauc": mean_pdauc,
        "mean_rankic": mean_rankic,
        "threshold": float(STAGE_S_AUC_SKILL),
        "null_verdict": null_gate.get("verdict"),
        "null_passed": bool(null_gate.get("passed")),
        "n_pdauc_days": int(pdauc_series.notna().sum()) if len(pdauc_series) else 0,
    }
    print(
        f"[HB] STAGE-S skill={has_skill} mean_pdauc={mean_pdauc} rankic={mean_rankic} "
        f"null={null_gate.get('verdict')}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage T — frozen regime gate, then books
    # ------------------------------------------------------------------
    print("[HB] Stage T regime gate (fixed)...", flush=True)
    ratio = ew_top50_btc_ratio(cleaned, pit50, btc_id)
    breadth = breadth_top100(cleaned, pit100)
    regime = regime_on_off(ratio, breadth)
    gate_on = regime["gate_on"]

    print("[HB] MODEL-V2 gated books...", flush=True)
    books = {h: [] for h in PHASE2_HORIZONS}
    for h in PHASE2_HORIZONS:
        preds = all_preds[h]
        if preds is None or preds.empty:
            continue
        for p_enter in P_ENTER_GRID:
            packed = run_gated_book(
                cleaned,
                pit50,
                preds,
                feat,
                btc_id,
                p_enter=float(p_enter),
                h=int(h),
                gate_on=gate_on,
                budget=float(REGIME_BUDGET),
            )
            books[h].append(packed)
            print(
                f"[HB] gated h={h} p={p_enter} rel={packed.get('rel_sharpe')} "
                f"tot={packed.get('book_total')} wbtc={packed.get('avg_w_btc')} "
                f"on={packed.get('gate_on_frac')}",
                flush=True,
            )

    head14 = pick_median_p_enter(books.get(PHASE2_PRIMARY_H) or [])
    if not isinstance(head14, dict) or head14.get("error") or not head14.get("start"):
        raise RuntimeError(f"headline book failed: {head14}")
    oos_start = pd.Timestamp(head14["start"], tz="UTC")
    print(f"[HB] naive v4 same-OOS start={oos_start.date()}", flush=True)
    naive_oos = naive_rotation_v3(cleaned, pit50, oos_start)
    btc_ref = _btc_ref(naive_oos)

    v14 = mechanical_verdicts(head14, naive_oos)
    if not gates_ok:
        v14 = dict(v14)
        v14["viable"] = False
        v14["replaces_floor"] = False
        v14["gates_blocked"] = True

    imps = mean_gain(all_metas.get(PHASE2_PRIMARY_H) or [], top_n=15)
    ctx_in_imp = [a for a, _ in imps if a in set(CTX_COLS)]
    if ctx_in_imp:
        raise RuntimeError(f"context features in Stage-S importances: {ctx_in_imp}")

    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "gates_ok": gates_ok,
        "n_features": len(STAGE_S_COLS),
        "n_train_rows": int(len(labeled)),
        "usable_from": USABLE_FROM,
        "btc_id": int(btc_id),
        "primary_h": PHASE2_PRIMARY_H,
        "feature_cols": list(STAGE_S_COLS),
        "naive_oos": naive_oos,
        "old50_rows": old50_sz["n_rows"],
        "old50_ids": old50_sz["n_ids"],
        "old50_med": old50_sz["median_per_date"],
        "old100_rows": old100_sz["n_rows"],
        "old100_ids": old100_sz["n_ids"],
        "old100_med": old100_sz["median_per_date"],
        "gate_on_frac_full": float(gate_on.mean()) if len(gate_on) else float("nan"),
    }

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_phase2b(
        rep_dir / "btcb_phase2b_report.md",
        autopsy=autopsy,
        clean_summary=clean_summary,
        floor50=floor50,
        floor100=floor100,
        naive=naive_full,
        naive_contrib=naive_full_tab,
        gates=cheap + [ctx_gate, seed_gate],
        null_gate=null_gate,
        skill=skill,
        headline=head14,
        grid=books.get(PHASE2_PRIMARY_H) or [],
        btc_ref=btc_ref,
        verdicts=v14,
        metas={str(h): all_metas[h] for h in all_metas},
        importances=imps,
        extra=extra,
    )
    plot_equity_gate(head14, naive_oos, gate_on, chart_dir / "btcb_p2b_equity.png")
    plot_calibration(all_metas.get(PHASE2_PRIMARY_H) or [], chart_dir / "btcb_p2b_calibration.png")
    plot_pdauc_series(pdauc_series, chart_dir / "btcb_p2b_pdauc.png")

    payload = {
        "criterion": PHASE2B_CRITERION,
        "death_convention": DEATH_CONVENTION,
        "gates_ok": gates_ok,
        "gates": _jsonable(cheap + [ctx_gate, seed_gate]),
        "null_gate": _jsonable(null_gate),
        "skill": _jsonable(skill),
        "verdicts": _jsonable(v14),
        "headline": _jsonable(head14),
        "grid_h14": [_jsonable(x) for x in (books.get(PHASE2_PRIMARY_H) or [])],
        "grid_h30": [_jsonable(x) for x in (books.get(30) or [])],
        "naive_v4_full": _jsonable(naive_full),
        "naive_v4_oos": _jsonable(naive_oos),
        "naive_v4_contrib": _jsonable(naive_full_tab),
        "autopsy": _jsonable(autopsy),
        "clean_summary": _jsonable({k: v for k, v in clean_summary.items() if k != "log"}),
        "splice_log": _jsonable(clog),
        "floor50": _jsonable(floor50),
        "floor100": _jsonable(floor100),
        "old_pit50": old50_sz,
        "old_pit100": old100_sz,
        "fold_meta": {str(h): _jsonable(all_metas[h]) for h in all_metas},
        "importances": [{"feature": a, "mean_gain": b} for a, b in imps],
        "extra": _jsonable({k: v for k, v in extra.items() if k != "naive_oos"}),
        "gpu_used": False,
    }
    (rep_dir / "btcb_phase2b_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (rep_dir / "btcb_phase2b_done.txt").write_text(
        json.dumps({"elapsed_sec": time.time() - t0, "gpu_used": False}, indent=2)
    )
    commit()

    viable_s = "VIABLE" if (v14.get("viable") and gates_ok) else "NOT VIABLE"
    repl_s = "REPLACES-FLOOR" if (v14.get("replaces_floor") and gates_ok) else "DOES-NOT-REPLACE-FLOOR"
    skill_s = "HAS-SELECTION-SKILL" if has_skill else "NO-SELECTION-SKILL"
    top_share = float(naive_full_tab.get("top_alt_share") or 0.0)
    print(f"NAIVE-V4 rel-Sharpe: {naive_full.get('rel_sharpe')}", flush=True)
    print(f"NAIVE-V4 live-benchmark: {naive_full.get('live_benchmark')}", flush=True)
    print(f"STAGE-S: {skill_s} mean-per-date-AUC={mean_pdauc}", flush=True)
    print(f"VERDICT: MODEL-V2 {viable_s}", flush=True)
    print(f"VERDICT: {repl_s}", flush=True)
    print(f"% time in BTC: {head14.get('avg_w_btc')}", flush=True)
    print(f"top naive-v4 contributor share: {top_share}", flush=True)
    print("COMBO untouched (v2.0-combo-final).", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "naive_v4_rel_sharpe": naive_full.get("rel_sharpe"),
        "naive_v4_live": bool(naive_full.get("live_benchmark")),
        "stage_s_skill": has_skill,
        "mean_pdauc": mean_pdauc,
        "viable": bool(v14.get("viable") and gates_ok),
        "replaces_floor": bool(v14.get("replaces_floor") and gates_ok),
        "gates_ok": gates_ok,
        "null_verdict": null_gate.get("verdict"),
        "avg_w_btc": head14.get("avg_w_btc"),
        "top_naive_v4_share": top_share,
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
    }


@app.local_entrypoint()
def main():
    print("[local] starting BTC-BEATER P2b (spawn, then wait)...", flush=True)
    fc = run_btcb_p2b.spawn()
    print(f"[local] spawned {getattr(fc, 'object_id', fc)}", flush=True)
    summary = fc.get()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    Path("universe").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase2b_report.md", "reports"),
        ("reports/btcb_phase2b_report.json", "reports"),
        ("charts/btcb_p2b_equity.png", "charts"),
        ("charts/btcb_p2b_calibration.png", "charts"),
        ("charts/btcb_p2b_pdauc.png", "charts"),
        ("btcb/universe/btcb_top50_floor.parquet", "universe"),
        ("btcb/universe/btcb_top100_floor.parquet", "universe"),
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
        for src in (art / "reports").glob("btcb_phase2b*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        for src in (art / "charts").glob("btcb_p2b*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] BTC-BEATER P2b complete.", flush=True)
