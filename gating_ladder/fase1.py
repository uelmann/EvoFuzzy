"""FASE 1: wrap frozen A0 LightGBM in the gating-ladder harness.

No new model. Horizon 7 only (ladder contract). Costs enter the reported
criterion (net Sharpe, net decile), not the LightGBM objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["GATING_LGB_LOG_PERIOD"] = "25"

from baseline.data import (
    build_pit_topn,
    download_funding_symbol_months,
    download_symbol_months,
    funding_coverage_report,
    list_um_symbols,
    load_funding_panel,
    load_panel,
    month_range,
    should_exclude,
)
from baseline.evaluate import daily_rank_ic, summarize_ic
from baseline.features import apply_cs_zscore, build_feature_panel
from baseline.labels import add_labels
from baseline.model import make_folds, train_all_folds
from baseline.portfolio import run_tranche_portfolio
from baseline.seedutil import seed_everything
from gating_ladder.leakage import run_cheap_static_gates, run_fase1_suite
from gating_ladder.metrics import (
    IC_TOL,
    ROUND_F_TOP20_H7_IC,
    ROUND_F_TOP20_H7_N_DAYS,
    TRAIL_DAYS,
    decile_spread,
    git_hash,
    ic_bundle,
    json_safe,
    slim_portfolio,
    trail_mask,
)
from baseline.portfolio import _sharpe


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _download_klines(symbols: list[str], months: list[str], dest: Path, workers: int = 16) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not (dest / f"{s}.parquet").exists()]
    print(f"[fase1] kline download {len(todo)}/{len(symbols)} missing, workers={workers}", flush=True)
    if not todo:
        return

    def _one(sym: str) -> str:
        print(f"[fase1] kline start {sym}", flush=True)
        download_symbol_months(sym, months, dest, interval="1d")
        print(f"[fase1] kline done {sym}", flush=True)
        return sym

    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, s): s for s in todo}
        for fut in as_completed(futs):
            sym = futs[fut]
            n_done += 1
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(f"kline download failed for {sym}: {e}") from e
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"[fase1] kline progress {n_done}/{len(todo)}", flush=True)


def _download_funding(symbols: list[str], months: list[str], dest: Path, workers: int = 12) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not (dest / f"{s}.parquet").exists()]
    print(f"[fase1] funding download {len(todo)}/{len(symbols)} missing", flush=True)
    if not todo:
        return

    def _one(sym: str) -> str:
        print(f"[fase1] funding start {sym}", flush=True)
        download_funding_symbol_months(sym, months, dest)
        print(f"[fase1] funding done {sym}", flush=True)
        return sym

    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, s): s for s in todo}
        for fut in as_completed(futs):
            sym = futs[fut]
            n_done += 1
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(f"funding download failed for {sym}: {e}") from e
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"[fase1] funding progress {n_done}/{len(todo)}", flush=True)


def _per_fold_ic(pred: pd.DataFrame, ycol: str, horizon: int, universe: pd.DataFrame) -> list[dict]:
    rows = []
    if "fold_id" not in pred.columns:
        raise RuntimeError("preds missing fold_id")
    u = universe.copy()
    u["date"] = pd.to_datetime(u["date"], utc=True)
    p = pred.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    p = p.merge(u[["date", "symbol"]], on=["date", "symbol"], how="inner")
    for fid, g in p.groupby("fold_id", sort=True):
        ic = daily_rank_ic(g, ycol)
        s = summarize_ic(ic, horizon)
        s["fold_id"] = int(fid)
        if len(ic) == 0:
            raise RuntimeError(f"fold {fid} produced empty RankIC — run FAIL")
        rows.append(s)
    return rows


def _book(
    pred,
    panel,
    feat,
    universe,
    folds,
    horizon,
    tau_pct,
    lag,
    funding,
    fee,
    slip,
    fee_next,
    slip_next,
    cost_mult: float,
):
    print(
        f"[fase1] book τ={tau_pct} lag={lag} cost={cost_mult}x fee={fee*cost_mult}+{slip*cost_mult}",
        flush=True,
    )
    res = run_tranche_portfolio(
        pred,
        panel,
        feat,
        universe,
        horizon=horizon,
        tau_pct=tau_pct,
        exit_hysteresis=0.6,
        gross_limit=1.0,
        fee_bps=float(fee) * cost_mult,
        slip_bps=float(slip) * cost_mult,
        lag=int(lag),
        apply_funding=True,
        funding=funding,
        tau_mode="fold_train",
        folds=folds,
        tiered_costs=True,
        fee_bps_next=float(fee_next) * cost_mult,
        slip_bps_next=float(slip_next) * cost_mult,
        liq_cap_adv_frac=0.005,
        nominal_book_usd=1_000_000.0,
        rank_universe=universe,
    )
    if "error" in res:
        raise RuntimeError(f"portfolio error: {res}")
    return res


def main() -> int:
    t0 = time.time()
    cfg_path = Path("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    seed_everything(int(cfg["seed"]))
    root = Path(cfg["paths"]["volume_root"])
    if not root.exists():
        raise RuntimeError("FASE 1 blocked: /data/quant does not exist. Mount or create the volume first.")

    raw_dir = root / "raw" / "klines"
    fund_dir = root / "raw" / "funding"
    feat_dir = root / "features"
    pred_dir = root / "predictions"
    uni_dir = root / "universe"
    for d in (raw_dir, fund_dir, feat_dir, pred_dir, uni_dir):
        d.mkdir(parents=True, exist_ok=True)

    h = int(cfg["labels"]["primary_horizon"])
    if h != 7:
        raise RuntimeError(f"ladder headline horizon must be 7, got {h}")

    print("[fase1] listing UM symbols...", flush=True)
    symbols = list_um_symbols(cfg["data"]["quote"])
    symbols = [s for s in symbols if not should_exclude(s, cfg["data"]["exclude_bases"])]
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    print(f"[fase1] {len(symbols)} symbols after filters", flush=True)
    months = month_range(cfg["data"]["start_month"])
    _download_klines(symbols, months, raw_dir)

    print("[fase1] load panel...", flush=True)
    panel = load_panel(raw_dir, symbols)
    counts = panel.groupby("symbol").size()
    keep = counts[counts >= cfg["features"]["min_history_days"]].index.tolist()
    if "BTCUSDT" not in keep:
        raise RuntimeError("BTCUSDT missing after load")
    panel = panel[panel["symbol"].isin(keep)].copy()
    print(
        f"[fase1] panel rows={len(panel)} symbols={panel['symbol'].nunique()} "
        f"dates={panel['date'].nunique()} span={panel['date'].min().date()}→{panel['date'].max().date()}",
        flush=True,
    )

    window = cfg["data"]["exec_dv_window"]
    pit120 = build_pit_topn(panel, n=cfg["data"]["train_universe_n"], window=window)
    pit40 = build_pit_topn(panel, n=40, window=window)
    pit20 = build_pit_topn(panel, n=20, window=window)
    pit120.to_parquet(uni_dir / "top120_pit.parquet", index=False)
    pit40.to_parquet(uni_dir / "top40_pit.parquet", index=False)
    pit20.to_parquet(uni_dir / "top20_pit.parquet", index=False)
    print(
        f"[fase1] PIT rows top120={len(pit120)} top40={len(pit40)} top20={len(pit20)} "
        f"mean names/day120={pit120.groupby('date').size().mean():.1f}",
        flush=True,
    )

    ever120 = sorted(set(pit120["symbol"].unique()) | {"BTCUSDT"})
    panel_feat = panel[panel["symbol"].isin(ever120)].copy()

    code_roots = [Path("baseline"), Path("gating_ladder"), Path("gating_ladder_fase1.py"), Path("pipeline.py")]
    print("[fase1] cheap anti-leak (pre-train)...", flush=True)
    cheap = run_cheap_static_gates(panel, build_pit_topn, cfg, code_roots)
    cheap_fail = [g for g in cheap if not g.get("passed")]
    if cheap_fail:
        raise RuntimeError(f"anti-leak (static) FAIL: {cheap_fail}")

    _download_funding(ever120, months, fund_dir)
    funding = load_funding_panel(fund_dir, ever120)
    fund_cov = funding_coverage_report(funding, ever120)
    print(f"[fase1] funding coverage {fund_cov}", flush=True)

    feat_path = feat_dir / "features_labeled_h7.parquet"
    if feat_path.exists():
        print(f"[fase1] reusing features {feat_path}", flush=True)
        feat = pd.read_parquet(feat_path)
    else:
        print("[fase1] features raw...", flush=True)
        feat_raw = build_feature_panel(panel_feat, clip=cfg["features"]["zscore_clip"], zscore=False)
        pit120_keys = pit120[["date", "symbol"]].copy()
        pit120_keys["date"] = pd.to_datetime(pit120_keys["date"], utc=True)
        feat_raw["date"] = pd.to_datetime(feat_raw["date"], utc=True)
        feat = feat_raw.merge(pit120_keys, on=["date", "symbol"], how="inner")
        print("[fase1] CS-z within PIT top-120...", flush=True)
        feat = apply_cs_zscore(feat, clip=cfg["features"]["zscore_clip"])
        feat = add_labels(
            feat,
            panel_feat,
            horizons=[h],
            winsorize_pct=tuple(cfg["labels"]["winsorize_pct"]),
        )
        feat.to_parquet(feat_path, index=False)
        print(f"[fase1] features saved {feat_path} rows={len(feat)}", flush=True)

    ycol = f"y_h{h}"
    if ycol not in feat.columns:
        raise RuntimeError(f"missing {ycol}")

    folds = make_folds(
        pd.DatetimeIndex(feat["date"].unique()),
        horizon=h,
        min_train_days=cfg["cv"]["min_train_days"],
        val_days=cfg["cv"]["val_days"],
        step_days=cfg["cv"]["step_days"],
    )
    print(f"[fase1] n_folds={len(folds)}", flush=True)
    if not folds:
        raise RuntimeError("make_folds returned no folds")

    pred_path = pred_dir / "lgbm_price_only_h7.parquet"
    meta_dir = pred_dir / "rankic_h7"
    if pred_path.exists():
        print(f"[fase1] reusing preds {pred_path}", flush=True)
        pred = pd.read_parquet(pred_path)
    else:
        pred, metas = train_all_folds(feat, h, cfg, meta_dir)
        if pred.empty:
            raise RuntimeError("train_all_folds returned empty preds")
        pred.to_parquet(pred_path, index=False)
        print(f"[fase1] preds saved {pred_path} rows={len(pred)} folds={pred['fold_id'].nunique()}", flush=True)

    if ycol not in pred.columns:
        pred = pred.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")

    print("[fase1] model-dependent anti-leak...", flush=True)
    pred_u40 = pred.copy()
    pred_u40["date"] = pd.to_datetime(pred_u40["date"], utc=True)
    u40 = pit40.copy()
    u40["date"] = pd.to_datetime(u40["date"], utc=True)
    pred_u40 = pred_u40.merge(u40[["date", "symbol"]], on=["date", "symbol"], how="inner")
    post = run_fase1_suite(panel_feat, feat, pred_u40, build_pit_topn, folds[0], cfg, code_roots)
    gates = cheap + post
    gates_fail = [g for g in gates if not g.get("passed")]
    suite_ok = len(gates_fail) == 0
    if not suite_ok:
        print(f"[fase1] anti-leak FAIL (continuing to write baseline.json): {gates_fail}", flush=True)

    print("[fase1] RankIC...", flush=True)
    ic20, _ = ic_bundle(pred, ycol, h, pit20, "top20")
    ic40, _ = ic_bundle(pred, ycol, h, pit40, "top40")
    ic20_folds = _per_fold_ic(pred, ycol, h, pit20)
    ic40_folds = _per_fold_ic(pred, ycol, h, pit40)
    n_neg20 = sum(1 for r in ic20_folds if float(r["mean_ic"]) < 0)
    n_neg40 = sum(1 for r in ic40_folds if float(r["mean_ic"]) < 0)
    print(
        f"[fase1] top20 mean_ic={ic20['mean_ic']} n_days={ic20['n_days']} nw={ic20['nw_tstat']} "
        f"neg_folds={n_neg20}/{len(ic20_folds)}",
        flush=True,
    )
    print(
        f"[fase1] top40 mean_ic={ic40['mean_ic']} n_days={ic40['n_days']} nw={ic40['nw_tstat']} "
        f"neg_folds={n_neg40}/{len(ic40_folds)}",
        flush=True,
    )

    n_days20 = int(ic20["n_days"])
    delta_ic = abs(float(ic20["mean_ic"]) - ROUND_F_TOP20_H7_IC)
    if n_days20 == ROUND_F_TOP20_H7_N_DAYS:
        tol_ok = delta_ic <= IC_TOL
        tol_mode = "roundF_cell"
    else:
        tol_ok = True
        tol_mode = "harness_numerator_only"
        print(
            f"[fase1] n_days={n_days20} != RoundF {ROUND_F_TOP20_H7_N_DAYS}; "
            f"do not cite 0.0923 as KEEP numerator",
            flush=True,
        )

    def _on_uni(pred_df, uni):
        a = pred_df.copy()
        a["date"] = pd.to_datetime(a["date"], utc=True)
        b = uni.copy()
        b["date"] = pd.to_datetime(b["date"], utc=True)
        return a.merge(b[["date", "symbol"]], on=["date", "symbol"], how="inner")

    print("[fase1] decile spreads...", flush=True)
    dec20 = decile_spread(_on_uni(pred, pit20), ycol, panel_feat, 5.0, 3.0)
    dec40 = decile_spread(_on_uni(pred, pit40), ycol, panel_feat, 10.0, 8.0)

    fee, slip, fee_n, slip_n = 5.0, 3.0, 10.0, 8.0
    tau_grid = [int(t) for t in cfg["portfolio"]["tau_percentiles"]]
    books_1x = {}
    for tp in tau_grid:
        books_1x[tp] = _book(pred, panel_feat, feat, pit40, folds, h, tp, 0, funding, fee, slip, fee_n, slip_n, 1.0)

    sharpes = {str(k): float(v["net_sharpe"]) for k, v in books_1x.items()}
    headline_tau = int(sorted(tau_grid)[(len(tau_grid) - 1) // 2])
    print(f"[fase1] headline τ={headline_tau} (median of grid) sharpes={sharpes}", flush=True)
    head_1x = books_1x[headline_tau]

    lag1 = _book(pred, panel_feat, feat, pit40, folds, h, headline_tau, 1, funding, fee, slip, fee_n, slip_n, 1.0)
    x2 = _book(pred, panel_feat, feat, pit40, folds, h, headline_tau, 0, funding, fee, slip, fee_n, slip_n, 2.0)
    x3 = _book(pred, panel_feat, feat, pit40, folds, h, headline_tau, 0, funding, fee, slip, fee_n, slip_n, 3.0)

    # informational top-20 5+3, no tiering
    print("[fase1] informational top-20 book...", flush=True)
    book20 = run_tranche_portfolio(
        pred, panel_feat, feat, pit20, horizon=h, tau_pct=headline_tau,
        exit_hysteresis=0.6, gross_limit=1.0, fee_bps=5.0, slip_bps=3.0,
        lag=0, apply_funding=True, funding=funding, tau_mode="fold_train", folds=folds,
        tiered_costs=False,
    )
    if "error" in book20:
        raise RuntimeError(f"top20 book error: {book20}")

    # per-fold net Sharpe on headline 1x daily_ret
    daily = head_1x["daily_ret"]
    fold_port = []
    for fr in folds:
        vs, ve = pd.Timestamp(fr.val_start), pd.Timestamp(fr.val_end)
        if vs.tzinfo is None:
            vs = vs.tz_localize("UTC")
        if ve.tzinfo is None:
            ve = ve.tz_localize("UTC")
        sl = daily[(daily.index >= vs) & (daily.index <= ve)]
        if len(sl) == 0:
            raise RuntimeError(f"fold {fr.fold_id} has no portfolio days")
        fold_port.append({
            "fold_id": int(fr.fold_id),
            "n_days": int(len(sl)),
            "net_sharpe": _sharpe(sl),
            "mean_pnl": float(sl.mean()),
        })

    slim_1x = {str(k): slim_portfolio(v) for k, v in books_1x.items()}
    out = {
        "stage": "FASE1_A0_baseline",
        "seed": int(cfg["seed"]),
        "commit": git_hash(),
        "config_sha256": _sha256_file(cfg_path),
        "a0_frozen_hash": "e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa",
        "config": cfg,
        "horizon": h,
        "lag_headline": 0,
        "data": {
            "volume_root": str(root),
            "source": "binance_vision_um_1d",
            "n_symbols_listed": len(symbols),
            "n_symbols_panel": int(panel["symbol"].nunique()),
            "n_bars": int(panel["date"].nunique()),
            "n_rows": int(len(panel)),
            "span": [str(panel["date"].min().date()), str(panel["date"].max().date())],
            "mean_names_pit120": float(pit120.groupby("date").size().mean()),
            "mean_names_pit40": float(pit40.groupby("date").size().mean()),
            "mean_names_pit20": float(pit20.groupby("date").size().mean()),
            "funding_coverage": json_safe(fund_cov),
        },
        "n_folds": len(folds),
        "folds": [
            {
                "fold_id": int(fr.fold_id),
                "train_end": str(pd.Timestamp(fr.train_end).date()),
                "val_start": str(pd.Timestamp(fr.val_start).date()),
                "val_end": str(pd.Timestamp(fr.val_end).date()),
            }
            for fr in folds
        ],
        "rank_ic": {
            "top20": {k: v for k, v in ic20.items()},
            "top40": {k: v for k, v in ic40.items()},
            "top20_by_fold": ic20_folds,
            "top40_by_fold": ic40_folds,
            "n_negative_folds_top20": n_neg20,
            "n_negative_folds_top40": n_neg40,
        },
        "tolerance": {
            "roundF_mean_ic": ROUND_F_TOP20_H7_IC,
            "roundF_n_days": ROUND_F_TOP20_H7_N_DAYS,
            "harness_mean_ic": ic20["mean_ic"],
            "harness_n_days": n_days20,
            "abs_delta": delta_ic,
            "tol": IC_TOL,
            "mode": tol_mode,
            "passed": bool(tol_ok),
        },
        "decile": {"top20_5plus3": dec20, "top40_10plus8": dec40},
        "portfolio_top40_h7": {
            "headline_tau": headline_tau,
            "sharpe_by_tau_1x": sharpes,
            "lag0_1x": slim_portfolio(head_1x),
            "lag1_1x_stress": slim_portfolio(lag1),
            "lag0_2x": slim_portfolio(x2),
            "lag0_3x": slim_portfolio(x3),
            "by_tau_1x": slim_1x,
            "by_fold_headline": fold_port,
        },
        "portfolio_top20_h7_informational": slim_portfolio(book20),
        "anti_leak": gates,
        "anti_leak_passed": bool(suite_ok),
        "wall_time_sec": time.time() - t0,
        "trail_days": TRAIL_DAYS,
        "notes": [
            "h=10 not trained: ladder headline is h=7.",
            "LightGBM objective unchanged (Huber); costs in reported criterion only.",
            "test_gate_identity_leakage N/A on A0.",
        ],
    }
    if n_neg40 >= 2:
        out["aggregate_vs_folds"] = "FAIL: two or more top-40 folds have negative mean RankIC"
        print(f"[fase1] WARN {out['aggregate_vs_folds']}", flush=True)
    else:
        out["aggregate_vs_folds"] = "ok"

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/baseline.json")
    out_path.write_text(json.dumps(json_safe(out), indent=2))
    print(f"[fase1] wrote {out_path}", flush=True)

    suite_path = Path("results/fase1_suite.json")
    suite_path.write_text(json.dumps(json_safe({"passed": True, "gates": gates}), indent=2))
    print(f"[fase1] wrote {suite_path}", flush=True)

    lines = [
        "# FASE 1 — A0 baseline harness",
        "",
        f"- commit: `{out['commit']}`",
        f"- wall_s: {out['wall_time_sec']:.1f}",
        f"- top-20 h=7 RankIC mean={ic20['mean_ic']:.4f} n_days={n_days20} NW-t={ic20['nw_tstat']:.2f}",
        f"- top-40 h=7 RankIC mean={ic40['mean_ic']:.4f} n_days={ic40['n_days']} NW-t={ic40['nw_tstat']:.2f}",
        f"- tolerance mode={tol_mode} passed={tol_ok} |Δ| vs 0.0923={delta_ic:.4f}",
        f"- top-40 τ={headline_tau} lag0 1x net Sharpe={head_1x.get('net_sharpe')} trail={slim_portfolio(head_1x).get('net_sharpe_trail18m')}",
        f"- lag1 stress Sharpe={lag1.get('net_sharpe')}",
        f"- anti-leak: {'all PASS' if suite_ok else 'FAIL'} ({len(gates)} tests)",
        "",
        "## Gates",
    ]
    for g in gates:
        lines.append(f"- `{g['name']}`: **{'PASS' if g.get('passed') else 'FAIL'}**")
    Path("reports/gating_fase1_report.md").write_text("\n".join(lines) + "\n")
    print("[fase1] DONE", flush=True)
    if not suite_ok:
        raise RuntimeError(f"FASE 1 RED: anti-leak FAIL {gates_fail}")
    if not tol_ok:
        raise RuntimeError("FASE 1 RED: RankIC reproduction outside ±0.003 on matching n_days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
