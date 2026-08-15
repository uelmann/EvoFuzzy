"""
BTC-BEATER Phase 3.c — Binance replay of SPREAD-LS.

BACKTEST ONLY. Same 2.c positions; Binance prices + native funding.
Replaces Phase 3.b. No MASTER book. CPU only. Zero GPU.
Usage: modal run --detach btcb_phase3c_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p3c"
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
        "reports/btcb_phase3c_addendum.md",
        remote_path="/root/btcb_phase3c_addendum.md",
    )
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("universe/btcb_top100_floor.parquet", remote_path="/root/btcb_top100_floor.parquet")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
)

app = modal.App(APP_NAME, image=image)

CMC_PANEL = Path("/data/quant/btcb/full/panel.parquet")
CMC_PRED = Path("/data/quant/btcb/phase2c/preds")
CMC_FEAT = Path("/data/quant/btcb/phase2b/feat_s.parquet")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StageHeartbeat:
    """60s heartbeat; loud warning if a stage is silent for 20 minutes."""

    def __init__(self, stage: str):
        self.stage = stage
        self.t0 = time.time()
        self.last = time.time()
        self.stop = threading.Event()
        self.warned = False
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
            if silent >= 20 * 60 and not self.warned:
                print(
                    f"[WARN] STAGE {self.stage} EXCEEDED 20 MIN WITHOUT LOG PROGRESS "
                    f"silent={silent:.0f}s — continuing",
                    flush=True,
                )
                self.warned = True

    def close(self) -> None:
        self.stop.set()
        print(f"[HB] STAGE END {self.stage} elapsed={time.time() - self.t0:.0f}s", flush=True)


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
        "daily_funding",
        "position_log",
        "ls_daily",
        "combo_daily",
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
    timeout=60 * 30,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=2,
    memory=4096,
    max_containers=40,
)
def download_one_spot(item: dict) -> dict:
    from baseline.data import download_spot_symbol_months, month_range

    symbol = item["symbol"]
    dest = Path("/data/quant/raw/spot_klines")
    dest.mkdir(parents=True, exist_ok=True)
    pq = dest / f"{symbol}.parquet"
    reused = pq.exists()
    t0 = time.time()
    try:
        path = download_spot_symbol_months(symbol, month_range(item["start_month"]), dest)
        empty = True
        n = 0
        import pandas as pd

        df = pd.read_parquet(path)
        empty = bool(df.empty)
        n = int(len(df))
        rec = {
            "symbol": symbol,
            "path": str(path),
            "reused": bool(reused),
            "empty": bool(empty),
            "n_rows": n,
            "elapsed": time.time() - t0,
            "ok": True,
        }
    except Exception as e:
        rec = {
            "symbol": symbol,
            "path": str(pq),
            "reused": bool(reused),
            "empty": True,
            "n_rows": 0,
            "elapsed": time.time() - t0,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }
        print(f"[spot] {symbol} FAIL {rec['error']}", flush=True)
        quant_vol.commit()
        return rec
    quant_vol.commit()
    print(f"[spot] {symbol} reused={reused} empty={empty} n={n}", flush=True)
    return rec


@app.function(
    timeout=60 * 60 * 4,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def run_btcb_p3c() -> dict:
    import numpy as np
    import pandas as pd
    import yaml

    from baseline.data import list_spot_symbols, load_funding_panel, load_panel
    from baseline.seedutil import seed_everything
    from btcb.binance_replay import (
        build_id_symbol_map,
        close_wide_from_panel,
        coverage_tables,
        discrepancy_tables,
        funding_wide_from_panel,
        needed_spot_symbols,
        replay_three_books,
        symbols_for_id,
    )
    from btcb.constants import (
        COMBO_SPREADLS_CORR,
        DEATH_CONVENTION,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_BETA_MATCH_DESIGNATION,
        PHASE3C_HOUSE_RULE,
        PHASE3C_MASTER_NOTE,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_N_DAYS,
        PHASE3C_REF_SHARPE,
        PHASE3C_REF_START,
        PHASE3C_VALIDATION,
        SEED,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.phase3c_report import (
        plot_hybrid_equity,
        plot_pnl_scatter,
        update_numbers_ledger,
        write_phase3c,
    )
    from btcb.spread_ls import (
        build_shortable,
        ew_basket,
        hash_pred_dir,
        load_twin_from_cache,
        run_spread_ls,
        squeeze_table,
    )

    t0 = time.time()
    seed_everything(SEED)
    addendum = Path("/root/btcb_phase3c_addendum.md").read_text()
    for txt in (
        PHASE3C_BETA_MATCH_DESIGNATION,
        PHASE3C_HOUSE_RULE,
        PHASE3C_MASTER_NOTE,
        PHASE3C_VALIDATION,
        DEATH_CONVENTION,
    ):
        if txt not in addendum:
            raise RuntimeError(f"Phase 3.c addendum missing freeze text: {txt[:80]}")
    print("[HB] BTC-BEATER P3c BACKTEST ONLY; Binance replay; zero GPU; no MASTER", flush=True)
    print(f"[HB] {PHASE3C_BETA_MATCH_DESIGNATION}", flush=True)
    print(f"[HB] {PHASE3C_HOUSE_RULE}", flush=True)
    print(f"[HB] {PHASE3C_MASTER_NOTE}", flush=True)
    print(f"[HB] {PHASE3C_VALIDATION}", flush=True)

    def commit():
        quant_vol.commit()

    if not CMC_PANEL.exists():
        raise RuntimeError(f"missing panel {CMC_PANEL}")
    cmc_panel_sha0 = _file_sha256(CMC_PANEL)
    cmc_feat_sha0 = _file_sha256(CMC_FEAT) if CMC_FEAT.exists() else None
    print(f"[HB] CMC READ-ONLY snapshot panel_sha256={cmc_panel_sha0} feat_sha256={cmc_feat_sha0}", flush=True)

    cfg = yaml.safe_load(Path("/root/config.yaml").read_text())
    start_month = str(cfg.get("data", {}).get("start_month") or "2019-09")

    panel_path = CMC_PANEL
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    print(f"[HB] btc_id={btc_id} panel_end_clip={end.date()} rows={len(panel)}", flush=True)

    def _load_pit() -> pd.DataFrame:
        for p in (
            Path("/data/quant/btcb/universe/btcb_top100_floor.parquet"),
            Path("/data/quant/universe/btcb_top100_floor.parquet"),
            Path("/root/btcb_top100_floor.parquet"),
        ):
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
                df["id"] = df["id"].astype(int)
                df = df[df["date"] <= end].copy()
                print(f"[HB] pit from {p} rows={len(df)}", flush=True)
                return df
        raise RuntimeError("missing floored PIT top-100")

    pit100 = _load_pit()
    print("[HB] re-applying frozen 2.b cleaner...", flush=True)
    cleaned, _clog = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()

    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if not feat_path.exists():
        raise RuntimeError(f"missing 2.b Stage-S features {feat_path}")
    feat = pd.read_parquet(feat_path)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat["id"] = feat["id"].astype(int)
    feat = feat[feat["date"] <= end].copy()

    pred_dir = Path("/data/quant/btcb/phase2c/preds")
    if not pred_dir.exists():
        raise RuntimeError(f"missing 2.c pred cache {pred_dir}")
    pred_hash = hash_pred_dir(pred_dir)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n_files={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(
            f"2.c cache hash mismatch got={pred_hash['sha256']} n={pred_hash['n_files']}"
        )

    twin = load_twin_from_cache(pred_dir, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()
    print(f"[HB] twin h={PHASE3C_REF_H} rows={len(twin)} dates={twin['date'].nunique()}", flush=True)

    raw_dir = Path("/data/quant/raw/klines")
    kline_syms = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    if not kline_syms:
        raise RuntimeError(f"no Binance UM klines in {raw_dir}")
    kline_panel = load_panel(raw_dir, kline_syms)
    kline_panel["date"] = pd.to_datetime(kline_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    kline_panel["symbol"] = kline_panel["symbol"].astype(str).str.upper()
    print(f"[HB] UM kline symbols={len(kline_syms)} rows={len(kline_panel)}", flush=True)
    shortable = build_shortable(cleaned, kline_panel, btc_id)

    fund_dir = Path("/data/quant/raw/funding")
    fund_syms = sorted(p.stem for p in fund_dir.glob("*.parquet")) if fund_dir.exists() else []
    funding = (
        load_funding_panel(fund_dir, fund_syms)
        if fund_syms
        else pd.DataFrame(columns=["date", "symbol", "funding_rate", "n_events"])
    )
    if not funding.empty:
        funding["date"] = pd.to_datetime(funding["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        funding["symbol"] = funding["symbol"].astype(str).str.upper()
    print(f"[HB] funding symbols={len(fund_syms)} rows={len(funding)}", flush=True)

    print("[HB] running β-matched h=14 engine (position log)...", flush=True)
    hb = StageHeartbeat("engine")
    packed = run_spread_ls(
        cleaned,
        pit100,
        twin,
        feat,
        shortable,
        btc_id,
        h=int(PHASE3C_REF_H),
        beta_matched=True,
        emit_position_log=True,
    )
    hb.close()
    if packed.get("error"):
        raise RuntimeError(f"engine failed: {packed}")
    if int(packed.get("btc_in_book_hits") or 0) != 0:
        raise RuntimeError("BTC leaked into SPREAD-LS")
    sh = float(packed.get("net_sharpe"))
    n_days = int(packed.get("n_days"))
    print(
        f"[HB] engine Sharpe={sh:.6f} trail={float(packed.get('net_sharpe_trail18m')):.6f} "
        f"n={n_days} start={packed.get('start')} end={packed.get('end')} "
        f"pos_sha={packed.get('position_sha256')}",
        flush=True,
    )
    if n_days != int(PHASE3C_REF_N_DAYS):
        raise RuntimeError(f"BOOK-CMC n_days={n_days} != ref {PHASE3C_REF_N_DAYS}")
    if packed.get("start") != PHASE3C_REF_START or packed.get("end") != PHASE3C_REF_END:
        raise RuntimeError(
            f"BOOK-CMC window {packed.get('start')}→{packed.get('end')} "
            f"!= {PHASE3C_REF_START}→{PHASE3C_REF_END}"
        )
    if abs(sh - float(PHASE3C_REF_SHARPE)) > 1e-6:
        raise RuntimeError(f"BOOK-CMC Sharpe {sh} != ref {PHASE3C_REF_SHARPE}")

    plog = packed["position_log"]
    out_p3c = Path("/data/quant/btcb/phase3c")
    out_p3c.mkdir(parents=True, exist_ok=True)
    plog.to_parquet(out_p3c / "position_log.parquet", index=False)
    commit()
    print(f"[HB] position log rows={len(plog)} unique_ids={plog['id'].nunique()}", flush=True)

    print("[HB] listing Binance spot USDT symbols...", flush=True)
    try:
        listed_spot = set(s.upper() for s in list_spot_symbols("USDT"))
        print(f"[HB] vision spot USDT symbols={len(listed_spot)}", flush=True)
    except Exception as e:
        listed_spot = None
        print(f"[HB] list_spot_symbols failed ({e}); will attempt CMC candidates", flush=True)

    need = needed_spot_symbols(plog, cleaned, listed_spot or set())
    if listed_spot is None:
        wanted = []
        seen = set()
        for iid in need["long_ids"]:
            for c in symbols_for_id(cleaned, int(iid)):
                if c not in seen:
                    seen.add(c)
                    wanted.append(c)
        need["symbols"] = wanted
    spot_dir = Path("/data/quant/raw/spot_klines")
    spot_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    reused = []
    for sym in need["symbols"]:
        if (spot_dir / f"{sym}.parquet").exists():
            reused.append(sym)
        else:
            todo.append({"symbol": sym, "start_month": start_month})
    print(
        f"[HB] spot needed={len(need['symbols'])} reused={len(reused)} download={len(todo)} "
        f"(idempotent: skip existing parquets)",
        flush=True,
    )
    dl_log = []
    if todo:
        hb_dl = StageHeartbeat("spot_download")
        for i in range(0, len(todo), 80):
            part = todo[i : i + 80]
            hb_dl.ping(f"batch {i//80 + 1} n={len(part)}")
            dl_log.extend(list(download_one_spot.map(part, order_outputs=False)))
            quant_vol.reload()
        hb_dl.close()
    else:
        print("[HB] spot download skipped — all needed parquets already cached", flush=True)
    quant_vol.reload()
    n_downloaded = int(sum(1 for r in dl_log if not r.get("reused")))
    n_empty = int(sum(1 for r in dl_log if r.get("empty")))
    print(f"[HB] spot download done new={n_downloaded} empty_markers={n_empty}", flush=True)
    (out_p3c / "spot_download_log.json").write_text(json.dumps(dl_log, indent=2, default=str))
    commit()

    spot_syms = sorted({p.stem.upper() for p in spot_dir.glob("*.parquet")})
    try:
        spot_panel = (
            load_panel(spot_dir, spot_syms)
            if spot_syms
            else pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "quote_volume", "symbol"]
            )
        )
    except RuntimeError:
        spot_panel = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "quote_volume", "symbol"]
        )
    if not spot_panel.empty:
        spot_panel["date"] = pd.to_datetime(spot_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        spot_panel["symbol"] = spot_panel["symbol"].astype(str).str.upper()
    n_spot_sym = int(spot_panel["symbol"].nunique()) if len(spot_panel) else 0
    print(f"[HB] spot panel symbols={n_spot_sym} rows={len(spot_panel)}", flush=True)

    all_ids = sorted(set(int(i) for i in plog["id"].unique()))
    nonempty_spot = set(spot_panel["symbol"].unique()) if len(spot_panel) else set()
    nonempty_perp = set(kline_panel["symbol"].unique())
    id_to_spot = build_id_symbol_map(all_ids, cleaned, set(spot_syms), nonempty_spot)
    id_to_perp = build_id_symbol_map(all_ids, cleaned, nonempty_perp, nonempty_perp)
    n_spot_mapped = sum(1 for v in id_to_spot.values() if v)
    n_perp_mapped = sum(1 for v in id_to_perp.values() if v)
    print(f"[HB] mapped spot={n_spot_mapped}/{len(all_ids)} perp={n_perp_mapped}/{len(all_ids)}", flush=True)

    cmc_close = cleaned.pivot(index="date", columns="id", values="close").sort_index()
    cmc_close.index = pd.to_datetime(cmc_close.index, utc=True).tz_convert("UTC").normalize()
    spot_wide = close_wide_from_panel(spot_panel, id_to_spot)
    perp_wide = close_wide_from_panel(kline_panel, id_to_perp)
    fund_wide = funding_wide_from_panel(funding, id_to_perp)
    print(
        f"[HB] wides cmc={cmc_close.shape} spot={spot_wide.shape} perp={perp_wide.shape} fund={fund_wide.shape}",
        flush=True,
    )

    print("[HB] replaying three books...", flush=True)
    hb_rp = StageHeartbeat("replay")
    replayed = replay_three_books(plog, cmc_close, spot_wide, perp_wide, fund_wide, packed)
    hb_rp.close()
    if replayed["sanity"].get("position_sha256") != packed.get("position_sha256"):
        raise RuntimeError("position log sha256 mismatch across books")
    print(
        f"[HB] positions byte-identical across three books sha256={packed.get('position_sha256')}",
        flush=True,
    )
    val = replayed["validation"]
    print(
        f"[HB] validation corr={val.get('corr')} sh_bn={val.get('sharpe_binance_only')} "
        f"sh_cmc={val.get('sharpe_cmc_subset')} validated={val.get('validated')}",
        flush=True,
    )

    id_to_sym = packed.get("id_to_sym") or {}
    for r in replayed.get("top_disagreements") or []:
        r["symbol"] = id_to_sym.get(int(r["id"]))
    cov = coverage_tables(plog, spot_wide, perp_wide, id_to_spot, id_to_perp, id_to_sym)
    print(
        f"[HB] coverage long={cov['long'].get('pct_replayable')} short={cov['short'].get('pct_replayable')}",
        flush=True,
    )

    disc = None
    if not val.get("validated"):
        disc = discrepancy_tables(replayed["daily"], plog, pit100, spot_wide, perp_wide, cmc_close)

    members100 = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit100.groupby("date")["id"]
    }
    basket = ew_basket(cmc_close, members100, list(cmc_close.index))
    squeeze_cmc = squeeze_table(replayed["cmc_net"], basket)
    squeeze_hyb = squeeze_table(replayed["hybrid_net"], basket)

    for tag in ("cmc", "hybrid", "binance_only"):
        replayed[tag]["forced_exits"] = packed.get("forced_exits")
        replayed[tag]["forced_covers"] = packed.get("forced_covers")
        replayed[tag]["beta_matched"] = True
        replayed[tag]["horizon"] = int(PHASE3C_REF_H)

    books = {
        "cmc": replayed["cmc"],
        "hybrid": replayed["hybrid"],
        "binance_only": replayed["binance_only"],
        "cmc_subset": replayed["cmc_subset"],
    }

    extra = {
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "btc_id": int(btc_id),
        "btc_hits": int(packed.get("btc_in_book_hits") or 0),
        "pred_sha256": pred_hash["sha256"],
        "pred_n_files": pred_hash["n_files"],
        "position_sha256": packed.get("position_sha256"),
        "max_abs_daily_diff": replayed["sanity"]["max_abs_daily_diff"],
        "ref_sharpe": float(PHASE3C_REF_SHARPE),
        "ref_n_days": int(PHASE3C_REF_N_DAYS),
        "n_spot_downloaded": n_downloaded,
        "n_spot_reused": int(len(reused)),
        "n_spot_attempted": int(len(need["symbols"])),
        "n_spot_empty": n_empty,
        "funding_events": replayed.get("funding_events"),
        "missing_funding_name_days": replayed.get("missing_funding_name_days"),
        "hybrid_flagged_share": replayed.get("hybrid_flagged_share"),
        "combo_corr": float(COMBO_SPREADLS_CORR),
        "id_to_spot": {str(k): v for k, v in id_to_spot.items() if v},
        "id_to_perp": {str(k): v for k, v in id_to_perp.items() if v},
    }

    ledger_path = Path("/root/numbers_ledger.md")
    ledger_block = update_numbers_ledger(
        ledger_path,
        validated=bool(val.get("validated")),
        hybrid=books["hybrid"],
        cmc=books["cmc"],
        extra=extra,
    )
    extra["ledger_note"] = (
        "OFFICIAL = BOOK-HYBRID (funding-on)."
        if val.get("validated")
        else "OFFICIAL SPREAD-LS RECORD SUSPENDED."
    )

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_phase3c(
        rep_dir / "btcb_phase3c_binance_replay.md",
        coverage=cov,
        books=books,
        validation=val,
        extra=extra,
        squeeze_cmc=squeeze_cmc,
        squeeze_hyb=squeeze_hyb,
        top_disagreements=replayed.get("top_disagreements") or [],
        discrepancy=disc,
    )
    plot_hybrid_equity(books["hybrid"]["equity"], chart_dir / "btcb_phase3c_hybrid_equity.png")
    plot_pnl_scatter(
        replayed["cmc_sub_net"],
        replayed["bn_net"],
        chart_dir / "btcb_phase3c_pnl_scatter.png",
        corr=val.get("corr"),
    )
    (rep_dir / "numbers_ledger.md").write_text(ledger_path.read_text())

    payload = {
        "addenda": {
            "beta_match": PHASE3C_BETA_MATCH_DESIGNATION,
            "house_rule": PHASE3C_HOUSE_RULE,
            "master_note": PHASE3C_MASTER_NOTE,
            "validation": PHASE3C_VALIDATION,
        },
        "validation": _jsonable(val),
        "coverage": _jsonable(cov),
        "books": {k: _jsonable(v) for k, v in books.items()},
        "squeeze_cmc": _jsonable(squeeze_cmc),
        "squeeze_hyb": _jsonable(squeeze_hyb),
        "top_disagreements": _jsonable(replayed.get("top_disagreements")),
        "discrepancy": _jsonable(disc) if disc is not None else None,
        "spot_download_log": _jsonable(dl_log),
        "sanity": _jsonable(replayed.get("sanity")),
        "extra": _jsonable(extra),
        "gpu_used": False,
        "pred_hash": {"sha256": pred_hash["sha256"], "n_files": pred_hash["n_files"]},
        "position_sha256": packed.get("position_sha256"),
        "ledger": ledger_block,
    }
    (rep_dir / "btcb_phase3c_binance_replay.json").write_text(json.dumps(payload, indent=2, default=str))
    commit()

    cmc_panel_sha1 = _file_sha256(CMC_PANEL)
    cmc_feat_sha1 = _file_sha256(CMC_FEAT) if CMC_FEAT.exists() else None
    pred_hash_end = hash_pred_dir(CMC_PRED)
    if cmc_panel_sha1 != cmc_panel_sha0:
        raise RuntimeError(f"CMC panel mutated during 3.c panel {cmc_panel_sha0} → {cmc_panel_sha1}")
    if cmc_feat_sha0 is not None and cmc_feat_sha1 != cmc_feat_sha0:
        raise RuntimeError("CMC/feat parquet mutated during 3.c")
    if pred_hash_end["sha256"] != PHASE2C_PRED_SHA256:
        raise RuntimeError("2.c pred cache mutated during 3.c")

    hyb = books["hybrid"]
    top = (replayed.get("top_disagreements") or [{}])[0]
    verdict_s = "PRICES ARE VALIDATED" if val.get("validated") else "PRICES ARE NOT VALIDATED"
    n_never_l = sum(1 for x in (cov["long"].get("never_listed") or []) if x.get("reason") == "never_listed")
    n_never_s = sum(1 for x in (cov["short"].get("never_listed") or []) if x.get("reason") == "never_listed")
    print(f"VALIDATION: {verdict_s}", flush=True)
    print(
        "OFFICIAL SPREAD-LS (funding-on hybrid): "
        f"Sharpe full={float(hyb.get('net_sharpe')):.3f} / trailing={float(hyb.get('net_sharpe_trail18m')):.3f} "
        f"MaxDD={100.0 * float(hyb.get('maxdd')):.1f}% "
        f"funding_pnl={float(hyb.get('funding_total_pnl') or 0):.4f} "
        f"funding_share_of_|gross|={float(hyb.get('funding_share_of_gross') or 0):.4f} "
        f"record={'OFFICIAL' if val.get('validated') else 'SUSPENDED'}",
        flush=True,
    )
    print(
        f"COVERAGE longs: {100.0 * float(cov['long'].get('pct_replayable')):.1f}% "
        f"({cov['long'].get('n_replayable')}/{cov['long'].get('n_name_days')} name-days) "
        f"never_listed={n_never_l}",
        flush=True,
    )
    print(
        f"COVERAGE shorts: {100.0 * float(cov['short'].get('pct_replayable')):.1f}% "
        f"({cov['short'].get('n_replayable')}/{cov['short'].get('n_name_days')} name-days) "
        f"never_listed={n_never_s}",
        flush=True,
    )
    print(
        f"TOP DISAGREEMENT: date={top.get('date')} {top.get('symbol')}({top.get('id')}) "
        f"side={top.get('side')} w={top.get('w')} d_r={top.get('d_r')} contrib_diff={top.get('contrib_diff')}",
        flush=True,
    )
    print(
        f"CMC RAW DATA READ-ONLY ASSERT: panel_sha256={cmc_panel_sha1} UNCHANGED; "
        f"feat_sha256={cmc_feat_sha1} UNCHANGED; "
        f"2.c preds sha256={pred_hash_end['sha256']} UNCHANGED; "
        f"writes only under /data/quant/raw/spot_klines and /data/quant/btcb/phase3c + reports/charts.",
        flush=True,
    )
    print("COMBO untouched. No MASTER. GPU=false.", flush=True)
    print(f"[HB] DONE elapsed={time.time()-t0:.1f}s gpu=false", flush=True)
    return {
        "verdict": verdict_s,
        "validated": bool(val.get("validated")),
        "corr": val.get("corr"),
        "sharpe_hybrid": hyb.get("net_sharpe"),
        "trail_hybrid": hyb.get("net_sharpe_trail18m"),
        "maxdd_hybrid": hyb.get("maxdd"),
        "funding_total": hyb.get("funding_total_pnl"),
        "coverage_long": cov["long"].get("pct_replayable"),
        "coverage_short": cov["short"].get("pct_replayable"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": False,
        "pred_sha256": pred_hash["sha256"],
        "position_sha256": packed.get("position_sha256"),
        "cmc_panel_sha256": cmc_panel_sha1,
        "cmc_untouched": True,
    }


@app.local_entrypoint()
def main():
    print("[local] starting Phase 3.c Binance replay (spawn, then wait)...", flush=True)
    fc = run_btcb_p3c.spawn()
    print(f"[local] spawned {getattr(fc, 'object_id', fc)}", flush=True)
    summary = fc.get()
    print("[local] syncing artifacts...", flush=True)
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_phase3c_binance_replay.md", "reports"),
        ("reports/btcb_phase3c_binance_replay.json", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_phase3c_hybrid_equity.png", "charts"),
        ("charts/btcb_phase3c_pnl_scatter.png", "charts"),
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
        for src in (art / "reports").glob("btcb_phase3c*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        led = art / "reports" / "numbers_ledger.md"
        if led.exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes(led.read_bytes())
        for src in (art / "charts").glob("btcb_phase3c*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())
    print(json.dumps(summary, indent=2, default=str))
    print("[local] Phase 3.c complete.", flush=True)
