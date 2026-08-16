"""
BTC-BEATER Phase 5 — CS-ATTN v0. Hourly panel (CPU) + cross-sectional attention (GPU).

BACKTEST / ANALYSIS ONLY. New Modal app. Frozen products read-only.
Usage: modal run btcb_phase5_pipeline.py

Stage B: 3×H100 (one seed each) + 1×H100 §B null in parallel, then CPU judge.
Architecture / seeds / folds / criteria frozen. Shared $80 cap including A10G sunk.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import modal

APP_NAME = "quant-btcb-p5-csattn"
VOL_Q = "quant-baseline"
quant_vol = modal.Volume.from_name(VOL_Q, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "matplotlib",
        "httpx",
        "pyyaml",
        "scikit-learn",
        "lightgbm",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/btcb_phase5_csattn_addendum.md", remote_path="/root/btcb_phase5_csattn_addendum.md")
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
    .add_local_file("universe/btcb_top100_floor.parquet", remote_path="/root/btcb_top100_floor.parquet")
)

gpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    .pip_install(
        "numpy",
        "pandas==2.2.2",
        "pyarrow",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "lightgbm",
        "httpx",
        "pyyaml",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("baseline", "btcb")
    .add_local_file("reports/btcb_phase5_csattn_addendum.md", remote_path="/root/btcb_phase5_csattn_addendum.md")
    .add_local_file("reports/numbers_ledger.md", remote_path="/root/numbers_ledger.md")
)

app = modal.App(APP_NAME)


class StageHeartbeat:
    """Heartbeat + kill if silent > 20 min (Phase 5 watchdog)."""

    def __init__(self, stage: str, kill_sec: int = 20 * 60):
        self.stage = stage
        self.kill_sec = int(kill_sec)
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
            if silent >= self.kill_sec:
                print(
                    f"[HB] WATCHDOG KILL stage={self.stage} silent={silent:.0f}s > {self.kill_sec}",
                    flush=True,
                )
                os._exit(2)

    def close(self) -> None:
        self.stop.set()
        print(f"[HB] STAGE END {self.stage} elapsed={time.time() - self.t0:.0f}s", flush=True)


def _fmt_local(x):
    try:
        if x is None:
            return "nan"
        v = float(x)
        if v != v:
            return "nan"
        return f"{v:.4f}"
    except Exception:
        return str(x)


def _jsonable(x, drop=None):
    import numpy as np
    import pandas as pd

    drop = drop or {"daily_ret", "equity", "id_to_sym", "btc_ret", "X"}
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
        return x if x == x else None
    return x


def _file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=cpu_image,
    timeout=60 * 45,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=2,
    memory=4096,
    max_containers=40,
)
def download_hourly_one(item: dict) -> dict:
    from baseline.data import month_range
    from btcb.hourly import download_hourly_symbol

    symbol = item["symbol"]
    kind = item.get("kind", "spot")
    dest = Path("/data/quant/raw/hourly_spot" if kind == "spot" else "/data/quant/raw/hourly_um")
    t0 = time.time()
    try:
        rec = download_hourly_symbol(symbol, month_range(item["start_month"]), dest, kind=kind)
        rec["elapsed"] = time.time() - t0
        rec["ok"] = True
    except Exception as e:
        rec = {
            "symbol": symbol,
            "kind": kind,
            "ok": False,
            "empty": True,
            "n_rows": 0,
            "reused": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed": time.time() - t0,
        }
        print(f"[hourly] {kind} {symbol} FAIL {rec['error']}", flush=True)
    quant_vol.commit()
    print(
        f"[hourly] {kind} {symbol} reused={rec.get('reused')} empty={rec.get('empty')} n={rec.get('n_rows')}",
        flush=True,
    )
    return rec


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 8,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=16,
    memory=65536,
)
def stage_a_hourly() -> dict:
    import pandas as pd

    from baseline.data import list_spot_symbols, list_um_symbols, month_range
    from btcb.constants import (
        CMC_PANEL_SHA256,
        DEATH_CONVENTION,
        PHASE5_CRITERION,
        PHASE5_HOURLY_START,
        PHASE5_NULL_GATE,
    )
    from btcb.features import btc_id_from_panel
    from btcb.hourly import (
        assemble_hourly_panel,
        audit_hourly_panel,
        build_sequence_cache,
        finalize_id_map,
        pick_id_sources,
        plan_symbol_jobs,
        write_hourly_report,
    )
    from btcb.hygiene import clean_panel
    from btcb.labels import add_twin_quintile_labels

    t0 = time.time()
    hb = StageHeartbeat("A-hourly")
    addendum = Path("/root/btcb_phase5_csattn_addendum.md").read_text()
    for txt in (PHASE5_CRITERION, PHASE5_NULL_GATE, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError(f"phase5 addendum missing freeze text: {txt[:80]}")
    print("[HB] BTC-BEATER P5 STAGE A hourly panel CPU; frozen products read-only", flush=True)
    print(f"[HB] {PHASE5_CRITERION}", flush=True)

    def commit():
        quant_vol.commit()

    cmc_path = Path("/data/quant/btcb/full/panel.parquet")
    if not cmc_path.exists():
        raise RuntimeError(f"missing CMC panel {cmc_path}")
    cmc_sha = _file_sha256(cmc_path)
    print(f"[HB] CMC READ-ONLY sha256={cmc_sha}", flush=True)
    if cmc_sha != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel hash mismatch {cmc_sha}")

    seq_dir = Path("/data/quant/btcb/phase5/seq")
    panel_path = Path("/data/quant/hourly/panel.parquet")
    seq_ok = (
        (seq_dir / "X.npy").exists()
        and (seq_dir / "index.parquet").exists()
        and (seq_dir / "meta.json").exists()
    )
    if seq_ok and panel_path.exists():
        hb.ping("reusing hourly panel + seq cache (skip download/assemble/rebuild)")
        audit_path = Path("/data/quant/hourly/audit.json")
        raw = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        audit = {k: v for k, v in raw.items() if k not in {"extra", "dl_log"}} if isinstance(raw, dict) else {}
        seq_meta = json.loads((seq_dir / "meta.json").read_text())
        extra_a = dict(raw.get("extra") or {}) if isinstance(raw, dict) else {}
        extra_a["reused"] = True
        extra_a["cmc_sha256"] = cmc_sha
        audit_light = {
            k: v
            for k, v in audit.items()
            if k not in {"gaps", "zero_volume", "alignment_violations", "coverage"}
        }
        summary = {
            "status": "ok",
            "elapsed_sec": time.time() - t0,
            "audit": _jsonable(audit_light),
            "seq_meta": _jsonable(seq_meta),
            "extra": extra_a,
            "n_download_log": 0,
            "btc_id": int(audit.get("btc_id") or 1),
            "gpu_used": False,
            "reused_cache": True,
        }
        hb.close()
        print(f"[HB] STAGE A REUSED elapsed={time.time()-t0:.1f}s seq_n={seq_meta.get('n_rows')}", flush=True)
        return summary

    panel = pd.read_parquet(cmc_path)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    btc_id = btc_id_from_panel(panel)
    hb.ping(f"btc_id={btc_id} panel_rows={len(panel)}")

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
            print(f"[HB] pit from {p} rows={len(pit)} ids={pit['id'].nunique()}", flush=True)
            break
    if pit is None:
        raise RuntimeError("missing floored PIT top-100")

    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    hb.ping("cleaned panel")

    print("[HB] listing Vision spot/um 1h symbols...", flush=True)
    try:
        spot_listed = set(s.upper() for s in list_spot_symbols("USDT"))
    except Exception as e:
        print(f"[HB] list_spot_symbols failed {e}", flush=True)
        spot_listed = set()
    try:
        um_listed = set(s.upper() for s in list_um_symbols("USDT"))
    except Exception as e:
        print(f"[HB] list_um_symbols failed {e}", flush=True)
        um_listed = set()
    hb.ping(f"listed spot={len(spot_listed)} um={len(um_listed)}")

    spot_dir = Path("/data/quant/raw/hourly_spot")
    um_dir = Path("/data/quant/raw/hourly_um")
    spot_dir.mkdir(parents=True, exist_ok=True)
    um_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_symbol_jobs(pit, cleaned, btc_id, spot_listed, um_listed, spot_dir, um_dir)
    print(
        f"[HB] ids={len(plan['all_ids'])} spot_wanted={len(plan['wanted_spot'])} "
        f"todo={len(plan['spot_todo'])} reuse={len(plan['spot_reuse'])}",
        flush=True,
    )
    dl_log = []
    todo = plan["spot_todo"]
    if todo:
        for i in range(0, len(todo), 80):
            part = todo[i : i + 80]
            hb.ping(f"spot batch {i//80 + 1} n={len(part)}")
            dl_log.extend(list(download_hourly_one.map(part, order_outputs=False)))
            quant_vol.reload()
    else:
        print("[HB] spot hourly all cached", flush=True)
    quant_vol.reload()
    hb.ping("spot downloads done")

    id_map, um_needed = pick_id_sources(plan["per_id"], btc_id, spot_dir, um_dir, um_listed)
    um_todo = []
    for sym in um_needed:
        if not (um_dir / f"{sym}.parquet").exists():
            um_todo.append({"symbol": sym, "kind": "um", "start_month": PHASE5_HOURLY_START})
    print(f"[HB] perp fallback needed={len(um_needed)} download={len(um_todo)}", flush=True)
    if um_todo:
        for i in range(0, len(um_todo), 80):
            part = um_todo[i : i + 80]
            hb.ping(f"um batch {i//80 + 1} n={len(part)}")
            dl_log.extend(list(download_hourly_one.map(part, order_outputs=False)))
            quant_vol.reload()
    quant_vol.reload()
    id_map = finalize_id_map(id_map, spot_dir, um_dir, btc_id)
    n_mapped = int(id_map["symbol"].notna().sum())
    n_um_ids = int((id_map["kind"] == "perp").sum())
    hb.ping(f"mapped {n_mapped}/{len(id_map)} um_ids={n_um_ids}")

    hourly = assemble_hourly_panel(id_map, spot_dir, um_dir)
    print(f"[HB] hourly panel rows={len(hourly)} ids={hourly['id'].nunique()}", flush=True)
    out_h = Path("/data/quant/hourly")
    out_h.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out_h / "panel.parquet", index=False)
    id_map.to_parquet(out_h / "id_map.parquet", index=False)
    commit()
    hb.ping("wrote panel.parquet")

    audit = audit_hourly_panel(hourly, cleaned, pit, btc_id)
    extra_a = {
        "n_new": int(sum(1 for r in dl_log if not r.get("reused") and r.get("ok"))),
        "n_reused": int(sum(1 for r in dl_log if r.get("reused"))),
        "n_empty": int(sum(1 for r in dl_log if r.get("empty"))),
        "n_um_ids": n_um_ids,
        "n_mapped": n_mapped,
        "cmc_sha256": cmc_sha,
        "months": month_range(PHASE5_HOURLY_START)[:1] + month_range(PHASE5_HOURLY_START)[-1:],
    }
    (out_h / "audit.json").write_text(json.dumps({**audit, "extra": extra_a, "dl_log": dl_log}, indent=2, default=str))
    rep_dir = Path("/data/quant/reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    write_hourly_report(rep_dir / "btcb_hourly_panel.md", audit, extra_a)
    commit()
    hb.ping(f"audit median_bps={audit.get('alignment_median_abs_bps')} pass={audit.get('alignment_pass')}")

    print("[HB] twin quintile labels on PIT + sequence cache...", flush=True)
    feat = pit[["date", "id"]].copy()
    if "symbol" in pit.columns:
        feat["symbol"] = pit["symbol"]
    else:
        feat["symbol"] = feat["id"].astype(str)
    labeled = add_twin_quintile_labels(feat, cleaned, btc_id, horizons=(14,))
    labeled = labeled[labeled["id"].astype(int) != int(btc_id)].copy()
    seq_dir = Path("/data/quant/btcb/phase5/seq")
    seq_meta = build_sequence_cache(hourly, labeled, btc_id, seq_dir, heartbeat=hb)
    commit()
    hb.ping(f"seq n={seq_meta.get('n_rows')}")

    summary = {
        "status": "ok",
        "elapsed_sec": time.time() - t0,
        "audit": _jsonable(audit),
        "seq_meta": _jsonable(seq_meta),
        "extra": extra_a,
        "n_download_log": len(dl_log),
        "btc_id": int(btc_id),
        "gpu_used": False,
    }
    (out_h / "stage_a.json").write_text(json.dumps(summary, indent=2, default=str))
    commit()
    hb.close()
    print(f"[HB] STAGE A DONE elapsed={time.time()-t0:.1f}s", flush=True)
    return summary


def _volume_commit() -> None:
    for i in range(6):
        try:
            quant_vol.commit()
            return
        except Exception as e:
            print(f"[HB] volume commit retry {i}: {e}", flush=True)
            time.sleep(min(2 ** i, 16))
            try:
                quant_vol.reload()
            except Exception:
                pass
    quant_vol.commit()


def _a10g_sunk_usd(fallback: float = 0.0) -> float:
    p = Path("/data/quant/btcb/phase5/gpu_spend.json")
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if d.get("a10g_sunk_usd") is not None:
                return float(d["a10g_sunk_usd"])
            if str(d.get("gpu") or "").upper() == "A10G" and d.get("usd") is not None:
                return float(d["usd"])
        except Exception:
            pass
    alt = Path("/data/quant/btcb/phase5/gpu_spend/a10g_sunk.json")
    if alt.exists():
        try:
            return float(json.loads(alt.read_text()).get("a10g_sunk_usd") or fallback)
        except Exception:
            pass
    return float(fallback)


class SharedGpuBudget:
    """$80 cap = A10G sunk + Σ (H100 worker hours) × list rate.

    Each parallel worker writes only its own `gpu_spend/worker_{id}.json` so
    volume commits do not clobber peer files.
    """

    def __init__(self, cap_usd: float, usd_per_hour: float, a10g_sunk: float, worker_id: str, reserve_usd: float):
        self.cap = float(cap_usd)
        self.rate = float(usd_per_hour)
        self.a10g_sunk = float(a10g_sunk)
        self.worker_id = str(worker_id)
        self.reserve = float(reserve_usd)
        self.t0 = time.time()
        self.aborted = False
        self.abort_reason = None
        self._peer_hours = 0.0
        self._peer_at = 0.0
        self.dir = Path("/data/quant/btcb/phase5/gpu_spend")
        self.dir.mkdir(parents=True, exist_ok=True)

    def my_hours(self) -> float:
        return (time.time() - self.t0) / 3600.0

    def _read_peer_hours(self, force: bool = False) -> float:
        now = time.time()
        if not force and (now - self._peer_at) < 20.0:
            return self._peer_hours
        try:
            quant_vol.reload()
        except Exception:
            pass
        total = 0.0
        if self.dir.is_dir():
            for p in self.dir.glob("worker_*.json"):
                try:
                    rec = json.loads(p.read_text())
                    wid = str(rec.get("worker_id") or p.stem.replace("worker_", ""))
                    if wid == self.worker_id:
                        continue
                    total += float(rec.get("hours") or 0.0)
                except Exception:
                    continue
        self._peer_hours = total
        self._peer_at = now
        return total

    def h100_hours_total(self, force: bool = False) -> float:
        return self._read_peer_hours(force=force) + self.my_hours()

    def usd(self, force: bool = False) -> float:
        return self.a10g_sunk + self.h100_hours_total(force=force) * self.rate

    def persist(self) -> dict:
        rec = {
            "worker_id": self.worker_id,
            "hours": self.my_hours(),
            "usd_own": self.my_hours() * self.rate,
            "gpu": "H100",
            "usd_per_hour": self.rate,
            "a10g_sunk_usd": self.a10g_sunk,
            "peer_hours": self._read_peer_hours(force=True),
            "total_usd_est": self.usd(force=True),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.dir / f"worker_{self.worker_id}.json").write_text(json.dumps(rec, indent=2))
        return rec

    def ok(self, reserve_usd: float | None = None) -> bool:
        if self.aborted:
            return False
        r = self.reserve if reserve_usd is None else float(reserve_usd)
        u = self.usd(force=True)
        if u + r >= self.cap:
            self.aborted = True
            self.abort_reason = (
                f"gpu spend {u:.2f}+reserve {r:.2f} >= cap {self.cap:.2f} "
                f"(a10g_sunk {self.a10g_sunk:.2f} + h100_h {self.h100_hours_total():.2f} × {self.rate:.2f})"
            )
            print(f"[HB] BUDGET ABORT {self.abort_reason}", flush=True)
            self.persist()
            _volume_commit()
            return False
        return True


def _load_folds_idx():
    import pandas as pd

    from btcb.constants import PHASE5_H
    from btcb.model import make_expanding_folds

    seq_dir = Path("/data/quant/btcb/phase5/seq")
    if not (seq_dir / "index.parquet").exists():
        raise RuntimeError("missing sequence cache — stage A must finish first")
    idx = pd.read_parquet(seq_dir / "index.parquet")
    idx["date"] = pd.to_datetime(idx["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    feat_path = Path("/data/quant/btcb/phase2b/feat_s.parquet")
    if feat_path.exists():
        feat_dates = pd.read_parquet(feat_path, columns=["date"])
        feat_dates["date"] = pd.to_datetime(feat_dates["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        fold_index = pd.DatetimeIndex(feat_dates["date"].unique())
        print(f"[HB] fold schedule from feat_s dates n={len(fold_index)} (same as 2.c)", flush=True)
    else:
        fold_index = pd.DatetimeIndex(idx["date"].unique())
        print("[HB] WARN feat_s missing; folds from sequence dates", flush=True)
    folds = make_expanding_folds(fold_index, horizon=int(PHASE5_H))
    print(f"[HB] folds={len(folds)} seq_rows={len(idx)} dates={idx['date'].nunique()}", flush=True)
    return seq_dir, idx, folds


def _smoke(hb: StageHeartbeat) -> dict:
    import torch

    from btcb.csattn import smoke_csattn

    smoke_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke_dev.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    hb.ping(f"smoke_csattn on {smoke_dev}")
    smoke = smoke_csattn(smoke_dev)
    hb.ping(f"smoke ok={smoke.get('ok')} cases={smoke.get('cases')}")
    return smoke


def _train_one(seq_dir, fold, seed, budget, hb, pred_root, tag="real", shuffle=False, shuf_seed=None):
    import pandas as pd

    from btcb.csattn import train_csattn_fold

    dest = pred_root / tag / f"seed{seed}" / f"fold{fold.fold_id}.parquet"
    mp = dest.with_suffix(".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        pred = pd.read_parquet(dest)
        meta = json.loads(mp.read_text()) if mp.exists() else {"status": "reuse"}
        meta["status"] = meta.get("status") or "reuse"
        hb.ping(f"reuse {tag} seed={seed} fold={fold.fold_id}")
        return pred, meta
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
            if meta.get("status") in {"empty", "empty_val"}:
                hb.ping(f"reuse-empty {tag} seed={seed} fold={fold.fold_id}")
                return pd.DataFrame(), meta
        except Exception:
            pass
    if not budget.ok():
        return pd.DataFrame(), {"status": "aborted_budget", "fold_id": fold.fold_id, "seed": seed}
    pred, meta = train_csattn_fold(
        seq_dir,
        fold,
        seed=int(seed),
        shuffle_labels=bool(shuffle),
        shuffle_seed=shuf_seed,
        heartbeat=hb,
        budget_ok=budget.ok,
    )
    if pred is not None and not pred.empty:
        pred.to_parquet(dest, index=False)
    mp.write_text(json.dumps(meta, indent=2, default=str))
    budget.persist()
    _volume_commit()
    return pred, meta


@app.function(
    image=cpu_image,
    timeout=120,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=1,
    memory=1024,
)
def persist_a10g_sunk(usd: float, hours: float, note: str = "") -> dict:
    quant_vol.reload()
    rec = {
        "a10g_sunk_usd": float(usd),
        "a10g_hours": float(hours),
        "a10g_usd_per_hour": 1.10,
        "gpu": "A10G",
        "usd": float(usd),
        "note": note or "sunk A10G sequential before H100 parallel",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    root = Path("/data/quant/btcb/phase5")
    root.mkdir(parents=True, exist_ok=True)
    spend_dir = root / "gpu_spend"
    spend_dir.mkdir(parents=True, exist_ok=True)
    (root / "gpu_spend.json").write_text(json.dumps(rec, indent=2))
    (spend_dir / "a10g_sunk.json").write_text(json.dumps(rec, indent=2))
    for p in spend_dir.glob("worker_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    _volume_commit()
    print(f"[HB] persisted A10G sunk ${usd:.2f} ({hours:.2f} h)", flush=True)
    return rec


@app.function(
    image=gpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": quant_vol},
    gpu="H100",
    memory=65536,
    max_containers=4,
)
def stage_b_train_seed(seed: int, a10g_sunk: float = 0.0) -> dict:
    from btcb.constants import (
        DEATH_CONVENTION,
        PHASE5_CRITERION,
        PHASE5_GPU_RESERVE_USD,
        PHASE5_GPU_USD_CAP,
        PHASE5_GPU_USD_PER_HOUR,
        PHASE5_NULL_GATE,
    )

    t0 = time.time()
    hb = StageHeartbeat(f"B-seed{seed}")
    quant_vol.reload()
    addendum = Path("/root/btcb_phase5_csattn_addendum.md").read_text()
    for txt in (PHASE5_CRITERION, PHASE5_NULL_GATE, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError("phase5 addendum missing freeze text")
    sunk = _a10g_sunk_usd(a10g_sunk)
    budget = SharedGpuBudget(
        PHASE5_GPU_USD_CAP,
        PHASE5_GPU_USD_PER_HOUR,
        sunk,
        worker_id=f"seed{int(seed)}",
        reserve_usd=PHASE5_GPU_RESERVE_USD,
    )
    print(
        f"[HB] BTC-BEATER P5 STAGE B seed={seed} GPU H100 cap=$80 a10g_sunk=${sunk:.2f}",
        flush=True,
    )
    print(f"[HB] {PHASE5_CRITERION}", flush=True)
    seq_dir, idx, folds = _load_folds_idx()
    hb.ping(f"n_folds={len(folds)} seq_rows={len(idx)}")
    smoke = _smoke(hb)
    pred_root = Path("/data/quant/btcb/phase5/preds")
    pred_root.mkdir(parents=True, exist_ok=True)
    metas = []
    n_written = 0
    for i, fr in enumerate(folds):
        if not budget.ok():
            metas.append({"fold_id": int(fr.fold_id), "status": "aborted_budget", "seed": int(seed)})
            break
        pred, meta = _train_one(seq_dir, fr, int(seed), budget, hb, pred_root, tag="real")
        metas.append(meta)
        ok_st = meta.get("status") in {"ok", "reuse", "empty", "empty_val"}
        if ok_st:
            n_written += 1
        n_left = len(folds) - (i + 1)
        last_h = float(meta.get("elapsed") or 0.0) / 3600.0
        hb.ping(
            f"done seed={seed} fold={fr.fold_id} n={meta.get('n_pred')} "
            f"usd={budget.usd():.2f} status={meta.get('status')} "
            f"fold_h={last_h:.2f} remaining_folds={n_left} eta_this_seed_h={n_left * last_h:.1f}"
        )
    complete = (not budget.aborted) and all(
        (m.get("status") in {"ok", "reuse", "empty", "empty_val"}) for m in metas
    ) and len(metas) >= len(folds)
    budget.persist()
    _volume_commit()
    rec = {
        "seed": int(seed),
        "status": "ok" if complete else ("aborted_budget" if budget.aborted else "incomplete"),
        "complete": bool(complete),
        "n_folds": len(folds),
        "n_done": n_written,
        "aborted": budget.aborted,
        "abort_reason": budget.abort_reason,
        "usd": budget.usd(force=True),
        "hours": budget.my_hours(),
        "a10g_sunk_usd": sunk,
        "smoke_ok": bool((smoke or {}).get("ok")),
        "elapsed_sec": time.time() - t0,
        "metas": _jsonable(metas),
        "gpu_used": True,
        "gpu_type": "H100",
    }
    hb.close()
    print(json.dumps({k: v for k, v in rec.items() if k != "metas"}, indent=2, default=str), flush=True)
    return rec


@app.function(
    image=gpu_image,
    timeout=60 * 60 * 24,
    retries=0,
    volumes={"/data/quant": quant_vol},
    gpu="H100",
    memory=65536,
    max_containers=4,
)
def stage_b_null(a10g_sunk: float = 0.0) -> dict:
    from btcb.constants import (
        DEATH_CONVENTION,
        PHASE5_CRITERION,
        PHASE5_GPU_RESERVE_USD,
        PHASE5_GPU_USD_CAP,
        PHASE5_GPU_USD_PER_HOUR,
        PHASE5_NULL_FOLDS,
        PHASE5_NULL_GATE,
        PHASE5_NULL_SHUFFLE_SEEDS,
    )
    from btcb.phase5_eval import fold_tail_ic_from_pred

    t0 = time.time()
    hb = StageHeartbeat("B-null")
    quant_vol.reload()
    addendum = Path("/root/btcb_phase5_csattn_addendum.md").read_text()
    for txt in (PHASE5_CRITERION, PHASE5_NULL_GATE, DEATH_CONVENTION):
        if txt not in addendum:
            raise RuntimeError("phase5 addendum missing freeze text")
    sunk = _a10g_sunk_usd(a10g_sunk)
    budget = SharedGpuBudget(
        PHASE5_GPU_USD_CAP,
        PHASE5_GPU_USD_PER_HOUR,
        sunk,
        worker_id="null",
        reserve_usd=PHASE5_GPU_RESERVE_USD,
    )
    print(f"[HB] BTC-BEATER P5 STAGE B null GPU H100 cap=$80 a10g_sunk=${sunk:.2f}", flush=True)
    print(f"[HB] {PHASE5_NULL_GATE}", flush=True)
    seq_dir, idx, folds = _load_folds_idx()
    hb.ping(f"n_folds={len(folds)}")
    smoke = _smoke(hb)
    pred_root = Path("/data/quant/btcb/phase5/preds")
    pred_root.mkdir(parents=True, exist_ok=True)
    fold_by_id = {int(f.fold_id): f for f in folds}
    null_ics = {int(f): [] for f in PHASE5_NULL_FOLDS}
    null_done = 0
    need = len(PHASE5_NULL_FOLDS) * len(PHASE5_NULL_SHUFFLE_SEEDS)
    jobs = []
    for fid in PHASE5_NULL_FOLDS:
        fr = fold_by_id.get(int(fid))
        if fr is None:
            print(f"[HB] null fold {fid} missing from schedule", flush=True)
            continue
        for sh in PHASE5_NULL_SHUFFLE_SEEDS:
            jobs.append((fr, int(sh)))
    for k, (fr, sh) in enumerate(jobs):
        if not budget.ok():
            break
        pred, meta = _train_one(
            seq_dir, fr, 42, budget, hb, pred_root, tag=f"null_sh{sh}", shuffle=True, shuf_seed=int(sh)
        )
        ic = fold_tail_ic_from_pred(pred)
        null_ics[int(fr.fold_id)].append(ic)
        if meta.get("status") in {"ok", "reuse", "empty", "empty_val"}:
            null_done += 1
        n_left = len(jobs) - (k + 1)
        last_h = float(meta.get("elapsed") or 0.0) / 3600.0
        hb.ping(
            f"null fold={fr.fold_id} shuf={sh} tailIC={ic} usd={budget.usd():.2f} "
            f"status={meta.get('status')} remaining={n_left} eta_h={n_left * last_h:.1f}"
        )
    complete = (not budget.aborted) and null_done >= need
    out = {
        "null_ics": {str(k): v for k, v in null_ics.items()},
        "null_done": null_done,
        "need": need,
        "complete": bool(complete),
        "status": "ok" if complete else ("aborted_budget" if budget.aborted else "incomplete"),
        "aborted": budget.aborted,
        "abort_reason": budget.abort_reason,
        "usd": budget.usd(force=True),
        "hours": budget.my_hours(),
        "a10g_sunk_usd": sunk,
        "smoke_ok": bool((smoke or {}).get("ok")),
        "elapsed_sec": time.time() - t0,
        "gpu_used": True,
        "gpu_type": "H100",
    }
    Path("/data/quant/btcb/phase5/null_ics.json").write_text(json.dumps(out, indent=2, default=str))
    budget.persist()
    _volume_commit()
    hb.close()
    print(json.dumps({k: v for k, v in out.items() if k != "null_ics"}, indent=2, default=str), flush=True)
    return out


def _aggregate_gpu(seed_recs: dict, null_rec: dict, a10g_sunk: float) -> dict:
    from btcb.constants import (
        PHASE5_A10G_USD_PER_HOUR,
        PHASE5_GPU_USD_CAP,
        PHASE5_GPU_USD_PER_HOUR,
    )

    spend_dir = Path("/data/quant/btcb/phase5/gpu_spend")
    workers = []
    h100_hours = 0.0
    if spend_dir.is_dir():
        for p in sorted(spend_dir.glob("worker_*.json")):
            try:
                rec = json.loads(p.read_text())
                workers.append(rec)
                h100_hours += float(rec.get("hours") or 0.0)
            except Exception:
                continue
    if not workers:
        for rec in list(seed_recs.values()) + ([null_rec] if null_rec else []):
            h100_hours += float((rec or {}).get("hours") or 0.0)
    h100_usd = h100_hours * float(PHASE5_GPU_USD_PER_HOUR)
    total = float(a10g_sunk) + h100_usd
    aborted = any(bool((r or {}).get("aborted")) for r in list(seed_recs.values()) + [null_rec])
    reasons = [str((r or {}).get("abort_reason")) for r in list(seed_recs.values()) + [null_rec] if (r or {}).get("abort_reason")]
    folds_done = []
    for s, rec in sorted(seed_recs.items(), key=lambda kv: int(kv[0])):
        for m in (rec or {}).get("metas") or []:
            folds_done.append({"seed": int(s), "fold_id": m.get("fold_id"), "status": m.get("status")})
    seeds_done = [int(s) for s, rec in seed_recs.items() if (rec or {}).get("complete")]
    a10g_hours = float(a10g_sunk) / float(PHASE5_A10G_USD_PER_HOUR) if a10g_sunk else 0.0
    return {
        "gpu_used": True,
        "gpu_type": "H100",
        "parallelism": "3×H100 seeds + 1×H100 null",
        "a10g_sunk_usd": float(a10g_sunk),
        "a10g_hours": a10g_hours,
        "a10g_usd_per_hour": PHASE5_A10G_USD_PER_HOUR,
        "h100_hours": h100_hours,
        "h100_usd": h100_usd,
        "usd_per_hour": PHASE5_GPU_USD_PER_HOUR,
        "gpu_hours": h100_hours + a10g_hours,
        "gpu_seconds": (h100_hours + a10g_hours) * 3600.0,
        "usd": total,
        "cap_usd": PHASE5_GPU_USD_CAP,
        "aborted": aborted,
        "abort_reason": "; ".join(reasons) if reasons else None,
        "folds_done": folds_done,
        "seeds_done": seeds_done,
        "null_done": int((null_rec or {}).get("null_done") or 0),
        "workers": _jsonable(workers),
        "seed_recs": {str(k): {kk: vv for kk, vv in (v or {}).items() if kk != "metas"} for k, v in seed_recs.items()},
        "null_rec": {k: v for k, v in (null_rec or {}).items() if k != "null_ics"},
    }


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 2,
    retries=0,
    volumes={"/data/quant": quant_vol},
    cpu=8,
    memory=32768,
)
def stage_b_judge(a10g_sunk: float = 0.0, seed_recs: dict | None = None, null_rec: dict | None = None) -> dict:
    import numpy as np
    import pandas as pd

    from baseline.data import load_panel
    from btcb.binance_replay import build_id_symbol_map, close_wide_from_panel
    from btcb.constants import (
        CMC_PANEL_SHA256,
        CSATTN_CONFIG,
        DEATH_CONVENTION,
        PHASE2C_PRED_N_FILES,
        PHASE2C_PRED_SHA256,
        PHASE3C_REF_END,
        PHASE3C_REF_H,
        PHASE3C_REF_START,
        PHASE5_CRITERION,
        PHASE5_GPU_USD_CAP,
        PHASE5_H,
        PHASE5_NULL_FOLDS,
        PHASE5_NULL_GATE,
        PHASE5_NULL_SHUFFLE_SEEDS,
        PHASE5_SEEDS,
    )
    from btcb.csattn import merge_seed_ensemble
    from btcb.features import btc_id_from_panel
    from btcb.hygiene import clean_panel
    from btcb.oracle_ladder import _as_utc, ffill_members, formation_dates
    from btcb.phase5_eval import (
        cell_stats,
        crude_book,
        fold_tail_ic_from_pred,
        load_manuel_score,
        mechanical_verdict,
        null_verdict,
        per_date_diagnostics,
        scores_from_twin,
        windowed_metrics,
    )
    from btcb.phase5_report import plot_overlap_cycles, plot_tail_ic_bars, update_ledger_phase5, write_phase5
    from btcb.spread_ls import hash_pred_dir, load_twin_from_cache
    from btcb.academic_factor import pit_members

    t0 = time.time()
    hb = StageHeartbeat("B-judge")
    quant_vol.reload()
    seed_recs = {int(k): v for k, v in (seed_recs or {}).items()}
    null_rec = null_rec or {}
    sunk = _a10g_sunk_usd(a10g_sunk)
    pred_root = Path("/data/quant/btcb/phase5/preds")

    concat = {}
    for seed in PHASE5_SEEDS:
        parts = []
        d = pred_root / "real" / f"seed{int(seed)}"
        if d.exists():
            for p in sorted(d.glob("fold*.parquet"), key=lambda x: int(x.stem.replace("fold", "") or 0)):
                if p.stat().st_size > 0:
                    parts.append(pd.read_parquet(p))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            df = df.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
            concat[int(seed)] = df
            df.to_parquet(pred_root / f"oos_seed{seed}.parquet", index=False)
        else:
            concat[int(seed)] = pd.DataFrame()
    ens = merge_seed_ensemble([concat[s] for s in PHASE5_SEEDS if not concat[s].empty])
    if not ens.empty:
        ens.to_parquet(pred_root / "oos_ensemble.parquet", index=False)

    null_ics = {int(f): [] for f in PHASE5_NULL_FOLDS}
    if (null_rec or {}).get("null_ics"):
        for k, v in (null_rec.get("null_ics") or {}).items():
            null_ics[int(k)] = list(v)
    else:
        nip = Path("/data/quant/btcb/phase5/null_ics.json")
        if nip.exists():
            blob = json.loads(nip.read_text())
            for k, v in (blob.get("null_ics") or {}).items():
                null_ics[int(k)] = list(v)
        for fid in PHASE5_NULL_FOLDS:
            if null_ics.get(int(fid)):
                continue
            for sh in PHASE5_NULL_SHUFFLE_SEEDS:
                p = pred_root / f"null_sh{sh}" / "seed42" / f"fold{int(fid)}.parquet"
                if p.exists() and p.stat().st_size > 0:
                    null_ics[int(fid)].append(fold_tail_ic_from_pred(pd.read_parquet(p)))
    null_done = int(sum(len(v) for v in null_ics.values()))
    need_null = len(PHASE5_NULL_FOLDS) * len(PHASE5_NULL_SHUFFLE_SEEDS)

    cmc_path = Path("/data/quant/btcb/full/panel.parquet")
    cmc_sha = _file_sha256(cmc_path)
    if cmc_sha != CMC_PANEL_SHA256:
        raise RuntimeError(f"CMC panel mutated {cmc_sha}")
    end = pd.Timestamp(PHASE3C_REF_END, tz="UTC")
    start = pd.Timestamp(PHASE3C_REF_START, tz="UTC")
    panel = pd.read_parquet(cmc_path)
    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    panel["id"] = panel["id"].astype(int)
    panel = panel[panel["date"] <= end].copy()
    btc_id = btc_id_from_panel(panel)
    pit = None
    for p in (
        Path("/data/quant/btcb/universe/btcb_top100_floor.parquet"),
        Path("/data/quant/universe/btcb_top100_floor.parquet"),
    ):
        if p.exists():
            pit = pd.read_parquet(p)
            pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            pit["id"] = pit["id"].astype(int)
            pit = pit[pit["date"] <= end].copy()
            break
    if pit is None:
        raise RuntimeError("missing PIT")
    cleaned, _ = clean_panel(panel, btc_id=btc_id)
    cleaned = cleaned[cleaned["date"] <= end].copy()

    pred_dir_2c = Path("/data/quant/btcb/phase2c/preds")
    pred_hash = hash_pred_dir(pred_dir_2c)
    print(f"[HB] 2.c cache sha256={pred_hash['sha256']} n={pred_hash['n_files']}", flush=True)
    if pred_hash["sha256"] != PHASE2C_PRED_SHA256 or int(pred_hash["n_files"]) != int(PHASE2C_PRED_N_FILES):
        raise RuntimeError(f"2.c cache mutated {pred_hash}")
    twin = load_twin_from_cache(pred_dir_2c, int(PHASE3C_REF_H))
    twin = twin[twin["date"] <= end].copy()

    spot_dir = Path("/data/quant/raw/spot_klines")
    if not (spot_dir / "BTCUSDT.parquet").exists():
        raise RuntimeError("BTCUSDT spot daily missing (3.c cache)")
    spot_syms = sorted(p.stem.upper() for p in spot_dir.glob("*.parquet"))
    spot_panel = load_panel(spot_dir, spot_syms)
    spot_panel["date"] = pd.to_datetime(spot_panel["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    spot_panel["symbol"] = spot_panel["symbol"].astype(str).str.upper()
    all_ids = sorted(set(int(i) for i in pit["id"].unique()) | {int(btc_id)})
    nonempty = set(spot_panel["symbol"].unique())
    id_to_spot = build_id_symbol_map(all_ids, cleaned, nonempty, nonempty)
    id_to_spot[int(btc_id)] = "BTCUSDT"
    close = close_wide_from_panel(spot_panel, id_to_spot)
    close = close[close.index <= end].sort_index()
    close.index = pd.DatetimeIndex([_as_utc(d) for d in close.index])
    dates = list(close.index)
    members = ffill_members(pit_members(pit, btc_id), dates)
    oos = [d for d in dates if d >= start]
    pairs14 = formation_dates(oos, int(PHASE5_H))
    hb.ping(f"judgment formations={len(pairs14)}")

    def _grid_row(score_at, label):
        daily = per_date_diagnostics(close, members, btc_id, score_at, pairs14)
        metrics = windowed_metrics(daily, start, end)
        book = crude_book(close, members, btc_id, score_at, pairs14, label)
        return metrics, book, daily

    grid = {}
    books = {}
    gbm_scores = scores_from_twin(twin, "spread")
    grid["gbm"], books["frozen GBM spread"], _ = _grid_row(gbm_scores, "gbm")
    hb.ping(f"GBM tailIC={grid['gbm']['full']['tail_ic_top']['mean']} ov={grid['gbm']['full']['overlap']['mean']}")

    seed_metrics = {}
    for seed in PHASE5_SEEDS:
        df = concat.get(int(seed))
        key = f"seed{seed}"
        if df is None or df.empty:
            grid[key] = {}
            seed_metrics[str(seed)] = {}
            continue
        sc = scores_from_twin(df, "spread")
        grid[key], books[f"CS-ATTN seed {seed}"], _ = _grid_row(sc, key)
        seed_metrics[str(seed)] = grid[key]
        hb.ping(f"seed {seed} tailIC={grid[key]['full']['tail_ic_top']['mean']}")
    if not ens.empty:
        grid["ensemble"], books["CS-ATTN 3-seed ensemble"], _ = _grid_row(
            scores_from_twin(ens, "spread"), "ensemble"
        )
    else:
        grid["ensemble"] = {}

    manuel = load_manuel_score(
        [
            Path("/data/quant/reports/manuel_score_falsification.json"),
            Path("/data/quant/reports/manuel_score_falsification.md"),
            Path("/root/manuel_score_falsification.json"),
        ]
    )
    if manuel and isinstance(manuel.get("row"), dict) and "full" in manuel["row"]:
        grid["manuel"] = manuel["row"]

    real42 = concat.get(42, pd.DataFrame())
    cells = []
    for fid in PHASE5_NULL_FOLDS:
        ics = null_ics.get(int(fid)) or []
        st = cell_stats(ics)
        real_ic = float("nan")
        if not real42.empty:
            sl = real42[real42["fold_id"].astype(int) == int(fid)] if "fold_id" in real42.columns else real42
            real_ic = fold_tail_ic_from_pred(sl)
        st["fold_id"] = int(fid)
        st["real_ic"] = real_ic
        st["exceeds"] = bool(
            np.isfinite(real_ic) and np.isfinite(st.get("p95", np.nan)) and float(real_ic) > float(st["p95"])
        )
        cells.append(st)
    null = null_verdict(cells)

    n_folds = 0
    for rec in seed_recs.values():
        n_folds = max(n_folds, int((rec or {}).get("n_folds") or 0))
    if n_folds <= 0:
        try:
            _, _, folds_sched = _load_folds_idx()
            n_folds = len(folds_sched)
        except Exception:
            n_folds = 0
    seeds_done = []
    for seed in PHASE5_SEEDS:
        d = pred_root / "real" / f"seed{int(seed)}"
        n_ok = len([p for p in d.glob("fold*.parquet") if p.exists() and p.stat().st_size > 0]) if d.exists() else 0
        rec = seed_recs.get(int(seed)) or {}
        if rec.get("complete") or (n_folds and n_ok >= n_folds):
            seeds_done.append(int(seed))

    gpu = _aggregate_gpu(seed_recs, null_rec, sunk)
    gpu["seeds_done"] = seeds_done
    gpu["null_done"] = null_done
    verdict = mechanical_verdict(grid.get("gbm") or {}, seed_metrics, grid.get("ensemble") or {}, null)
    incomplete_budget = bool(gpu.get("aborted")) or len(seeds_done) < 3
    incomplete_null = null_done < need_null
    if incomplete_budget or incomplete_null:
        verdict = dict(verdict)
        verdict["live"] = False
        verdict["verdict"] = "PARKED"
        extra_fail = []
        if incomplete_budget:
            extra_fail.append("incomplete_budget")
            verdict["clause_a"] = False
            verdict["clause_b"] = False
        if incomplete_null:
            extra_fail.append("incomplete_null")
            verdict["clause_c"] = False
        failed = list(verdict.get("failed_clauses") or [])
        for f in extra_fail:
            if f not in failed:
                failed.append(f)
        verdict["failed_clauses"] = failed
        verdict["record_ceiling"] = False
        verdict["ceiling_sentence"] = None

    audit = {}
    ap = Path("/data/quant/hourly/audit.json")
    if ap.exists():
        audit = json.loads(ap.read_text())
    seq_meta = {}
    smp = Path("/data/quant/btcb/phase5/seq/meta.json")
    if smp.exists():
        seq_meta = json.loads(smp.read_text())

    extra = {
        "elapsed_sec": time.time() - t0,
        "seq_meta": seq_meta,
        "cmc_sha256": cmc_sha,
        "pred_hash": pred_hash,
        "n_folds": n_folds,
        "manuel": manuel,
    }
    plain_bits = [
        f"CS-ATTN is {verdict.get('verdict')}",
        f"failed clauses {verdict.get('failed_clauses')}",
        f"GBM tail-IC(top) {_fmt_local(verdict.get('baseline_tail_ic_top'))} vs ensemble "
        f"{_fmt_local(verdict.get('ensemble_tail_ic_top'))} (Δ {_fmt_local(verdict.get('delta_tail_ic_top'))})",
        f"overlap GBM {_fmt_local(verdict.get('baseline_overlap'))} vs ensemble "
        f"{_fmt_local(verdict.get('ensemble_overlap'))} (Δ {_fmt_local(verdict.get('delta_overlap'))})",
        f"seed dispersion {_fmt_local(verdict.get('seed_dispersion'))}",
        f"null {null.get('verdict')}",
        f"GPU ${gpu.get('usd', 0):.2f} / ${PHASE5_GPU_USD_CAP:.0f}",
    ]
    if verdict.get("record_ceiling"):
        plain_bits.append(verdict.get("ceiling_sentence"))
    extra["plain"] = ". ".join(plain_bits) + "."

    rep_dir = Path("/data/quant/reports")
    chart_dir = Path("/data/quant/charts")
    for d in (rep_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_phase5(
        rep_dir / "btcb_phase5_csattn.md",
        audit_summary=audit,
        config=dict(CSATTN_CONFIG),
        grid=grid,
        null=null,
        verdict=verdict,
        books=books,
        gpu=gpu,
        extra=extra,
    )
    plot_tail_ic_bars(grid, chart_dir / "btcb_p5_tailic_top.png")
    plot_overlap_cycles(grid, chart_dir / "btcb_p5_overlap_cycle.png")
    ledger_src = Path("/root/numbers_ledger.md")
    ledger_dst = rep_dir / "numbers_ledger.md"
    if ledger_src.exists():
        ledger_dst.write_text(ledger_src.read_text())
        update_ledger_phase5(ledger_dst, verdict)

    payload = {
        "criterion": PHASE5_CRITERION,
        "null_gate": PHASE5_NULL_GATE,
        "death_convention": DEATH_CONVENTION,
        "config": dict(CSATTN_CONFIG),
        "verdict": _jsonable(verdict),
        "grid": _jsonable(grid),
        "null": _jsonable(null),
        "books": {k: _jsonable(v) for k, v in books.items()},
        "gpu": _jsonable(gpu),
        "audit": _jsonable(audit),
        "extra": _jsonable(extra),
        "seed_recs": {str(k): _jsonable(v) for k, v in seed_recs.items()},
    }
    (rep_dir / "btcb_phase5_csattn.json").write_text(json.dumps(payload, indent=2, default=str))
    Path("/data/quant/btcb/phase5/gpu_spend.json").write_text(
        json.dumps(
            {
                "a10g_sunk_usd": sunk,
                "h100_hours": gpu.get("h100_hours"),
                "h100_usd": gpu.get("h100_usd"),
                "usd": gpu.get("usd"),
                "aborted": gpu.get("aborted"),
                "verdict": verdict.get("verdict"),
                "gpu": "H100",
            },
            indent=2,
        )
    )
    _volume_commit()
    print(f"VERDICT: {verdict.get('verdict')} clauses_failed={verdict.get('failed_clauses')}", flush=True)
    print(
        f"BASELINE vs ENSEMBLE tail-IC(top) {verdict.get('baseline_tail_ic_top')} → "
        f"{verdict.get('ensemble_tail_ic_top')} Δ={verdict.get('delta_tail_ic_top')} "
        f"overlap {verdict.get('baseline_overlap')} → {verdict.get('ensemble_overlap')} "
        f"Δ={verdict.get('delta_overlap')}",
        flush=True,
    )
    print(f"SEED DISP {verdict.get('seed_dispersion')} per_seed={verdict.get('per_seed_tail_ic_top')}", flush=True)
    print(f"NULL {null.get('verdict')} gpu_usd={gpu.get('usd')} aborted={gpu.get('aborted')}", flush=True)
    print("COMBO / SPREAD-LS / LONG-TIDE / 2.c cache untouched.", flush=True)
    hb.close()
    return {
        "verdict": verdict.get("verdict"),
        "failed_clauses": verdict.get("failed_clauses"),
        "delta_tail_ic_top": verdict.get("delta_tail_ic_top"),
        "delta_overlap": verdict.get("delta_overlap"),
        "seed_dispersion": verdict.get("seed_dispersion"),
        "null_verdict": null.get("verdict"),
        "gpu_usd": gpu.get("usd"),
        "aborted": gpu.get("aborted"),
        "elapsed_sec": time.time() - t0,
        "gpu_used": True,
        "seeds_done": seeds_done,
        "null_done": null_done,
    }


def _pull_artifacts() -> None:
    import shutil
    import subprocess

    art = Path("artifacts")
    Path("reports").mkdir(exist_ok=True)
    Path("charts").mkdir(exist_ok=True)
    pulls = [
        ("reports/btcb_hourly_panel.md", "reports"),
        ("reports/btcb_phase5_csattn.md", "reports"),
        ("reports/btcb_phase5_csattn.json", "reports"),
        ("reports/numbers_ledger.md", "reports"),
        ("charts/btcb_p5_tailic_top.png", "charts"),
        ("charts/btcb_p5_overlap_cycle.png", "charts"),
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
        for src in (art / "reports").glob("btcb_*"):
            (opt / "reports" / src.name).write_bytes(src.read_bytes())
        if (art / "reports" / "numbers_ledger.md").exists():
            (opt / "reports" / "numbers_ledger.md").write_bytes((art / "reports" / "numbers_ledger.md").read_bytes())
        for src in (art / "charts").glob("btcb_p5*"):
            (opt / "charts" / src.name).write_bytes(src.read_bytes())
            (opt / "screenshots" / src.name).write_bytes(src.read_bytes())


@app.local_entrypoint()
def main():
    print("[local] BTC-BEATER P5 CS-ATTN — Stage A CPU + 3×H100 seeds + H100 null + CPU judge", flush=True)
    print("[local] STAGE A hourly panel (CPU)...", flush=True)
    fa = stage_a_hourly.spawn()
    print(f"[local] spawned A {getattr(fa, 'object_id', fa)}", flush=True)
    a_sum = fa.get()
    print(json.dumps(a_sum, indent=2, default=str)[:2000], flush=True)

    sunk_path = Path("artifacts/a10g_sunk.json")
    if sunk_path.exists():
        sunk_blob = json.loads(sunk_path.read_text())
        a10g_usd = float(sunk_blob.get("usd") or sunk_blob.get("a10g_sunk_usd") or 0.0)
        a10g_hours = float(sunk_blob.get("hours") or sunk_blob.get("a10g_hours") or 0.0)
        note = str(sunk_blob.get("note") or "")
    else:
        a10g_usd, a10g_hours = 10.85, 9.86
        note = "A10G sequential seed42 folds 0-22 (default if local sunk file missing)"
    print(f"[local] persist A10G sunk ${a10g_usd:.2f} ({a10g_hours:.2f} h)", flush=True)
    persist_a10g_sunk.remote(a10g_usd, a10g_hours, note)

    seeds_left = [42, 43, 44]
    null_left = True
    seed_recs: dict = {}
    null_rec: dict = {}
    for attempt in range(1, 4):
        if not seeds_left and not null_left:
            break
        print(
            f"[local] STAGE B attempt {attempt}/3 H100 parallel seeds={seeds_left} null={null_left} "
            f"(fold parquets resume, shared $80 cap)",
            flush=True,
        )
        handles = {s: stage_b_train_seed.spawn(int(s), float(a10g_usd)) for s in seeds_left}
        h_null = stage_b_null.spawn(float(a10g_usd)) if null_left else None
        for s, h in handles.items():
            print(f"[local] spawned seed {s} {getattr(h, 'object_id', h)}", flush=True)
        if h_null is not None:
            print(f"[local] spawned null {getattr(h_null, 'object_id', h_null)}", flush=True)
        next_seeds = []
        for s, h in handles.items():
            try:
                rec = h.get()
            except Exception as e:
                print(f"[local] seed {s} attempt {attempt} raised {type(e).__name__}: {e}", flush=True)
                rec = {
                    "seed": int(s),
                    "status": "error",
                    "complete": False,
                    "error": f"{type(e).__name__}: {e}",
                    "aborted": True,
                }
            seed_recs[int(s)] = rec
            print(f"[local] seed {s} complete={rec.get('complete')} status={rec.get('status')} n_done={rec.get('n_done')}", flush=True)
            if not rec.get("complete"):
                next_seeds.append(int(s))
        if h_null is not None:
            try:
                null_rec = h_null.get()
            except Exception as e:
                print(f"[local] null attempt {attempt} raised {type(e).__name__}: {e}", flush=True)
                null_rec = {"status": "error", "complete": False, "error": f"{type(e).__name__}: {e}", "aborted": True}
            print(f"[local] null complete={null_rec.get('complete')} done={null_rec.get('null_done')}/{null_rec.get('need')}", flush=True)
            null_left = not bool(null_rec.get("complete"))
        seeds_left = next_seeds
        if seeds_left or null_left:
            print("[local] B incomplete — another GPU window (shared $80 cap)", flush=True)

    print("[local] STAGE B judge (CPU)...", flush=True)
    try:
        b_sum = stage_b_judge.remote(
            float(a10g_usd),
            {str(k): v for k, v in seed_recs.items()},
            null_rec,
        )
    except Exception as e:
        print(f"[local] judge raised {type(e).__name__}: {e}", flush=True)
        b_sum = {
            "verdict": "PARKED",
            "failed_clauses": ["incomplete_budget"],
            "error": f"{type(e).__name__}: {e}",
            "aborted": True,
            "gpu_usd": a10g_usd,
        }
    print("[local] syncing artifacts...", flush=True)
    _pull_artifacts()
    print(json.dumps(b_sum, indent=2, default=str))
    print(
        f"FINAL {b_sum.get('verdict')} failed={b_sum.get('failed_clauses')} "
        f"ΔtailIC={b_sum.get('delta_tail_ic_top')} Δov={b_sum.get('delta_overlap')} "
        f"disp={b_sum.get('seed_dispersion')} null={b_sum.get('null_verdict')} "
        f"gpu_usd={b_sum.get('gpu_usd')}",
        flush=True,
    )
    print("[local] BTC-BEATER P5 complete.", flush=True)
