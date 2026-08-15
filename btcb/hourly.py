"""Phase 5 hourly Binance 1h panel: Vision download, audit, sequence cache."""

from __future__ import annotations

import json
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.data import VISION_FILE, _kline_zip_url, month_range
from btcb.binance_replay import candidate_usdt_symbols, symbols_for_id
from btcb.constants import (
    PHASE5_ALIGN_BPS,
    PHASE5_BARS_PER_DAY,
    PHASE5_CHANNELS,
    PHASE5_HOURLY_START,
    PHASE5_MIN_BARS_FRAC,
    PHASE5_N_CHANNELS,
    PHASE5_SEQ_LEN,
    PHASE5_VOL_Z_WINDOW,
    PHASE3C_NAME_TIERS,
)


KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
]


def _log(msg: str) -> None:
    print(f"[hourly {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_kline_zip(zip_path: Path) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh, header=None)
    except Exception as e:
        _log(f"parse skip {zip_path.name}: {e}")
        return None
    if df.shape[1] < 8:
        return None
    if len(df) and str(df.iloc[0, 0]).lower().startswith("open"):
        df = df.iloc[1:]
    df = df.iloc[:, :11].copy()
    df.columns = KLINE_COLS
    return df


def download_hourly_symbol(
    symbol: str,
    months: list[str],
    dest_dir: Path,
    kind: str = "spot",
) -> dict:
    """Idempotent per-symbol 1h cache. Returns metadata; empty parquet is a marker."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_pq = dest_dir / f"{symbol}.parquet"
    if out_pq.exists():
        try:
            n = int(pd.read_parquet(out_pq, columns=["ts"]).shape[0]) if out_pq.stat().st_size > 0 else 0
        except Exception:
            n = 0
        return {
            "symbol": symbol,
            "kind": kind,
            "path": str(out_pq),
            "reused": True,
            "n_rows": n,
            "empty": n == 0,
            "ok": True,
        }

    frames: list[pd.DataFrame] = []
    raw_dir = dest_dir / "raw" / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_dl = 0
    n_skip = 0
    import httpx

    for ym in months:
        url, zip_name = _kline_zip_url(symbol, "1h", ym, kind=kind)
        zip_path = raw_dir / zip_name
        if not zip_path.exists():
            try:
                with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                    if r.status_code == 404:
                        n_skip += 1
                        continue
                    r.raise_for_status()
                    zip_path.write_bytes(r.read())
                    n_dl += 1
            except Exception as e:
                _log(f"{kind} {symbol} {ym} skip: {e}")
                n_skip += 1
                continue
        parsed = _parse_kline_zip(zip_path)
        if parsed is None or parsed.empty:
            n_skip += 1
            continue
        frames.append(parsed)

    empty_cols = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "symbol",
        "kind",
    ]
    if not frames:
        pd.DataFrame(columns=empty_cols).to_parquet(out_pq, index=False)
        return {
            "symbol": symbol,
            "kind": kind,
            "path": str(out_pq),
            "reused": False,
            "n_rows": 0,
            "empty": True,
            "ok": True,
            "n_downloaded_zips": n_dl,
            "n_skip": n_skip,
        }

    all_df = pd.concat(frames, ignore_index=True)
    all_df["open_time"] = pd.to_numeric(all_df["open_time"], errors="coerce")
    all_df = all_df.dropna(subset=["open_time"])
    if all_df.empty:
        pd.DataFrame(columns=empty_cols).to_parquet(out_pq, index=False)
        return {
            "symbol": symbol,
            "kind": kind,
            "path": str(out_pq),
            "reused": False,
            "n_rows": 0,
            "empty": True,
            "ok": True,
        }
    ot = all_df["open_time"].to_numpy(dtype="float64")
    ms = np.where(ot > 1e14, ot / 1000.0, ot)
    all_df["ts"] = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    all_df = all_df.dropna(subset=["ts"])
    lo = pd.Timestamp("2017-01-01", tz="UTC")
    hi = pd.Timestamp("2030-12-31", tz="UTC")
    all_df = all_df[(all_df["ts"] >= lo) & (all_df["ts"] <= hi)]
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    all_df["symbol"] = symbol
    all_df["kind"] = kind
    all_df = (
        all_df[empty_cols]
        .dropna(subset=["ts", "close"])
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    all_df.to_parquet(out_pq, index=False)
    _log(f"{kind} {symbol} wrote n={len(all_df)} dl_zips={n_dl}")
    return {
        "symbol": symbol,
        "kind": kind,
        "path": str(out_pq),
        "reused": False,
        "n_rows": int(len(all_df)),
        "empty": False,
        "ok": True,
        "n_downloaded_zips": n_dl,
        "n_skip": n_skip,
    }


def _nonempty_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        df = pd.read_parquet(path)
        return len(df) > 0
    except Exception:
        return False


def plan_symbol_jobs(
    pit: pd.DataFrame,
    panel: pd.DataFrame,
    btc_id: int,
    spot_listed: set[str],
    um_listed: set[str],
    spot_dir: Path,
    um_dir: Path,
) -> dict:
    """Map every PIT id (+BTC) to ordered Binance candidates; list download jobs."""
    all_ids = sorted(set(int(i) for i in pit["id"].unique()) | {int(btc_id)})
    per_id: dict[int, list[str]] = {}
    wanted_spot: list[str] = []
    seen_s: set[str] = set()
    for iid in all_ids:
        if int(iid) == int(btc_id):
            cands = ["BTCUSDT"]
        else:
            cands = symbols_for_id(panel, int(iid))
            if not cands:
                last = panel.loc[panel["id"].astype(int) == int(iid), "symbol"]
                if len(last):
                    cands = candidate_usdt_symbols(str(last.iloc[-1]))
        per_id[int(iid)] = cands
        for c in cands:
            if spot_listed and c not in spot_listed:
                continue
            if c not in seen_s:
                seen_s.add(c)
                wanted_spot.append(c)
    # always try BTC
    if "BTCUSDT" not in seen_s:
        wanted_spot.insert(0, "BTCUSDT")
        seen_s.add("BTCUSDT")

    months = month_range(PHASE5_HOURLY_START)
    spot_todo = []
    spot_reuse = []
    for sym in wanted_spot:
        pq = spot_dir / f"{sym}.parquet"
        if pq.exists():
            spot_reuse.append(sym)
        else:
            spot_todo.append({"symbol": sym, "kind": "spot", "start_month": PHASE5_HOURLY_START})
    return {
        "all_ids": all_ids,
        "per_id": per_id,
        "wanted_spot": wanted_spot,
        "spot_todo": spot_todo,
        "spot_reuse": spot_reuse,
        "um_listed": sorted(um_listed),
        "spot_listed_n": int(len(spot_listed)),
        "um_listed_n": int(len(um_listed)),
        "n_months": int(len(months)),
        "months_start": months[0] if months else None,
        "months_end": months[-1] if months else None,
    }


def pick_id_sources(
    per_id: dict[int, list[str]],
    btc_id: int,
    spot_dir: Path,
    um_dir: Path,
    um_listed: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """After spot downloads: assign spot or queue perp fallback."""
    rows = []
    um_needed: list[str] = []
    seen_u: set[str] = set()
    for iid, cands in per_id.items():
        chosen = None
        kind = None
        for c in cands:
            pq = spot_dir / f"{c}.parquet"
            if _nonempty_parquet(pq):
                chosen, kind = c, "spot"
                break
        if chosen is None:
            for c in cands:
                if c in um_listed or True:
                    if c not in seen_u:
                        seen_u.add(c)
                        um_needed.append(c)
        rows.append({"id": int(iid), "symbol": chosen, "kind": kind, "candidates": ",".join(cands)})
    # BTC forced
    btc_row = next(r for r in rows if r["id"] == int(btc_id))
    if btc_row["symbol"] is None:
        um_needed = ["BTCUSDT"] + [s for s in um_needed if s != "BTCUSDT"]
    map_df = pd.DataFrame(rows)
    return map_df, um_needed


def finalize_id_map(map_df: pd.DataFrame, spot_dir: Path, um_dir: Path, btc_id: int) -> pd.DataFrame:
    out = map_df.copy()
    for i, row in out.iterrows():
        if row["symbol"] and row["kind"]:
            continue
        cands = str(row["candidates"] or "").split(",") if row["candidates"] else []
        picked = None
        kind = None
        for c in cands:
            if not c:
                continue
            if _nonempty_parquet(spot_dir / f"{c}.parquet"):
                picked, kind = c, "spot"
                break
        if picked is None:
            for c in cands:
                if not c:
                    continue
                if _nonempty_parquet(um_dir / f"{c}.parquet"):
                    picked, kind = c, "perp"
                    break
        out.at[i, "symbol"] = picked
        out.at[i, "kind"] = kind
    if int(btc_id) in set(out["id"].astype(int)):
        idx = out.index[out["id"].astype(int) == int(btc_id)][0]
        if _nonempty_parquet(spot_dir / "BTCUSDT.parquet"):
            out.at[idx, "symbol"] = "BTCUSDT"
            out.at[idx, "kind"] = "spot"
        elif _nonempty_parquet(um_dir / "BTCUSDT.parquet"):
            out.at[idx, "symbol"] = "BTCUSDT"
            out.at[idx, "kind"] = "perp"
    return out


def assemble_hourly_panel(id_map: pd.DataFrame, spot_dir: Path, um_dir: Path) -> pd.DataFrame:
    parts = []
    for row in id_map.itertuples(index=False):
        if not row.symbol or not row.kind:
            continue
        src = spot_dir if row.kind == "spot" else um_dir
        pq = src / f"{row.symbol}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if df.empty:
            continue
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["id"] = int(row.id)
        df["source"] = str(row.kind)
        df["binance_symbol"] = str(row.symbol)
        parts.append(df)
    if not parts:
        raise RuntimeError("hourly panel empty — no symbol caches loaded")
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = panel["ts"].dt.tz_convert("UTC").dt.normalize()
    panel = panel.sort_values(["id", "ts"]).drop_duplicates(["id", "ts"]).reset_index(drop=True)
    return panel


def audit_hourly_panel(
    hourly: pd.DataFrame,
    daily_panel: pd.DataFrame,
    pit: pd.DataFrame,
    btc_id: int,
) -> dict:
    """Gaps, duplicates, zero-volume runs, daily-close alignment."""
    h = hourly.copy()
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    h["date"] = pd.to_datetime(h["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    h["id"] = h["id"].astype(int)
    h["year"] = h["ts"].dt.year

    dup = int(h.duplicated(["id", "ts"]).sum())
    gap_rows = []
    zv_rows = []
    for iid, g in h.groupby("id", sort=False):
        g = g.sort_values("ts")
        ts = g["ts"]
        if len(ts) < 2:
            continue
        dlt = ts.diff().dt.total_seconds() / 3600.0
        n_gap = int((dlt > 1.5).sum())
        hours_missing = float((dlt[dlt > 1.5] - 1.0).clip(lower=0).sum())
        for yr, gy in g.groupby("year"):
            expected = 24 * (366 if yr % 4 == 0 else 365)
            gap_rows.append(
                {
                    "id": int(iid),
                    "year": int(yr),
                    "n_bars": int(len(gy)),
                    "expected": int(expected),
                    "coverage": float(len(gy) / max(expected, 1)),
                    "n_gap_events": int((gy["ts"].diff().dt.total_seconds() / 3600.0 > 1.5).sum()) if len(gy) > 1 else 0,
                }
            )
        vol = g["volume"].to_numpy(dtype=float)
        run = 0
        max_run = 0
        n_runs24 = 0
        for v in vol:
            if not np.isfinite(v) or v <= 0:
                run += 1
                max_run = max(max_run, run)
            else:
                if run >= 24:
                    n_runs24 += 1
                run = 0
        if run >= 24:
            n_runs24 += 1
        zv_rows.append(
            {
                "id": int(iid),
                "max_zero_vol_run": int(max_run),
                "n_zero_vol_runs_ge24": int(n_runs24),
                "n_gap_events": n_gap,
                "hours_missing": hours_missing,
            }
        )

    # last hourly close of UTC day vs daily panel close
    last_h = h.sort_values("ts").groupby(["id", "date"], sort=False).tail(1)
    last_h = last_h[["id", "date", "close"]].rename(columns={"close": "h_close"})
    d = daily_panel.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    d["id"] = d["id"].astype(int)
    d = d[["date", "id", "close"]].rename(columns={"close": "d_close"})
    m = last_h.merge(d, on=["id", "date"], how="inner")
    m["h_close"] = pd.to_numeric(m["h_close"], errors="coerce")
    m["d_close"] = pd.to_numeric(m["d_close"], errors="coerce")
    m = m[(m["d_close"] > 0) & m["h_close"].notna()]
    m["abs_bps"] = (m["h_close"] / m["d_close"] - 1.0).abs() * 1e4
    median_bps = float(m["abs_bps"].median()) if len(m) else float("nan")
    per_id_med = m.groupby("id")["abs_bps"].median()
    viol_ids = per_id_med[per_id_med >= float(PHASE5_ALIGN_BPS)]
    viol = [
        {"id": int(i), "median_abs_bps": float(v), "n": int((m["id"] == i).sum())}
        for i, v in viol_ids.sort_values(ascending=False).items()
    ]

    pit_c = pit.copy()
    pit_c["date"] = pd.to_datetime(pit_c["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit_c["id"] = pit_c["id"].astype(int)
    pit_c["year"] = pit_c["date"].dt.year
    have = set(int(i) for i in h["id"].unique())
    cov_rows = []
    years = sorted(int(y) for y in pit_c["year"].unique())
    for yr in years:
        py = pit_c[pit_c["year"] == yr]
        for lo, hi, name in PHASE3C_NAME_TIERS:
            if "rank" in py.columns:
                sl = py[(py["rank"] >= lo) & (py["rank"] <= hi)]
            else:
                sl = py
            ids = set(int(i) for i in sl["id"].unique())
            n_have = len(ids & have)
            cov_rows.append(
                {
                    "year": yr,
                    "tier": name,
                    "n_ids": int(len(ids)),
                    "n_with_hourly": int(n_have),
                    "frac": float(n_have / max(len(ids), 1)),
                }
            )
        ids = set(int(i) for i in py["id"].unique())
        cov_rows.append(
            {
                "year": yr,
                "tier": "all",
                "n_ids": int(len(ids)),
                "n_with_hourly": int(len(ids & have)),
                "frac": float(len(ids & have) / max(len(ids), 1)),
            }
        )

    src_counts = h.groupby("source")["id"].nunique().to_dict() if "source" in h.columns else {}
    return {
        "n_rows": int(len(h)),
        "n_ids": int(h["id"].nunique()),
        "n_symbols": int(h["binance_symbol"].nunique()) if "binance_symbol" in h.columns else None,
        "ts_min": str(h["ts"].min()),
        "ts_max": str(h["ts"].max()),
        "n_duplicate_bars": dup,
        "source_id_counts": {str(k): int(v) for k, v in src_counts.items()},
        "alignment_n_overlap_days": int(len(m)),
        "alignment_median_abs_bps": median_bps,
        "alignment_pass": bool(np.isfinite(median_bps) and median_bps < float(PHASE5_ALIGN_BPS)),
        "alignment_threshold_bps": float(PHASE5_ALIGN_BPS),
        "alignment_violations_n": int(len(viol)),
        "alignment_violations": viol[:40],
        "zero_volume": zv_rows,
        "gaps": gap_rows,
        "coverage": cov_rows,
        "btc_id": int(btc_id),
        "btc_in_panel": bool(int(btc_id) in have),
    }


def write_hourly_report(path: Path, audit: dict, extra: dict) -> None:
    lines = [
        "# BTC-BEATER Phase 5 — hourly panel audit",
        "",
        "BACKTEST/ANALYSIS ONLY. Binance 1h klines, spot primary, USDT-M perp fallback.",
        "",
        f"- rows = `{audit.get('n_rows')}` ids = `{audit.get('n_ids')}` symbols = `{audit.get('n_symbols')}`",
        f"- span = `{audit.get('ts_min')}` → `{audit.get('ts_max')}`",
        f"- duplicate bars = `{audit.get('n_duplicate_bars')}`",
        f"- sources (ids) = `{audit.get('source_id_counts')}`",
        f"- BTC in panel = `{audit.get('btc_in_panel')}` (id `{audit.get('btc_id')}`)",
        "",
        "## Daily-close alignment",
        "",
        f"Median |hourly last-bar / daily-panel close − 1| on overlapping (id, date) = "
        f"`{audit.get('alignment_median_abs_bps')}` bps "
        f"(need < `{audit.get('alignment_threshold_bps')}` bps). "
        f"pass=`{audit.get('alignment_pass')}` n=`{audit.get('alignment_n_overlap_days')}` "
        f"violating ids = `{audit.get('alignment_violations_n')}`.",
        "",
        "| id | median |Δ| bps | n overlap |",
        "|----|----------------|-----------|",
    ]
    for v in (audit.get("alignment_violations") or [])[:25]:
        lines.append(f"| {v.get('id')} | {v.get('median_abs_bps'):.3f} | {v.get('n')} |")
    if not audit.get("alignment_violations"):
        lines.append("| — | none | — |")
    lines += [
        "",
        "## Coverage per year / PIT tier (ids ever in floored top-100 that year)",
        "",
        "| year | tier | n ids | with hourly | frac |",
        "|------|------|-------|-------------|------|",
    ]
    for r in audit.get("coverage") or []:
        lines.append(
            f"| {r.get('year')} | {r.get('tier')} | {r.get('n_ids')} | {r.get('n_with_hourly')} | "
            f"{float(r.get('frac') or 0):.3f} |"
        )
    zv = audit.get("zero_volume") or []
    n_zv = sum(1 for r in zv if int(r.get("n_zero_vol_runs_ge24") or 0) > 0)
    lines += [
        "",
        f"Zero-volume runs ≥24h: `{n_zv}` / `{len(zv)}` ids.",
        f"Gap events (hour jumps >1.5h): `{int(sum(int(r.get('n_gap_events') or 0) for r in zv))}`.",
        "",
        f"Downloads logged: new=`{extra.get('n_new')}` reused=`{extra.get('n_reused')}` "
        f"empty=`{extra.get('n_empty')}` um_fallback_ids=`{extra.get('n_um_ids')}`.",
        "",
        "Idempotent per-symbol parquet cache under `/data/quant/raw/hourly_spot` and "
        "`/data/quant/raw/hourly_um`. Panel: `/data/quant/hourly/panel.parquet`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _channels_for_symbol(g: pd.DataFrame, btc_log: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    g = g.sort_values("ts")
    ts = pd.to_datetime(g["ts"], utc=True)
    close = g["close"].to_numpy(dtype=np.float64)
    high = g["high"].to_numpy(dtype=np.float64)
    low = g["low"].to_numpy(dtype=np.float64)
    vol = g["volume"].to_numpy(dtype=np.float64)
    tb = g["taker_buy_base"].to_numpy(dtype=np.float64) if "taker_buy_base" in g.columns else np.full_like(vol, np.nan)
    logc = np.log(np.clip(close, 1e-18, None))
    log_ret = np.diff(logc, prepend=logc[0])
    log_ret[0] = 0.0
    hl = np.log(np.clip(high, 1e-18, None) / np.clip(low, 1e-18, None))
    hl = np.clip(np.nan_to_num(hl, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 2.0)
    w = int(PHASE5_VOL_Z_WINDOW)
    vol_s = pd.Series(vol)
    mu = vol_s.rolling(w, min_periods=max(24, w // 4)).mean()
    sd = vol_s.rolling(w, min_periods=max(24, w // 4)).std(ddof=0)
    vol_z = ((vol_s - mu) / sd.replace(0.0, np.nan)).clip(-5, 5).to_numpy(dtype=np.float64)
    vol_z = np.nan_to_num(vol_z, nan=0.0)
    share = np.divide(tb, vol, out=np.full_like(vol, 0.5), where=vol > 0)
    share = np.clip(np.nan_to_num(share, nan=0.5), 0.0, 1.0)
    ts_idx = pd.DatetimeIndex(ts)
    btc_al = btc_log.reindex(ts_idx).to_numpy(dtype=np.float64) if btc_log is not None else np.zeros(len(ts))
    vs = log_ret - np.nan_to_num(btc_al, nan=0.0)
    x = np.stack([log_ret, hl, vol_z, share, vs], axis=1).astype(np.float32)
    return ts.to_numpy(), x


def btc_hourly_logret(hourly: pd.DataFrame, btc_id: int) -> pd.Series:
    g = hourly[hourly["id"].astype(int) == int(btc_id)].sort_values("ts")
    if g.empty:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(g["ts"], utc=True)
    close = g["close"].to_numpy(dtype=np.float64)
    logc = np.log(np.clip(close, 1e-18, None))
    lr = np.diff(logc, prepend=logc[0])
    lr[0] = 0.0
    return pd.Series(lr, index=pd.DatetimeIndex(ts), dtype=float)


def build_sequence_cache(
    hourly: pd.DataFrame,
    labeled: pd.DataFrame,
    btc_id: int,
    out_dir: Path,
    heartbeat=None,
) -> dict:
    """X.npy (N, 504, 5) + index.parquet for labeled (date, id) rows with enough hourly history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hourly.copy()
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    h["id"] = h["id"].astype(int)
    lab = labeled.copy()
    lab["date"] = pd.to_datetime(lab["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    lab["id"] = lab["id"].astype(int)
    btc_lr = btc_hourly_logret(h, btc_id)
    min_bars = int(PHASE5_SEQ_LEN * float(PHASE5_MIN_BARS_FRAC))

    grouped = {int(i): g.sort_values("ts") for i, g in h.groupby("id", sort=False)}
    ch_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ids = sorted(set(int(i) for i in lab["id"].unique()) & set(grouped))
    t0 = time.time()
    for k, iid in enumerate(ids, 1):
        ch_cache[iid] = _channels_for_symbol(grouped[iid], btc_lr)
        if heartbeat and k % 25 == 0:
            heartbeat.ping(f"channels {k}/{len(ids)}")
        if k % 40 == 0:
            _log(f"channels {k}/{len(ids)}")

    # count then write
    items: list[tuple] = []
    for iid, g in lab.groupby("id", sort=False):
        iid = int(iid)
        if iid not in ch_cache:
            continue
        ts, x = ch_cache[iid]
        ts_d = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert("UTC").normalize()
        # searchsorted on dates
        dates = pd.to_datetime(g["date"], utc=True).dt.tz_convert("UTC").dt.normalize().to_numpy()
        y_top = g["y_h14"].to_numpy() if "y_h14" in g.columns else np.full(len(g), np.nan)
        y_bot = g["y_bot_h14"].to_numpy() if "y_bot_h14" in g.columns else np.full(len(g), np.nan)
        ex = g["excess_h14"].to_numpy() if "excess_h14" in g.columns else np.full(len(g), np.nan)
        sym = g["symbol"].astype(str).to_numpy() if "symbol" in g.columns else np.array(["?"] * len(g))
        for dt, yt, yb, e, sy in zip(dates, y_top, y_bot, ex, sym):
            dt = pd.Timestamp(dt)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            # last index with calendar date <= dt
            j = int(np.searchsorted(ts_d, dt, side="right") - 1)
            if j < 0:
                continue
            start = max(0, j - PHASE5_SEQ_LEN + 1)
            sl = x[start : j + 1]
            if sl.shape[0] < min_bars:
                continue
            items.append((dt, iid, sl, yt, yb, e, sy, int(sl.shape[0])))
        if heartbeat:
            heartbeat.ping(f"index id={iid} n={len(items)}")

    n = len(items)
    if n == 0:
        raise RuntimeError("hourly sequence cache empty")
    x_path = out_dir / "X.npy"
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n, PHASE5_SEQ_LEN, PHASE5_N_CHANNELS))
    rows = []
    for i, (dt, iid, sl, yt, yb, e, sy, slen) in enumerate(items):
        pad = PHASE5_SEQ_LEN - sl.shape[0]
        win = np.zeros((PHASE5_SEQ_LEN, PHASE5_N_CHANNELS), dtype=np.float32)
        win[pad:] = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
        X[i] = win
        rows.append(
            {
                "row_id": i,
                "date": dt,
                "id": int(iid),
                "symbol": sy,
                "seq_len": slen,
                "y_h14": float(yt) if yt is not None and np.isfinite(yt) else np.nan,
                "y_bot_h14": float(yb) if yb is not None and np.isfinite(yb) else np.nan,
                "excess_h14": float(e) if e is not None and np.isfinite(e) else np.nan,
            }
        )
        if heartbeat and (i + 1) % 20000 == 0:
            heartbeat.ping(f"write {i+1}/{n}")
    X.flush()
    del X
    idx = pd.DataFrame(rows)
    idx["date"] = pd.to_datetime(idx["date"], utc=True)
    idx.to_parquet(out_dir / "index.parquet", index=False)
    meta = {
        "n_rows": int(len(idx)),
        "seq_len": PHASE5_SEQ_LEN,
        "n_channels": PHASE5_N_CHANNELS,
        "channels": list(PHASE5_CHANNELS),
        "nbytes": int(n * PHASE5_SEQ_LEN * PHASE5_N_CHANNELS * 4),
        "elapsed_sec": time.time() - t0,
        "n_ids": int(idx["id"].nunique()),
        "n_dates": int(idx["date"].nunique()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _log(f"seq cache n={meta['n_rows']} nbytes={meta['nbytes']/1e9:.2f}GB")
    return meta
