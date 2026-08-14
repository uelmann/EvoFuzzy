"""Full-map target construction, credit guard, resumable OHLCV download."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from btcb.cmc_client import CmcPublic, fetch_id_history
from btcb.constants import (
    DOWNLOAD_MAX_YEARS,
    DOWNLOAD_PERIOD_DAYS,
    DOWNLOAD_SLEEP_S,
    HTTP_HARD_STOP,
    SNAPSHOT_END,
    SNAPSHOT_START,
    SNAPSHOT_TOPN,
)


def quarter_end_dates(start=SNAPSHOT_START, end=SNAPSHOT_END) -> list[date]:
    qs = [(3, 31), (6, 30), (9, 30), (12, 31)]
    y, q = start
    ey, eq = end
    out: list[date] = []
    while (y, q) <= (ey, eq):
        m, d = qs[q - 1]
        out.append(date(y, m, d))
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out


def _listing_rows(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for x in raw:
        quotes = x.get("quotes") or []
        q0 = quotes[0] if quotes else {}
        rows.append(
            {
                "id": int(x["id"]),
                "name": x.get("name"),
                "symbol": x.get("symbol"),
                "slug": x.get("slug"),
                "rank": int(x.get("cmcRank") or 0),
                "marketCap": float(q0.get("marketCap") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def fetch_snapshot(api: CmcPublic, d: date, topn: int = SNAPSHOT_TOPN) -> tuple[pd.DataFrame, str]:
    for back in range(0, 7):
        dd = d - timedelta(days=back)
        iso = dd.isoformat()
        raw = api.fetch_historical_listing(iso, start=1, limit=topn)
        if raw:
            df = _listing_rows(raw)
            if len(df):
                return df, iso
    return pd.DataFrame(), d.isoformat()


def credit_guard(*, n_ids: int, n_cached: int, n_snapshots: int, max_years: int = DOWNLOAD_MAX_YEARS) -> dict:
    """Project HTTP volume. Existing script is public data-api (credit_count=0)."""
    windows = int((max_years * 365.25 + DOWNLOAD_PERIOD_DAYS - 1) // DOWNLOAD_PERIOD_DAYS)
    remaining_ids = max(0, int(n_ids) - int(n_cached))
    http_remaining = remaining_ids * windows
    http_total = 2 + int(n_snapshots) + int(n_ids) * windows
    pro_keys = [k for k in ("CMC_PRO_API_KEY", "COINMARKETCAP_API_KEY", "CMC_API_KEY") if os.environ.get(k)]
    plan = "public-data-api (existing download_cmc_kucoin.py; credit_count=0; no plan meter)"
    if pro_keys:
        plan += f"; Pro key present in env ({','.join(pro_keys)}) but NOT used for this job"
    hard_stop = http_remaining > HTTP_HARD_STOP
    proposal = None
    if hard_stop:
        proposal = {
            "reason": f"remaining HTTP GETs {http_remaining} > {HTTP_HARD_STOP}",
            "options": [
                "Reduce snapshot union from top-500 to top-300 (quarterly).",
                "Switch snapshots from quarterly to year-end only (2017–2025).",
                "Cap target ids at current-828 union top-300-per-snapshot.",
            ],
        }
    return {
        "plan": plan,
        "credits_available": None,
        "credits_projected": 0,
        "http_hard_stop_threshold": HTTP_HARD_STOP,
        "windows_per_id": windows,
        "n_ids": int(n_ids),
        "n_cached": int(n_cached),
        "remaining_ids": remaining_ids,
        "http_remaining": int(http_remaining),
        "http_total_if_cold": int(http_total),
        "hard_stop": bool(hard_stop),
        "reduction_proposal": proposal,
        "pro_keys_ignored": pro_keys,
    }


def load_current_828(path: Path) -> set[int]:
    df = pd.read_csv(path)
    col = "id" if "id" in df.columns else "cryptocurrency_id"
    return set(int(x) for x in df[col].dropna())


def build_target_ids(
    api: CmcPublic,
    snap_dir: Path,
    current_ids: set[int],
    *,
    topn: int = SNAPSHOT_TOPN,
) -> tuple[set[int], list[dict]]:
    snap_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    union: set[int] = set(current_ids)
    for d in quarter_end_dates():
        cache = snap_dir / f"{d.isoformat()}.parquet"
        meta_p = snap_dir / f"{d.isoformat()}.json"
        if cache.exists():
            df = pd.read_parquet(cache)
            used = json.loads(meta_p.read_text())["used_date"] if meta_p.exists() else d.isoformat()
            print(f"[HB] snapshot reuse {d} n={len(df)}", flush=True)
        else:
            df, used = fetch_snapshot(api, d, topn=topn)
            if df.empty:
                print(f"[HB] snapshot EMPTY {d}", flush=True)
            else:
                df.to_parquet(cache, index=False)
            meta_p.write_text(
                json.dumps(
                    {
                        "quarter_end": d.isoformat(),
                        "used_date": used,
                        "n": int(len(df)),
                        "topn": int(topn),
                        "source": f"{api.__class__.__name__} listings/historical",
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
            print(f"[HB] snapshot {d} used={used} n={len(df)}", flush=True)
        ids = set(int(x) for x in df["id"]) if len(df) else set()
        n_new = len(ids - union)
        union |= ids
        used_date = used
        if meta_p.exists():
            used_date = json.loads(meta_p.read_text()).get("used_date", used)
        provenance.append(
            {
                "quarter_end": d.isoformat(),
                "used_date": used_date,
                "n": int(len(df)),
                "n_new": int(n_new),
            }
        )
    return union, provenance


def download_ohlcv(
    api: CmcPublic,
    cmap: pd.DataFrame,
    target_ids: list[int],
    ohlcv_dir: Path,
    state_path: Path,
    *,
    commit_fn=None,
    save_every: int = 25,
    max_years: int = DOWNLOAD_MAX_YEARS,
) -> dict:
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    state = {"completed": [], "empty": [], "failed": []}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    done = set(int(x) for x in state.get("completed", [])) | set(int(x) for x in state.get("empty", []))
    done |= {int(p.stem) for p in ohlcv_dir.glob("*.parquet") if p.stem.isdigit()}
    done |= {int(p.stem) for p in ohlcv_dir.glob("*.empty") if p.stem.isdigit()}
    meta_by_id = cmap.set_index("id").to_dict("index") if len(cmap) else {}
    todo = [i for i in target_ids if i not in done]
    print(f"[HB] ohlcv todo={len(todo)} already={len(done)} total={len(target_ids)}", flush=True)
    t0 = datetime.now(timezone.utc)
    for n, cid in enumerate(todo, 1):
        cache = ohlcv_dir / f"{cid}.parquet"
        empty_mark = ohlcv_dir / f"{cid}.empty"
        if cache.exists() or empty_mark.exists():
            try:
                if empty_mark.exists() or (cache.exists() and len(pd.read_parquet(cache)) == 0):
                    state.setdefault("empty", []).append(int(cid))
                else:
                    state.setdefault("completed", []).append(int(cid))
                continue
            except Exception:
                pass
        row = meta_by_id.get(int(cid), {})
        meta = {
            "name": row.get("name"),
            "symbol": row.get("symbol"),
            "slug": row.get("slug"),
        }
        try:
            hist = fetch_id_history(api, int(cid), max_years=max_years, meta=meta)
            if hist is None or hist.empty:
                empty_mark.touch()
                state.setdefault("empty", []).append(int(cid))
                print(f"[HB] empty id={cid} {meta.get('symbol')} ({n}/{len(todo)})", flush=True)
            else:
                hist.to_parquet(cache, index=False)
                state.setdefault("completed", []).append(int(cid))
                last = str(pd.Timestamp(hist["timestamp"].max()).date())
                print(
                    f"[HB] id={cid} {meta.get('symbol')} rows={len(hist)} last={last} "
                    f"({n}/{len(todo)}) http={api.http_count}",
                    flush=True,
                )
        except Exception as exc:
            state.setdefault("failed", []).append({"id": int(cid), "err": str(exc)[:200]})
            print(f"[HB] FAIL id={cid} {exc}", flush=True)
        if n % save_every == 0:
            state["http_count"] = api.http_count
            state["credit_count"] = api.credit_count
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            state_path.write_text(json.dumps(state))
            if commit_fn:
                commit_fn()
            elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
            print(f"[HB] checkpoint n={n} elapsed={elapsed:.0f}s http={api.http_count}", flush=True)
    state["http_count"] = api.http_count
    state["credit_count"] = api.credit_count
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state))
    if commit_fn:
        commit_fn()
    return state


def seed_from_existing_panel(panel_path: Path, ohlcv_dir: Path, target_ids: set[int]) -> int:
    """Copy already-downloaded 828-coin history into per-id cache (resume)."""
    if not panel_path.exists():
        return 0
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(panel_path)
    if "id" not in df.columns and "cryptocurrency_id" in df.columns:
        df["id"] = df["cryptocurrency_id"].astype(int)
    n = 0
    want = set(int(x) for x in target_ids)
    cols = [
        "cryptocurrency_id", "timestamp", "open", "high", "low", "close", "volume", "marketCap",
        "currency_name", "currency_symbol", "currency_slug",
    ]
    for cid, g in df.groupby("id"):
        cid = int(cid)
        if cid not in want:
            continue
        dest = ohlcv_dir / f"{cid}.parquet"
        if dest.exists() or (ohlcv_dir / f"{cid}.empty").exists():
            continue
        gg = g.copy()
        if "cryptocurrency_id" not in gg.columns:
            gg["cryptocurrency_id"] = cid
        if "timestamp" not in gg.columns:
            gg["timestamp"] = gg["date"]
        for c, src in (
            ("currency_name", "name"),
            ("currency_symbol", "symbol"),
            ("currency_slug", "slug"),
        ):
            if c not in gg.columns and src in gg.columns:
                gg[c] = gg[src]
        keep = [c for c in cols if c in gg.columns]
        out = gg[keep].drop_duplicates("timestamp").sort_values("timestamp")
        if out.empty:
            continue
        out.to_parquet(dest, index=False)
        n += 1
    print(f"[HB] seeded {n} ids from existing panel {panel_path}", flush=True)
    return n


def assemble_panel(ohlcv_dir: Path, cmap: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    files = sorted(ohlcv_dir.glob("*.parquet"))
    chunks = []
    for i, p in enumerate(files, 1):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        chunks.append(df)
        if i % 200 == 0:
            print(f"[HB] assemble {i}/{len(files)} chunks={len(chunks)}", flush=True)
    if not chunks:
        raise RuntimeError("no OHLCV parquet files to assemble")
    panel = pd.concat(chunks, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["timestamp"], utc=True).dt.normalize()
    panel["id"] = panel["cryptocurrency_id"].astype(int)
    panel["symbol"] = panel["currency_symbol"].astype(str)
    panel["name"] = panel["currency_name"].astype(str)
    panel["slug"] = panel["currency_slug"].astype(str)
    for c in ("open", "high", "low", "close", "volume", "marketCap"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["dv"] = panel["volume"].astype(float)
    panel["mcap"] = panel["marketCap"].astype(float)
    last = panel.groupby("id")["date"].max().rename("last_available_date")
    ls = cmap[["id", "listing_status"]].drop_duplicates("id") if "listing_status" in cmap.columns else pd.DataFrame(columns=["id", "listing_status"])
    panel = panel.merge(ls, on="id", how="left")
    panel = panel.merge(last.reset_index(), on="id", how="left")
    panel["listing_status"] = panel["listing_status"].fillna("unknown")
    panel = panel.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"[HB] panel rows={len(panel)} ids={panel['id'].nunique()} → {out_path}", flush=True)
    return panel
