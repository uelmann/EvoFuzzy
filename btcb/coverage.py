"""True-top-N coverage vs external CMC historical snapshots (gate v2)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from btcb.constants import BLOCK_BEFORE, COVERAGE_NS, COVERAGE_THRESH


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def coverage_at(
    panel: pd.DataFrame,
    snap: pd.DataFrame,
    D,
    ns=COVERAGE_NS,
    slack_days: int = 2,
) -> dict:
    D = _as_utc(D).normalize()
    dates = pd.to_datetime(panel["date"], utc=True)
    near = panel[(dates >= D - pd.Timedelta(days=slack_days)) & (dates <= D + pd.Timedelta(days=slack_days))]
    have = set(int(x) for x in near["id"].unique())
    out = {"D": str(D.date()), "n_panel_near": int(len(have))}
    for n in ns:
        top = snap[snap["rank"] <= int(n)] if "rank" in snap.columns else snap.head(int(n))
        ids = [int(x) for x in top["id"].head(int(n))]
        hit = sum(1 for i in ids if i in have)
        out[f"top{n}"] = {"need": int(len(ids)), "hit": int(hit), "frac": float(hit / max(len(ids), 1))}
    return out


def scan_usable_from_snapshots(panel: pd.DataFrame, snap_dir: Path) -> dict:
    files = sorted(snap_dir.glob("*.parquet"))
    rows = []
    for p in files:
        qe = p.stem
        meta = {}
        mj = p.with_suffix(".json")
        if mj.exists():
            meta = json.loads(mj.read_text())
        used = meta.get("used_date", qe)
        snap = pd.read_parquet(p)
        if snap.empty:
            continue
        cov = coverage_at(panel, snap, used)
        cov["quarter_end"] = qe
        cov["used_date"] = used
        cov["pass100"] = bool(cov["top100"]["frac"] >= COVERAGE_THRESH)
        rows.append(cov)
    rows.sort(key=lambda r: r["used_date"])
    usable = None
    for i, r in enumerate(rows):
        later = rows[i:]
        if later and all(x["pass100"] for x in later):
            usable = r
            break
    blocked = False
    fiction = False
    if usable is None:
        blocked = True
        verdict = "BLOCKED"
        usable_from = None
    else:
        d = usable["used_date"][:7]
        if pd.Timestamp(usable["used_date"]) > pd.Timestamp(BLOCK_BEFORE):
            blocked = True
            verdict = "BLOCKED"
            usable_from = None
        else:
            verdict = f"USABLE-FROM-{d}"
            usable_from = usable["used_date"]
    return {
        "verdict": verdict,
        "blocked": blocked,
        "fiction_2018_2020": fiction,
        "usable_from": usable_from,
        "usable_from_month": None if not usable_from else usable_from[:7],
        "threshold": COVERAGE_THRESH,
        "n_snapshots": len(rows),
        "scan": rows,
    }
