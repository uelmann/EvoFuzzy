"""Phase 0 dataset audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    BLOCK_BEFORE,
    ENDED_LAG_DAYS,
    FICTION_BEFORE,
    LISTED_SLACK_DAYS,
    NAMED_GRAVEYARD,
    PIT_DATE_FRAC,
    PIT_DV_MIN_PERIODS,
    PIT_DV_WINDOW,
    PIT_MIN_NAMES,
    PRESENT_FRAC,
    SAMPLE_N,
    SAMPLE_TOPN,
    SAMPLE_YEARS,
    SCAN_START,
    SEED,
    STABLE_OR_WRAP,
    SURVIVOR_LAG_DAYS,
)


def _utc(s) -> pd.DatetimeIndex | pd.Timestamp:
    t = pd.to_datetime(s, utc=True)
    return t


def is_excluded_symbol(sym: str) -> bool:
    u = str(sym).upper()
    if u in STABLE_OR_WRAP:
        return True
    if u.endswith("USD") and u not in {"SAND", "BAND"} and len(u) <= 5:
        # crude: true USD stables already listed; leave SAND/BAND
        pass
    return False


def load_cmc_panel(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    df["symbol"] = df["currency_symbol"].astype(str)
    df["name"] = df["currency_name"].astype(str)
    df["slug"] = df["currency_slug"].astype(str)
    df["id"] = df["cryptocurrency_id"].astype(int)
    for c in ("open", "high", "low", "close", "volume", "marketCap"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # CMC volume is USD; dollar volume for PIT
    df["dv"] = df["volume"].astype(float)
    df["mcap"] = df["marketCap"].astype(float)
    df = df.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    return df.reset_index(drop=True)


def schema_report(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_ids": int(df["id"].nunique()),
        "n_symbols": int(df["symbol"].nunique()),
        "n_slugs": int(df["slug"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "columns": [
            "cryptocurrency_id", "timestamp", "open", "high", "low", "close",
            "volume", "marketCap", "currency_name", "currency_symbol", "currency_slug",
        ],
        "null_frac": {c: float(df[c].isna().mean()) for c in ("open", "high", "low", "close", "volume", "mcap")},
        "dup_symbol_ids": int(df.groupby("symbol")["id"].nunique().gt(1).sum()),
        "dup_id_symbols": int(df.groupby("id")["symbol"].nunique().gt(1).sum()),
    }


def per_coin_span(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("id").agg(
        symbol=("symbol", "last"),
        name=("name", "last"),
        slug=("slug", "last"),
        first=("date", "min"),
        last=("date", "max"),
        n=("date", "size"),
        last_close=("close", "last"),
        last_mcap=("mcap", "last"),
    )
    cal = (g["last"] - g["first"]).dt.days + 1
    g["calendar_days"] = cal.astype(int)
    g["gap_days"] = (g["calendar_days"] - g["n"]).clip(lower=0).astype(int)
    g["gap_frac"] = g["gap_days"] / g["calendar_days"].replace(0, np.nan)
    return g.reset_index()


def find_named(span: pd.DataFrame, df: pd.DataFrame, data_end: pd.Timestamp) -> list[dict]:
    rows = []
    for spec in NAMED_GRAVEYARD:
        q = spec["query"].upper()
        slugs = {s.lower() for s in spec["slugs"]}
        hit = span[(span["symbol"].str.upper() == q) | (span["slug"].str.lower().isin(slugs))]
        if hit.empty:
            hit = span[span["name"].str.contains(spec["query"], case=False, na=False)]
        if hit.empty:
            rows.append(
                {
                    "query": spec["query"],
                    "event": spec["event"],
                    "present": False,
                    "ids": [],
                    "symbol": None,
                    "slug": None,
                    "first": None,
                    "last": None,
                    "n": 0,
                    "terminal": "MISSING",
                    "crash_note": None,
                }
            )
            continue
        for _, r in hit.iterrows():
            term = _terminal(r["last"], data_end)
            crash_note = None
            cr = spec.get("expect_crash")
            if cr:
                t0 = pd.Timestamp(cr, tz="UTC")
                sub = df[(df["id"] == r["id"]) & (df["date"] >= t0 - pd.Timedelta(days=40)) & (df["date"] <= t0 + pd.Timedelta(days=40))]
                if len(sub) >= 5:
                    mx, mn = float(sub["close"].max()), float(sub["close"].min())
                    crash_note = f"window {cr} max={mx:.6g} min={mn:.6g} min/max={mn/mx if mx else float('nan'):.4f}"
            rows.append(
                {
                    "query": spec["query"],
                    "event": spec["event"],
                    "present": True,
                    "ids": [int(r["id"])],
                    "symbol": r["symbol"],
                    "slug": r["slug"],
                    "name": r["name"],
                    "first": str(r["first"].date()),
                    "last": str(r["last"].date()),
                    "n": int(r["n"]),
                    "terminal": term,
                    "crash_note": crash_note,
                }
            )
    return rows


def _terminal(last: pd.Timestamp, data_end: pd.Timestamp) -> str:
    lag = int((data_end - last).days)
    if lag <= SURVIVOR_LAG_DAYS:
        return "SURVIVOR"
    if lag >= ENDED_LAG_DAYS:
        return "ENDED"
    return "TRUNCATED"


def year_end_top(df: pd.DataFrame, year: int, n: int) -> pd.DataFrame:
    target = pd.Timestamp(f"{year}-12-31", tz="UTC")
    sl = df[df["date"] <= target]
    if sl.empty:
        return pd.DataFrame()
    d = sl["date"].max()
    day = sl[sl["date"] == d].dropna(subset=["mcap"])
    day = day[~day["symbol"].str.upper().isin(STABLE_OR_WRAP)]
    day = day[day["symbol"].str.upper() != "BTC"]
    return day.sort_values("mcap", ascending=False).head(int(n))


def draw_sample(df: pd.DataFrame, span: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for y in SAMPLE_YEARS:
        t = year_end_top(df, y, SAMPLE_TOPN)
        if not t.empty:
            t = t.assign(sample_year=y)
            parts.append(t[["id", "symbol", "name", "slug", "mcap", "sample_year", "date"]])
            print(f"[HB] year-end {y} session={t['date'].iloc[0].date()} n={len(t)}", flush=True)
    if not parts:
        return pd.DataFrame()
    uni = pd.concat(parts, ignore_index=True)
    ids = uni["id"].drop_duplicates().to_numpy()
    rng = np.random.RandomState(SEED)
    if len(ids) > SAMPLE_N:
        pick = rng.choice(ids, size=SAMPLE_N, replace=False)
    else:
        pick = ids
    pick = set(int(x) for x in pick)
    out = span[span["id"].isin(pick)].copy()
    yrs = uni[uni["id"].isin(pick)].groupby("id")["sample_year"].apply(lambda s: ",".join(str(int(x)) for x in sorted(set(s))))
    out = out.merge(yrs.rename("in_years").reset_index(), on="id", how="left")
    return out


def sample_ok_from(row, D: pd.Timestamp, data_end: pd.Timestamp) -> bool:
    first, last = row["first"], row["last"]
    in_era = first <= pd.Timestamp("2020-12-31", tz="UTC")
    listed_by_d = first <= D + pd.Timedelta(days=LISTED_SLACK_DAYS)
    if not (in_era or listed_by_d):
        return False
    term = _terminal(last, data_end)
    return term in {"SURVIVOR", "ENDED"}


def pit_coverage(dv_med: pd.DataFrame, D: pd.Timestamp, min_names: int) -> dict:
    idx = dv_med.index[dv_med.index >= D]
    if len(idx) == 0:
        return {"frac": 0.0, "n_dates": 0, "n_ok": 0}
    cnt = dv_med.loc[idx].notna().sum(axis=1)
    n_ok = int((cnt >= min_names).sum())
    return {"frac": float(n_ok / len(idx)), "n_dates": int(len(idx)), "n_ok": n_ok}


def scan_usable_start(sample: pd.DataFrame, dv_med: pd.DataFrame, data_end: pd.Timestamp) -> dict:
    starts = pd.date_range(SCAN_START, BLOCK_BEFORE, freq="MS", tz="UTC")
    rows = []
    need = PRESENT_FRAC * max(len(sample), 1)
    for D in starts:
        n_ok = int(sum(sample_ok_from(r, D, data_end) for _, r in sample.iterrows())) if len(sample) else 0
        frac = n_ok / max(len(sample), 1)
        pit = pit_coverage(dv_med, D, PIT_MIN_NAMES)
        ok = (frac >= PRESENT_FRAC) and (pit["frac"] >= PIT_DATE_FRAC)
        rows.append(
            {
                "D": str(D.date())[:7],
                "D_iso": str(D.date()),
                "sample_ok": n_ok,
                "sample_n": int(len(sample)),
                "sample_frac": float(frac),
                "pit_frac": pit["frac"],
                "pit_n_ok": pit["n_ok"],
                "pass": bool(ok),
            }
        )
    passed = [r for r in rows if r["pass"]]
    fiction = False
    blocked = False
    usable = None
    if passed:
        usable = passed[0]
        if pd.Timestamp(usable["D_iso"], tz="UTC") >= pd.Timestamp(FICTION_BEFORE, tz="UTC"):
            fiction = True
        if pd.Timestamp(usable["D_iso"], tz="UTC") >= pd.Timestamp(BLOCK_BEFORE, tz="UTC"):
            blocked = True
            usable = None
    else:
        blocked = True
        fiction = True
    if blocked:
        verdict = "BLOCKED"
    elif fiction:
        verdict = f"USABLE-FROM-{usable['D']} (2018–2020 FICTION)"
    else:
        verdict = f"USABLE-FROM-{usable['D']}"
    return {
        "verdict": verdict,
        "blocked": blocked,
        "fiction_2018_2020": fiction and not blocked,
        "usable_from": None if blocked else usable["D_iso"],
        "usable_from_month": None if blocked else usable["D"],
        "need_sample_ok": float(need),
        "scan": rows,
    }


def quality_flags(df: pd.DataFrame, span: pd.DataFrame) -> dict:
    # large one-day close jumps (redenomination / split suspects)
    df = df.sort_values(["id", "date"])
    rets = df.groupby("id")["close"].pct_change()
    jump = df.assign(abs_ret=rets.abs())
    suspects = (
        jump[jump["abs_ret"] > 5]
        .groupby(["id", "symbol", "slug"], as_index=False)
        .agg(n_jumps=("abs_ret", "size"), max_abs_ret=("abs_ret", "max"), date=("date", "first"))
    )
    stables_present = sorted(set(span["symbol"].str.upper()) & STABLE_OR_WRAP)
    wrapped = [s for s in stables_present if s in {"WBTC", "WETH", "STETH", "WBETH", "BTCB", "RENBTC", "CBETH"}]
    stables = [s for s in stables_present if s not in set(wrapped)]
    dup_sym = (
        span.groupby("symbol")["id"].nunique().reset_index(name="n_ids")
    )
    dup_sym = dup_sym[dup_sym["n_ids"] > 1]
    return {
        "stables_in_panel": stables,
        "wrapped_in_panel": wrapped,
        "dup_tickers": dup_sym.to_dict("records"),
        "redenom_suspects": suspects.head(40).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        "n_redenom_suspects": int(len(suspects)),
        "gap_p50": float(span["gap_frac"].median()) if len(span) else float("nan"),
        "gap_p90": float(span["gap_frac"].quantile(0.9)) if len(span) else float("nan"),
        "n_high_gap": int((span["gap_frac"] > 0.05).sum()) if len(span) else 0,
    }


def binance_agreement(cmc: pd.DataFrame, bnc: pd.DataFrame, symbols: list[str]) -> dict:
    rows = []
    for sym in symbols:
        c = cmc[cmc["symbol"].str.upper() == sym][["date", "close"]].drop_duplicates("date").set_index("date")["close"].sort_index()
        b = bnc[bnc["symbol"].str.upper().str.replace("USDT", "", regex=False) == sym]
        if b.empty:
            # panel symbols are BTCUSDT
            b = bnc[bnc["symbol"].str.upper() == f"{sym}USDT"]
        if b.empty or c.empty:
            rows.append({"symbol": sym, "present": False})
            continue
        b = b.drop_duplicates("date").set_index("date")["close"].sort_index()
        c_ret = c.pct_change()
        b_ret = b.pct_change()
        a, bb = c_ret.align(b_ret, join="inner")
        m = a.notna() & bb.notna()
        a, bb = a[m], bb[m]
        if len(a) < 30:
            rows.append({"symbol": sym, "present": True, "n": int(len(a)), "corr": float("nan")})
            continue
        corr = float(a.corr(bb))
        mad = float((a - bb).abs().max())
        rows.append(
            {
                "symbol": sym,
                "present": True,
                "n": int(len(a)),
                "corr": corr,
                "max_abs_diff": mad,
                "mean_abs_diff": float((a - bb).abs().mean()),
            }
        )
    corrs = [r["corr"] for r in rows if r.get("present") and np.isfinite(r.get("corr", float("nan")))]
    med = float(np.median(corrs)) if corrs else float("nan")
    return {
        "rows": rows,
        "n_compared": int(len(corrs)),
        "median_corr": med,
        "suspect": bool(np.isfinite(med) and med < 0.99),
        "min_corr": float(np.min(corrs)) if corrs else float("nan"),
    }
