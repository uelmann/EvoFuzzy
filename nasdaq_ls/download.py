"""Yahoo Finance Nasdaq-100 panel + spliced market series."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nasdaq_ls.constants import (
    DATA_DIR,
    FALLBACK_TICKERS,
    MARKET_IXIC,
    MARKET_PATH,
    MARKET_QQQ,
    PANEL_PATH,
    PRICE_START,
    TICKERS_PATH,
)


def _log(msg: str) -> None:
    print(f"[nasdaq {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _yahoo_symbol(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return t.replace(".", "-")


def fetch_nasdaq100_tickers() -> tuple[list[str], str]:
    """Live Wikipedia scrape; fallback to the baked-in list."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
    except Exception as e:
        _log(f"wikipedia read_html failed: {e}")
        tables = []
    tickers: list[str] = []
    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        ticker_col = None
        for i, c in enumerate(cols):
            if c in {"ticker", "symbol", "ticker symbol"} or "ticker" in c:
                ticker_col = tbl.columns[i]
                break
        if ticker_col is None:
            continue
        raw = tbl[ticker_col].astype(str).str.strip()
        cand = [
            _yahoo_symbol(x)
            for x in raw
            if x and x.lower() not in {"nan", "ticker", "symbol"} and x.isascii()
        ]
        cand = [c for c in cand if 1 <= len(c) <= 6 and c.replace("-", "").isalnum()]
        if len(cand) >= 80:
            tickers = sorted(set(cand))
            break
    if len(tickers) >= 80:
        _log(f"wikipedia nasdaq-100 n={len(tickers)}")
        return tickers, "wikipedia"
    _log(f"wikipedia insufficient (n={len(tickers)}); using fallback n={len(FALLBACK_TICKERS)}")
    return sorted(set(FALLBACK_TICKERS)), "fallback"


def _flatten_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        top = [str(x).lower() for x in raw.columns.get_level_values(0)]
        # yfinance default: attributes at level 0, tickers at level 1
        if any(x in top for x in ("open", "close", "adj close", "high", "low", "volume")):
            for t in tickers:
                try:
                    sub = raw.xs(t, axis=1, level=1, drop_level=True)
                except Exception:
                    continue
                part = _one_ohlcv(sub, t)
                if part is not None:
                    frames.append(part)
        else:
            for t in tickers:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t]
                part = _one_ohlcv(sub, t)
                if part is not None:
                    frames.append(part)
    else:
        part = _one_ohlcv(raw, tickers[0] if len(tickers) == 1 else "UNK")
        if part is not None:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _one_ohlcv(sub: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Build one symbol. Working `close` is Yahoo Adj Close (total return)."""
    if sub is None or sub.empty:
        return None
    d = sub.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    raw_col = "close" if "close" in d.columns else None
    adj_col = "adj_close" if "adj_close" in d.columns else ("adjclose" if "adjclose" in d.columns else None)
    if raw_col is None and adj_col is None:
        return None
    raw = pd.to_numeric(d[raw_col], errors="coerce") if raw_col else None
    adj = pd.to_numeric(d[adj_col], errors="coerce") if adj_col else None
    if adj is None or not np.isfinite(adj).any():
        adj = raw
    if raw is None or not np.isfinite(raw).any():
        raw = adj
    if adj is None or raw is None:
        return None
    vol = pd.to_numeric(d["volume"], errors="coerce") if "volume" in d.columns else pd.Series(np.nan, index=d.index)
    factor = adj / raw.replace(0, np.nan)
    o = pd.to_numeric(d["open"], errors="coerce") if "open" in d.columns else adj
    h = pd.to_numeric(d["high"], errors="coerce") if "high" in d.columns else adj
    l = pd.to_numeric(d["low"], errors="coerce") if "low" in d.columns else adj
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    idx = pd.DatetimeIndex(idx).tz_localize("UTC").normalize()
    out = pd.DataFrame(
        {
            "date": np.asarray(idx),
            "symbol": symbol,
            "open": np.asarray(o * factor, dtype=float),
            "high": np.asarray(h * factor, dtype=float),
            "low": np.asarray(l * factor, dtype=float),
            "close": np.asarray(adj, dtype=float),
            "adj_close": np.asarray(adj, dtype=float),
            "close_raw": np.asarray(raw, dtype=float),
            "volume": np.asarray(vol, dtype=float),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    out = out[out["close"] > 0]
    if out.empty:
        return None
    # Liquidity rank uses unadjusted close × volume, not Adj Close.
    out["dollar_volume"] = out["close_raw"] * out["volume"].fillna(0.0)
    return out


def _download_group(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    _log(f"yfinance download n={len(tickers)} start={start}")
    raw = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        group_by="column",
        threads=True,
        progress=False,
    )
    panel = _flatten_yf(raw, tickers)
    got = sorted(panel["symbol"].unique()) if not panel.empty else []
    missing = [t for t in tickers if t not in set(got)]
    if missing:
        _log(f"batch missing n={len(missing)}; retrying singles")
        parts = [panel] if not panel.empty else []
        for t in missing:
            try:
                one = yf.download(
                    tickers=t,
                    start=start,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                p = _flatten_yf(one, [t])
                if not p.empty:
                    parts.append(p)
                else:
                    _log(f"skip {t}: empty")
            except Exception as e:
                _log(f"skip {t}: {e}")
        panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return panel


def splice_market(ixic: pd.Series, qqq: pd.Series) -> pd.Series:
    ixic = pd.to_numeric(ixic, errors="coerce").dropna().sort_index()
    qqq = pd.to_numeric(qqq, errors="coerce").dropna().sort_index()
    if qqq.empty:
        return ixic
    if ixic.empty:
        return qqq
    start = qqq.index[0]
    pre = ixic[ixic.index < start]
    if pre.empty:
        return qqq
    anchor = ixic[ixic.index <= start]
    if anchor.empty or float(anchor.iloc[-1]) == 0:
        return pd.concat([pre, qqq]).sort_index()
    scale = float(qqq.iloc[0]) / float(anchor.iloc[-1])
    head = pre * scale
    return pd.concat([head, qqq]).sort_index()


def download_all(force: bool = False) -> dict:
    dest = Path(DATA_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    panel_path = Path(PANEL_PATH)
    market_path = Path(MARKET_PATH)
    tickers_path = Path(TICKERS_PATH)

    if panel_path.exists() and market_path.exists() and tickers_path.exists() and not force:
        meta = json.loads(tickers_path.read_text())
        _log(f"using cached panel rows={meta.get('n_rows')} tickers={len(meta.get('tickers') or [])}")
        return meta

    tickers, source = fetch_nasdaq100_tickers()
    panel = _download_group(tickers, PRICE_START)
    if panel.empty:
        raise RuntimeError("empty Yahoo panel")

    mkt_panel = _download_group([MARKET_QQQ, MARKET_IXIC], PRICE_START)
    if mkt_panel.empty:
        raise RuntimeError("empty Yahoo market panel")
    close_w = mkt_panel.pivot(index="date", columns="symbol", values="close").sort_index()
    qqq = close_w[MARKET_QQQ] if MARKET_QQQ in close_w.columns else pd.Series(dtype=float)
    ixic = close_w[MARKET_IXIC] if MARKET_IXIC in close_w.columns else pd.Series(dtype=float)
    market = splice_market(ixic, qqq)
    market.name = "close"
    market_df = market.rename("close").to_frame().reset_index()
    if "date" not in market_df.columns:
        market_df = market_df.rename(columns={market_df.columns[0]: "date"})
    market_df["date"] = pd.to_datetime(market_df["date"], utc=True).dt.normalize()

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    market_df.to_parquet(market_path, index=False)
    meta = {
        "source": source,
        "tickers": sorted(panel["symbol"].unique()),
        "requested": tickers,
        "n_rows": int(len(panel)),
        "n_symbols": int(panel["symbol"].nunique()),
        "min_date": str(panel["date"].min()),
        "max_date": str(panel["date"].max()),
        "market_min": str(market_df["date"].min()),
        "market_max": str(market_df["date"].max()),
        "price_start": PRICE_START,
        "returns": "yahoo_adj_close",
        "dollar_volume": "unadjusted_close_times_volume",
    }
    tickers_path.write_text(json.dumps(meta, indent=2))
    _log(
        f"saved panel symbols={meta['n_symbols']} rows={meta['n_rows']} "
        f"{meta['min_date'][:10]}→{meta['max_date'][:10]} source={source}"
    )
    return meta
