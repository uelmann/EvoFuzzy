"""Public CoinMarketCap data-api client (same surface as download_cmc_kucoin).

The existing script has no Pro API key and no credit plan. Responses report
credit_count=0. Paid Pro historical endpoints are never used here.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE = "https://api.coinmarketcap.com/data-api/v3"
CRYPTOS_JSON = "https://s3.coinmarketcap.com/generated/core/crypto/cryptos.json"
CONVERT_USD = 2781

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://coinmarketcap.com",
    "referer": "https://coinmarketcap.com/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}


def _status_label(is_active, status) -> str:
    try:
        ia = int(is_active)
    except Exception:
        ia = 1 if is_active else 0
    try:
        st = int(status) if status is not None else (1 if ia else 3)
    except Exception:
        st = 1 if ia else 3
    if ia == 1 or st == 1:
        return "active"
    if st == 2:
        return "untracked"
    return "inactive"


class CmcPublic:
    def __init__(self, sleep_s: float = 0.12, retries: int = 6):
        self.sleep_s = float(sleep_s)
        self.retries = int(retries)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.http_count = 0
        self.credit_count = 0

    def get(self, url: str, params: dict | None = None) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                self.http_count += 1
                if resp.status_code in (429, 503):
                    wait = 2 ** attempt
                    print(f"[HB] HTTP {resp.status_code} sleep {wait}s {url}", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(self.sleep_s)
                js = resp.json()
                st = js.get("status") if isinstance(js, dict) else None
                if isinstance(st, dict):
                    self.credit_count += int(st.get("credit_count") or 0)
                return js
            except Exception as exc:
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET failed {url}: {last_err}")

    def fetch_full_map(self) -> pd.DataFrame:
        js = self.get(CRYPTOS_JSON)
        fields = js["fields"]
        rows = [dict(zip(fields, v)) for v in js["values"]]
        df = pd.DataFrame(rows)
        df["id"] = df["id"].astype(int)
        df["listing_status"] = [
            _status_label(a, s) for a, s in zip(df["is_active"], df["status"])
        ]
        df["first_historical_data"] = pd.to_datetime(df["first_historical_data"], utc=True, errors="coerce")
        df["last_historical_data"] = pd.to_datetime(df["last_historical_data"], utc=True, errors="coerce")
        return df

    def fetch_historical_listing(self, date_iso: str, start: int = 1, limit: int = 500) -> list[dict]:
        js = self.get(
            f"{BASE}/cryptocurrency/listings/historical",
            params={"date": date_iso, "start": int(start), "limit": int(limit)},
        )
        return list(js.get("data") or [])

    def fetch_ohlcv_window(self, crypto_id: int, time_start: int, time_end: int) -> list[dict]:
        js = self.get(
            f"{BASE}/cryptocurrency/historical",
            params={
                "id": int(crypto_id),
                "convertId": CONVERT_USD,
                "timeStart": int(time_start),
                "timeEnd": int(time_end),
                "interval": "1d",
            },
        )
        data = js.get("data") or {}
        return list(data.get("quotes") or [])


def quotes_to_frame(crypto_id: int, quotes: list[dict], meta: dict | None = None) -> pd.DataFrame:
    rows = []
    meta = meta or {}
    for q in quotes:
        qq = q.get("quote") or {}
        ts = q.get("timeOpen") or qq.get("timestamp")
        rows.append(
            {
                "cryptocurrency_id": int(crypto_id),
                "timestamp": ts,
                "open": qq.get("open"),
                "high": qq.get("high"),
                "low": qq.get("low"),
                "close": qq.get("close"),
                "volume": qq.get("volume"),
                "marketCap": qq.get("marketCap"),
                "currency_name": meta.get("name"),
                "currency_symbol": meta.get("symbol"),
                "currency_slug": meta.get("slug"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_id_history(
    api: CmcPublic,
    crypto_id: int,
    *,
    max_years: int = 12,
    period_days: int = 180,
    meta: dict | None = None,
) -> pd.DataFrame:
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    oldest = today - timedelta(days=int(365.25 * max_years))
    cursor = today
    chunks: list[pd.DataFrame] = []
    found_any = False
    empty_after = 0
    while cursor > oldest:
        end_time = int(cursor.timestamp())
        start_time = int((cursor - timedelta(days=period_days)).timestamp())
        quotes = api.fetch_ohlcv_window(int(crypto_id), start_time, end_time)
        if quotes:
            found_any = True
            empty_after = 0
            chunks.append(quotes_to_frame(crypto_id, quotes, meta))
        elif found_any:
            empty_after += 1
            if empty_after >= 2:
                break
        cursor = cursor - timedelta(days=period_days)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df
