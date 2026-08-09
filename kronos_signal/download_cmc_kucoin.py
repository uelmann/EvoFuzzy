"""Download daily OHLCV+marketCap history via CoinMarketCap data-api (as in KuCoin BT notebook).

Universe = unique base currencies listed on KuCoin spot (CMC exchange market-pairs).
Saves long-form CSV compatible with the notebook's historical_data.csv.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("cmc_download")

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "historical_data.csv"
ARTIFACT_OUT = Path("/opt/cursor/artifacts/crypto_data/historical_data.csv")


class CoinMarketCapAPI:
    def __init__(self, sleep_s: float = 0.12):
        self.base_url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency"
        self.sleep_s = sleep_s
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "origin": "https://coinmarketcap.com",
            "referer": "https://coinmarketcap.com/",
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(self, url: str, params: dict | None = None, retries: int = 4) -> dict:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                if resp.status_code in (429, 503):
                    wait = 2 ** attempt
                    log.warning("HTTP %s — sleep %ss", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(self.sleep_s)
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET failed {url}: {last_err}")

    def get_kucoin_currencies(self, exchange_slug: str = "kucoin") -> list[dict]:
        url = "https://api.coinmarketcap.com/data-api/v3/exchange/market-pairs/latest"
        start = 1
        limit = 100
        all_pairs: list[dict] = []
        while True:
            log.info("Fetching KuCoin pairs %s–%s", start, start + limit - 1)
            data = self._get(
                url,
                {
                    "slug": exchange_slug,
                    "category": "spot",
                    "start": str(start),
                    "limit": str(limit),
                },
            )
            pairs = (data.get("data") or {}).get("marketPairs") or []
            if not pairs:
                break
            all_pairs.extend(pairs)
            start += limit
            if len(pairs) < limit:
                break

        seen: set[int] = set()
        currencies: list[dict] = []
        for entry in all_pairs:
            cid = entry["baseCurrencyId"]
            if cid in seen:
                continue
            seen.add(cid)
            currencies.append(
                {
                    "id": cid,
                    "name": entry["baseCurrencyName"],
                    "slug": entry["baseCurrencySlug"],
                    "symbol": entry["baseSymbol"],
                }
            )
        log.info("Unique KuCoin base currencies: %s", len(currencies))
        return currencies

    def fetch_listings_mcap(self, max_pages: int = 20) -> dict[int, float]:
        """Return map crypto_id -> latest USD market cap (for ranking)."""
        mcap: dict[int, float] = {}
        start = 1
        limit = 100
        for _ in range(max_pages):
            data = self._get(
                f"{self.base_url}/listing",
                {
                    "start": start,
                    "limit": limit,
                    "sortBy": "market_cap",
                    "sortType": "desc",
                    "convert": "USD",
                    "cryptoType": "all",
                    "tagType": "all",
                    "audited": "false",
                },
            )
            rows = (data.get("data") or {}).get("cryptoCurrencyList") or []
            if not rows:
                break
            for row in rows:
                quotes = row.get("quotes") or []
                usd = next((q for q in quotes if q.get("name") == "USD"), None)
                if usd and usd.get("marketCap") is not None:
                    mcap[int(row["id"])] = float(usd["marketCap"])
            start += limit
            if len(rows) < limit:
                break
        log.info("Fetched market-cap ranks for %s coins", len(mcap))
        return mcap

    def fetch_historical_data(
        self,
        crypto_id: int,
        convert_id: int = 2781,
        start_date: str = "2016-01-01",
    ) -> pd.DataFrame:
        """Daily OHLCV+mcap from start_date (default 2016) through today."""
        today = datetime.now(timezone.utc).replace(tzinfo=None)
        oldest = datetime.fromisoformat(start_date)
        processed: list[dict] = []
        cursor = today
        period_days = 180

        while cursor > oldest:
            end_time = int(cursor.timestamp())
            start_dt = max(cursor - timedelta(days=period_days), oldest)
            start_time = int(start_dt.timestamp())
            data = self._get(
                f"{self.base_url}/historical",
                {
                    "id": crypto_id,
                    "convertId": convert_id,
                    "timeStart": start_time,
                    "timeEnd": end_time,
                    "interval": "1d",
                },
            )
            quotes = ((data.get("data") or {}).get("quotes")) or []
            if not quotes:
                break
            for quote in quotes:
                q = quote["quote"]
                processed.append(
                    {
                        "cryptocurrency_id": crypto_id,
                        "timestamp": quote["timeOpen"],
                        "open": q["open"],
                        "high": q["high"],
                        "low": q["low"],
                        "close": q["close"],
                        "volume": q["volume"],
                        "marketCap": q["marketCap"],
                    }
                )
            cursor = start_dt
            if start_dt <= oldest:
                break

        if not processed:
            return pd.DataFrame()
        df = pd.DataFrame(processed)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        # Keep only bars on/after start_date
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.loc[ts >= pd.Timestamp(start_date, tz="UTC")].reset_index(drop=True)
        return df



STABLE_SYMBOLS = {
    "USDT",
    "USDC",
    "DAI",
    "TUSD",
    "FDUSD",
    "USDE",
    "USD1",
    "USDD",
    "BUSD",
    "USDG",
    "PYUSD",
    "USD",
    "EUR",
    "U",
    "STABLE",
    "XAUT",
    "PAXG",
}


def download_universe(
    out_path: Path,
    max_coins: int | None = None,
    skip_stables: bool = True,
    sleep_s: float = 0.12,
    save_every: int = 10,
    start_date: str = "2016-01-01",
) -> pd.DataFrame:
    """Download KuCoin-listed bases (notebook recipe). max_coins=None → all."""
    api = CoinMarketCapAPI(sleep_s=sleep_s)
    currencies = api.get_kucoin_currencies()
    mcap = api.fetch_listings_mcap(max_pages=40)

    df_cur = pd.DataFrame(currencies)
    df_cur["market_cap"] = df_cur["id"].map(mcap)
    df_cur = df_cur.sort_values("market_cap", ascending=False, na_position="last")
    if skip_stables:
        df_cur = df_cur[~df_cur["symbol"].astype(str).str.upper().isin(STABLE_SYMBOLS)]
    if max_coins is not None:
        df_cur = df_cur.head(max_coins)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_name("universe_meta.csv")
    df_cur.to_csv(meta_path, index=False)
    log.info(
        "Universe size=%s start_date=%s → %s",
        len(df_cur),
        start_date,
        meta_path,
    )

    chunks: list[pd.DataFrame] = []
    total = len(df_cur)
    for i, (_, row) in enumerate(df_cur.iterrows(), start=1):
        cid = int(row["id"])
        name = row["name"]
        log.info("[%s/%s] %s (%s) id=%s", i, total, name, row["symbol"], cid)
        try:
            hist = api.fetch_historical_data(cid, start_date=start_date)
            if hist.empty:
                log.warning("No history for %s", name)
                continue
            hist["currency_name"] = name
            hist["currency_symbol"] = row["symbol"]
            hist["currency_slug"] = row["slug"]
            chunks.append(hist)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed %s: %s", name, exc)
            continue

        if i % save_every == 0 and chunks:
            partial = pd.concat(chunks, ignore_index=True)
            partial.to_csv(out_path, index=False)
            log.info("Checkpoint saved %s rows → %s", len(partial), out_path)

    if not chunks:
        raise RuntimeError("No historical data downloaded")

    historical_df = pd.concat(chunks, ignore_index=True)
    historical_df.to_csv(out_path, index=False)
    log.info(
        "Done: %s rows, %s symbols, %s→%s → %s",
        len(historical_df),
        historical_df["currency_symbol"].nunique(),
        historical_df["timestamp"].min(),
        historical_df["timestamp"].max(),
        out_path,
    )
    return historical_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--max-coins",
        type=int,
        default=60,
        help="Top-N by market cap among KuCoin listings. Use 0 for ALL (notebook default).",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="Earliest daily bar to keep (UTC ISO date).",
    )
    p.add_argument("--include-stables", action="store_true")
    p.add_argument("--sleep", type=float, default=0.12)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--also-artifact", action="store_true", default=True)
    args = p.parse_args()

    max_coins = None if args.max_coins == 0 else args.max_coins
    df = download_universe(
        out_path=args.out,
        max_coins=max_coins,
        skip_stables=not args.include_stables,
        sleep_s=args.sleep,
        save_every=args.save_every,
        start_date=args.start_date,
    )
    if args.also_artifact:
        ARTIFACT_OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ARTIFACT_OUT, index=False)
        log.info("Also wrote %s", ARTIFACT_OUT)


if __name__ == "__main__":
    main()
