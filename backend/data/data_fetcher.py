"""
Data Fetcher
============
Loads OHLCV candlestick data using Yahoo Finance's public v8 chart API
directly via `requests`.  This avoids the reliability issues of the
`yfinance` library on cloud servers and local environments where Yahoo
blocks or throttles automated requests.

Supports multiple timeframes: 1m, 5m, 15m, 1h, 1d.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from config import SUPPORTED_TIMEFRAMES, TIMEFRAME_PERIODS

logger = logging.getLogger(__name__)

# ── Yahoo Finance v8 Chart API ────────────────────────────────────────────────
_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.5",
}

# Map our timeframe strings to Yahoo's interval + range params
_PERIOD_MAP = {
    "1m":  {"interval": "1m",  "range": "7d"},
    "5m":  {"interval": "5m",  "range": "60d"},
    "15m": {"interval": "15m", "range": "60d"},
    "1h":  {"interval": "1h",  "range": "730d"},
    "1d":  {"interval": "1d",  "range": "5y"},
}


def _yahoo_chart(symbol: str, interval: str, range_str: str) -> pd.DataFrame:
    """
    Hit Yahoo Finance v8 chart API and return an OHLCV DataFrame.
    """
    url = _BASE_URL.format(symbol=symbol)
    params = {
        "interval": interval,
        "range": range_str,
        "includePrePost": "false",
        "events": "",
    }

    resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Navigate the JSON response
    result = data.get("chart", {}).get("result")
    if not result:
        err_msg = data.get("chart", {}).get("error", {}).get("description", "Unknown error")
        raise RuntimeError(f"Yahoo Finance API error for '{symbol}': {err_msg}")

    result = result[0]
    timestamps = result.get("timestamp")
    if not timestamps:
        raise RuntimeError(f"No timestamp data for '{symbol}' ({interval}/{range_str})")

    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Open":   quote.get("open",   []),
        "High":   quote.get("high",   []),
        "Low":    quote.get("low",    []),
        "Close":  quote.get("close",  []),
        "Volume": quote.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))

    df.index = df.index.tz_convert(None)  # strip timezone for consistency
    df = df.dropna()
    return df


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    period: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance with retry logic.

    Parameters
    ----------
    symbol    : Ticker symbol, e.g. "AAPL".
    timeframe : One of SUPPORTED_TIMEFRAMES ("1m", "5m", "15m", "1h", "1d").
    period    : Override the look-back range (e.g. "60d", "5y").

    Returns
    -------
    DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Choose from {SUPPORTED_TIMEFRAMES}."
        )

    mapping = _PERIOD_MAP.get(timeframe, {"interval": "1d", "range": "1y"})
    interval = mapping["interval"]
    range_str = period or mapping["range"]

    logger.info("Fetching %s | interval=%s | range=%s", symbol, interval, range_str)

    # Retry up to 3 times with exponential backoff
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            df = _yahoo_chart(symbol, interval, range_str)
            if len(df) > 0:
                logger.info("Got %d bars for %s on attempt %d", len(df), symbol, attempt)
                return df
            logger.warning("Attempt %d: empty data for %s", attempt, symbol)
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d failed for %s: %s", attempt, symbol, e)

        if attempt < 3:
            time.sleep(2 * attempt)

    detail = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(
        f"No data returned for symbol='{symbol}', "
        f"timeframe='{timeframe}', range='{range_str}'.{detail}"
    )


def get_current_price(symbol: str) -> float:
    """Return the latest closing price for a symbol."""
    df = _yahoo_chart(symbol, "1d", "5d")
    return float(df["Close"].iloc[-1])
