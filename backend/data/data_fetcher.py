"""
Data Fetcher
============
Loads OHLCV candlestick data from Yahoo Finance via yfinance.
Supports multiple timeframes: 1m, 5m, 15m, 1h, 1d.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from config import SUPPORTED_TIMEFRAMES, TIMEFRAME_PERIODS

logger = logging.getLogger(__name__)


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
    period    : Override the look-back period (e.g. "60d", "5y").

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

    _period = period or TIMEFRAME_PERIODS.get(timeframe, "1y")

    logger.info("Fetching %s | %s | period=%s", symbol, timeframe, _period)

    # Retry up to 3 times — Yahoo Finance can be flaky on cloud servers
    last_error = None
    for attempt in range(1, 4):
        try:
            df: pd.DataFrame = yf.download(
                symbol,
                period=_period,
                interval=timeframe,
                progress=False,
                timeout=30,
            )

            if df is not None and not df.empty:
                break

            logger.warning(
                "Attempt %d: empty result for %s (tf=%s, period=%s)",
                attempt, symbol, timeframe, _period,
            )
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d failed for %s: %s", attempt, symbol, e)

        if attempt < 3:
            time.sleep(2 * attempt)  # back off 2s, 4s
    else:
        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(
            f"No data returned for symbol='{symbol}', "
            f"timeframe='{timeframe}', period='{_period}'.{detail}"
        )

    # Handle multi-level columns that yf.download sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalise column names (handle lowercase from some yfinance versions)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "open":
            col_map[col] = "Open"
        elif lower == "high":
            col_map[col] = "High"
        elif lower == "low":
            col_map[col] = "Low"
        elif lower == "close":
            col_map[col] = "Close"
        elif lower == "volume":
            col_map[col] = "Volume"
    df = df.rename(columns=col_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns {missing} in data for {symbol}. "
            f"Available: {list(df.columns)}"
        )

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    logger.info("Fetched %d bars for %s [%s]", len(df), symbol, timeframe)
    return df


def get_current_price(symbol: str) -> float:
    """Return the latest closing price for a symbol."""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    return float(info.last_price)
