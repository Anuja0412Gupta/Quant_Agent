"""
Data Fetcher
============
Loads OHLCV candlestick data from Yahoo Finance via yfinance.
Supports multiple timeframes: 1m, 5m, 15m, 1h, 1d.
"""

from __future__ import annotations

import logging
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
    Download OHLCV data from Yahoo Finance.

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

    ticker = yf.Ticker(symbol)
    df: pd.DataFrame = ticker.history(period=_period, interval=timeframe)

    if df.empty:
        raise RuntimeError(
            f"No data returned for symbol='{symbol}', "
            f"timeframe='{timeframe}', period='{_period}'."
        )

    # Normalise column names
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    logger.info("Fetched %d bars for %s [%s]", len(df), symbol, timeframe)
    return df


def get_current_price(symbol: str) -> float:
    """Return the latest closing price for a symbol."""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    return float(info.last_price)
