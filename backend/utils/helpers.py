"""Shared helper utilities for QuantAgent backend."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def direction_to_num(signal: str) -> float:
    """Map a signal string to a numeric direction {-1, 0, 1}."""
    mapping = {
        "bullish": 1.0, "uptrend": 1.0,
        "bearish": -1.0, "downtrend": -1.0,
        "neutral": 0.0, "sideways": 0.0,
    }
    return mapping.get(signal.lower(), 0.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, min_periods=period).mean()


def hurst_exponent(ts: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent using rescaled-range analysis.
    H < 0.5 → mean-reverting, H ≈ 0.5 → random walk, H > 0.5 → trending.
    """
    lags   = range(min_lag, max_lag)
    tau    = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    valid  = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return 0.5
    log_lags = np.log([v[0] for v in valid])
    log_tau  = np.log([v[1] for v in valid])
    poly     = np.polyfit(log_lags, log_tau, 1)
    return float(poly[0])
