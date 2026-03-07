"""
Market Regime Agent
===================
Classifies the current market regime using:
- ATR (Average True Range) for volatility
- Rolling variance of returns
- Hurst Exponent (trending vs mean-reverting)

Regimes:
- "trending"       : Hurst > 0.6, directional moves
- "mean_reverting" : Hurst < 0.4, oscillating
- "high_volatility": ATR-ratio above threshold
- "low_volatility" : ATR-ratio below threshold

Output schema:
{
  "regime": str,
  "hurst":  float,
  "atr":    float,
  "atr_ratio": float,
  "rolling_variance": float,
  "confidence": float [0-1],
  "explanation": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import ATR_PERIOD, HURST_MAX_LAGS, HURST_MIN_LAGS, REGIME_VARIANCE_WINDOW
from utils.helpers import clamp, compute_atr, hurst_exponent

logger = logging.getLogger(__name__)

_HIGH_VOLATILITY_THRESHOLD = 0.025   # ATR / price > 2.5 %
_LOW_VOLATILITY_THRESHOLD  = 0.008   # ATR / price < 0.8 %
_TRENDING_HURST            = 0.58
_MEAN_REV_HURST            = 0.42


def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect market regime from OHLCV data.

    Parameters
    ----------
    df : OHLCV DataFrame.

    Returns
    -------
    dict with keys: regime, hurst, atr, atr_ratio, rolling_variance, confidence, explanation.
    """
    min_bars = max(ATR_PERIOD + 5, HURST_MAX_LAGS + 5, REGIME_VARIANCE_WINDOW + 5)
    if len(df) < min_bars:
        return _unknown("Not enough data for regime detection")

    close = df["Close"]
    last_price = float(close.iloc[-1])

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr_series = compute_atr(df, period=ATR_PERIOD)
    atr_val    = float(atr_series.dropna().iloc[-1])
    atr_ratio  = atr_val / last_price if last_price else 0.0

    # ── Rolling return variance ────────────────────────────────────────────────
    returns           = close.pct_change().dropna()
    rolling_var       = float(
        returns.rolling(REGIME_VARIANCE_WINDOW).var().dropna().iloc[-1]
    )

    # ── Hurst exponent ────────────────────────────────────────────────────────
    log_prices = np.log(close.values.astype(float))
    hurst = hurst_exponent(log_prices, HURST_MIN_LAGS, HURST_MAX_LAGS)

    # ── Classify ─────────────────────────────────────────────────────────────
    if atr_ratio > _HIGH_VOLATILITY_THRESHOLD:
        regime = "high_volatility"
        confidence = clamp((atr_ratio - _HIGH_VOLATILITY_THRESHOLD) * 40 + 0.5, 0.5, 0.95)
    elif atr_ratio < _LOW_VOLATILITY_THRESHOLD:
        regime = "low_volatility"
        confidence = clamp((_LOW_VOLATILITY_THRESHOLD - atr_ratio) * 80 + 0.5, 0.5, 0.90)
    elif hurst > _TRENDING_HURST:
        regime = "trending"
        confidence = clamp((hurst - _TRENDING_HURST) * 5 + 0.5, 0.5, 0.92)
    elif hurst < _MEAN_REV_HURST:
        regime = "mean_reverting"
        confidence = clamp((_MEAN_REV_HURST - hurst) * 5 + 0.5, 0.5, 0.92)
    else:
        regime = "trending"   # ambiguous → default to trending
        confidence = 0.45

    explanation = (
        f"ATR={atr_val:.4f} ({atr_ratio:.2%} of price), "
        f"Hurst={hurst:.3f}, "
        f"Rolling variance={rolling_var:.6f}. "
        f"Regime classified as '{regime.upper()}'."
    )

    logger.debug("RegimeAgent: %s (conf=%.2f, H=%.3f)", regime, confidence, hurst)

    return {
        "regime":            regime,
        "hurst":             round(hurst, 4),
        "atr":               round(atr_val, 4),
        "atr_ratio":         round(atr_ratio, 6),
        "rolling_variance":  round(rolling_var, 8),
        "confidence":        round(confidence, 4),
        "explanation":       explanation,
    }


def _unknown(reason: str) -> Dict[str, Any]:
    return {
        "regime":            "trending",
        "hurst":             0.5,
        "atr":               0.0,
        "atr_ratio":         0.0,
        "rolling_variance":  0.0,
        "confidence":        0.0,
        "explanation":       reason,
    }
