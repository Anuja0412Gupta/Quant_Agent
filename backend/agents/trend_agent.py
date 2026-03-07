"""
Trend Agent
===========
Detects overall market direction using polynomial regression trendlines,
support/resistance levels, slope and trend strength measurements.

Output schema:
{
  "trend": "uptrend" | "downtrend" | "sideways",
  "slope": float,
  "strength": float [0-1],
  "confidence": float [0-1],
  "support": float,
  "resistance": float,
  "explanation": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import SUPPORT_RESIST_WINDOW, TREND_LOOKBACK, TREND_POLY_DEGREE
from utils.helpers import clamp

logger = logging.getLogger(__name__)


def _compute_support_resistance(
    df: pd.DataFrame, window: int
) -> tuple[float, float]:
    """Rolling min/max over *window* bars as proxy for support/resistance."""
    support    = float(df["Low"].rolling(window).min().iloc[-1])
    resistance = float(df["High"].rolling(window).max().iloc[-1])
    return support, resistance


def _polynomial_trend(
    closes: np.ndarray, degree: int
) -> tuple[float, float]:
    """
    Fit a polynomial of *degree* through the closing prices.

    Returns
    -------
    slope      : first derivative (rate of change) at the last bar
    r_squared  : goodness of fit (proxy for trend strength)
    """
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, degree)

    # Slope at the last point = derivative of polynomial at x[-1]
    deriv_coeffs = np.polyder(coeffs)
    slope = float(np.polyval(deriv_coeffs, x[-1]))

    # R² for the fit
    fitted      = np.polyval(coeffs, x)
    ss_res      = np.sum((closes - fitted) ** 2)
    ss_tot      = np.sum((closes - closes.mean()) ** 2)
    r_squared   = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return slope, float(clamp(r_squared, 0.0, 1.0))


def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse trend direction and strength.

    Parameters
    ----------
    df : OHLCV DataFrame.

    Returns
    -------
    dict with keys: trend, slope, strength, confidence, support, resistance, explanation.
    """
    if len(df) < TREND_LOOKBACK:
        return _unknown("Not enough data for trend analysis")

    recent = df.iloc[-TREND_LOOKBACK:]
    closes = recent["Close"].values.astype(float)

    slope, r_squared = _polynomial_trend(closes, TREND_POLY_DEGREE)

    # Normalise slope to price scale (% per bar)
    slope_pct = (slope / closes[-1]) * 100.0

    support, resistance = _compute_support_resistance(df, SUPPORT_RESIST_WINDOW)

    # Classify trend
    if slope_pct > 0.05:
        trend = "uptrend"
    elif slope_pct < -0.05:
        trend = "downtrend"
    else:
        trend = "sideways"

    # Strength: weighted combination of |slope_pct| and R²
    strength = clamp(abs(slope_pct) * 5 * r_squared, 0.0, 1.0)

    # Confidence based on R² and magnitude of slope
    confidence = clamp(r_squared * 0.6 + strength * 0.4, 0.0, 1.0)

    explanation = (
        f"Polynomial trendline (degree={TREND_POLY_DEGREE}) over {TREND_LOOKBACK} bars: "
        f"slope={slope_pct:+.3f}%/bar, R²={r_squared:.2f}. "
        f"Support={support:.2f}, Resistance={resistance:.2f}. "
        f"Trend classified as {trend.upper()}."
    )

    logger.debug("TrendAgent: %s slope=%.4f strength=%.2f", trend, slope, strength)

    return {
        "trend":      trend,
        "slope":      round(float(slope_pct), 6),
        "strength":   round(strength, 4),
        "confidence": round(confidence, 4),
        "support":    round(support, 4),
        "resistance": round(resistance, 4),
        "explanation": explanation,
    }


def _unknown(reason: str) -> Dict[str, Any]:
    return {
        "trend":      "sideways",
        "slope":      0.0,
        "strength":   0.0,
        "confidence": 0.0,
        "support":    0.0,
        "resistance": 0.0,
        "explanation": reason,
    }
