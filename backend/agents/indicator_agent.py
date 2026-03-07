"""
Indicator Agent
===============
Computes momentum-based technical indicators and produces a directional signal.

Indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator (%K / %D)
- ROC (Rate of Change)
- Williams %R

Output schema:
{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float [0-1],
  "indicator_values": {...},
  "explanation": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import (
    MACD_FAST, MACD_SIGNAL, MACD_SLOW,
    ROC_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PERIOD,
    STOCH_D, STOCH_K, WILLIAMS_PERIOD,
)
from utils.helpers import clamp, safe_float

logger = logging.getLogger(__name__)


# ── Pure-pandas indicator helpers (no pandas_ta needed) ──────────────────────

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) as pd.Series."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3):
    """Returns (%K, %D) as pd.Series."""
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d


def _roc(close: pd.Series, length: int = 10) -> pd.Series:
    """Rate of change (percentage)."""
    prev = close.shift(length)
    return 100.0 * (close - prev) / prev.replace(0, np.nan)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 14) -> pd.Series:
    """Williams %R (range -100 to 0)."""
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    return -100.0 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


# ── Main entry point ─────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all technical indicators on *df* and return a consolidated signal.

    Parameters
    ----------
    df : OHLCV DataFrame (columns: Open, High, Low, Close, Volume).

    Returns
    -------
    dict with keys: signal, confidence, indicator_values, explanation.
    """
    if len(df) < 50:
        return _neutral("Not enough data (need ≥ 50 bars)")

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_series = _rsi(close, length=RSI_PERIOD)
    rsi = safe_float(rsi_series.iloc[-1])

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_line, macd_sig_line, macd_hist_line = _macd(
        close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL
    )
    macd_val    = safe_float(macd_line.iloc[-1])
    macd_sig    = safe_float(macd_sig_line.iloc[-1])
    macd_hist   = safe_float(macd_hist_line.iloc[-1])
    prev_hist   = safe_float(macd_hist_line.iloc[-2])

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_k_series, stoch_d_series = _stochastic(high, low, close, k=STOCH_K, d=STOCH_D)
    stoch_k  = safe_float(stoch_k_series.iloc[-1])
    stoch_d  = safe_float(stoch_d_series.iloc[-1])

    # ── ROC ───────────────────────────────────────────────────────────────────
    roc_series = _roc(close, length=ROC_PERIOD)
    roc = safe_float(roc_series.iloc[-1])

    # ── Williams %R ───────────────────────────────────────────────────────────
    willr_series = _williams_r(high, low, close, length=WILLIAMS_PERIOD)
    willr = safe_float(willr_series.iloc[-1])

    # ── Score each indicator [-1, +1] ─────────────────────────────────────────

    scores: list[float] = []

    # RSI
    if rsi >= RSI_OVERBOUGHT:
        scores.append(-1.0)
    elif rsi <= RSI_OVERSOLD:
        scores.append(1.0)
    elif rsi > 50:
        scores.append(0.3)
    elif rsi < 50:
        scores.append(-0.3)
    else:
        scores.append(0.0)

    # MACD histogram crossover
    if macd_hist > 0 and prev_hist <= 0:
        scores.append(1.0)       # fresh bullish crossover
    elif macd_hist < 0 and prev_hist >= 0:
        scores.append(-1.0)      # fresh bearish crossover
    elif macd_hist > 0:
        scores.append(0.4)
    elif macd_hist < 0:
        scores.append(-0.4)
    else:
        scores.append(0.0)

    # Stochastic
    if stoch_k < 20 and stoch_k > stoch_d:
        scores.append(1.0)
    elif stoch_k > 80 and stoch_k < stoch_d:
        scores.append(-1.0)
    elif stoch_k > 50:
        scores.append(0.2)
    else:
        scores.append(-0.2)

    # ROC
    roc_score = clamp(roc / 5.0, -1.0, 1.0)
    scores.append(roc_score)

    # Williams %R  (range -100..0 ; above -20 OB, below -80 OS)
    if willr > -20:
        scores.append(-0.8)
    elif willr < -80:
        scores.append(0.8)
    else:
        w_score = (willr + 50) / 50.0   # scale to [-1, 1]
        scores.append(clamp(w_score, -1.0, 1.0))

    avg_score = float(np.mean(scores))

    # ── Signal & confidence ───────────────────────────────────────────────────
    if avg_score > 0.15:
        signal = "bullish"
    elif avg_score < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    confidence = clamp(abs(avg_score), 0.0, 1.0)

    explanation = (
        f"RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}, "
        f"Stoch_K={stoch_k:.1f}, ROC={roc:.2f}%, Williams_R={willr:.1f}. "
        f"Composite score={avg_score:.3f} → {signal.upper()}."
    )

    logger.debug("IndicatorAgent: %s (conf=%.2f)", signal, confidence)

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "indicator_values": {
            "rsi":        round(rsi, 2),
            "macd":       round(macd_val, 4),
            "macd_signal":round(macd_sig, 4),
            "macd_hist":  round(macd_hist, 4),
            "stoch_k":    round(stoch_k, 2),
            "stoch_d":    round(stoch_d, 2),
            "roc":        round(roc, 2),
            "williams_r": round(willr, 2),
        },
        "explanation": explanation,
    }


def _neutral(reason: str) -> Dict[str, Any]:
    return {
        "signal": "neutral",
        "confidence": 0.0,
        "indicator_values": {},
        "explanation": reason,
    }
