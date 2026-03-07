"""
Pattern Agent
=============
Detects classical chart patterns from OHLC data using heuristic rules.

Patterns detected:
- Double Top
- Double Bottom
- Head and Shoulders
- Bullish Flag
- Triangle Consolidation

Output schema:
{
  "pattern": str,
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float [0-1],
  "explanation": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from utils.helpers import clamp

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _local_peaks(series: np.ndarray, order: int = 5) -> np.ndarray:
    return argrelextrema(series, np.greater, order=order)[0]

def _local_troughs(series: np.ndarray, order: int = 5) -> np.ndarray:
    return argrelextrema(series, np.less, order=order)[0]

def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return abs(a - b) / b


# ── Pattern detectors ─────────────────────────────────────────────────────────

def _check_double_top(closes: np.ndarray, highs: np.ndarray) -> Tuple[bool, float]:
    peaks = _local_peaks(highs, order=4)
    if len(peaks) < 2:
        return False, 0.0
    p1, p2 = highs[peaks[-2]], highs[peaks[-1]]
    similarity = 1.0 - _pct_diff(p1, p2)
    # Need valley between them and second peak recent
    if similarity > 0.97 and peaks[-1] > peaks[-2]:
        return True, clamp(similarity, 0.6, 0.95)
    return False, 0.0


def _check_double_bottom(closes: np.ndarray, lows: np.ndarray) -> Tuple[bool, float]:
    troughs = _local_troughs(lows, order=4)
    if len(troughs) < 2:
        return False, 0.0
    t1, t2 = lows[troughs[-2]], lows[troughs[-1]]
    similarity = 1.0 - _pct_diff(t1, t2)
    if similarity > 0.97 and troughs[-1] > troughs[-2]:
        return True, clamp(similarity, 0.6, 0.95)
    return False, 0.0


def _check_head_and_shoulders(highs: np.ndarray) -> Tuple[bool, float]:
    peaks = _local_peaks(highs, order=3)
    if len(peaks) < 3:
        return False, 0.0
    left, head, right = highs[peaks[-3]], highs[peaks[-2]], highs[peaks[-1]]
    if head > left and head > right:
        shoulder_similarity = 1.0 - _pct_diff(left, right)
        if shoulder_similarity > 0.92:
            return True, clamp(shoulder_similarity * 0.9, 0.55, 0.90)
    return False, 0.0


def _check_bullish_flag(closes: np.ndarray, highs: np.ndarray) -> Tuple[bool, float]:
    n = len(closes)
    if n < 30:
        return False, 0.0
    # Strong upward pole in first 2/3, slight downward consolidation in last 1/3
    split = n * 2 // 3
    pole   = closes[split] - closes[0]
    recent = closes[-1]   - closes[split]
    if pole > 0 and recent < 0:
        pole_pct   = pole   / closes[0]
        recent_pct = abs(recent) / closes[split]
        if pole_pct > 0.05 and recent_pct < pole_pct * 0.5:
            conf = clamp(pole_pct * 5, 0.5, 0.88)
            return True, conf
    return False, 0.0


def _check_triangle(highs: np.ndarray, lows: np.ndarray) -> Tuple[bool, float]:
    if len(highs) < 20:
        return False, 0.0
    # Check if highs are declining and lows are rising (symmetrical triangle)
    recent_highs = highs[-20:]
    recent_lows  = lows[-20:]
    high_slope = np.polyfit(range(20), recent_highs, 1)[0]
    low_slope  = np.polyfit(range(20), recent_lows,  1)[0]
    if high_slope < -0.01 and low_slope > 0.01:
        conf = clamp(abs(high_slope - low_slope) / 2, 0.4, 0.80)
        return True, conf
    return False, 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect chart patterns in *df* and return the strongest pattern found.

    Parameters
    ----------
    df : OHLCV DataFrame.

    Returns
    -------
    dict with keys: pattern, signal, confidence, explanation.
    """
    if len(df) < 30:
        return _neutral("Not enough data for pattern detection")

    # Use a recent window for efficiency
    window = min(len(df), 80)
    sub     = df.iloc[-window:].copy()

    closes = sub["Close"].values
    highs  = sub["High"].values
    lows   = sub["Low"].values

    candidates: list[Tuple[str, str, float]] = []  # (pattern, signal, confidence)

    ok, conf = _check_double_top(closes, highs)
    if ok:
        candidates.append(("Double Top", "bearish", conf))

    ok, conf = _check_double_bottom(closes, lows)
    if ok:
        candidates.append(("Double Bottom", "bullish", conf))

    ok, conf = _check_head_and_shoulders(highs)
    if ok:
        candidates.append(("Head and Shoulders", "bearish", conf))

    ok, conf = _check_bullish_flag(closes, highs)
    if ok:
        candidates.append(("Bullish Flag", "bullish", conf))

    ok, conf = _check_triangle(highs, lows)
    if ok:
        candidates.append(("Triangle Consolidation", "neutral", conf))

    if not candidates:
        return _neutral("No significant chart pattern detected")

    # Pick highest confidence pattern
    best = max(candidates, key=lambda x: x[2])
    pattern, signal, confidence = best

    explanation = (
        f"Detected '{pattern}' pattern over the last {window} bars. "
        f"Signal: {signal.upper()} with {confidence:.0%} confidence."
    )

    logger.debug("PatternAgent: %s %s (conf=%.2f)", pattern, signal, confidence)

    return {
        "pattern":     pattern,
        "signal":      signal,
        "confidence":  round(confidence, 4),
        "explanation": explanation,
    }


def _neutral(reason: str) -> Dict[str, Any]:
    return {
        "pattern":     "None",
        "signal":      "neutral",
        "confidence":  0.0,
        "explanation": reason,
    }
