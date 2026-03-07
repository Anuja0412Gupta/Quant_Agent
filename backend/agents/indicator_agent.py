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
import pandas_ta as ta

from config import (
    MACD_FAST, MACD_SIGNAL, MACD_SLOW,
    ROC_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_PERIOD,
    STOCH_D, STOCH_K, WILLIAMS_PERIOD,
)
from utils.helpers import clamp, safe_float

logger = logging.getLogger(__name__)


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
    rsi_series = ta.rsi(close, length=RSI_PERIOD)
    rsi = safe_float(rsi_series.iloc[-1])

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macd_col    = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    signal_col  = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    hist_col    = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    macd_val    = safe_float(macd_df[macd_col].iloc[-1])
    macd_sig    = safe_float(macd_df[signal_col].iloc[-1])
    macd_hist   = safe_float(macd_df[hist_col].iloc[-1])
    prev_hist   = safe_float(macd_df[hist_col].iloc[-2])

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_df = ta.stoch(high, low, close, k=STOCH_K, d=STOCH_D)
    stoch_k_col = f"STOCHk_{STOCH_K}_{STOCH_D}_3"
    stoch_d_col = f"STOCHd_{STOCH_K}_{STOCH_D}_3"

    stoch_k  = safe_float(stoch_df[stoch_k_col].iloc[-1])
    stoch_d  = safe_float(stoch_df[stoch_d_col].iloc[-1])

    # ── ROC ───────────────────────────────────────────────────────────────────
    roc_series = ta.roc(close, length=ROC_PERIOD)
    roc = safe_float(roc_series.iloc[-1])

    # ── Williams %R ───────────────────────────────────────────────────────────
    willr_series = ta.willr(high, low, close, length=WILLIAMS_PERIOD)
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
