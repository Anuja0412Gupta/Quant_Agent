"""
Decision Agent
==============
Combines signals from all agents using confidence-weighted fusion to
produce a final trade decision.

Fusion formula:
  Final Score = Σ (weight_i × confidence_i × direction_i)

Output schema:
{
  "action": "LONG" | "SHORT" | "NO_TRADE",
  "entry": float,
  "stop_loss": float,
  "take_profit": float,
  "confidence": float [0-1],
  "score": float,
  "reasoning": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from config import (
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    BUY_THRESHOLD, DIRECTION_MAP,
    DEFAULT_AGENT_WEIGHTS, SELL_THRESHOLD,
)
from utils.helpers import clamp, compute_atr

logger = logging.getLogger(__name__)


def _signal_to_direction(signal: str) -> float:
    return DIRECTION_MAP.get(signal.lower(), 0.0)


def run(
    df: pd.DataFrame,
    indicator_result: Dict[str, Any],
    pattern_result:   Dict[str, Any],
    trend_result:     Dict[str, Any],
    regime_result:    Dict[str, Any],
    agent_weights:    Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Produce a final trade decision by fusing agent outputs.

    Parameters
    ----------
    df               : OHLCV DataFrame (used for ATR-based SL/TP).
    indicator_result : Output from indicator_agent.run().
    pattern_result   : Output from pattern_agent.run().
    trend_result     : Output from trend_agent.run().
    regime_result    : Output from market_regime_agent.run().
    agent_weights    : Optional override weights {agent_name: weight}.

    Returns
    -------
    Full decision dict.
    """
    weights = agent_weights or DEFAULT_AGENT_WEIGHTS

    last_price = float(df["Close"].iloc[-1])

    # ── Compute ATR for SL/TP ─────────────────────────────────────────────────
    atr_series = compute_atr(df, 14)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else last_price * 0.01

    # ── Agent contributions ───────────────────────────────────────────────────
    agents: list[tuple[str, float, float]] = [
        ("indicator", weights.get("indicator", 0.30),
         indicator_result.get("confidence", 0.0)),
        ("pattern",   weights.get("pattern",   0.25),
         pattern_result.get("confidence", 0.0)),
        ("trend",     weights.get("trend",     0.25),
         trend_result.get("confidence", 0.0)),
        ("regime",    weights.get("regime",    0.20),
         regime_result.get("confidence", 0.0)),
    ]

    signals = {
        "indicator": indicator_result.get("signal", "neutral"),
        "pattern":   pattern_result.get("signal",   "neutral"),
        "trend":     trend_result.get("trend",       "sideways"),
        "regime":    regime_result.get("regime",     "trending"),
    }

    total_weight = sum(w for _, w, _ in agents)
    score = 0.0
    parts: list[str] = []

    for name, weight, conf in agents:
        sig   = signals[name]
        dirn  = _signal_to_direction(sig)
        contrib = weight * conf * dirn
        score  += contrib
        parts.append(
            f"{name.capitalize()}: {sig} (w={weight:.2f}, c={conf:.2f}, d={dirn:+.2f} → {contrib:+.4f})"
        )

    if total_weight > 0:
        score /= total_weight

    # ── Determine action ──────────────────────────────────────────────────────
    if score > BUY_THRESHOLD:
        action     = "LONG"
        stop_loss  = last_price - ATR_SL_MULTIPLIER * atr
        take_profit = last_price + ATR_TP_MULTIPLIER * atr
    elif score < SELL_THRESHOLD:
        action     = "SHORT"
        stop_loss  = last_price + ATR_SL_MULTIPLIER * atr
        take_profit = last_price - ATR_TP_MULTIPLIER * atr
    else:
        action      = "NO_TRADE"
        stop_loss   = last_price
        take_profit = last_price

    confidence = clamp(abs(score), 0.0, 1.0)

    reasoning = (
        f"Weighted fusion score={score:+.4f}. "
        + " | ".join(parts)
        + f". ATR={atr:.4f}. "
        + (f"Entry at {last_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}." if action != "NO_TRADE" else "Score insufficient — NO_TRADE.")
    )

    logger.debug("DecisionAgent: %s score=%.4f conf=%.2f", action, score, confidence)

    return {
        "action":      action,
        "entry":       round(last_price, 4),
        "stop_loss":   round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "confidence":  round(confidence, 4),
        "score":       round(float(score), 6),
        "reasoning":   reasoning,
    }
