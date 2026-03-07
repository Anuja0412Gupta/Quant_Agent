"""
Disagreement Model
==================
Measures disagreement between agents using variance of their confidence scores.

  Disagreement Index = variance(agent_confidences)

If the index exceeds a configurable threshold the system either:
  1. Reduces position size by POSITION_REDUCE_FACTOR, or
  2. Recommends NO_TRADE when disagreement is very high.

Output schema:
{
  "disagreement_index": float,
  "high_disagreement": bool,
  "position_multiplier": float [0-1],
  "recommendation": "PROCEED" | "REDUCE" | "NO_TRADE",
  "explanation": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from config import DISAGREEMENT_THRESHOLD, POSITION_REDUCE_FACTOR

logger = logging.getLogger(__name__)

_NO_TRADE_THRESHOLD = DISAGREEMENT_THRESHOLD * 2.5


def run(
    indicator_result: Dict[str, Any],
    pattern_result:   Dict[str, Any],
    trend_result:     Dict[str, Any],
    regime_result:    Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute disagreement index and produce a position sizing recommendation.

    Parameters
    ----------
    *_result : Outputs from each sub-agent ({ "confidence": float, ... }).

    Returns
    -------
    dict with keys: disagreement_index, high_disagreement,
                    position_multiplier, recommendation, explanation.
    """
    confidences: List[float] = [
        float(indicator_result.get("confidence", 0.0)),
        float(pattern_result.get("confidence",   0.0)),
        float(trend_result.get("confidence",      0.0)),
        float(regime_result.get("confidence",     0.0)),
    ]

    # Direction alignment (convert signals to ±1 / 0)
    from utils.helpers import direction_to_num
    from config import DIRECTION_MAP

    signal_map = {
        "indicator": indicator_result.get("signal", "neutral"),
        "pattern":   pattern_result.get("signal",   "neutral"),
        "trend":     trend_result.get("trend",       "sideways"),
        "regime":    regime_result.get("regime",     "trending"),
    }
    directions = [direction_to_num(s) for s in signal_map.values()]

    # Disagreement index = variance of (confidence × direction) products
    weighted = [c * d for c, d in zip(confidences, directions)]
    disagreement_index = float(np.var(weighted))

    high_disagreement = disagreement_index > DISAGREEMENT_THRESHOLD

    if disagreement_index > _NO_TRADE_THRESHOLD:
        recommendation   = "NO_TRADE"
        position_mult    = 0.0
    elif high_disagreement:
        recommendation   = "REDUCE"
        position_mult    = POSITION_REDUCE_FACTOR
    else:
        recommendation   = "PROCEED"
        position_mult    = 1.0

    explanation = (
        f"Agent confidences: {[round(c, 2) for c in confidences]}. "
        f"Weighted signals: {[round(w, 2) for w in weighted]}. "
        f"Disagreement Index={disagreement_index:.4f} "
        f"(threshold={DISAGREEMENT_THRESHOLD}). "
        f"Recommendation: {recommendation}, position_multiplier={position_mult:.1f}."
    )

    logger.debug(
        "DisagreementModel: idx=%.4f high=%s rec=%s",
        disagreement_index, high_disagreement, recommendation,
    )

    return {
        "disagreement_index":  round(disagreement_index, 6),
        "high_disagreement":   high_disagreement,
        "position_multiplier": position_mult,
        "recommendation":      recommendation,
        "explanation":         explanation,
    }
