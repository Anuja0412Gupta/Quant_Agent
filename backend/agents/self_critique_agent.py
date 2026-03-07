"""
Self Critique Agent
===================
Evaluates which agent signals were correct after a trade closes,
and updates per-agent confidence weights using a simple RL-style
reward signal.

The weights are persisted to a JSON file so they carry across sessions.

Output schema:
{
  "updated_weights": dict,
  "corrections": {agent: float},
  "explanation": str
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from config import CRITIQUE_WEIGHTS_FILE, DEFAULT_AGENT_WEIGHTS
from utils.helpers import clamp

logger = logging.getLogger(__name__)

_LEARNING_RATE = 0.05   # weight update step


def _load_weights() -> Dict[str, float]:
    """Load agent weights from disk, falling back to defaults."""
    if os.path.exists(CRITIQUE_WEIGHTS_FILE):
        try:
            with open(CRITIQUE_WEIGHTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_AGENT_WEIGHTS)


def _save_weights(weights: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(CRITIQUE_WEIGHTS_FILE), exist_ok=True)
    with open(CRITIQUE_WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)


def _agent_was_correct(
    prediction: str,      # "bullish" | "bearish" | "neutral"
    trade_result: str,    # "profit" | "loss" | "breakeven"
) -> float:
    """
    Return +1 if agent was directionally correct, -1 if wrong, 0 if neutral.
    """
    if prediction in ("bullish", "uptrend") and trade_result == "profit":
        return 1.0
    if prediction in ("bearish", "downtrend") and trade_result == "profit":
        return -1.0
    if prediction in ("bullish", "uptrend") and trade_result == "loss":
        return -1.0
    if prediction in ("bearish", "downtrend") and trade_result == "loss":
        return 1.0
    return 0.0


def run(
    indicator_result: Dict[str, Any],
    pattern_result:   Dict[str, Any],
    trend_result:     Dict[str, Any],
    regime_result:    Dict[str, Any],
    trade_result:     str,    # "profit" | "loss" | "breakeven"
    pnl_pct:          float = 0.0,   # actual PnL % for scaling updates
) -> Dict[str, Any]:
    """
    Evaluate agent signals against trade outcome and update weights.

    Parameters
    ----------
    *_result   : Prior agent outputs.
    trade_result : "profit", "loss", or "breakeven".
    pnl_pct    : Actual PnL expressed as a fraction (e.g. 0.02 = +2%).

    Returns
    -------
    dict with updated weights, corrections, and explanation.
    """
    weights = _load_weights()

    predictions = {
        "indicator": indicator_result.get("signal", "neutral"),
        "pattern":   pattern_result.get("signal",   "neutral"),
        "trend":     trend_result.get("trend",       "sideways"),
        "regime":    regime_result.get("regime",     "trending"),
    }

    corrections: Dict[str, float] = {}
    parts: list[str] = []

    pnl_scale = clamp(abs(pnl_pct) * 10, 0.1, 2.0)   # bigger trades → larger update

    for agent, pred in predictions.items():
        correctness = _agent_was_correct(pred, trade_result)
        delta       = _LEARNING_RATE * correctness * pnl_scale
        old_w       = weights.get(agent, DEFAULT_AGENT_WEIGHTS.get(agent, 0.25))
        new_w       = clamp(old_w + delta, 0.05, 0.70)
        weights[agent]    = round(new_w, 4)
        corrections[agent] = round(delta, 4)
        parts.append(
            f"{agent}: pred={pred} → {'✓' if correctness > 0 else ('✗' if correctness < 0 else '~')} "
            f"delta={delta:+.4f} ({old_w:.4f} → {new_w:.4f})"
        )

    # Normalise weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}

    _save_weights(weights)

    explanation = (
        f"Trade result: {trade_result} (PnL={pnl_pct:+.2%}). "
        + " | ".join(parts)
        + f". Normalised weights: {weights}."
    )

    logger.info("SelfCritiqueAgent: weights updated → %s", weights)

    return {
        "updated_weights": weights,
        "corrections":     corrections,
        "explanation":     explanation,
    }
