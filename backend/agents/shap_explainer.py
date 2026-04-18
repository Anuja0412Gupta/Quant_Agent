"""
SHAP Explainer
==============
Computes SHAP values for the RL policy to explain which agent features
drove each RL action. Used in:
  - Analyze Tab → SHAPPanel.jsx
  - RL Brain Tab

Uses shap.KernelExplainer wrapping the RL model's forward pass.
Falls back to a gradient-free permutation importance method if SHAP
is not installed.

Output schema:
{
  "shap_values": {
    "indicator_signal":   float,
    "indicator_conf":     float,
    "pattern_signal":     float,
    "pattern_conf":       float,
    "trend_signal":       float,
    "trend_conf":         float,
    "regime_signal":      float,
    "regime_conf":        float,
    "atr_normalized":     float,
    "disagreement_score": float,
    "hurst":              float,
    "rolling_return_5d":  float,
    "rolling_return_20d": float,
    "rolling_vol_20d":    float,
    "drawdown":           float,
    "portfolio_cash_pct": float,
  },
  "top_features": [ { "feature": str, "shap": float, "direction": "positive"|"negative" } ],
  "action":       float,    ← rl_action used as explainer target
  "method":       "shap_kernel" | "permutation",
  "explanation":  str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from agents.feature_engineering import STATE_DIM, build_state_vector

logger = logging.getLogger(__name__)

_FEATURE_NAMES = [
    "indicator_signal",   "indicator_conf",
    "pattern_signal",     "pattern_conf",
    "trend_signal",       "trend_conf",
    "regime_signal",      "regime_conf",
    "atr_normalized",     "disagreement_score",
    "hurst",              "rolling_return_5d",
    "rolling_return_20d", "rolling_vol_20d",
    "drawdown",           "portfolio_cash_pct",
]


def _permutation_importance(
    predict_fn,
    obs: np.ndarray,
    n_repeats: int = 50,
) -> np.ndarray:
    """
    Model-agnostic permutation-based feature attribution.
    
    Shuffles each feature n_repeats times and measures the change in
    predicted rl_action magnitude. Mean absolute change = importance.
    """
    baseline = predict_fn(obs)[0]
    importances = np.zeros(STATE_DIM, dtype=np.float32)
    for feat_idx in range(STATE_DIM):
        diffs = []
        for _ in range(n_repeats):
            perturbed = obs.copy()
            perturbed[feat_idx] = np.random.uniform(-1.0, 1.0)
            perturbed_out = predict_fn(perturbed)[0]
            diffs.append(abs(perturbed_out - baseline))
        importances[feat_idx] = float(np.mean(diffs))
    return importances


def explain(
    indicator_result:   Dict[str, Any],
    pattern_result:     Dict[str, Any],
    trend_result:       Dict[str, Any],
    regime_result:      Dict[str, Any],
    disagreement_score: float = 0.0,
    drawdown:           float = 0.0,
    price_series=None,
    rl_controller=None,   # RLMetaController instance
    n_background: int = 20,
) -> Dict[str, Any]:
    """
    Compute SHAP (or permutation) feature attributions for the current RL action.

    Parameters
    ----------
    *_result         : Agent outputs
    disagreement_score, drawdown : RL state inputs
    price_series     : Close price history for rolling features
    rl_controller    : RLMetaController instance (uses module singleton if None)
    n_background     : Number of background samples for SHAP KernelExplainer

    Returns
    -------
    dict — see module docstring schema.
    """
    if rl_controller is None:
        from agents.rl_meta_controller import get_controller
        rl_controller = get_controller()

    obs = build_state_vector(
        indicator_result, pattern_result, trend_result, regime_result,
        disagreement_score=disagreement_score,
        drawdown=drawdown,
        price_series=price_series,
    )

    # Determine active regime + model
    active_regime = rl_controller._select_regime(regime_result) if rl_controller._enabled else "trending"
    model_dict = rl_controller.models if rl_controller._enabled else {}
    model = model_dict.get(active_regime)

    # ── Prediction function wrapper ──────────────────────────────────────────
    def _predict(x: np.ndarray) -> np.ndarray:
        """Returns rl_action for a batch or single obs."""
        if model is None:
            return np.zeros(1, dtype=np.float32)
        x2 = np.atleast_2d(x).astype(np.float32)
        actions = []
        for row in x2:
            try:
                raw, _ = model.predict(row, deterministic=True)
                actions.append(float(raw[0]))   # rl_action (direction component)
            except Exception:
                actions.append(0.0)
        return np.array(actions, dtype=np.float32)

    baseline_action = float(_predict(obs)[0])

    # ── Try SHAP first, fall back to permutation ──────────────────────────────
    method = "permutation"
    shap_vals = np.zeros(STATE_DIM, dtype=np.float32)

    try:
        import shap  # optional dependency

        # Background dataset: perturbations around the current observation
        rng  = np.random.default_rng(42)
        bg   = obs[np.newaxis, :] + rng.normal(0, 0.1, (n_background, STATE_DIM))
        bg   = np.clip(bg, -1.0, 1.0).astype(np.float32)

        explainer = shap.KernelExplainer(_predict, bg)
        raw_shap  = explainer.shap_values(obs[np.newaxis, :], nsamples=100, silent=True)
        shap_vals = np.array(raw_shap).flatten()[:STATE_DIM]
        method    = "shap_kernel"
        logger.info("SHAP KernelExplainer completed for regime=%s", active_regime)

    except ImportError:
        logger.info("SHAP not installed — using permutation importance fallback.")
        shap_vals = _permutation_importance(_predict, obs)
    except Exception as e:
        logger.warning("SHAP failed (%s) — using permutation fallback.", e)
        shap_vals = _permutation_importance(_predict, obs)

    # ── Build output ──────────────────────────────────────────────────────────
    shap_dict = {name: round(float(v), 6) for name, v in zip(_FEATURE_NAMES, shap_vals)}

    # Top 5 features by absolute SHAP value
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features: List[Dict[str, Any]] = [
        {
            "feature":    feat,
            "shap":       round(val, 6),
            "direction":  "positive" if val >= 0 else "negative",
        }
        for feat, val in sorted_feats[:6]
    ]

    # Human-readable explanation of top driver
    top_feat, top_val = sorted_feats[0] if sorted_feats else ("unknown", 0.0)
    direction_word = "increased" if top_val > 0 else "decreased"
    explanation = (
        f"RL action={baseline_action:.3f}. "
        f"Top driver: '{top_feat}' {direction_word} action magnitude by {abs(top_val):.4f}. "
        f"Method: {method}. Regime: {active_regime}."
    )

    return {
        "shap_values":  shap_dict,
        "top_features": top_features,
        "action":       round(baseline_action, 4),
        "active_regime": active_regime,
        "method":       method,
        "explanation":  explanation,
    }
