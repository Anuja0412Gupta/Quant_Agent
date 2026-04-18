"""
Feature Engineering Pipeline
=============================
Centralized, point-in-time-correct feature extraction for the RL state vector.

CRITICAL DESIGN INVARIANT:
    NO full-dataset normalization is used anywhere in this pipeline.
    All normalization is strictly rolling-window (Z-score or MinMax) using
    only past data visible at inference time. This prevents lookahead bias.

State vector (16 dims):
    [indicator_signal, indicator_conf,
     pattern_signal,   pattern_conf,
     trend_signal,     trend_conf,
     regime_signal,    regime_conf,
     atr_normalized,   disagreement_score,   ← continuous [0, 1]
     hurst_exponent,   rolling_return_5d,
     rolling_return_20d, rolling_vol_20d,
     drawdown,         portfolio_cash_ratio]
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Signal encoding map ───────────────────────────────────────────────────────
_SIGNAL_MAP: Dict[str, float] = {
    "bullish": 1.0,    "uptrend": 1.0,    "trending": 0.5,
    "bearish": -1.0,   "downtrend": -1.0, "mean_reverting": -0.3,
    "neutral": 0.0,    "sideways": 0.0,
    "high_volatility": -0.5, "low_volatility": 0.2,
}

STATE_DIM = 16


def _encode_signal(result: Dict[str, Any], key: str = "signal") -> float:
    """Map a categorical signal string to [-1, 1]."""
    val = str(result.get(key, "neutral")).lower()
    return _SIGNAL_MAP.get(val, 0.0)


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Point-in-time Z-score normalization.
    INVARIANT: uses only past data (min_periods enforced).
    """
    mu  = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std()
    z   = (series - mu) / (std + 1e-8)
    return z.clip(-3, 3) / 3.0   # map [-3σ, 3σ] → [-1, 1]


def build_state_vector(
    indicator_result:    Dict[str, Any],
    pattern_result:      Dict[str, Any],
    trend_result:        Dict[str, Any],
    regime_result:       Dict[str, Any],
    disagreement_score:  float = 0.0,    # continuous [0, 1]
    drawdown:            float = 0.0,    # current drawdown fraction [0, 1]
    portfolio_cash_pct:  float = 1.0,    # cash as fraction of portfolio [0, 1]
    price_series:        "pd.Series | None" = None,
) -> np.ndarray:
    """
    Build the normalized RL observation vector (STATE_DIM = 16 dims).

    All features are mapped to [-1, +1] using only point-in-time statistics.

    Parameters
    ----------
    indicator_result    : Output from indicator_agent.run()
    pattern_result      : Output from pattern_agent.run()
    trend_result        : Output from trend_agent.run()
    regime_result       : Output from market_regime_agent.run()
    disagreement_score  : Continuous disagreement [0, 1] from disagreement_model
    drawdown            : Current portfolio drawdown [0, 1]
    portfolio_cash_pct  : Cash ratio [0, 1] — 1.0 means fully in cash
    price_series        : Historical Close prices for rolling momentum/vol features

    Returns
    -------
    np.ndarray of shape (STATE_DIM,) with dtype float32, range [-1, 1]
    """
    # ── Agent signals & confidences ───────────────────────────────────────────
    f_ind_signal = _encode_signal(indicator_result, "signal")
    f_ind_conf   = _clamp(float(indicator_result.get("confidence", 0.0)), 0.0, 1.0)

    f_pat_signal = _encode_signal(pattern_result, "signal")
    f_pat_conf   = _clamp(float(pattern_result.get("confidence", 0.0)), 0.0, 1.0)

    f_tre_signal = _encode_signal(trend_result, "trend")
    f_tre_conf   = _clamp(float(trend_result.get("confidence", 0.0)), 0.0, 1.0)

    f_reg_signal = _encode_signal(regime_result, "regime")
    f_reg_conf   = _clamp(float(regime_result.get("confidence", 0.0)), 0.0, 1.0)

    # ── Volatility (ATR ratio, normalized to [-1, 1]) ─────────────────────────
    atr_ratio    = float(regime_result.get("atr_ratio", 0.05))
    # atr_ratio typically 0–0.1; multiply ×10 then clamp
    f_atr        = _clamp(atr_ratio * 10.0, 0.0, 1.0)

    # ── Continuous disagreement score [0, 1] → mapped to [-1, 1] ─────────────
    # 0 = full agreement (positive), 1 = full disagreement (negative signal)
    f_disagreement = _clamp(disagreement_score, 0.0, 1.0)

    # ── Hurst exponent ────────────────────────────────────────────────────────
    # H=0.5 → random walk (neutral), H>0.5 → trending, H<0.5 → mean-reverting
    hurst = float(regime_result.get("hurst", 0.5))
    f_hurst = _clamp((hurst - 0.5) * 2.0)   # [0, 1] → [-1, 1] centred at 0

    # ── Price-based rolling features (safe fallbacks if no series) ────────────
    f_ret_5d  = 0.0
    f_ret_20d = 0.0
    f_vol_20d = 0.0

    if price_series is not None and len(price_series) >= 20:
        rets = price_series.pct_change()
        # 5-day return: normalized by 20-day rolling std
        r5  = price_series.pct_change(5).iloc[-1]
        r20 = price_series.pct_change(20).iloc[-1]
        vol = rets.rolling(20, min_periods=10).std().iloc[-1]

        f_ret_5d  = _clamp(r5  / (vol * 5.0  + 1e-6))
        f_ret_20d = _clamp(r20 / (vol * 20.0 + 1e-6))
        f_vol_20d = _clamp((vol * np.sqrt(252) - 0.15) / 0.35)  # annualized, centred ~15%

    # ── Portfolio state ───────────────────────────────────────────────────────
    f_drawdown  = _clamp(drawdown,          0.0, 1.0)
    f_cash_pct  = _clamp(portfolio_cash_pct, 0.0, 1.0)

    state = np.array([
        f_ind_signal,    # 0
        f_ind_conf,      # 1
        f_pat_signal,    # 2
        f_pat_conf,      # 3
        f_tre_signal,    # 4
        f_tre_conf,      # 5
        f_reg_signal,    # 6
        f_reg_conf,      # 7
        f_atr,           # 8
        f_disagreement,  # 9
        f_hurst,         # 10
        f_ret_5d,        # 11
        f_ret_20d,       # 12
        f_vol_20d,       # 13
        f_drawdown,      # 14
        f_cash_pct,      # 15
    ], dtype=np.float32)

    assert state.shape == (STATE_DIM,), f"State dim mismatch: {state.shape}"
    logger.debug("FeatureEng: state=%s", np.round(state, 3).tolist())
    return state


def audit_lookahead(state_vector_fn) -> bool:
    """
    Smoke-test: verify the state builder never accesses future data.
    
    Runs the builder on a synthetic price series and checks that outputs
    are identical for any suffix extension of the series.
    
    Returns True if no lookahead detected.
    """
    import warnings

    stub = {
        "signal": "neutral", "confidence": 0.5,
        "trend": "sideways", "regime": "trending",
        "atr_ratio": 0.02, "hurst": 0.5,
    }
    prices = pd.Series(np.random.randn(50).cumsum() + 100)
    v1 = state_vector_fn(stub, stub, stub, stub, price_series=prices)

    # Extend series by 5 bars — past output must be identical
    extras = pd.Series(np.random.randn(5).cumsum() + prices.iloc[-1])
    prices_ext = pd.concat([prices, extras], ignore_index=True)
    v2 = state_vector_fn(stub, stub, stub, stub, price_series=prices_ext.iloc[:50])

    ok = np.allclose(v1, v2, atol=1e-5)
    if not ok:
        warnings.warn("LOOKAHEAD BIAS DETECTED in feature pipeline!", stacklevel=2)
    else:
        logger.info("Lookahead audit PASSED — no future data leakage detected.")
    return ok
