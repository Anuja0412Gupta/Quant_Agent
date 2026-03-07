"""
Risk Management Agent
=====================
Implements dynamic position sizing using:
- Fractional Kelly Criterion
- Volatility-adjusted position sizing
- Maximum drawdown enforcement

Output schema:
{
  "shares": int,
  "position_value": float,
  "position_pct": float,
  "kelly_fraction": float,
  "method": str,
  "explanation": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import (
    KELLY_FRACTION, MAX_DRAWDOWN_PCT, MAX_POSITION_PCT,
    PORTFOLIO_VALUE,
)
from utils.helpers import clamp, compute_atr

logger = logging.getLogger(__name__)


def _kelly_position(
    win_rate: float,
    avg_win:  float,
    avg_loss: float,
) -> float:
    """
    Full Kelly fraction:  K = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
    Returns fractional Kelly (scaled by KELLY_FRACTION constant).
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.1
    k = (win_rate / avg_loss) - ((1.0 - win_rate) / avg_win)
    return clamp(k * KELLY_FRACTION, 0.0, MAX_POSITION_PCT)


def _volatility_position(
    atr: float,
    price: float,
    portfolio: float,
    risk_pct: float = 0.01,      # risk 1 % of portfolio per trade
) -> tuple[float, float]:
    """
    Volatility-adjusted sizing: shares such that 1 ATR move = risk_pct of portfolio.
    """
    risk_dollars = portfolio * risk_pct
    if atr <= 0 or price <= 0:
        return 0, 0
    shares = risk_dollars / atr
    value  = shares * price
    return shares, clamp(value / portfolio, 0.0, MAX_POSITION_PCT)


def run(
    df: pd.DataFrame,
    decision_result:     Dict[str, Any],
    disagreement_result: Dict[str, Any],
    portfolio_value:     float = PORTFOLIO_VALUE,
    current_drawdown:    float = 0.0,      # current % drawdown (0-1)
) -> Dict[str, Any]:
    """
    Calculate position size respecting Kelly, volatility, and drawdown limits.

    Parameters
    ----------
    df                   : OHLCV DataFrame.
    decision_result      : Output from decision_agent.run().
    disagreement_result  : Output from disagreement_model.run().
    portfolio_value      : Current virtual portfolio value in $.
    current_drawdown     : Current drawdown fraction (0-1).

    Returns
    -------
    dict with position sizing details.
    """
    action = decision_result.get("action", "NO_TRADE")
    if action == "NO_TRADE":
        return _zero("Decision agent returned NO_TRADE.")

    # Enforce drawdown limit
    if current_drawdown >= MAX_DRAWDOWN_PCT:
        return _zero(
            f"Max drawdown {MAX_DRAWDOWN_PCT:.0%} breached "
            f"(current={current_drawdown:.2%}). Trading halted."
        )

    last_price = float(df["Close"].iloc[-1])
    atr_series = compute_atr(df, 14)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else last_price * 0.01

    # ── Kelly (estimated from recent win/loss) ─────────────────────────────
    returns   = df["Close"].pct_change().dropna()
    pos_ret   = returns[returns > 0]
    neg_ret   = returns[returns < 0]
    win_rate  = len(pos_ret) / max(len(returns), 1)
    avg_win   = float(pos_ret.mean()) if not pos_ret.empty else 0.01
    avg_loss  = float(abs(neg_ret.mean())) if not neg_ret.empty else 0.01

    kelly_frac   = _kelly_position(win_rate, avg_win, avg_loss)
    kelly_value  = portfolio_value * kelly_frac
    kelly_shares = kelly_value / last_price if last_price else 0

    # ── Volatility sizing ─────────────────────────────────────────────────
    vol_shares, vol_pct = _volatility_position(atr, last_price, portfolio_value)

    # ── Choose conservative of the two ───────────────────────────────────
    chosen_shares = min(kelly_shares, vol_shares)
    chosen_pct    = (chosen_shares * last_price) / portfolio_value

    # ── Apply disagreement multiplier ──────────────────────────────────────
    pos_mult = disagreement_result.get("position_multiplier", 1.0)
    chosen_shares = int(chosen_shares * pos_mult)
    chosen_pct   *= pos_mult

    chosen_shares = max(0, chosen_shares)
    chosen_value  = chosen_shares * last_price

    recommendation = disagreement_result.get("recommendation", "PROCEED")
    if recommendation == "NO_TRADE":
        return _zero("Disagreement model recommends NO_TRADE.")

    explanation = (
        f"Kelly fraction={kelly_frac:.4f} → {kelly_shares:.1f} shares, "
        f"Volatility sizing → {vol_shares:.1f} shares (vol_pct={vol_pct:.2%}). "
        f"Conservative choice: {chosen_shares} shares. "
        f"Disagreement multiplier applied: ×{pos_mult:.1f}. "
        f"Final position: {chosen_shares} shares × ${last_price:.2f} = ${chosen_value:.2f} "
        f"({chosen_pct:.2%} of portfolio)."
    )

    logger.debug("RiskAgent: %d shares (%s pct=%.2f)", chosen_shares, action, chosen_pct)

    return {
        "shares":          chosen_shares,
        "position_value":  round(chosen_value, 2),
        "position_pct":    round(float(chosen_pct), 4),
        "kelly_fraction":  round(kelly_frac, 4),
        "method":          "min(kelly, volatility) × disagreement_multiplier",
        "explanation":     explanation,
    }


def _zero(reason: str) -> Dict[str, Any]:
    return {
        "shares":          0,
        "position_value":  0.0,
        "position_pct":    0.0,
        "kelly_fraction":  0.0,
        "method":          "none",
        "explanation":     reason,
    }
