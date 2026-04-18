"""
Risk Management Agent — v2
============================
Complete risk layer upgrade with VaR, trailing stops, hard RL override,
and volatility-based exposure scaling.

NEW in v2:
  - Rolling VaR (95th percentile of returns)
  - Trailing stop tracker
  - Hard override of unsafe RL actions before execution
  - Volatility-scaled exposure
  - `current_var` and `stop_alert` fields in output

Output schema:
{
  "shares":           int,
  "position_value":   float,
  "position_pct":     float,
  "kelly_fraction":   float,
  "current_var":      float,   ← NEW: 95th-pct VaR as fraction of portfolio
  "stop_loss":        float,   ← price level
  "take_profit":      float,   ← price level
  "trailing_stop":    float,   ← price level (current trailing stop)
  "stop_alert":       bool,    ← NEW: True if price within 0.5% of stop
  "hard_override":    bool,    ← NEW: True if RL action blocked by risk
  "override_reason":  str,
  "method":           str,
  "explanation":      str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    KELLY_FRACTION, MAX_DRAWDOWN_PCT, MAX_POSITION_PCT,
    PORTFOLIO_VALUE, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
)
from utils.helpers import clamp, compute_atr

logger = logging.getLogger(__name__)

# VaR parameters
_VAR_WINDOW       = 60    # rolling window for VaR computation
_VAR_CONFIDENCE   = 0.95  # 95th percentile
_STOP_ALERT_PCT   = 0.005 # 0.5% proximity triggers stop alert
_MAX_VAR_PCT      = 0.03  # 3% daily VaR limit — trades blocked if exceeded


def _compute_rolling_var(returns: pd.Series, window: int = _VAR_WINDOW) -> float:
    """
    Compute rolling 95th-percentile VaR (Value at Risk) as a positive fraction.
    Uses only historical data (no lookahead).
    
    Returns the magnitude of the 5th-percentile return (worst loss).
    """
    if len(returns) < max(10, window // 4):
        return 0.02   # default 2% if insufficient history
    recent = returns.dropna().tail(window)
    var_95 = float(-np.percentile(recent, 5))   # positive number = loss
    return float(np.clip(var_95, 0.0, 0.5))


def _kelly_position(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_win <= 0 or avg_loss <= 0:
        return 0.1
    k = (win_rate / avg_loss) - ((1.0 - win_rate) / avg_win)
    return clamp(k * KELLY_FRACTION, 0.0, MAX_POSITION_PCT)


def _volatility_position(
    atr: float,
    price: float,
    portfolio: float,
    risk_pct: float = 0.01,
) -> Tuple[float, float]:
    risk_dollars = portfolio * risk_pct
    if atr <= 0 or price <= 0:
        return 0, 0
    shares = risk_dollars / atr
    value  = shares * price
    return shares, clamp(value / portfolio, 0.0, MAX_POSITION_PCT)


def _compute_stops(
    price: float,
    atr: float,
    action: str,
    high_water_mark: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute stop-loss, take-profit, and trailing stop levels.
    
    Returns (stop_loss, take_profit, trailing_stop)
    """
    if action == "LONG":
        stop_loss    = price - atr * ATR_SL_MULTIPLIER
        take_profit  = price + atr * ATR_TP_MULTIPLIER
        # Trailing stop: if we have a high-water mark, trail from there
        trail_base   = high_water_mark if high_water_mark else price
        trailing_stop = trail_base - atr * ATR_SL_MULTIPLIER
    elif action == "SHORT":
        stop_loss    = price + atr * ATR_SL_MULTIPLIER
        take_profit  = price - atr * ATR_TP_MULTIPLIER
        trail_base   = high_water_mark if high_water_mark else price
        trailing_stop = trail_base + atr * ATR_SL_MULTIPLIER
    else:
        stop_loss = take_profit = trailing_stop = price

    return stop_loss, take_profit, trailing_stop


def run(
    df: pd.DataFrame,
    decision_result:     Dict[str, Any],
    disagreement_result: Dict[str, Any],
    portfolio_value:     float = PORTFOLIO_VALUE,
    current_drawdown:    float = 0.0,
    high_water_mark:     Optional[float] = None,
    rl_result:           Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate position size and apply risk controls.

    Parameters
    ----------
    df                 : OHLCV DataFrame
    decision_result    : Output from decision_agent.run()
    disagreement_result: Output from disagreement_model.run()
    portfolio_value    : Current virtual portfolio value ($)
    current_drawdown   : Current drawdown fraction [0, 1]
    high_water_mark    : Price high-water mark for trailing stop
    rl_result          : Output from rl_meta_controller.run() for hard override check

    Returns
    -------
    dict with full risk metrics — see module docstring.
    """
    action = decision_result.get("action", "NO_TRADE")

    # ── Compute VaR first (always) ─────────────────────────────────────────────
    returns    = df["Close"].pct_change().dropna()
    current_var = _compute_rolling_var(returns)

    last_price = float(df["Close"].iloc[-1])
    atr_series = compute_atr(df, 14)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else last_price * 0.01
    stop_loss, take_profit, trailing_stop = _compute_stops(
        last_price, atr, action, high_water_mark
    )
    stop_alert = abs(last_price - stop_loss) / max(last_price, 1e-6) < _STOP_ALERT_PCT

    # ── Hard override checks ───────────────────────────────────────────────────
    hard_override   = False
    override_reason = ""

    if action == "NO_TRADE":
        return _zero("Decision agent returned NO_TRADE.", current_var, stop_loss,
                     take_profit, trailing_stop, stop_alert)

    if current_drawdown >= MAX_DRAWDOWN_PCT:
        return _zero(
            f"Max drawdown {MAX_DRAWDOWN_PCT:.0%} breached (current={current_drawdown:.2%}). "
            "Trading halted.",
            current_var, stop_loss, take_profit, trailing_stop, stop_alert,
        )

    if current_var > _MAX_VAR_PCT:
        return _zero(
            f"VaR limit breached: daily VaR={current_var:.2%} > threshold={_MAX_VAR_PCT:.2%}. "
            "Trade blocked.",
            current_var, stop_loss, take_profit, trailing_stop, stop_alert,
        )

    # Check RL hard override: if RL says FLAT but decision agent says LONG/SHORT
    if rl_result is not None:
        rl_direction = rl_result.get("direction", "FLAT")
        effective_pos = rl_result.get("effective_position", 1.0)
        if rl_direction == "FLAT" or effective_pos < 0.05:
            hard_override   = True
            override_reason = (
                f"RL policy says FLAT (effective_position={effective_pos:.3f}). "
                "Trade blocked by RL risk gate."
            )
            return _zero(override_reason, current_var, stop_loss, take_profit,
                         trailing_stop, stop_alert, hard_override)

    # ── Kelly sizing ───────────────────────────────────────────────────────────
    pos_ret   = returns[returns > 0]
    neg_ret   = returns[returns < 0]
    win_rate  = len(pos_ret) / max(len(returns), 1)
    avg_win   = float(pos_ret.mean())  if not pos_ret.empty else 0.01
    avg_loss  = float(abs(neg_ret.mean())) if not neg_ret.empty else 0.01

    kelly_frac   = _kelly_position(win_rate, avg_win, avg_loss)
    kelly_value  = portfolio_value * kelly_frac
    kelly_shares = kelly_value / last_price if last_price else 0

    # ── Volatility sizing ──────────────────────────────────────────────────────
    vol_shares, vol_pct = _volatility_position(atr, last_price, portfolio_value)

    # ── Conservative of the two ───────────────────────────────────────────────
    chosen_shares = min(kelly_shares, vol_shares)
    chosen_pct    = (chosen_shares * last_price) / portfolio_value

    # ── Apply disagreement multiplier ─────────────────────────────────────────
    pos_mult = disagreement_result.get("position_multiplier", 1.0)
    chosen_shares = int(chosen_shares * pos_mult)
    chosen_pct   *= pos_mult

    # ── Apply RL effective position scaling ───────────────────────────────────
    if rl_result is not None:
        rl_scale      = float(rl_result.get("effective_position", 1.0))
        chosen_shares = int(chosen_shares * rl_scale)
        chosen_pct   *= rl_scale

    chosen_shares = max(0, chosen_shares)
    chosen_value  = chosen_shares * last_price

    if disagreement_result.get("recommendation") == "NO_TRADE":
        return _zero("Disagreement model recommends NO_TRADE.", current_var, stop_loss,
                     take_profit, trailing_stop, stop_alert)

    explanation = (
        f"Kelly={kelly_frac:.4f} → {kelly_shares:.1f} shares. "
        f"Vol-sizing → {vol_shares:.1f} shares. "
        f"Conservative: {chosen_shares} shares. "
        f"Disagreement mult: ×{pos_mult:.1f}. "
        f"VaR (95%): {current_var:.2%}. "
        f"Stop: {stop_loss:.2f}, TP: {take_profit:.2f}, Trail: {trailing_stop:.2f}. "
        f"Final: {chosen_shares}×${last_price:.2f}=${chosen_value:.2f} ({chosen_pct:.2%})."
    )

    return {
        "shares":          chosen_shares,
        "position_value":  round(chosen_value, 2),
        "position_pct":    round(float(chosen_pct), 4),
        "kelly_fraction":  round(kelly_frac, 4),
        "current_var":     round(current_var, 4),
        "stop_loss":       round(stop_loss, 4),
        "take_profit":     round(take_profit, 4),
        "trailing_stop":   round(trailing_stop, 4),
        "stop_alert":      stop_alert,
        "hard_override":   hard_override,
        "override_reason": override_reason,
        "method":          "min(kelly,vol) × disagreement × rl_scale | VaR-capped",
        "explanation":     explanation,
    }


def _zero(
    reason: str,
    current_var:   float = 0.0,
    stop_loss:     float = 0.0,
    take_profit:   float = 0.0,
    trailing_stop: float = 0.0,
    stop_alert:    bool  = False,
    hard_override: bool  = False,
) -> Dict[str, Any]:
    return {
        "shares":          0,
        "position_value":  0.0,
        "position_pct":    0.0,
        "kelly_fraction":  0.0,
        "current_var":     round(current_var, 4),
        "stop_loss":       round(stop_loss, 4),
        "take_profit":     round(take_profit, 4),
        "trailing_stop":   round(trailing_stop, 4),
        "stop_alert":      stop_alert,
        "hard_override":   hard_override,
        "override_reason": reason,
        "method":          "none",
        "explanation":     reason,
    }
