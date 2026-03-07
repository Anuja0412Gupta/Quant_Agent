"""
Backtesting Engine
==================
Runs the full QuantAgent pipeline on historical OHLCV data bar-by-bar,
simulating trades and computing performance metrics.

Metrics computed:
- Total return
- Win rate
- Sharpe ratio
- Max drawdown
- Profit factor
- Number of trades

Output schema:
{
  "metrics": { ... },
  "equity_curve": [float],
  "trades": [ { ... } ],
  "symbol": str,
  "timeframe": str,
  "period": str
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import BACKTEST_COMMISSION, BACKTEST_INITIAL_CAPITAL
from utils.helpers import clamp

logger = logging.getLogger(__name__)

# Minimum bars required to run all agents
_MIN_WARMUP = 60


def _compute_sharpe(returns: List[float], periods_per_year: int = 252) -> float:
    arr = np.array(returns)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return float((arr.mean() / arr.std()) * np.sqrt(periods_per_year))


def _compute_max_drawdown(equity: List[float]) -> float:
    eq  = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())


def run(
    df: pd.DataFrame,
    symbol:    str = "AAPL",
    timeframe: str = "1d",
    period:    str = "5y",
    agent_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run full QuantAgent simulation on historical data.

    Parameters
    ----------
    df            : Full OHLCV DataFrame.
    symbol        : Ticker symbol (for reporting).
    timeframe     : Timeframe string (for reporting).
    period        : Period string (for reporting).
    agent_weights : Optional fixed agent weights override.

    Returns
    -------
    dict with metrics, equity_curve, trades lists.
    """
    # Import agents here to avoid circular imports at module level
    from agents import (
        indicator_agent, pattern_agent, trend_agent,
        market_regime_agent, decision_agent, disagreement_model,
    )

    if len(df) < _MIN_WARMUP + 10:
        return {"error": f"Need at least {_MIN_WARMUP + 10} bars for backtesting."}

    capital    = float(BACKTEST_INITIAL_CAPITAL)
    equity     = [capital]
    trades: List[Dict[str, Any]] = []
    returns    = []

    position: Optional[Dict[str, Any]] = None  # open position

    for i in range(_MIN_WARMUP, len(df)):
        bar   = df.iloc[:i]
        price = float(df["Close"].iloc[i])

        # ── Close open position ────────────────────────────────────────────────
        if position is not None:
            entry_price = position["entry"]
            shares      = position["shares"]
            action_type = position["action"]

            if action_type == "LONG":
                pnl_per_share = price - entry_price
            else:
                pnl_per_share = entry_price - price

            pnl    = pnl_per_share * shares
            comm   = abs(price * shares) * BACKTEST_COMMISSION
            net_pnl = pnl - comm
            capital += net_pnl
            ret    = net_pnl / max(equity[-1], 1.0)

            trades.append({
                "bar":      i,
                "action":   action_type,
                "entry":    round(entry_price, 4),
                "exit":     round(price, 4),
                "shares":   shares,
                "pnl":      round(net_pnl, 2),
                "return":   round(ret, 4),
                "result":   "profit" if net_pnl > 0 else "loss",
            })
            returns.append(ret)
            position = None

        equity.append(capital)

        # ── Run agent pipeline ─────────────────────────────────────────────────
        try:
            ind_r    = indicator_agent.run(bar)
            pat_r    = pattern_agent.run(bar)
            tre_r    = trend_agent.run(bar)
            reg_r    = market_regime_agent.run(bar)
            dis_r    = disagreement_model.run(ind_r, pat_r, tre_r, reg_r)
            dec_r    = decision_agent.run(bar, ind_r, pat_r, tre_r, reg_r, agent_weights)
        except Exception as e:
            logger.debug("Backtest bar %d skipped: %s", i, e)
            continue

        action = dec_r.get("action", "NO_TRADE")
        if dis_r.get("recommendation") in ("NO_TRADE",):
            action = "NO_TRADE"

        if action in ("LONG", "SHORT"):
            risk_pct = 0.02    # risk 2 % of capital per trade
            shares   = max(1, int((capital * risk_pct) / max(price, 1)))
            position = {
                "action":  action,
                "entry":   price,
                "shares":  shares,
                "sl":      dec_r.get("stop_loss", price),
                "tp":      dec_r.get("take_profit", price),
            }

    # Close final open position at last bar
    if position is not None:
        price = float(df["Close"].iloc[-1])
        entry_price = position["entry"]
        shares      = position["shares"]
        action_type = position["action"]
        pnl = ((price - entry_price) if action_type == "LONG" else (entry_price - price)) * shares
        comm  = abs(price * shares) * BACKTEST_COMMISSION
        net_pnl = pnl - comm
        capital += net_pnl
        trades.append({
            "bar":    len(df) - 1,
            "action": action_type,
            "entry":  round(entry_price, 4),
            "exit":   round(price, 4),
            "shares": shares,
            "pnl":    round(net_pnl, 2),
            "return": round(net_pnl / max(equity[-1], 1), 4),
            "result": "profit" if net_pnl > 0 else "loss",
        })
        returns.append(net_pnl / max(equity[-1], 1))
        equity.append(capital)

    # ── Metrics ────────────────────────────────────────────────────────────────
    n_trades  = len(trades)
    winners   = [t for t in trades if t["result"] == "profit"]
    losers    = [t for t in trades if t["result"] == "loss"]
    win_rate  = len(winners) / max(n_trades, 1)

    gross_profit = sum(t["pnl"] for t in winners)
    gross_loss   = abs(sum(t["pnl"] for t in losers))
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    total_return = (capital - BACKTEST_INITIAL_CAPITAL) / BACKTEST_INITIAL_CAPITAL
    sharpe       = _compute_sharpe(returns)
    max_dd       = _compute_max_drawdown(equity)

    metrics = {
        "total_return":   round(total_return, 4),
        "win_rate":       round(win_rate, 4),
        "sharpe_ratio":   round(sharpe, 4),
        "max_drawdown":   round(max_dd, 4),
        "profit_factor":  round(profit_factor, 4),
        "n_trades":       n_trades,
        "final_capital":  round(capital, 2),
    }

    logger.info(
        "Backtest [%s %s]: trades=%d, return=%.2f%%, sharpe=%.2f, dd=%.2f%%",
        symbol, timeframe,
        n_trades, total_return * 100, sharpe, max_dd * 100,
    )

    return {
        "metrics":      metrics,
        "equity_curve": [round(v, 2) for v in equity],
        "trades":       trades[-50:],    # last 50 for API response size
        "symbol":       symbol,
        "timeframe":    timeframe,
        "period":       period,
    }
