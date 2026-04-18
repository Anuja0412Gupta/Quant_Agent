"""
Reward Function
================
Production-grade, research-level reward function for the QuantAgent RL policy.

Reward formula:
    R = return - λ₁·drawdown_penalty - λ₂·volatility_penalty
          - λ₃·transaction_cost - λ₄·overtrading_penalty

Fee model (Zerodha-calibrated defaults):
    Brokerage : 0.03%  (min ₹20 per order, capped at 0.03%)
    STT        : 0.025% on sell-side (equity delivery)
    Total      : ~0.055% per round-trip

Ablation variants (for reward ablation study):
    FULL         : all penalty terms active
    NO_DRAWDOWN  : λ₁ = 0
    NO_VOLATILITY: λ₂ = 0
    NO_COST      : λ₃ = 0
    NO_OVERTRADE : λ₄ = 0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Fee model constants (Zerodha-calibrated) ──────────────────────────────────
BROKERAGE_RATE  = 0.0003   # 0.03%
STT_RATE        = 0.00025  # 0.025% (sell-side, equity)
SLIPPAGE_BASE   = 0.0002   # base slippage (0.02%), volume-adjusted further


class AblationVariant(str, Enum):
    FULL         = "full"
    NO_DRAWDOWN  = "no_drawdown"
    NO_VOLATILITY= "no_volatility"
    NO_COST      = "no_cost"
    NO_OVERTRADE = "no_overtrade"


@dataclass
class RewardConfig:
    """Penalty weights and fee parameters for the reward function."""
    lambda_drawdown:  float = 2.0    # λ₁ — drawdown penalty weight
    lambda_volatility: float = 0.5   # λ₂ — excess volatility penalty weight
    lambda_cost:      float = 1.0    # λ₃ — transaction cost penalty weight
    lambda_overtrade: float = 0.3    # λ₄ — overtrading penalty weight
    target_vol:       float = 0.15   # annualized target volatility (15%)
    brokerage_rate:   float = BROKERAGE_RATE
    stt_rate:         float = STT_RATE
    slippage_base:    float = SLIPPAGE_BASE
    ablation:         AblationVariant = AblationVariant.FULL


@dataclass
class TradeContext:
    """
    Context for a single RL step (bar-by-bar during training or backtesting).
    
    Attributes
    ----------
    pnl_pct      : Realized P&L as fraction of position value this bar
    position_size: Fraction of portfolio in the position [0, 1]
    drawdown     : Current portfolio drawdown fraction [0, 1]
    volatility   : Rolling annualized volatility (e.g. 0.20 = 20%)
    is_trade     : Whether a trade (open/close) occurred this bar
    trade_value  : Notional trade value (for fee calculation)
    bars_in_trade: Consecutive bars holding without action (for overtrade penalty)
    volume_ratio : Trade volume / avg daily volume (slippage scaling)
    """
    pnl_pct:       float = 0.0
    position_size: float = 0.0
    drawdown:      float = 0.0
    volatility:    float = 0.20
    is_trade:      bool  = False
    trade_value:   float = 0.0
    bars_in_trade: int   = 0
    volume_ratio:  float = 1.0


def compute_transaction_cost(
    trade_value: float,
    volume_ratio: float = 1.0,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """
    Compute realistic round-trip transaction cost as a fraction of trade value.

    Includes:
    - Brokerage (0.03% × 2 for buy + sell)
    - STT on sell leg
    - Volume-adjusted slippage (linear market impact)
    
    Returns cost as a positive fraction (to be subtracted from reward).
    """
    brokerage = cfg.brokerage_rate * 2           # buy + sell
    stt       = cfg.stt_rate                     # sell-side only
    slippage  = cfg.slippage_base * np.sqrt(max(volume_ratio, 0.1))
    return float(brokerage + stt + slippage)


def compute_reward(
    ctx: TradeContext,
    cfg: Optional[RewardConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the RL reward for a single timestep.

    Returns a dict with the total reward and each penalty component,
    enabling reward-curve visualization and ablation analysis.

    Parameters
    ----------
    ctx : TradeContext — market and portfolio state for this bar
    cfg : RewardConfig — penalty weights and fee model

    Returns
    -------
    {
      "reward": float,
      "return_component": float,
      "drawdown_penalty": float,
      "volatility_penalty": float,
      "cost_penalty": float,
      "overtrade_penalty": float,
    }
    """
    if cfg is None:
        cfg = RewardConfig()

    # ── Base return component ─────────────────────────────────────────────────
    # Risk-adjusted: raw PnL / realized volatility
    risk_adj_return = ctx.pnl_pct / max(ctx.volatility, 0.01)

    # ── Drawdown penalty ──────────────────────────────────────────────────────
    # Exponential to strongly penalize deep drawdowns
    drawdown_penalty = 0.0
    if cfg.ablation != AblationVariant.NO_DRAWDOWN:
        drawdown_penalty = cfg.lambda_drawdown * (ctx.drawdown ** 2)

    # ── Volatility penalty ────────────────────────────────────────────────────
    # Penalize excess volatility above target (encourages smooth equity curve)
    volatility_penalty = 0.0
    if cfg.ablation != AblationVariant.NO_VOLATILITY:
        excess_vol = max(0.0, ctx.volatility - cfg.target_vol)
        volatility_penalty = cfg.lambda_volatility * excess_vol * ctx.position_size

    # ── Transaction cost penalty ──────────────────────────────────────────────
    # Only applied on bars where a trade is executed
    cost_penalty = 0.0
    if cfg.ablation != AblationVariant.NO_COST and ctx.is_trade and ctx.trade_value > 0:
        cost_fraction  = compute_transaction_cost(ctx.trade_value, ctx.volume_ratio, cfg)
        cost_penalty   = cfg.lambda_cost * cost_fraction

    # ── Overtrading penalty ───────────────────────────────────────────────────
    # Penalize frequent trades (churning) — accumulates when position changes rapidly
    overtrade_penalty = 0.0
    if cfg.ablation != AblationVariant.NO_OVERTRADE:
        if ctx.is_trade and ctx.bars_in_trade < 3:
            # Penalty for closing/reversing within 3 bars
            overtrade_penalty = cfg.lambda_overtrade * (1.0 / max(ctx.bars_in_trade, 1))

    # ── Total reward ──────────────────────────────────────────────────────────
    reward = (
        risk_adj_return
        - drawdown_penalty
        - volatility_penalty
        - cost_penalty
        - overtrade_penalty
    )
    reward = float(np.clip(reward, -10.0, 10.0))

    logger.debug(
        "Reward=%.4f  ret=%.4f  dd_pen=%.4f  vol_pen=%.4f  cost=%.4f  ot=%.4f",
        reward, risk_adj_return, drawdown_penalty,
        volatility_penalty, cost_penalty, overtrade_penalty,
    )

    return {
        "reward":               reward,
        "return_component":     round(float(risk_adj_return),    6),
        "drawdown_penalty":     round(float(drawdown_penalty),   6),
        "volatility_penalty":   round(float(volatility_penalty), 6),
        "cost_penalty":         round(float(cost_penalty),       6),
        "overtrade_penalty":    round(float(overtrade_penalty),  6),
    }


def run_ablation(
    trade_contexts: List[TradeContext],
    base_cfg: Optional[RewardConfig] = None,
) -> Dict[str, Any]:
    """
    Run all 5 ablation variants over a list of TradeContexts.
    
    Returns cumulative reward and simulated Sharpe for each variant.
    Used by the /rl/ablation endpoint and the RL Brain Tab ablation bar chart.
    
    Returns
    -------
    {
      "variants": {
        "full":           { "total_reward": float, "sharpe": float, "rewards": [float] },
        "no_drawdown":    { ... },
        "no_volatility":  { ... },
        "no_cost":        { ... },
        "no_overtrade":   { ... },
      }
    }
    """
    if base_cfg is None:
        base_cfg = RewardConfig()

    results: Dict[str, Any] = {}

    for variant in AblationVariant:
        cfg = RewardConfig(
            lambda_drawdown   = base_cfg.lambda_drawdown,
            lambda_volatility = base_cfg.lambda_volatility,
            lambda_cost       = base_cfg.lambda_cost,
            lambda_overtrade  = base_cfg.lambda_overtrade,
            target_vol        = base_cfg.target_vol,
            brokerage_rate    = base_cfg.brokerage_rate,
            stt_rate          = base_cfg.stt_rate,
            slippage_base     = base_cfg.slippage_base,
            ablation          = variant,
        )
        step_rewards = [compute_reward(ctx, cfg)["reward"] for ctx in trade_contexts]
        arr = np.array(step_rewards)
        sharpe = float((arr.mean() / (arr.std() + 1e-8)) * np.sqrt(252)) if len(arr) > 1 else 0.0

        results[variant.value] = {
            "total_reward": round(float(arr.sum()), 4),
            "mean_reward":  round(float(arr.mean()), 6),
            "sharpe":       round(sharpe, 4),
            "rewards":      [round(float(r), 4) for r in step_rewards[:200]],  # cap for API
        }
        logger.info("Ablation [%s]: sharpe=%.3f total=%.2f", variant.value, sharpe, arr.sum())

    return {"variants": results}
