"""
QuantAgent v3.0 — Almgren-Chriss Optimal Execution Agent
==========================================================
Almgren & Chriss (2001) closed-form trajectory optimization.

When the RL policy says "buy X% of portfolio", this agent computes
the optimal schedule to accumulate that position over T bars to
minimize: E[cost] + risk_aversion * Var[cost].

Solution: closed-form hyperbolic-sinh trajectory (monotone, front-loaded).

Integrated into TradingEnv.step() for trades > 2% of portfolio.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSchedule:
    """Output of AlmgrenChrissExecutor.execute()."""
    trades_pct_per_bar: List[float]   # fraction of portfolio to trade each bar
    n_bars:             int
    total_shares:       int
    estimated_cost_bps: float         # expected total market impact in bps
    participation_rate: float         # avg fraction of daily volume per bar


class AlmgrenChrissExecutor:
    """
    Optimal liquidation/acquisition trajectory using Almgren-Chriss (2001).

    Minimizes: E[cost] + risk_aversion * Var[cost]

    Cost components:
      - Temporary impact: eta * (trade_rate / volume) per bar  [short-term]
      - Permanent impact: gamma * cumulative_trade             [market depth]

    Solution: closed-form hyperbolic-sine trajectory, front-loaded to
    reduce variance of remaining inventory.

    Parameters
    ----------
    risk_aversion:  λ in the AC objective (higher = more front-loading)
    sigma:          daily return volatility of the asset
    eta:            temporary impact coefficient  (~2.5e-7 for liquid US stocks)
    gamma:          permanent impact coefficient  (~2.5e-8 for liquid US stocks)
    """

    def __init__(self,
                 risk_aversion: float = 1e-6,
                 sigma: float = 0.02,
                 eta: float = 2.5e-7,
                 gamma: float = 2.5e-8):
        self.risk_aversion = risk_aversion
        self.sigma         = sigma
        self.eta           = eta
        self.gamma         = gamma

    def compute_trajectory(self, target_shares: int,
                            T_bars: int,
                            tau: float = 1.0) -> np.ndarray:
        """
        Compute optimal trade schedule (shares per bar).

        Parameters
        ----------
        target_shares:  total shares to acquire (positive = buy)
        T_bars:         number of bars to spread execution over
        tau:            time step size (1.0 = 1 bar)

        Returns
        -------
        np.ndarray of shape (T_bars,): shares to trade at each bar.
        Sum ≈ target_shares (small rounding error).
        """
        X = float(target_shares)
        T = float(T_bars)
        sign = 1 if X >= 0 else -1
        X_abs = abs(X)

        if T <= 0 or X_abs < 1:
            return np.array([float(target_shares)])

        # κ² = λσ² / η   (AC 2001, eq. 20)
        kappa_sq = (self.risk_aversion * self.sigma ** 2) / (self.eta + 1e-12)
        kappa    = math.sqrt(max(kappa_sq, 1e-10))

        # Remaining inventory at each time step t ∈ {0, 1, ..., T}
        t_arr = np.arange(T_bars + 1, dtype=float) * tau
        denom = math.sinh(kappa * T * tau)

        if denom < 1e-10:
            # kappa ≈ 0: uniform VWAP execution
            x_t = X_abs * (1.0 - t_arr / (T * tau))
        else:
            x_t = X_abs * np.sinh(kappa * (T * tau - t_arr)) / denom

        x_t = np.clip(x_t, 0.0, X_abs)

        # Trade at each bar = decrease in inventory (must be positive = buying)
        trades = np.diff(x_t)              # (T_bars,)
        trades = -trades                    # negative diff = we're selling inventory
        trades = np.clip(trades, 0.0, X_abs)

        # Normalize to ensure exact total
        total = trades.sum()
        if total > 1e-6:
            trades = trades * (X_abs / total)

        return (sign * trades).reshape(T_bars)

    def execute(self,
                target_position_pct: float,
                portfolio_value: float,
                current_price: float,
                avg_daily_volume: float,
                current_sigma: Optional[float] = None,
                max_participation_rate: float = 0.10) -> ExecutionSchedule:
        """
        Compute the optimal execution schedule for a given target position.

        Parameters
        ----------
        target_position_pct:    e.g. 0.08 = 8% of portfolio to acquire
        portfolio_value:        current portfolio value in dollars
        current_price:          current asset price
        avg_daily_volume:       average shares traded per day
        current_sigma:          override volatility estimate (uses self.sigma if None)
        max_participation_rate: max fraction of daily volume per bar

        Returns
        -------
        ExecutionSchedule with trade schedule and cost estimate
        """
        sigma = current_sigma if current_sigma is not None else self.sigma

        target_dollar  = abs(target_position_pct) * portfolio_value
        target_shares  = max(1, int(target_dollar / max(current_price, 1e-8)))
        max_per_bar    = max(1, int(avg_daily_volume * max_participation_rate))

        # Minimum bars needed: ceiling of target_shares / max_per_bar
        T_bars = max(1, int(math.ceil(target_shares / max_per_bar)))
        T_bars = min(T_bars, 20)  # cap at ~4 trading weeks

        # Use updated sigma for trajectory
        executor = AlmgrenChrissExecutor(
            risk_aversion=self.risk_aversion,
            sigma=sigma,
            eta=self.eta,
            gamma=self.gamma,
        )
        trades = executor.compute_trajectory(target_shares, T_bars)

        # Estimated cost (temporary + permanent impact)
        total_cost_dollar = 0.0
        for t_shares in trades:
            t_abs = abs(t_shares)
            # Temporary impact: η * (t_shares / avg_volume) per bar
            temp_impact = self.eta * (t_abs / (avg_daily_volume + 1e-8))
            # Permanent impact: γ * cumulative_shares over avg_daily_dollar_vol
            perm_impact = self.gamma * t_abs
            bar_cost = (temp_impact + perm_impact) * current_price
            total_cost_dollar += bar_cost

        cost_bps = (total_cost_dollar / max(target_dollar, 1.0)) * 10_000.0
        participation = float(np.mean(np.abs(trades)) / (avg_daily_volume + 1e-8))

        # Convert share amounts to portfolio fraction per bar
        trades_pct = [float(t * current_price / max(portfolio_value, 1.0)) for t in trades]

        return ExecutionSchedule(
            trades_pct_per_bar=trades_pct,
            n_bars=T_bars,
            total_shares=target_shares,
            estimated_cost_bps=round(cost_bps, 2),
            participation_rate=round(participation, 4),
        )


# ── Module-level default executor ────────────────────────────────────────────

_default_executor = AlmgrenChrissExecutor(
    risk_aversion=1e-6,
    sigma=0.02,
    eta=2.5e-7,
    gamma=2.5e-8,
)


def should_use_ac(position_delta_pct: float, threshold: float = 0.02) -> bool:
    """Returns True if Almgren-Chriss execution should be used (trade > 2% of portfolio)."""
    return abs(position_delta_pct) > threshold


def get_execution_schedule(target_pct: float,
                            portfolio_value: float,
                            price: float,
                            avg_volume: float,
                            sigma: Optional[float] = None) -> ExecutionSchedule:
    """Convenience wrapper using default executor parameters."""
    return _default_executor.execute(
        target_position_pct=target_pct,
        portfolio_value=portfolio_value,
        current_price=price,
        avg_daily_volume=avg_volume,
        current_sigma=sigma,
    )


# ── Type hint fix ──────────────────────────────────────────────────────────────
from typing import Optional  # noqa: E402  (needed for Optional in execute signature)
