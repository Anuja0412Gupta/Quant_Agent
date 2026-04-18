"""
QuantAgent v3.0 — Reward Function (4-component simplified)
============================================================
Replaces the 10-component version.

Components (in priority order):
  1. r_core    — Sortino proxy (penalize downside only)
  2. r_tc      — Transaction cost (linear in turnover)
  3. r_conv    — Conviction bonus (log-scaled holding bonus for winners)
  4. r_ic      — IC alignment (reward sign-agreement with high-confidence signals)

Lagrangian constraints applied via LagrangianConstraintManager:
  - max_drawdown < 0.20  (Lagrange multiplier λ_dd)
  - CVaR(95%) < 0.04     (Lagrange multiplier λ_cvar)

NOTE: Sentiment signals (news, Reddit, SEC) enter through state dims [21-28].
      The policy learns to use them. Reward shaping for sentiment is removed
      to avoid circular incentives.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional

import numpy as np

from shared_types import EnvState
from rl.rl_policy import LagrangianConstraintManager

logger = logging.getLogger(__name__)


def compute_reward(state: EnvState,
                   constraint_mgr: Optional[LagrangianConstraintManager] = None
                   ) -> float:
    """
    Compute scalar reward for one environment step.

    Parameters
    ----------
    state           : EnvState typed dataclass with all required fields.
    constraint_mgr  : Lagrangian constraint manager (updates λ externally).

    Returns
    -------
    float: augmented reward in [-10, 10]
    """
    # ── 1. r_core: Sortino proxy ─────────────────────────────────────────────
    downside = [r for r in state.recent_20_returns if r < 0]
    if len(downside) >= 2:
        downside_std = float(np.std(downside))
    else:
        downside_std = 1e-6

    r_core = float(state.log_return) / (downside_std + 1e-6)
    r_core = float(np.clip(r_core, -3.0, 3.0))

    # ── 2. r_tc: Transaction cost ─────────────────────────────────────────────
    r_tc = -0.001 * abs(state.position_delta)

    # ── 3. r_conv: Conviction bonus ──────────────────────────────────────────
    if state.unrealized_pnl > 0 and state.holding_period > 5:
        r_conv = 0.01 * math.log1p(state.holding_period)
    else:
        r_conv = 0.0

    # ── 4. r_ic: Information coefficient alignment ───────────────────────────
    # Reward when current return direction agrees with high-confidence consensus
    consensus = float(state.agent_consensus)
    sign_agree = float(np.sign(state.log_return)) * consensus
    r_ic = 0.05 * sign_agree * abs(consensus)
    r_ic = float(np.clip(r_ic, -0.5, 0.5))

    # ── Base reward clipped ───────────────────────────────────────────────────
    base = float(np.clip(r_core + r_tc + r_conv + r_ic, -5.0, 5.0))

    # ── Apply Lagrangian constraint penalties ─────────────────────────────────
    if constraint_mgr is not None:
        augmented = constraint_mgr.augmented_reward(
            base_reward=base,
            step_dd=state.current_drawdown,
            step_cvar=state.running_cvar_95,
        )
    else:
        augmented = base

    return float(np.clip(augmented, -10.0, 10.0))


def compute_episode_stats(episode_returns: List[float]) -> dict:
    """
    Compute episode-level statistics for Lagrangian multiplier update.
    """
    arr = np.array(episode_returns, dtype=float)
    if len(arr) == 0:
        return {"max_drawdown": 0.0, "cvar_95": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "total_return": 0.0}

    # Max drawdown
    cumret = np.cumprod(1.0 + arr) - 1.0
    equity = 1.0 + cumret
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / np.where(peak == 0, 1.0, peak)
    max_dd = float(abs(dd.min()))

    # CVaR 95%
    sorted_r = np.sort(arr)
    threshold_idx = max(1, int(len(sorted_r) * 0.05))
    cvar_95 = float(abs(sorted_r[:threshold_idx].mean()))

    # Sharpe (annualized daily)
    sharpe = float((arr.mean() / (arr.std() + 1e-8)) * math.sqrt(252)) \
             if len(arr) >= 2 else 0.0

    # Sortino
    down = arr[arr < 0]
    sortino = float((arr.mean() / (down.std() + 1e-8)) * math.sqrt(252)) \
              if len(down) >= 2 else 0.0

    total_ret = float(np.prod(1.0 + arr) - 1.0)

    return {
        "max_drawdown": round(max_dd, 6),
        "cvar_95":      round(cvar_95, 6),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),
        "total_return": round(total_ret, 4),
    }
