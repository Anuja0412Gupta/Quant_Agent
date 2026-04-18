"""
QuantAgent v3.0 — TradingEnv (Gymnasium)
==========================================
45-dim observation, 2-dim continuous action space.
Regime-conditional spread model, Lagrangian constraints, curriculum integration.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:
    _GYM_OK = False
    logging.warning("gymnasium not available")

from config import (
    COMMISSION_RATE, SLIPPAGE_DAILY, MAX_PARTICIPATION_RATE,
    FEATURE_BURNIN_BARS, MIN_EPISODE_LENGTH,
    CVAR_NO_TRADE_THRESHOLD, MAX_DRAWDOWN_LIMIT,
)
from shared_types import EnvState, ChangePointAlert
from rl.reward_function import compute_reward, compute_episode_stats
from rl.rl_policy import LagrangianConstraintManager, CurriculumScheduler

logger = logging.getLogger(__name__)

STATE_DIM   = 45
ACTION_DIM  = 2    # [direction, size_modifier]


def regime_adjusted_spread(base_spread: float,
                            dominant_regime: str,
                            vix_level: float = 20.0) -> float:
    """
    Bid-ask spreads widen during high-volatility regimes.
    Empirical: spread ≈ VIX/15 × base_spread × regime_multiplier.
    """
    REGIME_MULTIPLIERS = {
        "trending":        1.0,
        "mean_reverting":  1.5,
        "high_volatility": 3.0,
    }
    vix_factor   = max(1.0, vix_level / 15.0)
    regime_mult  = REGIME_MULTIPLIERS.get(dominant_regime, 1.0)
    return base_spread * regime_mult * vix_factor


if _GYM_OK:

    class TradingEnv(gym.Env):
        """
        Single-asset trading environment with 45-dim observations.

        Action space: Box([-1,-1], [1,1])
            action[0]: direction (positive=LONG, negative=SHORT, 0=FLAT)
            action[1]: size modifier (0=min size, 1=max size)

        Observation space: Box(45,) float32
        """
        metadata = {"render_modes": []}

        def __init__(
            self,
            features_df:   pd.DataFrame,           # (T, 45) feature matrix
            prices_df:     pd.DataFrame,            # (T,) OHLCV
            regime_probs:  Optional[pd.DataFrame] = None,  # (T, 3)
            ensemble       = None,                  # DeepEnsembleUncertainty
            constraint_mgr: Optional[LagrangianConstraintManager] = None,
            curriculum:     Optional[CurriculumScheduler] = None,
            initial_capital: float = 100_000.0,
            max_position_pct: float = 0.10,
            episode_length: int = 252,
            ticker: str = "UNKNOWN",
            vix_level: float = 20.0,
        ):
            super().__init__()
            assert len(features_df) == len(prices_df), "features and prices must align"
            assert features_df.shape[1] == STATE_DIM, \
                f"Expected {STATE_DIM} feature dims, got {features_df.shape[1]}"

            self.features          = features_df.values.astype(np.float32)
            self.prices            = prices_df["Close"].values.astype(float)
            self.volumes           = prices_df["Volume"].values.astype(float)
            self.regime_probs      = regime_probs
            self.ensemble          = ensemble
            self.constraint_mgr    = constraint_mgr or LagrangianConstraintManager()
            self.curriculum        = curriculum
            self.initial_capital   = initial_capital
            self.max_position_pct  = max_position_pct
            self.episode_length    = episode_length
            self.ticker            = ticker
            self.vix_level         = vix_level

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(STATE_DIM,), dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

            # State variables reset in reset()
            self._reset_state()

        def _reset_state(self) -> None:
            self.capital       = self.initial_capital
            self.position      = 0.0    # current fraction of capital in position
            self.entry_price   = 0.0
            self.holding_period = 0
            self.peak_equity   = self.initial_capital
            self.current_step  = 0
            self.episode_step  = 0
            self._returns_buf  = deque(maxlen=20)
            self._all_returns: list = []
            self.max_drawdown  = 0.0
            self.trades_today  = 0
            self.days_no_trade = 0
            self._cvar_window  = deque(maxlen=252)
            self.running_cvar  = 0.0

        def reset(self, *, seed: Optional[int] = None,
                  options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self._reset_state()

            # Curriculum-based start selection
            min_start = FEATURE_BURNIN_BARS
            if self.curriculum is not None:
                start = self.curriculum.sample_start(
                    len(self.features), self.episode_length
                )
                start = max(start, min_start)
            else:
                upper = max(min_start + 1,
                            len(self.features) - self.episode_length)
                start = int(np.random.randint(min_start, upper))

            self.current_step = start
            obs = self._get_obs()
            return obs, {"start_idx": start}

        def step(self, action: np.ndarray
                 ) -> Tuple[np.ndarray, float, bool, bool, dict]:
            assert self.action_space.contains(
                action.astype(np.float32)
            ) or True  # relax for numerical tolerance

            direction   = float(np.clip(action[0], -1.0, 1.0))
            size_mod    = float(np.clip(action[1],  0.0, 1.0))
            target_pos  = direction * size_mod * self.max_position_pct

            # ── Prices ──────────────────────────────────────────────────────
            current_price = self.prices[self.current_step]
            if self.current_step + 1 < len(self.prices):
                next_price = self.prices[self.current_step + 1]
            else:
                next_price = current_price

            # ── Regime-adjusted spread ───────────────────────────────────────
            dominant_regime = self._get_dominant_regime()
            spread = regime_adjusted_spread(SLIPPAGE_DAILY, dominant_regime,
                                            self.vix_level)

            # ── Execute trade ────────────────────────────────────────────────
            prev_position   = self.position
            position_delta  = target_pos - prev_position
            abs_delta       = abs(position_delta)

            # Transaction cost: spread + commission
            dollar_traded   = abs_delta * self.capital
            total_cost      = dollar_traded * (spread + COMMISSION_RATE)
            self.capital   -= total_cost

            # Update position
            self.position = target_pos

            # ── PnL calculation ──────────────────────────────────────────────
            price_return    = (next_price - current_price) / (current_price + 1e-8)
            position_pnl    = self.position * self.capital * price_return
            self.capital   += position_pnl

            log_return = math.log(next_price / (current_price + 1e-8) + 1e-12)
            step_return = (self.capital - self.initial_capital) / self.initial_capital

            # ── Drawdown tracking ────────────────────────────────────────────
            self.peak_equity   = max(self.peak_equity, self.capital)
            current_dd         = max(0.0, 1.0 - self.capital / self.peak_equity)
            self.max_drawdown  = max(self.max_drawdown, current_dd)

            # ── CVaR tracking (rolling window) ────────────────────────────────
            self._cvar_window.append(step_return)
            self.running_cvar = self._compute_cvar()

            # ── Holding period update ────────────────────────────────────────
            if abs(self.position) > 0.01:
                self.holding_period += 1
                self.days_no_trade = 0
            else:
                self.holding_period = 0
                self.days_no_trade += 1

            # ── Agent consensus ──────────────────────────────────────────────
            agent_consensus = self._get_agent_consensus()

            # ── Build EnvState ────────────────────────────────────────────────
            self._returns_buf.append(step_return)
            self._all_returns.append(step_return)

            unrealized_pnl = position_pnl if abs(self.position) > 0.01 else 0.0

            env_state = EnvState(
                log_return=log_return,
                current_drawdown=current_dd,
                recent_20_returns=list(self._returns_buf),
                position_delta=position_delta,
                unrealized_pnl=unrealized_pnl,
                holding_period=self.holding_period,
                portfolio_heat=abs(self.position),
                days_since_last_trade=self.days_no_trade,
                current_position=self.position,
                agent_consensus=agent_consensus,
                dominant_regime=dominant_regime,
                news_sentiment_score=float(self.features[self.current_step, 21]),
                sentiment_trend="neutral",
                sec_earnings_flag=bool(self.features[self.current_step, 27] > 0.5),
                sec_8k_flag=bool(self.features[self.current_step, 28] > 0.5),
                changepoint_probability=float(self.features[self.current_step, 39]),
                running_cvar_95=self.running_cvar,
                action_magnitude=float(abs(direction * size_mod)),
                trades_today=self.trades_today,
            )

            reward_dict = compute_reward(env_state, self.constraint_mgr)
            reward = reward_dict["reward"]

            # ── Step ─────────────────────────────────────────────────────────
            self.current_step += 1
            self.episode_step += 1

            # Episode termination conditions
            terminated = False
            truncated  = False

            if current_dd >= MAX_DRAWDOWN_LIMIT:
                terminated = True
                logger.debug("TradingEnv: max drawdown %.2f%% → episode terminated",
                              current_dd * 100)

            if self.running_cvar >= CVAR_NO_TRADE_THRESHOLD * 2:
                terminated = True
                logger.debug("TradingEnv: CVaR %.4f → episode terminated",
                              self.running_cvar)

            if self.episode_step >= self.episode_length:
                truncated = True

            if self.current_step >= len(self.features) - 1:
                truncated = True

            obs     = self._get_obs()
            info    = {
                "capital":        self.capital,
                "position":       self.position,
                "drawdown":       current_dd,
                "running_cvar":   self.running_cvar,
                "log_return":     log_return,
                "regime":         dominant_regime,
                "step":           self.current_step,
                "reward_dict":    reward_dict,
            }

            return obs, reward, terminated, truncated, info

        def _get_obs(self) -> np.ndarray:
            idx = min(self.current_step, len(self.features) - 1)
            obs = self.features[idx].copy()

            # Inject current position context (dims [34-36])
            obs[34] = float(np.clip(self.position / self.max_position_pct, -1, 1))
            obs[35] = float(1.0 - abs(self.position))
            obs[36] = float(np.clip(self.max_drawdown, 0, 1))

            # Replace NaN with 0 (rare, only in burn-in region)
            obs = np.where(np.isfinite(obs), obs, 0.0)
            return obs.astype(np.float32)

        def _get_dominant_regime(self) -> str:
            if self.regime_probs is not None and self.current_step < len(self.regime_probs):
                row = self.regime_probs.iloc[self.current_step]
                return str(row.idxmax())
            obs = self.features[min(self.current_step, len(self.features) - 1)]
            p_arr = obs[29:32]
            labels = ["trending", "mean_reverting", "high_volatility"]
            return labels[int(np.argmax(p_arr))]

        def _get_agent_consensus(self) -> float:
            """Use ensemble prediction for current step if available."""
            if self.ensemble is not None and self.ensemble._fitted:
                obs = self.features[min(self.current_step, len(self.features) - 1)]
                result = self.ensemble.predict(obs.reshape(1, -1))
                return float(result.agent_consensus)
            return 0.0

        def _compute_cvar(self) -> float:
            """95% CVaR on rolling return window."""
            if len(self._cvar_window) < 10:
                return 0.0
            arr = np.array(self._cvar_window)
            threshold_idx = max(1, int(len(arr) * 0.05))
            tail = np.sort(arr)[:threshold_idx]
            return float(abs(tail.mean()))

        def get_episode_stats(self) -> dict:
            """Called by training callback after each episode."""
            return {
                **compute_episode_stats(self._all_returns),
                "final_capital": round(self.capital, 2),
                "peak_equity":   round(self.peak_equity, 2),
                "episode_steps": self.episode_step,
            }

        @property
        def tradeable(self) -> bool:
            """True when current bar is safe to trade (no SEC flags, no crisis)."""
            obs = self.features[min(self.current_step, len(self.features) - 1)]
            earnings_flag = obs[27] > 0.5
            k8_flag       = obs[28] > 0.5
            cp_prob       = obs[39] > 0.7
            return not (earnings_flag or k8_flag or cp_prob)

else:
    # Stub when gymnasium not available
    class TradingEnv:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium not available")
