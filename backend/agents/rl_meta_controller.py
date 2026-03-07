"""
RL Meta-Controller
==================
A Proximal Policy Optimization (PPO) agent implemented with Stable Baselines3
that learns to adjust agent weights, position size multipliers, and whether to
trade at all — based on market state features.

──────────────────────────────────────────────────────────────────────────────
State  (observation vector, 12 dims):
  [indicator_signal, indicator_conf,
   pattern_signal,   pattern_conf,
   trend_signal,     trend_conf,
   regime_signal,    regime_conf,
   volatility,       disagreement_index,
   hurst_exponent,   drawdown]

Action (continuous, 6 dims, each [0, 1]):
  [w_indicator, w_pattern, w_trend, w_regime, position_multiplier, trade_flag]

Reward:
  risk_adjusted_return - drawdown_penalty
  = (PnL / volatility) - (drawdown * 2)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from config import (
    DEFAULT_AGENT_WEIGHTS, RL_LEARNING_RATE,
    RL_MODEL_PATH, RL_TIMESTEPS,
)
from utils.helpers import clamp

logger = logging.getLogger(__name__)

_SIGNAL_MAP = {
    "bullish": 1.0,  "uptrend": 1.0,  "trending": 0.5,
    "bearish": -1.0, "downtrend": -1.0, "mean_reverting": -0.3,
    "neutral": 0.0,  "sideways": 0.0,
    "high_volatility": -0.5, "low_volatility": 0.2,
}

_RL_AVAILABLE = False   # set to True after first successful import


def _try_import_rl():
    """Lazily import heavy RL deps; return True on success."""
    global _RL_AVAILABLE
    if _RL_AVAILABLE:
        return True
    try:
        import gymnasium          # noqa: F401
        import stable_baselines3  # noqa: F401
        _RL_AVAILABLE = True
        logger.info("RL dependencies loaded successfully.")
    except Exception as e:
        logger.warning("RL dependencies unavailable (%s). Using default weights.", e)
        _RL_AVAILABLE = False
    return _RL_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# Custom Gymnasium Environment (only constructed when RL is available)
# ══════════════════════════════════════════════════════════════════════════════

def _build_env():
    """Build and return a DummyVecEnv wrapping TradingEnv."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, obs_dim: int = 12, action_dim: int = 6):
            super().__init__()
            self.obs_dim     = obs_dim
            self._current_obs: np.ndarray = np.zeros(obs_dim, dtype=np.float32)
            self._reward: float = 0.0
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self._current_obs, {}

        def step(self, action):
            return self._current_obs, self._reward, True, False, {}

        def set_state(self, obs, reward):
            self._current_obs = obs.astype(np.float32)
            self._reward = reward

    return DummyVecEnv([lambda: TradingEnv()]), TradingEnv


# ══════════════════════════════════════════════════════════════════════════════
# RL Meta-Controller
# ══════════════════════════════════════════════════════════════════════════════

class RLMetaController:
    """Wraps a PPO agent for agent-weight and position-size control."""

    def __init__(self):
        self._enabled = _try_import_rl()
        self.env      = None
        self.model    = None
        if self._enabled:
            try:
                self.env, self._TradingEnv = _build_env()
                self.model = self._load_or_create()
            except Exception as e:
                logger.warning("Could not initialise RL model: %s", e)
                self._enabled = False

    # ── Model I/O ─────────────────────────────────────────────────────────────

    def _load_or_create(self):
        from stable_baselines3 import PPO
        model_zip = RL_MODEL_PATH + ".zip"
        if os.path.exists(model_zip):
            logger.info("Loading existing RL model from %s", model_zip)
            return PPO.load(RL_MODEL_PATH, env=self.env)
        logger.info("Creating new PPO model")
        return PPO(
            "MlpPolicy", self.env,
            learning_rate=RL_LEARNING_RATE,
            n_steps=64, batch_size=16, n_epochs=4, verbose=0,
        )

    def save(self):
        if not self._enabled or self.model is None:
            return
        os.makedirs(os.path.dirname(RL_MODEL_PATH) or ".", exist_ok=True)
        self.model.save(RL_MODEL_PATH)
        logger.info("RL model saved to %s", RL_MODEL_PATH)

    # ── Observation builder ───────────────────────────────────────────────────

    @staticmethod
    def build_observation(
        indicator_result: Dict[str, Any],
        pattern_result:   Dict[str, Any],
        trend_result:     Dict[str, Any],
        regime_result:    Dict[str, Any],
        disagreement_idx: float = 0.0,
        drawdown:         float = 0.0,
    ) -> np.ndarray:
        def sig(r, key="signal"):
            return _SIGNAL_MAP.get(str(r.get(key, "neutral")).lower(), 0.0)

        return np.array([
            sig(indicator_result),
            clamp(indicator_result.get("confidence", 0.0)),
            sig(pattern_result),
            clamp(pattern_result.get("confidence", 0.0)),
            sig(trend_result, key="trend"),
            clamp(trend_result.get("confidence", 0.0)),
            sig(regime_result, key="regime"),
            clamp(regime_result.get("confidence", 0.0)),
            clamp(regime_result.get("atr_ratio", 0.0) * 20),
            clamp(disagreement_idx * 25),
            clamp(regime_result.get("hurst", 0.5)),
            clamp(drawdown),
        ], dtype=np.float32)

    # ── Default fallback weights (when RL is not available) ───────────────────

    @staticmethod
    def _default_output() -> Dict[str, Any]:
        return {
            "weights":              dict(DEFAULT_AGENT_WEIGHTS),
            "position_multiplier":  1.0,
            "should_trade":         True,
            "raw_action":           [0.3, 0.25, 0.25, 0.2, 1.0, 1.0],
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_action_weights(
        self,
        indicator_result: Dict[str, Any],
        pattern_result:   Dict[str, Any],
        trend_result:     Dict[str, Any],
        regime_result:    Dict[str, Any],
        disagreement_idx: float = 0.0,
        drawdown:         float = 0.0,
    ) -> Dict[str, Any]:
        if not self._enabled or self.model is None:
            return self._default_output()

        obs = self.build_observation(
            indicator_result, pattern_result,
            trend_result, regime_result,
            disagreement_idx, drawdown,
        )

        try:
            raw_action, _ = self.model.predict(obs, deterministic=True)
        except Exception as e:
            logger.warning("RL predict failed: %s", e)
            return self._default_output()

        raw_action = np.clip(raw_action, 0.0, 1.0)
        w_indicator, w_pattern, w_trend, w_regime, pos_mult, trade_flag = raw_action

        raw_w = np.array([w_indicator, w_pattern, w_trend, w_regime]) + 0.05
        raw_w /= raw_w.sum()

        return {
            "weights": {
                "indicator": round(float(raw_w[0]), 4),
                "pattern":   round(float(raw_w[1]), 4),
                "trend":     round(float(raw_w[2]), 4),
                "regime":    round(float(raw_w[3]), 4),
            },
            "position_multiplier":  round(float(pos_mult), 4),
            "should_trade":         bool(trade_flag > 0.5),
            "raw_action":           raw_action.tolist(),
        }

    # ── Online learning step ──────────────────────────────────────────────────

    def update(
        self,
        indicator_result, pattern_result,
        trend_result, regime_result,
        disagreement_idx, drawdown,
        pnl_pct, volatility,
    ) -> float:
        if not self._enabled or self.model is None:
            return 0.0
        reward = (pnl_pct / max(volatility, 1e-6)) - (drawdown * 2.0)
        reward = clamp(float(reward), -10.0, 10.0)
        obs    = self.build_observation(
            indicator_result, pattern_result,
            trend_result, regime_result,
            disagreement_idx, drawdown,
        )
        env_inner = self.env.envs[0]
        env_inner.set_state(obs, reward)
        try:
            self.model.learn(total_timesteps=16, reset_num_timesteps=False)
        except Exception as e:
            logger.warning("RL learn step failed: %s", e)
        return reward

    def train(self, timesteps: int = RL_TIMESTEPS):
        if not self._enabled or self.model is None:
            logger.warning("RL not available — skipping training.")
            return
        logger.info("Training RL meta-controller for %d timesteps …", timesteps)
        self.model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
        self.save()


# ── Module-level singleton ────────────────────────────────────────────────────
_controller: Optional[RLMetaController] = None


def get_controller() -> RLMetaController:
    global _controller
    if _controller is None:
        _controller = RLMetaController()
    return _controller


def run(
    indicator_result: Dict[str, Any],
    pattern_result:   Dict[str, Any],
    trend_result:     Dict[str, Any],
    regime_result:    Dict[str, Any],
    disagreement_idx: float = 0.0,
    drawdown:         float = 0.0,
) -> Dict[str, Any]:
    """Convenience function — returns adjusted weights from RL policy."""
    ctrl = get_controller()
    return ctrl.get_action_weights(
        indicator_result, pattern_result,
        trend_result, regime_result,
        disagreement_idx, drawdown,
    )
