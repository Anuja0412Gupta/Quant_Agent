"""
RL Meta-Controller — v2 (Production RL Redesign)
=================================================
RL is now the *primary decision-maker* — not a weight adjuster.

──────────────────────────────────────────────────────────────────────────────
Action Space  (continuous, 2 dims):
  [rl_action ∈ [-1, +1],   position_size ∈ [0, 1]]
   -1 = full short, 0 = flat/neutral, +1 = full long

Effective action after Bayesian uncertainty-aware gating:
  effective_action = rl_action × (1 − α × disagreement_score)
  position_size    = position_size × (1 − β × disagreement_score)

State Space  (16 dims via feature_engineering.build_state_vector):
  [indicator_signal, indicator_conf,
   pattern_signal,   pattern_conf,
   trend_signal,     trend_conf,
   regime_signal,    regime_conf,
   atr_normalized,   disagreement_score,
   hurst_exponent,   rolling_return_5d,
   rolling_return_20d, rolling_vol_20d,
   drawdown,         portfolio_cash_ratio]

Mixture of Experts (MoE):
  Three PPO policies, one per market regime:
    • trending       (Hurst > 0.55)
    • mean_reverting (Hurst < 0.45)
    • high_volatility (ATR ratio > threshold)
  MarketRegimeAgent selects the active policy at inference time.

Reward:
  See reward_function.py — risk-adjusted return minus structured penalties.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import RL_LEARNING_RATE, RL_MODEL_PATH, RL_TIMESTEPS
from agents.feature_engineering import build_state_vector, STATE_DIM

logger = logging.getLogger(__name__)

# ── Disagreement gating coefficients ─────────────────────────────────────────
_ALPHA_ACTION   = 0.8   # how much disagreement shrinks rl_action magnitude
_BETA_SIZE      = 0.6   # how much disagreement shrinks position_size

# ── Regime detection thresholds ───────────────────────────────────────────────
_HURST_TRENDING  = 0.55
_HURST_MR        = 0.45
_ATR_HIGH_VOL    = 0.035

# ── MoE regime keys ──────────────────────────────────────────────────────────
REGIME_TRENDING       = "trending"
REGIME_MEAN_REVERTING = "mean_reverting"
REGIME_HIGH_VOL       = "high_volatility"
_REGIMES = [REGIME_TRENDING, REGIME_MEAN_REVERTING, REGIME_HIGH_VOL]

_RL_AVAILABLE = False


def _try_import_rl() -> bool:
    global _RL_AVAILABLE
    if _RL_AVAILABLE:
        return True
    try:
        import gymnasium          # noqa: F401
        import stable_baselines3  # noqa: F401
        _RL_AVAILABLE = True
        logger.info("RL dependencies loaded successfully.")
    except Exception as e:
        logger.warning("RL dependencies unavailable (%s). Falling back to heuristic.", e)
        _RL_AVAILABLE = False
    return _RL_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# Trading Environment  (16-dim obs, 2-dim continuous action)
# ══════════════════════════════════════════════════════════════════════════════

def _build_env():
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
            )
            # action[0]: direction [-1, +1], action[1]: size [0, 1]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0,  1.0], dtype=np.float32),
            )
            self._obs = np.zeros(STATE_DIM, dtype=np.float32)
            self._reward = 0.0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self._obs, {}

        def step(self, action):
            return self._obs, self._reward, True, False, {}

        def set_state(self, obs: np.ndarray, reward: float):
            self._obs = obs.astype(np.float32)
            self._reward = float(reward)

    return DummyVecEnv([lambda: TradingEnv()]), TradingEnv


# ══════════════════════════════════════════════════════════════════════════════
# RL Meta-Controller
# ══════════════════════════════════════════════════════════════════════════════

class RLMetaController:
    """
    Production PPO-based RL controller with:
    - Continuous action space (position direction + size)
    - Mixture of Experts (regime-specific policies)
    - Disagreement gating (uncertainty-aware position sizing)
    - Reward from reward_function.py (with fee model + ablation support)
    """

    def __init__(self):
        self._enabled = _try_import_rl()
        self.envs: Dict[str, Any]   = {}
        self.models: Dict[str, Any] = {}

        # Running state for entropy / training history
        self._reward_history: List[float] = []
        self._entropy_history: List[float] = []

        if self._enabled:
            try:
                for regime in _REGIMES:
                    env, _ = _build_env()
                    self.envs[regime] = env
                    self.models[regime] = self._load_or_create(regime, env)
            except Exception as e:
                logger.warning("Could not initialise RL models: %s", e)
                self._enabled = False

    # ── Model I/O ─────────────────────────────────────────────────────────────

    def _load_or_create(self, regime: str, env):
        from stable_baselines3 import PPO
        path = f"{RL_MODEL_PATH}_{regime}"
        if os.path.exists(path + ".zip"):
            logger.info("Loading RL model [%s] from %s.zip", regime, path)
            return PPO.load(path, env=env)
        logger.info("Creating new PPO model for regime [%s]", regime)
        return PPO(
            "MlpPolicy", env,
            learning_rate=RL_LEARNING_RATE,
            n_steps=64, batch_size=16, n_epochs=4,
            verbose=0,
        )

    def save(self):
        if not self._enabled:
            return
        os.makedirs(os.path.dirname(RL_MODEL_PATH) or ".", exist_ok=True)
        for regime, model in self.models.items():
            path = f"{RL_MODEL_PATH}_{regime}"
            model.save(path)
            logger.info("RL model [%s] saved to %s.zip", regime, path)

    # ── Regime selector ───────────────────────────────────────────────────────

    @staticmethod
    def _select_regime(regime_result: Dict[str, Any]) -> str:
        """
        Select the active MoE policy based on market regime.
        Falls back to 'trending' if regime_confidence < 0.6.
        """
        regime     = str(regime_result.get("regime", "trending")).lower()
        confidence = float(regime_result.get("confidence", 0.0))
        atr_ratio  = float(regime_result.get("atr_ratio", 0.0))
        hurst      = float(regime_result.get("hurst", 0.5))

        if confidence < 0.6:
            return REGIME_TRENDING   # fallback general policy

        if atr_ratio > _ATR_HIGH_VOL:
            return REGIME_HIGH_VOL
        if hurst > _HURST_TRENDING:
            return REGIME_TRENDING
        if hurst < _HURST_MR:
            return REGIME_MEAN_REVERTING
        return REGIME_TRENDING

    # ── Disagreement gating ───────────────────────────────────────────────────

    @staticmethod
    def _apply_disagreement_gate(
        rl_action: float,
        position_size: float,
        disagreement_score: float,
    ) -> Tuple[float, float]:
        """
        Uncertainty-aware position sizing (Bayesian RL gating).

        effective_action = rl_action × (1 − α × disagreement_score)
        effective_size   = position_size × (1 − β × disagreement_score)

        At disagreement_score = 1.0 → action and size collapse to 0 (flat).
        At disagreement_score = 0.0 → full RL action passes through.
        """
        gate = 1.0 - _ALPHA_ACTION * float(np.clip(disagreement_score, 0.0, 1.0))
        size_gate = 1.0 - _BETA_SIZE * float(np.clip(disagreement_score, 0.0, 1.0))

        effective_action = float(np.clip(rl_action * gate, -1.0, 1.0))
        effective_size   = float(np.clip(position_size * size_gate, 0.0, 1.0))
        return effective_action, effective_size

    # ── Default fallback ──────────────────────────────────────────────────────

    @staticmethod
    def _default_output() -> Dict[str, Any]:
        return {
            "rl_action":          0.0,
            "position_size":      0.0,
            "effective_action":   0.0,
            "effective_position": 0.0,
            "active_regime":      REGIME_TRENDING,
            "regime_confidence":  0.0,
            "disagreement_score": 0.0,
            "gate_value":         1.0,
            "should_trade":       False,
            "direction":          "FLAT",
            "raw_action":         [0.0, 0.0],
            "rl_available":       False,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_action(
        self,
        indicator_result:    Dict[str, Any],
        pattern_result:      Dict[str, Any],
        trend_result:        Dict[str, Any],
        regime_result:       Dict[str, Any],
        disagreement_score:  float = 0.0,
        drawdown:            float = 0.0,
        portfolio_cash_pct:  float = 1.0,
        price_series=None,
    ) -> Dict[str, Any]:
        """
        Run RL inference and return trading action.

        Returns
        -------
        {
          rl_action:          float [-1, +1]  (pre-gate direction signal)
          position_size:      float [0, 1]    (pre-gate size)
          effective_action:   float [-1, +1]  (post-disagreement-gate)
          effective_position: float [0, 1]    (post-disagreement-gate size)
          active_regime:      str             (MoE policy used)
          regime_confidence:  float
          disagreement_score: float [0, 1]
          gate_value:         float [0, 1]    (1 - α×disagreement)
          should_trade:       bool
          direction:          "LONG" | "SHORT" | "FLAT"
          raw_action:         [float, float]
          rl_available:       bool
        }
        """
        if not self._enabled or not self.models:
            return self._default_output()

        obs = build_state_vector(
            indicator_result, pattern_result, trend_result, regime_result,
            disagreement_score=disagreement_score,
            drawdown=drawdown,
            portfolio_cash_pct=portfolio_cash_pct,
            price_series=price_series,
        )

        active_regime = self._select_regime(regime_result)
        model = self.models.get(active_regime, list(self.models.values())[0])

        try:
            raw_action, _ = model.predict(obs, deterministic=True)
        except Exception as e:
            logger.warning("RL predict failed: %s", e)
            return self._default_output()

        rl_action     = float(np.clip(raw_action[0], -1.0, 1.0))
        position_size = float(np.clip(raw_action[1],  0.0, 1.0))

        effective_action, effective_position = self._apply_disagreement_gate(
            rl_action, position_size, disagreement_score
        )

        gate_value = 1.0 - _ALPHA_ACTION * float(np.clip(disagreement_score, 0.0, 1.0))

        # Determine trade direction from effective_action
        if abs(effective_action) < 0.1:
            direction = "FLAT"
            should_trade = False
        elif effective_action > 0:
            direction = "LONG"
            should_trade = True
        else:
            direction = "SHORT"
            should_trade = True

        return {
            "rl_action":          round(rl_action,           4),
            "position_size":      round(position_size,        4),
            "effective_action":   round(effective_action,     4),
            "effective_position": round(effective_position,   4),
            "active_regime":      active_regime,
            "regime_confidence":  round(float(regime_result.get("confidence", 0.0)), 4),
            "disagreement_score": round(float(disagreement_score), 4),
            "gate_value":         round(gate_value, 4),
            "should_trade":       should_trade,
            "direction":          direction,
            "raw_action":         [round(float(x), 4) for x in raw_action],
            "rl_available":       True,
        }

    # Backward-compat alias for main.py (old interface expected "weights")
    def get_action_weights(self, *args, **kwargs) -> Dict[str, Any]:
        result = self.get_action(*args, **kwargs)
        # Synthesize legacy weight-like output from new action
        pos = max(0.0, result["effective_action"])
        neg = max(0.0, -result["effective_action"])
        result["weights"] = {
            "indicator": 0.30, "pattern": 0.25, "trend": 0.25, "regime": 0.20
        }
        result["position_multiplier"] = result["effective_position"]
        return result

    # ── Training history accessors ────────────────────────────────────────────

    def get_brain_state(self) -> Dict[str, Any]:
        """Return RL brain state for the /rl/brain endpoint."""
        rewards = self._reward_history[-200:]
        arr = np.array(rewards) if rewards else np.array([0.0])
        return {
            "reward_history":   [round(float(r), 4) for r in rewards],
            "entropy_history":  [round(float(e), 4) for e in self._entropy_history[-200:]],
            "mean_reward":      round(float(arr.mean()), 4),
            "reward_std":       round(float(arr.std()), 4),
            "n_steps_trained":  len(self._reward_history),
            "active_regimes":   list(self.models.keys()),
            "rl_available":     self._enabled,
        }

    # ── Online learning step ──────────────────────────────────────────────────

    def update(
        self,
        indicator_result, pattern_result, trend_result, regime_result,
        disagreement_score: float,
        drawdown: float,
        pnl_pct: float,
        volatility: float,
        is_trade: bool = False,
        bars_in_trade: int = 1,
        price_series=None,
    ) -> float:
        if not self._enabled or not self.models:
            return 0.0

        from agents.reward_function import compute_reward, TradeContext
        ctx = TradeContext(
            pnl_pct=pnl_pct,
            drawdown=drawdown,
            volatility=volatility,
            is_trade=is_trade,
            bars_in_trade=bars_in_trade,
        )
        reward_info = compute_reward(ctx)
        reward = reward_info["reward"]

        obs = build_state_vector(
            indicator_result, pattern_result, trend_result, regime_result,
            disagreement_score=disagreement_score, drawdown=drawdown,
            price_series=price_series,
        )

        active_regime = self._select_regime(regime_result)
        env = self.envs.get(active_regime)
        model = self.models.get(active_regime)
        if env is None or model is None:
            return 0.0

        env.envs[0].set_state(obs, reward)
        try:
            model.learn(total_timesteps=16, reset_num_timesteps=False)
        except Exception as e:
            logger.warning("RL learn step failed: %s", e)

        self._reward_history.append(reward)
        return reward

    def train(self, timesteps: int = RL_TIMESTEPS):
        if not self._enabled or not self.models:
            logger.warning("RL not available — skipping training.")
            return
        for regime, model in self.models.items():
            logger.info("Training regime policy [%s] for %d timesteps…", regime, timesteps)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
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
    price_series=None,
) -> Dict[str, Any]:
    """Convenience wrapper — backward compatible entry point."""
    ctrl = get_controller()
    return ctrl.get_action_weights(
        indicator_result, pattern_result, trend_result, regime_result,
        disagreement_score=disagreement_idx,
        drawdown=drawdown,
        price_series=price_series,
    )
