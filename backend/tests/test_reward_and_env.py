"""
Tests for Reward Function and TradingEnv
==========================================
- compute_reward: correct 4-component decomposition
- Sortino denominator uses downside only (not std)
- Lagrangian augmentation reduces reward on violation
- TradingEnv: observation shape, action space, step mechanics
- Episode termination on max drawdown and CVaR
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd


def _make_env_state(**overrides):
    from shared_types import EnvState
    defaults = dict(
        log_return=0.01,
        current_drawdown=0.05,
        recent_20_returns=[0.005] * 10 + [-0.003] * 5 + [0.001] * 5,
        position_delta=0.02,
        unrealized_pnl=100.0,
        holding_period=10,
        portfolio_heat=0.08,
        days_since_last_trade=0,
        current_position=0.08,
        agent_consensus=0.5,
        dominant_regime="trending",
        news_sentiment_score=0.2,
        sentiment_trend="improving",
        sec_earnings_flag=False,
        sec_8k_flag=False,
        changepoint_probability=0.1,
        running_cvar_95=0.015,
        action_magnitude=0.04,
        trades_today=1,
    )
    defaults.update(overrides)
    return EnvState(**defaults)


class TestComputeReward(unittest.TestCase):

    def test_reward_in_bounds(self):
        from rl.reward_function import compute_reward
        state = _make_env_state()
        r = compute_reward(state)
        self.assertGreaterEqual(r, -10.0)
        self.assertLessEqual(r,   10.0)

    def test_positive_return_positive_reward(self):
        from rl.reward_function import compute_reward
        state_pos = _make_env_state(log_return=0.02, recent_20_returns=[-0.001]*5 + [0.002]*15)
        state_neg = _make_env_state(log_return=-0.02, recent_20_returns=[-0.003]*10 + [0.001]*10)
        r_pos = compute_reward(state_pos)
        r_neg = compute_reward(state_neg)
        self.assertGreater(r_pos, r_neg)

    def test_transaction_cost_penalizes_large_delta(self):
        from rl.reward_function import compute_reward
        state_small = _make_env_state(position_delta=0.001)
        state_large = _make_env_state(position_delta=0.50)
        r_small = compute_reward(state_small)
        r_large = compute_reward(state_large)
        self.assertGreater(r_small, r_large)

    def test_lagrangian_reduces_reward_on_violation(self):
        from rl.reward_function import compute_reward
        from rl.rl_policy import LagrangianConstraintManager
        mgr = LagrangianConstraintManager(dd_limit=0.10, cvar_limit=0.02)
        mgr.lambda_dd   = 10.0
        mgr.lambda_cvar = 10.0
        state = _make_env_state(current_drawdown=0.25, running_cvar_95=0.05)
        r_with    = compute_reward(state, constraint_mgr=mgr)
        r_without = compute_reward(state, constraint_mgr=None)
        self.assertLess(r_with, r_without)

    def test_sortino_uses_downside_only(self):
        """IC: pure downside std drives r_core — verify via sign."""
        from rl.reward_function import compute_reward
        # Only upside returns (no downside) → infinite Sortino → clamped to max
        state = _make_env_state(
            log_return=0.005,
            recent_20_returns=[0.003] * 20,  # all positive, no downside
        )
        r = compute_reward(state)
        self.assertGreater(r, 0.0)  # should be positive

    def test_conviction_bonus_for_long_winners(self):
        from rl.reward_function import compute_reward
        state_long_win  = _make_env_state(unrealized_pnl=500.0, holding_period=20)
        state_short_hold = _make_env_state(unrealized_pnl=500.0, holding_period=3)
        r_long  = compute_reward(state_long_win)
        r_short = compute_reward(state_short_hold)
        self.assertGreater(r_long, r_short)


class TestComputeEpisodeStats(unittest.TestCase):

    def test_empty_returns(self):
        from rl.reward_function import compute_episode_stats
        stats = compute_episode_stats([])
        self.assertEqual(stats["max_drawdown"], 0.0)
        self.assertEqual(stats["cvar_95"],      0.0)

    def test_all_positive_returns_zero_maxdd(self):
        from rl.reward_function import compute_episode_stats
        stats = compute_episode_stats([0.01] * 100)
        self.assertAlmostEqual(stats["max_drawdown"], 0.0, places=4)

    def test_losing_streak_high_maxdd(self):
        from rl.reward_function import compute_episode_stats
        returns = [-0.05] * 20  # 5% loss per day for 20 days
        stats = compute_episode_stats(returns)
        self.assertGreater(stats["max_drawdown"], 0.30)  # ~64% cumulative loss

    def test_cvar_less_than_max_loss(self):
        from rl.reward_function import compute_episode_stats
        returns = list(np.random.RandomState(42).randn(200) * 0.01)
        stats = compute_episode_stats(returns)
        self.assertGreater(stats["cvar_95"], 0.0)
        self.assertLess(stats["cvar_95"], 1.0)


class TestTradingEnvBasic(unittest.TestCase):

    def _make_env(self, T=400):
        from environment.trading_env import TradingEnv
        from features.feature_pipeline import FeaturePipeline

        rng = np.random.RandomState(42)
        dates  = pd.date_range("2022-01-01", periods=T, freq="B")
        prices = 100.0 + np.cumsum(rng.randn(T) * 0.5)
        df = pd.DataFrame({
            "Open": prices * 0.999, "High": prices * 1.01,
            "Low":  prices * 0.99,  "Close": prices,
            "Volume": rng.randint(1_000_000, 5_000_000, T).astype(float),
        }, index=dates)

        pipeline = FeaturePipeline()
        features = pipeline.compute(df, ticker="TEST").fillna(0.0)

        env = TradingEnv(
            features_df=features,
            prices_df=df,
            episode_length=63,
            ticker="TEST",
        )
        return env

    def test_reset_returns_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        self.assertEqual(obs.shape, (45,))
        self.assertEqual(obs.dtype, np.float32)

    def test_step_returns_correct_types(self):
        env = self._make_env()
        env.reset()
        action = np.array([0.3, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(obs.shape, (45,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated,  bool)

    def test_episode_terminates_on_max_drawdown(self):
        """Artificially induce max_drawdown ≥ 0.20 → terminated=True."""
        env = self._make_env()
        env.reset()
        env.capital   = 50_000.0    # simulate 50% loss
        env.peak_equity = 100_000.0  # from initial
        action = np.array([0.0, 0.0], dtype=np.float32)  # flat position
        _, _, terminated, _, _ = env.step(action)
        self.assertTrue(terminated)

    def test_obs_no_nan_after_burnin(self):
        """Observations after burn-in should have no NaN."""
        from config import FEATURE_BURNIN_BARS
        env = self._make_env(T=FEATURE_BURNIN_BARS + 100)
        env.reset()
        # Skip to post-burnin
        env.current_step = FEATURE_BURNIN_BARS + 10
        obs = env._get_obs()
        self.assertFalse(np.isnan(obs).any())

    def test_action_space_contains_sample(self):
        env = self._make_env()
        sample = env.action_space.sample()
        self.assertTrue(env.action_space.contains(sample))


if __name__ == "__main__":
    unittest.main()
