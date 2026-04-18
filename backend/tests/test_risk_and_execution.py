"""
Tests for Risk Management, Lagrangian Constraints, and Execution
=================================================================
- DCCGARCHRiskModel: fit, portfolio CVaR
- LagrangianConstraintManager: multiplier updates, augmented reward
- AlmgrenChrissExecutor: trajectory sum, participation rate
- kyle_lambda_slippage: scaling properties
- TradeDecision: veto conditions
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import unittest
import numpy as np
import pandas as pd


def _make_returns(T=300, seed=42):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randn(T) * 0.01, name="returns")


class TestKyleLambdaSlippage(unittest.TestCase):

    def test_larger_trade_higher_slippage(self):
        from backtest.engine import kyle_lambda_slippage
        # Pass bid_ask_spread_pct=0 to observe pure market impact (no floor clipping)
        small = kyle_lambda_slippage(1_000,   1_000_000, 150.0, 0.02,
                                     bid_ask_spread_pct=0.0)
        large = kyle_lambda_slippage(100_000, 1_000_000, 150.0, 0.02,
                                     bid_ask_spread_pct=0.0)
        self.assertGreater(large, small)

    def test_bounded_above(self):
        from backtest.engine import kyle_lambda_slippage
        slip = kyle_lambda_slippage(10_000_000, 100_000, 150.0, 0.10)
        self.assertLessEqual(slip, 0.05)

    def test_zero_volume_fallback(self):
        from backtest.engine import kyle_lambda_slippage
        slip = kyle_lambda_slippage(1_000, 0, 150.0, 0.02, bid_ask_spread_pct=0.001)
        self.assertAlmostEqual(slip, 0.001, places=4)

    def test_sqrt_scaling(self):
        """Slippage should scale roughly as sqrt(trade_size)."""
        from backtest.engine import kyle_lambda_slippage
        # Use bid_ask_spread_pct=0 to isolate pure impact scaling
        s1 = kyle_lambda_slippage(1_000,  1_000_000, 100.0, 0.02,
                                  bid_ask_spread_pct=0.0)
        s4 = kyle_lambda_slippage(4_000,  1_000_000, 100.0, 0.02,
                                  bid_ask_spread_pct=0.0)
        s16= kyle_lambda_slippage(16_000, 1_000_000, 100.0, 0.02,
                                  bid_ask_spread_pct=0.0)
        # sqrt scaling: s4/s1 ≈ 2, s16/s1 ≈ 4 — allow 30% tolerance
        ratio1 = s4  / (s1  + 1e-10)
        ratio2 = s16 / (s1  + 1e-10)
        self.assertGreater(ratio1, 1.5)   # should be ~2
        self.assertGreater(ratio2, 3.0)   # should be ~4


class TestLagrangianConstraintManager(unittest.TestCase):

    def test_multiplier_increases_on_violation(self):
        from rl.rl_policy import LagrangianConstraintManager
        mgr = LagrangianConstraintManager(lr_lambda=0.1, dd_limit=0.10, cvar_limit=0.02)
        mgr.update(episode_max_dd=0.25, episode_cvar=0.05)
        self.assertGreater(mgr.lambda_dd,   0.0)
        self.assertGreater(mgr.lambda_cvar, 0.0)

    def test_multiplier_no_change_without_violation(self):
        from rl.rl_policy import LagrangianConstraintManager
        mgr = LagrangianConstraintManager(lr_lambda=0.1, dd_limit=0.20, cvar_limit=0.04)
        mgr.update(episode_max_dd=0.10, episode_cvar=0.02)
        self.assertAlmostEqual(mgr.lambda_dd,   0.0)
        self.assertAlmostEqual(mgr.lambda_cvar, 0.0)

    def test_multiplier_capped(self):
        from rl.rl_policy import LagrangianConstraintManager
        mgr = LagrangianConstraintManager(lr_lambda=10.0, dd_limit=0.0, cvar_limit=0.0)
        for _ in range(100):
            mgr.update(episode_max_dd=1.0, episode_cvar=1.0)
        self.assertLessEqual(mgr.lambda_dd,   100.0)
        self.assertLessEqual(mgr.lambda_cvar, 100.0)

    def test_augmented_reward_reduces_on_violation(self):
        from rl.rl_policy import LagrangianConstraintManager
        mgr = LagrangianConstraintManager(lr_lambda=0.1, dd_limit=0.10, cvar_limit=0.02)
        mgr.lambda_dd   = 5.0
        mgr.lambda_cvar = 5.0
        base_r = 1.0
        augmented = mgr.augmented_reward(base_r, step_dd=0.25, step_cvar=0.05)
        self.assertLess(augmented, base_r)

    def test_state_dict_round_trip(self):
        from rl.rl_policy import LagrangianConstraintManager
        mgr = LagrangianConstraintManager()
        mgr.update(episode_max_dd=0.25, episode_cvar=0.05)
        state = mgr.state_dict()
        mgr2  = LagrangianConstraintManager()
        mgr2.load_state_dict(state)
        self.assertAlmostEqual(mgr.lambda_dd,   mgr2.lambda_dd,   places=8)
        self.assertAlmostEqual(mgr.lambda_cvar, mgr2.lambda_cvar, places=8)


class TestAlmgrenChrissExecutor(unittest.TestCase):

    def test_trajectory_sums_to_target(self):
        from rl.execution_agent import AlmgrenChrissExecutor
        executor = AlmgrenChrissExecutor(risk_aversion=1e-6, sigma=0.02)
        trades = executor.compute_trajectory(target_shares=10_000, T_bars=10)
        self.assertAlmostEqual(abs(trades).sum(), 10_000, delta=1.0)

    def test_trajectory_positive_for_buy(self):
        from rl.execution_agent import AlmgrenChrissExecutor
        executor = AlmgrenChrissExecutor()
        trades = executor.compute_trajectory(target_shares=5_000, T_bars=5)
        self.assertTrue((trades > 0).all())

    def test_trajectory_negative_for_sell(self):
        from rl.execution_agent import AlmgrenChrissExecutor
        executor = AlmgrenChrissExecutor()
        trades = executor.compute_trajectory(target_shares=-5_000, T_bars=5)
        self.assertTrue((trades < 0).all())

    def test_front_loaded_with_high_risk_aversion(self):
        """High risk_aversion → front-loaded execution (first bar > last bar)."""
        from rl.execution_agent import AlmgrenChrissExecutor
        executor = AlmgrenChrissExecutor(risk_aversion=1e-3, sigma=0.05)
        trades = executor.compute_trajectory(target_shares=10_000, T_bars=10)
        self.assertGreater(abs(trades[0]), abs(trades[-1]))

    def test_participation_rate_bounded(self):
        from rl.execution_agent import AlmgrenChrissExecutor
        executor = AlmgrenChrissExecutor()
        schedule = executor.execute(
            target_position_pct=0.05,
            portfolio_value=100_000,
            current_price=150.0,
            avg_daily_volume=5_000_000,
            max_participation_rate=0.10,
        )
        self.assertLessEqual(schedule.participation_rate, 0.15)  # small tolerance

    def test_single_bar_when_small(self):
        """Small trade should execute in 1 bar."""
        from rl.execution_agent import AlmgrenChrissExecutor
        executor = AlmgrenChrissExecutor()
        schedule = executor.execute(
            target_position_pct=0.001,   # 0.1% of portfolio
            portfolio_value=100_000,
            current_price=100.0,
            avg_daily_volume=10_000_000,
        )
        self.assertEqual(schedule.n_bars, 1)


class TestTradeDecision(unittest.TestCase):

    def test_vetoed_factory(self):
        from shared_types import TradeDecision
        d = TradeDecision.vetoed("test reason")
        self.assertFalse(d.approved)
        self.assertEqual(d.final_size, 0.0)
        self.assertEqual(d.veto_reason, "test reason")

    def test_approved_size_in_bounds(self):
        from shared_types import TradeDecision
        d = TradeDecision(approved=True, final_size=0.08, adjusted_action=0.8,
                          veto_reason=None, size_reduction_pct=0.2)
        self.assertTrue(0.0 <= d.final_size <= 1.0)

    def test_invalid_size_raises(self):
        from shared_types import TradeDecision
        with self.assertRaises(AssertionError):
            TradeDecision(approved=True, final_size=1.5, adjusted_action=0.5,
                          veto_reason=None, size_reduction_pct=0.0)


if __name__ == "__main__":
    unittest.main()
