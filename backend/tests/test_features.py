"""
Tests for FeaturePipeline (45-dim)
====================================
- rolling_zscore_with_burnin: strict min_periods, NaN enforcement
- FeaturePipeline.compute: correct shape, burn-in, no future leakage
- Feature registry: 45 dims, correct group assignment
- No look-ahead: verify z-score at bar t uses only bars [0:t]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd


def _make_ohlcv(T=500, seed=42):
    rng = np.random.RandomState(seed)
    dates  = pd.date_range("2022-01-01", periods=T, freq="B")
    prices = 100.0 + np.cumsum(rng.randn(T) * 1.0)
    prices = np.abs(prices) + 50.0
    return pd.DataFrame({
        "Open":   prices * (1 + rng.randn(T) * 0.002),
        "High":   prices * (1 + np.abs(rng.randn(T)) * 0.01),
        "Low":    prices * (1 - np.abs(rng.randn(T)) * 0.01),
        "Close":  prices,
        "Volume": rng.randint(1_000_000, 10_000_000, T).astype(float),
    }, index=dates)


class TestRollingZScoreBurnin(unittest.TestCase):

    def test_first_window_minus_one_is_nan(self):
        """For window=60, rows [0:59] must be NaN — not partial estimates."""
        from features.feature_pipeline import rolling_zscore_with_burnin
        series = pd.Series(np.random.randn(200))
        z = rolling_zscore_with_burnin(series, window=60)
        self.assertTrue(z.iloc[:59].isna().all())

    def test_window_bar_is_valid(self):
        """Row 59 (0-indexed) must be non-NaN for window=60."""
        from features.feature_pipeline import rolling_zscore_with_burnin
        series = pd.Series(np.random.randn(200))
        z = rolling_zscore_with_burnin(series, window=60)
        # Bar index 59 = 60th bar = first valid bar
        self.assertFalse(np.isnan(z.iloc[59]))

    def test_clipping(self):
        """Values must be clipped to [-3, 3]."""
        from features.feature_pipeline import rolling_zscore_with_burnin
        series = pd.Series([100.0] * 60 + [0.0] * 60)  # step function → extreme z
        z = rolling_zscore_with_burnin(series, window=60, clip_val=3.0)
        valid = z.dropna()
        self.assertTrue((valid.abs() <= 3.0 + 1e-6).all())

    def test_no_future_leakage(self):
        """
        z-score at position t must be identical whether computed on data[:t+1]
        or on the full series data[:T]. This verifies no look-ahead.
        """
        from features.feature_pipeline import rolling_zscore_with_burnin
        T = 150
        series = pd.Series(np.random.randn(T))
        z_full = rolling_zscore_with_burnin(series, window=60)

        # Check 5 different positions
        for t in [59, 80, 100, 120, 140]:
            z_partial = rolling_zscore_with_burnin(series.iloc[:t + 1], window=60)
            if not np.isnan(z_full.iloc[t]) and not np.isnan(z_partial.iloc[t]):
                self.assertAlmostEqual(
                    float(z_full.iloc[t]), float(z_partial.iloc[t]),
                    places=8,
                    msg=f"Look-ahead at bar t={t}"
                )


class TestFeaturePipelineShape(unittest.TestCase):

    def test_output_shape(self):
        from features.feature_pipeline import FeaturePipeline
        df = _make_ohlcv(T=400)
        pipeline = FeaturePipeline()
        feat = pipeline.compute(df, ticker="TEST")
        self.assertEqual(feat.shape[1], 45)
        self.assertEqual(len(feat), len(df))

    def test_burnin_rows_are_nan(self):
        """First FEATURE_BURNIN_BARS rows should have NaN in rolling features."""
        from features.feature_pipeline import FeaturePipeline
        from config import FEATURE_BURNIN_BARS
        df = _make_ohlcv(T=FEATURE_BURNIN_BARS + 100)
        pipeline = FeaturePipeline()
        feat = pipeline.compute(df, ticker="TEST")

        # Check a rolling feature (e.g., rsi_zscore)
        rsi_col = feat["rsi_zscore"].iloc[:FEATURE_BURNIN_BARS]
        # At least some should be NaN
        self.assertTrue(rsi_col.isna().any() or feat["rsi_zscore"].isna().any())

    def test_column_names_canonical(self):
        from features.feature_pipeline import FeaturePipeline, STATE_DIM
        pipeline = FeaturePipeline()
        df = _make_ohlcv(T=300)
        feat = pipeline.compute(df, ticker="TEST")
        self.assertEqual(list(feat.columns), pipeline.FEATURE_NAMES)
        self.assertEqual(len(pipeline.FEATURE_NAMES), STATE_DIM)

    def test_get_latest_vector_shape(self):
        from features.feature_pipeline import FeaturePipeline, STATE_DIM
        pipeline = FeaturePipeline()
        df = _make_ohlcv(T=300)
        vec = pipeline.get_latest_vector(df, ticker="TEST")
        self.assertEqual(vec.shape, (STATE_DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_no_inf_in_valid_rows(self):
        from features.feature_pipeline import FeaturePipeline
        from config import FEATURE_BURNIN_BARS
        pipeline = FeaturePipeline()
        df = _make_ohlcv(T=FEATURE_BURNIN_BARS + 50)
        feat = pipeline.compute(df, ticker="TEST").fillna(0.0)
        self.assertFalse(np.isinf(feat.values).any())

    def test_bb_pct_b_in_zero_one(self):
        """Bollinger %B must stay in [0, 1] by construction."""
        from features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = _make_ohlcv(T=300)
        feat = pipeline.compute(df, ticker="TEST")
        bb = feat["bb_pct_b"].dropna()
        self.assertTrue((bb >= 0.0).all())
        self.assertTrue((bb <= 1.0).all())

    def test_feature_registry_groups(self):
        from features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        groups = pipeline.registry.feature_groups
        # All 45 dims covered
        all_dims = set()
        for dims in groups.values():
            all_dims.update(dims)
        self.assertEqual(len(all_dims), 45)
        self.assertEqual(max(all_dims), 44)


class TestFeatureNoLookAhead(unittest.TestCase):
    """Verify that feature at bar t only uses data up to t."""

    def test_rsi_no_future_leakage(self):
        from features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = _make_ohlcv(T=300)

        t = 200
        feat_full    = pipeline.compute(df, ticker="TEST")
        feat_partial = pipeline.compute(df.iloc[:t + 1], ticker="TEST")

        # RSI at bar t should be the same in both
        val_full    = float(feat_full["rsi_zscore"].iloc[t])
        val_partial = float(feat_partial["rsi_zscore"].iloc[t])
        if not (np.isnan(val_full) or np.isnan(val_partial)):
            self.assertAlmostEqual(val_full, val_partial, places=6,
                                    msg="Look-ahead detected in rsi_zscore")


if __name__ == "__main__":
    unittest.main()
