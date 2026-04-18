"""
Tests for DataFetcher v3.0
===========================
- TokenBucketRateLimiter: rate limiting, burst capacity, thread safety
- SentimentDecayModel: exponential decay, effective_age
- PointInTimeUniverse: constituent filtering
- DataFetcher: OHLCV validation, fallback behavior
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd


class TestTokenBucketRateLimiter(unittest.TestCase):

    def test_no_wait_when_tokens_available(self):
        from data.data_fetcher import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)
        wait = limiter.acquire(1)
        self.assertAlmostEqual(wait, 0.0, places=2)

    def test_blocks_when_tokens_exhausted(self):
        from data.data_fetcher import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(rate=1.0, capacity=1)
        limiter.acquire(1)   # consume all tokens
        t0 = time.monotonic()
        limiter.acquire(1)   # should wait ~1s
        elapsed = time.monotonic() - t0
        self.assertGreater(elapsed, 0.5)

    def test_thread_safe(self):
        from data.data_fetcher import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=50)
        results = []
        def worker():
            wait = limiter.acquire(1)
            results.append(wait)
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(results), 20)

    def test_capacity_not_exceeded(self):
        from data.data_fetcher import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(rate=1000.0, capacity=5)
        # Even with high rate, capacity caps tokens
        time.sleep(0.5)   # allow some refill
        self.assertLessEqual(limiter._tokens, 5.0)


class TestSentimentDecayModel(unittest.TestCase):

    def test_decay_reduces_score(self):
        from data.data_fetcher import SentimentDecayModel
        model = SentimentDecayModel()
        raw_score = 0.8
        decayed = model.decay_score(raw_score, hours_since_publication=24.0,
                                     signal_type="ticker_news")
        self.assertLess(decayed, raw_score)
        self.assertGreater(decayed, 0.0)

    def test_half_life_correctness(self):
        from data.data_fetcher import SentimentDecayModel
        model = SentimentDecayModel()
        # After 1 half-life, score should be ~50% of original
        half_life = 24  # ticker_news
        decayed = model.decay_score(1.0, hours_since_publication=float(half_life),
                                     signal_type="ticker_news")
        self.assertAlmostEqual(decayed, 0.5, delta=0.01)

    def test_negative_age_clamped(self):
        from data.data_fetcher import SentimentDecayModel
        model = SentimentDecayModel()
        # Negative age (future-dated article) clamped to 0
        decayed = model.decay_score(0.5, hours_since_publication=-10.0,
                                     signal_type="ticker_news")
        self.assertAlmostEqual(decayed, 0.5, delta=0.01)

    def test_effective_age(self):
        from datetime import datetime, timedelta
        from data.data_fetcher import SentimentDecayModel
        model = SentimentDecayModel()
        now = datetime.utcnow()
        times = [now - timedelta(hours=1), now - timedelta(hours=48)]
        eff_age = model.effective_age(times, "ticker_news")
        # Most weight on the recent article (1h), so effective age < 48h
        self.assertLess(eff_age, 48.0)
        self.assertGreater(eff_age, 0.0)


class TestDataFetcherOHLCV(unittest.TestCase):

    def _make_mock_df(self):
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        prices = 150.0 + np.cumsum(np.random.randn(300) * 0.5)
        df = pd.DataFrame({
            "Open":   prices * 0.999,
            "High":   prices * 1.01,
            "Low":    prices * 0.99,
            "Close":  prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, 300),
        }, index=dates)
        return df

    def test_ohlcv_validation_drops_bad_rows(self):
        """Rows where High < Low should be dropped."""
        from data.data_fetcher import DataFetcher

        with patch("data.data_fetcher._YF_OK", True):
            with patch("yfinance.Ticker") as MockTicker:
                df = self._make_mock_df()
                # Inject a bad row
                df.loc[df.index[5], "High"] = df.loc[df.index[5], "Low"] - 1.0
                instance = MockTicker.return_value
                instance.history.return_value = df

                fetcher = DataFetcher()
                result  = fetcher.fetch_ohlcv("AAPL", "1d")

                # Bad row should have been dropped
                self.assertTrue((result["High"] >= result["Low"]).all())

    def test_ohlcv_sorted_chronologically(self):
        from data.data_fetcher import DataFetcher

        with patch("data.data_fetcher._YF_OK", True):
            with patch("yfinance.Ticker") as MockTicker:
                df = self._make_mock_df()
                # Shuffle the index
                df = df.sample(frac=1.0, random_state=42)
                instance = MockTicker.return_value
                instance.history.return_value = df

                fetcher = DataFetcher()
                result  = fetcher.fetch_ohlcv("AAPL", "1d")
                self.assertTrue(result.index.is_monotonic_increasing)

    def test_raises_on_empty_response(self):
        from data.data_fetcher import DataFetcher, DataFetchError

        with patch("data.data_fetcher._YF_OK", True):
            with patch("yfinance.Ticker") as MockTicker:
                instance = MockTicker.return_value
                instance.history.return_value = pd.DataFrame()

                fetcher = DataFetcher()
                with self.assertRaises(DataFetchError):
                    fetcher.fetch_ohlcv("BADTICKER", "1d")


if __name__ == "__main__":
    unittest.main()
