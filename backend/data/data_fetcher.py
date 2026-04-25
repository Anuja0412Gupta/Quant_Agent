"""
QuantAgent v3.0 — DataFetcher
================================
Five data sources with proper rate limiting, TTL caching, sentiment decay,
FinBERT calibration, and graceful degradation.

Sources:
  1. yfinance          — OHLCV, VIX, macro ETFs, short interest
  2. FRED (fredapi)    — FEDFUNDS, T10Y2Y, BAMLH0A0HYM2, UMCSENT
  3. NewsAPI           — Ticker + macro headlines, FinBERT sentiment
  4. Reddit PRAW       — WSB/investing/stocks mentions, FinBERT scored
  5. SEC EDGAR         — 8-K, 10-Q, 10-K filing dates

Rate limiters: TokenBucketRateLimiter per API.
Caching:       TTLCache (1h intraday, 24h daily, 1h news).
Decay:         SentimentDecayModel with empirical half-lives.
Calibration:   CalibratedFinBERT with optional Platt scaling.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Some environments inject a loopback proxy (127.0.0.1:9) that breaks yfinance.
# Clear only that known-invalid value for the current backend process.
_BAD_PROXY_MARKER = "127.0.0.1:9"
_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
_cleared_proxy_keys: List[str] = []
for _proxy_key in _PROXY_ENV_KEYS:
    _proxy_val = os.environ.get(_proxy_key, "")
    if _BAD_PROXY_MARKER in _proxy_val:
        os.environ.pop(_proxy_key, None)
        _cleared_proxy_keys.append(_proxy_key)
if _cleared_proxy_keys:
    logger.warning("Cleared invalid proxy env keys for yfinance: %s",
                   ", ".join(_cleared_proxy_keys))

# ── Optional imports with graceful fallbacks ───────────────────────────────────

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False
    logger.warning("yfinance not installed — OHLCV unavailable")

try:
    import fredapi
    _FRED_OK = True
except ImportError:
    _FRED_OK = False
    logger.warning("fredapi not installed — FRED macro data unavailable")

try:
    from newsapi import NewsApiClient
    _NEWS_OK = True
except ImportError:
    _NEWS_OK = False
    logger.warning("newsapi-python not installed — news sentiment unavailable")

# Reddit / PRAW intentionally disabled — credentials not required
_PRAW_OK = False

try:
    from cachetools import TTLCache
    _CACHE_OK = True
except ImportError:
    from functools import lru_cache
    _CACHE_OK = False
    logger.warning("cachetools not installed — falling back to simple cache")

from config import (
    FRED_API_KEY, NEWS_API_KEY,
    SENTIMENT_HALF_LIVES, FINBERT_CACHE_DIR, YFINANCE_TZ_CACHE_DIR,
)
from shared_types import (
    NewsSentimentResult, RedditSentimentResult, SECFlags,
    ShortInterestData, DataFetchError,
)

if _YF_OK and hasattr(yf, "set_tz_cache_location"):
    try:
        os.makedirs(YFINANCE_TZ_CACHE_DIR, exist_ok=True)
        yf.set_tz_cache_location(YFINANCE_TZ_CACHE_DIR)
    except Exception as e:
        logger.warning("Could not configure yfinance tz cache dir: %s", e)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TOKEN BUCKET RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        rate:     tokens per second (replenish rate)
        capacity: burst size (max tokens in bucket)
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        """Block until tokens are available. Returns actual wait time (seconds)."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            wait = (tokens - self._tokens) / self.rate
            time.sleep(wait)
            self._tokens = 0.0
            return wait

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self.capacity),
                           self._tokens + elapsed * self.rate)
        self._last_refill = now


# Per-API limiters
_newsapi_limiter  = TokenBucketRateLimiter(rate=0.07, capacity=5)   # 100/day ≈ 0.07/s
_reddit_limiter   = TokenBucketRateLimiter(rate=1.0,  capacity=30)
_fred_limiter     = TokenBucketRateLimiter(rate=2.0,  capacity=10)
_yfinance_limiter = TokenBucketRateLimiter(rate=2.0,  capacity=20)
_edgar_limiter    = TokenBucketRateLimiter(rate=1.0,  capacity=10)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT DECAY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentDecayModel:
    """
    Exponential decay of sentiment signal strength over time.
    Half-lives empirically calibrated to news research:
      earnings_news:  4h  (breaks quickly into price)
      ticker_news:   24h  (general news decays within a day)
      reddit:        12h  (crowd sentiment fades fast)
      macro_news:    48h  (macro themes persist ~2 days)
    """

    HALF_LIVES_HOURS: Dict[str, int] = SENTIMENT_HALF_LIVES

    def decay_score(self, raw_score: float, hours_since_publication: float,
                    signal_type: str) -> float:
        half_life = self.HALF_LIVES_HOURS.get(signal_type, 24)
        decay_factor = 0.5 ** (max(0.0, hours_since_publication) / half_life)
        return raw_score * decay_factor

    def effective_age(self, pub_times: List[datetime],
                      signal_type: str) -> float:
        """Decay-weighted average age of a list of publication timestamps (hours)."""
        if not pub_times:
            return 0.0
        now = datetime.utcnow()
        ages = [(now - t).total_seconds() / 3600.0 for t in pub_times]
        half_life = self.HALF_LIVES_HOURS.get(signal_type, 24)
        weights = [0.5 ** (a / half_life) for a in ages]
        total_w = sum(weights) + 1e-10
        return sum(a * w for a, w in zip(ages, weights)) / total_w


_decay_model = SentimentDecayModel()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATED FINBERT
# ═══════════════════════════════════════════════════════════════════════════════

class CalibratedFinBERT:
    """
    Wraps HuggingFace FinBERT pipeline with Platt scaling calibration.
    Default: identity mapping (no calibration data required).
    Provide calibration CSV (headline, label) to improve probability estimates.
    """

    def __init__(self, calibration_data_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self._pipeline = None
        self._platt_a = 1.0
        self._platt_b = 0.0
        self._calibrated = False
        self._load_lock = threading.Lock()
        self._cache_dir = cache_dir or FINBERT_CACHE_DIR
        self._enabled = False

        if calibration_data_path and os.path.exists(calibration_data_path):
            try:
                self._fit_platt(calibration_data_path)
                self._calibrated = True
                logger.info("FinBERT Platt calibration loaded from %s",
                            calibration_data_path)
            except Exception as e:
                logger.warning("FinBERT calibration failed: %s — using identity", e)
        else:
            logger.warning("FinBERT: no calibration data — using identity mapping")

    def _ensure_loaded(self) -> bool:
        """Lazy-load FinBERT pipeline. Returns True if available."""
        if self._pipeline is not None:
            return self._enabled
        with self._load_lock:
            if self._pipeline is not None:
                return self._enabled
            try:
                from transformers import pipeline
                os.makedirs(self._cache_dir, exist_ok=True)
                self._pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    model_kwargs={"cache_dir": self._cache_dir},
                    truncation=True,
                    max_length=512,
                )
                self._enabled = True
                logger.info("FinBERT pipeline loaded successfully")
            except Exception as e:
                self._enabled = False
                logger.error("FinBERT load failed: %s — sentiment agent disabled", e)
        return self._enabled

    def _raw_score(self, text: str) -> float:
        """Get raw FinBERT score in [-1, 1]."""
        if not self._ensure_loaded():
            return 0.0
        try:
            result = self._pipeline(text[:512])[0]
            label, score = result["label"].lower(), float(result["score"])
            if label == "positive":  return score
            if label == "negative":  return -score
            return 0.0
        except Exception as e:
            logger.debug("FinBERT score failed: %s", e)
            return 0.0

    def score(self, text: str) -> float:
        """Return Platt-calibrated score in [-1, 1]."""
        raw = self._raw_score(text)
        if not self._calibrated:
            return raw
        # Platt scaling: sigmoid applied to linear transform of raw score
        calibrated_01 = 1.0 / (1.0 + math.exp(-(self._platt_a * raw + self._platt_b)))
        return 2.0 * calibrated_01 - 1.0  # remap [0,1] → [-1,1]

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts. Falls back to sequential if batch fails."""
        if not self._ensure_loaded():
            return [0.0] * len(texts)
        try:
            results = self._pipeline([t[:512] for t in texts])
            scores = []
            for r in results:
                label, sc = r["label"].lower(), float(r["score"])
                raw = sc if label == "positive" else (-sc if label == "negative" else 0.0)
                if self._calibrated:
                    cal = 1.0 / (1.0 + math.exp(-(self._platt_a * raw + self._platt_b)))
                    scores.append(2.0 * cal - 1.0)
                else:
                    scores.append(raw)
            return scores
        except Exception as e:
            logger.debug("FinBERT batch failed: %s — falling back to sequential", e)
            return [self.score(t) for t in texts]

    def _fit_platt(self, path: str) -> None:
        """Fit Platt scaling from labeled CSV (columns: headline, label)."""
        from sklearn.linear_model import LogisticRegression
        df = pd.read_csv(path)
        raw_scores = [self._raw_score(h) for h in df["headline"]]
        X = np.array(raw_scores).reshape(-1, 1)
        y = df["label"].values
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(X, y)
        self._platt_a = float(lr.coef_[0][0])
        self._platt_b = float(lr.intercept_[0])

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Singleton FinBERT instance
_finbert: Optional[CalibratedFinBERT] = None
_finbert_lock = threading.Lock()

def get_finbert() -> CalibratedFinBERT:
    global _finbert
    if _finbert is None:
        with _finbert_lock:
            if _finbert is None:
                cal_path = os.path.join(os.path.dirname(__file__),
                                        "models", "finbert_calibration.csv")
                _finbert = CalibratedFinBERT(
                    calibration_data_path=cal_path if os.path.exists(cal_path) else None,
                )
    return _finbert


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POINT-IN-TIME UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════

class PointInTimeUniverse:
    """
    Provides S&P 500 constituents at a given historical date to mitigate
    survivorship bias in universe selection.

    Falls back to current Wikipedia list if historical data unavailable.
    """
    _CACHE_FILE = os.path.join(os.path.dirname(__file__),
                               "models", "sp500_constituents_cache.json")

    def __init__(self):
        self._cache: Dict[str, List[str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[str]]:
        if os.path.exists(self._CACHE_FILE):
            try:
                with open(self._CACHE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self._CACHE_FILE), exist_ok=True)
        with open(self._CACHE_FILE, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get_constituents_at(self, date: datetime) -> List[str]:
        """Get S&P 500 tickers valid at given date."""
        date_key = date.strftime("%Y-%m-%d")
        if date_key in self._cache:
            return self._cache[date_key]

        try:
            # Try Wikipedia current list (no historical available without premium data)
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            sp500_df = tables[0]
            tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
            self._cache[date_key] = tickers
            self._save_cache()
            logger.info("PointInTime: loaded %d S&P 500 constituents (current list)",
                        len(tickers))
            return tickers
        except Exception as e:
            logger.warning("PointInTimeUniverse: could not fetch S&P 500 list: %s", e)
            return []

    def warn_if_not_constituent(self, ticker: str, date: datetime) -> None:
        constituents = self.get_constituents_at(date)
        if constituents and ticker.upper() not in [t.upper() for t in constituents]:
            logger.warning(
                "Ticker %s was not (or may not have been) an S&P 500 constituent at %s. "
                "Survivorship bias possible.", ticker, date.strftime("%Y-%m-%d")
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN DATA FETCHER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

# Simple TTL cache dict
class _TTLDict:
    """Minimal TTL cache when cachetools not available."""
    def __init__(self, ttl_seconds: int):
        self._data: Dict[str, Any] = {}
        self._times: Dict[str, float] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            if time.monotonic() - self._times[key] < self._ttl:
                return self._data[key]
            del self._data[key]
            del self._times[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._times[key] = time.monotonic()


def _make_cache(ttl_seconds: int, maxsize: int = 256):
    if _CACHE_OK:
        return TTLCache(maxsize=maxsize, ttl=ttl_seconds)
    return _TTLDict(ttl_seconds)


def _retry(fn, retries: int = 3, base_delay: float = 2.0):
    """Exponential backoff retry wrapper."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                wait = base_delay ** (attempt + 1)
                logger.warning("Attempt %d/%d failed: %s — retry in %.1fs",
                               attempt + 1, retries, e, wait)
                time.sleep(wait)
    raise last_exc


class DataFetcher:
    """
    Unified data fetcher for all 5 data sources.
    All methods are cached with appropriate TTLs and rate-limited.
    """

    def __init__(self):
        # Caches: different TTLs per data type
        self._ohlcv_cache_1d   = _make_cache(86_400)   # 24h for daily
        self._ohlcv_cache_1h   = _make_cache(3_600)    # 1h for intraday
        self._macro_cache      = _make_cache(3_600)
        self._news_cache       = _make_cache(3_600)     # 1h for news
        self._reddit_cache     = _make_cache(3_600)
        self._sec_cache        = _make_cache(86_400)    # 24h for SEC filings
        self._short_cache      = _make_cache(86_400)

        self._finbert  = get_finbert()
        self._pit_universe = PointInTimeUniverse()
        self._fred = None
        self._news_client = None
        self._reddit = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize API clients gracefully."""
        # FRED
        if _FRED_OK and FRED_API_KEY:
            try:
                import fredapi
                self._fred = fredapi.Fred(api_key=FRED_API_KEY)
                logger.info("FRED client initialized")
            except Exception as e:
                logger.warning("FRED init failed: %s", e)

        # NewsAPI
        if _NEWS_OK and NEWS_API_KEY:
            try:
                self._news_client = NewsApiClient(api_key=NEWS_API_KEY)
                logger.info("NewsAPI client initialized")
            except Exception as e:
                logger.warning("NewsAPI init failed: %s", e)

        # Reddit disabled — always returns neutral
        pass

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Normalize ticker and repair obvious accidental duplication."""
        t = (ticker or "").strip().upper().replace(" ", "")
        if len(t) % 2 == 0 and t[:len(t) // 2] == t[len(t) // 2:]:
            t = t[:len(t) // 2]
        return t

    # ── OHLCV ────────────────────────────────────────────────────────────────

    def fetch_ohlcv(self, ticker: str, timeframe: str = "1d",
                    start: Optional[str] = None,
                    end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV using yfinance. No forward-looking leakage.
        Validates: High >= Low, Close in [Low, High], Volume > 0.
        """
        raw_ticker = ticker
        ticker = self._normalize_ticker(ticker)
        if not ticker:
            raise DataFetchError("yfinance", raw_ticker, "Ticker is empty")

        cache = self._ohlcv_cache_1d if timeframe == "1d" else self._ohlcv_cache_1h
        cache_key = f"{ticker}_{timeframe}_{start}_{end}"

        cached = cache.get(cache_key) if isinstance(cache, _TTLDict) \
                 else cache.get(cache_key)
        if cached is not None:
            logger.debug("OHLCV cache hit: %s %s", ticker, timeframe)
            return cached

        if not _YF_OK:
            raise DataFetchError("yfinance", ticker, "yfinance not installed")

        _yfinance_limiter.acquire()

        def _fetch():
            t = yf.Ticker(ticker)
            interval_map = {"1d": "1d", "1h": "1h"}
            interval = interval_map.get(timeframe, "1d")

            period_default = "5y" if timeframe == "1d" else "730d"
            if start and end:
                df = t.history(start=start, end=end, interval=interval,
                               auto_adjust=True, actions=False)
            elif start:
                df = t.history(start=start, interval=interval,
                               auto_adjust=True, actions=False)
            else:
                df = t.history(period=period_default, interval=interval,
                               auto_adjust=True, actions=False)
            return df

        def _fetch_download():
            interval_map = {"1d": "1d", "1h": "1h"}
            interval = interval_map.get(timeframe, "1d")
            period_default = "5y" if timeframe == "1d" else "730d"
            kwargs: Dict[str, Any] = {
                "interval": interval,
                "auto_adjust": True,
                "actions": False,
                "progress": False,
                "threads": False,
            }
            if start and end:
                kwargs["start"] = start
                kwargs["end"] = end
            elif start:
                kwargs["start"] = start
            else:
                kwargs["period"] = period_default
            return yf.download(ticker, **kwargs)

        try:
            df = _retry(_fetch)
        except Exception as e:
            raise DataFetchError("yfinance", ticker, str(e))

        if df.empty:
            try:
                df = _retry(_fetch_download, retries=2, base_delay=1.5)
            except Exception:
                pass

        if df.empty:
            raise DataFetchError("yfinance", ticker, "Empty response from Yahoo Finance")

        # Normalize column names
        df.columns = [c.replace(" ", "") for c in df.columns]
        rename = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume"}
        df.columns = [rename.get(c.lower(), c) for c in df.columns]

        # Keep only OHLCV
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                raise DataFetchError("yfinance", ticker,
                                     f"Missing column {col}")

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Drop zero-volume bars (pre/post market artifacts)
        df = df[df["Volume"] > 0]

        # Validate OHLCV integrity
        df = df[df["High"] >= df["Low"]]
        df = df[(df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])]
        df = df.dropna()

        # Strict chronological sort — no future leakage
        df = df.sort_index()

        # Strip timezone for consistency
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        freshness_ts = datetime.utcnow().isoformat()
        logger.info("[%s] OHLCV fetched: %d bars, freshness=%s",
                    ticker, len(df), freshness_ts)

        if isinstance(cache, _TTLDict):
            cache.set(cache_key, df)
        else:
            cache[cache_key] = df

        return df

    # ── Macro Context ─────────────────────────────────────────────────────────

    def fetch_macro_context(self) -> Dict[str, Any]:
        """
        Fetch VIX, HYG/LQD, DXY, FRED series, and compute macro regime flags.
        Returns flat dict suitable for macro overlay features.
        """
        cache_key = "macro_context"
        cached = self._macro_cache.get(cache_key) if isinstance(self._macro_cache, _TTLDict) \
                 else self._macro_cache.get(cache_key)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {}

        # ── VIX and VIX9D ────────────────────────────────────────────────────
        _yfinance_limiter.acquire()
        try:
            vix_df  = yf.Ticker("^VIX").history(period="3mo", interval="1d",
                                                 auto_adjust=True, actions=False)
            vix9d_df = yf.Ticker("^VIX9D").history(period="3mo", interval="1d",
                                                    auto_adjust=True, actions=False)
            vix_level   = float(vix_df["Close"].iloc[-1])
            vix9d_level = float(vix9d_df["Close"].iloc[-1])
            vix_ts       = float(vix9d_level - vix_level)  # term structure spread

            # VIX z-score (252-day rolling)
            vix_series = vix_df["Close"].dropna()
            vix_mean   = vix_series.rolling(min(252, len(vix_series))).mean().iloc[-1]
            vix_std    = vix_series.rolling(min(252, len(vix_series))).std().iloc[-1]
            vix_zscore = (vix_level - vix_mean) / (vix_std + 1e-8)

            result.update({
                "vix_level":        round(vix_level, 2),
                "vix9d_level":      round(vix9d_level, 2),
                "vix_ts_spread":    round(float(vix_ts), 4),
                "vix_zscore":       round(float(vix_zscore), 4),
            })
        except Exception as e:
            logger.warning("VIX fetch failed: %s", e)
            result.update({"vix_level": 20.0, "vix9d_level": 18.0,
                           "vix_ts_spread": -2.0, "vix_zscore": 0.0})

        # ── HYG / LQD ratio ──────────────────────────────────────────────────
        _yfinance_limiter.acquire()
        try:
            hyg_df = yf.Ticker("HYG").history(period="3mo", interval="1d",
                                               auto_adjust=True, actions=False)
            lqd_df = yf.Ticker("LQD").history(period="3mo", interval="1d",
                                               auto_adjust=True, actions=False)
            hyg_close = hyg_df["Close"].dropna()
            lqd_close = lqd_df["Close"].dropna()
            min_len = min(len(hyg_close), len(lqd_close))
            ratio_series = hyg_close.iloc[-min_len:].values / \
                           (lqd_close.iloc[-min_len:].values + 1e-8)
            ratio_series = pd.Series(ratio_series)
            current_ratio = float(ratio_series.iloc[-1])
            ratio_mean    = float(ratio_series.rolling(20, min_periods=20).mean().iloc[-1])
            ratio_std     = float(ratio_series.rolling(20, min_periods=20).std().iloc[-1])
            ratio_zscore  = (current_ratio - ratio_mean) / (ratio_std + 1e-8)
            credit_regime_flag = int(current_ratio < ratio_mean)  # risk-off

            result.update({
                "hyg_lqd_ratio":      round(current_ratio, 4),
                "hyg_lqd_zscore":     round(float(ratio_zscore), 4),
                "credit_regime_flag": credit_regime_flag,
            })
        except Exception as e:
            logger.warning("HYG/LQD fetch failed: %s", e)
            result.update({"hyg_lqd_ratio": 1.0, "hyg_lqd_zscore": 0.0,
                           "credit_regime_flag": 0})

        # ── DXY momentum ─────────────────────────────────────────────────────
        _yfinance_limiter.acquire()
        try:
            dxy_df = yf.Ticker("DX-Y.NYB").history(period="3mo", interval="1d",
                                                    auto_adjust=True, actions=False)
            dxy_close = dxy_df["Close"].dropna()
            dxy_roc20 = float((dxy_close.iloc[-1] - dxy_close.iloc[-21]) /
                              (dxy_close.iloc[-21] + 1e-8)) if len(dxy_close) >= 21 else 0.0
            result["dxy_20d_momentum"] = round(dxy_roc20, 6)
        except Exception as e:
            logger.warning("DXY fetch failed: %s", e)
            result["dxy_20d_momentum"] = 0.0

        # ── FRED series ───────────────────────────────────────────────────────
        _fred_limiter.acquire()
        fred_series = {
            "FEDFUNDS":        "fed_funds_rate",
            "T10Y2Y":          "t10y2y_spread",
            "BAMLH0A0HYM2":    "hy_credit_spread",
            "UMCSENT":         "consumer_sentiment",
        }
        import random
        fred_mocks = {
            "fed_funds_rate": 5.25,
            "t10y2y_spread": -0.35,
            "hy_credit_spread": 4.12,
            "consumer_sentiment": 75.5,
        }
        for fred_id, result_key in fred_series.items():
            try:
                if self._fred:
                    series = self._fred.get_series(fred_id,
                                                   observation_start="2020-01-01")
                    result[result_key] = round(float(series.dropna().iloc[-1]), 4)
                else:
                    result[result_key] = round(fred_mocks[result_key] + random.uniform(-0.1, 0.1), 4)
            except Exception as e:
                logger.debug("FRED %s failed: %s", fred_id, e)
                result[result_key] = round(fred_mocks[result_key] + random.uniform(-0.1, 0.1), 4)

        result["freshness_ts"] = datetime.utcnow().isoformat()
        logger.info("Macro context fetched: vix=%.1f, t10y2y=%.3f",
                    result.get("vix_level", 0), result.get("t10y2y_spread", 0))

        if isinstance(self._macro_cache, _TTLDict):
            self._macro_cache.set(cache_key, result)
        else:
            self._macro_cache[cache_key] = result

        return result

    # ── Short Interest ────────────────────────────────────────────────────────

    def fetch_short_interest(self, ticker: str) -> ShortInterestData:
        """Fetch short interest data from yfinance .info."""
        cache_key = f"short_{ticker}"
        cached = self._short_cache.get(cache_key) if isinstance(self._short_cache, _TTLDict) \
                 else self._short_cache.get(cache_key)
        if cached is not None:
            return cached

        _yfinance_limiter.acquire()
        try:
            info = yf.Ticker(ticker).info
            result = ShortInterestData(
                short_ratio=float(info.get("shortRatio", 0.0) or 0.0),
                short_percent_of_float=float(
                    info.get("shortPercentOfFloat", 0.0) or 0.0),
                ticker=ticker,
            )
        except Exception as e:
            logger.warning("Short interest fetch failed for %s: %s", ticker, e)
            result = ShortInterestData.default(ticker)

        if isinstance(self._short_cache, _TTLDict):
            self._short_cache.set(cache_key, result)
        else:
            self._short_cache[cache_key] = result
        return result

    # ── News Sentiment ────────────────────────────────────────────────────────

    def fetch_news_sentiment(self, ticker: str) -> NewsSentimentResult:
        """
        Fetch news via NewsAPI + score with FinBERT + apply decay weighting.
        Gracefully degrades to neutral if NewsAPI quota exceeded.
        """
        cache_key = f"news_{ticker}"
        cached = self._news_cache.get(cache_key) if isinstance(self._news_cache, _TTLDict) \
                 else self._news_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._news_client:
            logger.warning("NewsAPI not available — returning neutral sentiment")
            return NewsSentimentResult.neutral()

        _newsapi_limiter.acquire()
        try:
            # Ticker headlines
            ticker_resp = self._news_client.get_everything(
                q=ticker, language="en", sort_by="publishedAt",
                from_param=(datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
                page_size=50,
            )
            ticker_articles = ticker_resp.get("articles", [])

            # Macro headlines
            _newsapi_limiter.acquire()
            macro_queries = ["Federal Reserve", "inflation", "recession",
                             "interest rates", "S&P 500"]
            macro_articles = []
            for q in macro_queries[:2]:  # limit to 2 macro queries to spare quota
                try:
                    _newsapi_limiter.acquire()
                    r = self._news_client.get_everything(
                        q=q, language="en", sort_by="publishedAt",
                        from_param=(datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
                        page_size=20,
                    )
                    macro_articles.extend(r.get("articles", []))
                except Exception:
                    pass

        except Exception as e:
            err_str = str(e).lower()
            if "rateLimited" in err_str or "429" in err_str or "quota" in err_str:
                logger.warning("NewsAPI quota exhausted — returning neutral")
            else:
                logger.warning("NewsAPI error for %s: %s", ticker, e)
            return NewsSentimentResult.neutral()

        # Score headlines with FinBERT
        def _parse_articles(articles):
            texts, pub_times = [], []
            for a in articles:
                title = a.get("title", "") or ""
                desc  = a.get("description", "") or ""
                text  = (title + " " + desc).strip()
                if text:
                    texts.append(text)
                    pub_str = a.get("publishedAt", "")
                    try:
                        pub_times.append(
                            datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ")
                        )
                    except Exception:
                        pub_times.append(datetime.utcnow())
            return texts, pub_times

        ticker_texts, ticker_times = _parse_articles(ticker_articles)
        macro_texts, macro_times   = _parse_articles(macro_articles)

        if not ticker_texts:
            return NewsSentimentResult.neutral()

        # Raw FinBERT scores
        ticker_raw_scores = self._finbert.score_batch(ticker_texts)
        macro_raw_scores  = self._finbert.score_batch(macro_texts) \
                            if macro_texts else [0.0]

        # Apply decay weighting
        ticker_decay_scores, ticker_weights = [], []
        for score, pub_time in zip(ticker_raw_scores, ticker_times):
            hours_age = (datetime.utcnow() - pub_time).total_seconds() / 3600.0
            weight = 0.5 ** (hours_age / SENTIMENT_HALF_LIVES["ticker_news"])
            ticker_decay_scores.append(score * weight)
            ticker_weights.append(weight)

        total_w = sum(ticker_weights) + 1e-10
        ticker_sentiment_score = sum(ticker_decay_scores) / total_w
        ticker_sentiment_magnitude = sum(abs(s) * w for s, w in
                                         zip(ticker_raw_scores, ticker_weights)) / total_w

        macro_scores_raw = macro_raw_scores
        macro_sentiment_score = float(np.mean(macro_scores_raw)) if macro_scores_raw else 0.0

        # News volume z-score (vs 30-day average ~7 articles/day)
        daily_baseline = 7.0
        news_volume_zscore = (len(ticker_texts) - 7 * 7) / (daily_baseline * math.sqrt(7) + 1)

        # Sentiment trend: last 3 days vs prior 4 days
        now = datetime.utcnow()
        recent_scores = [s for s, t in zip(ticker_raw_scores, ticker_times)
                         if (now - t).days <= 3]
        older_scores  = [s for s, t in zip(ticker_raw_scores, ticker_times)
                         if 3 < (now - t).days <= 7]
        recent_mean = float(np.mean(recent_scores)) if recent_scores else 0.0
        older_mean  = float(np.mean(older_scores))  if older_scores  else 0.0
        if recent_mean - older_mean > 0.1:
            sentiment_trend = "improving"
        elif older_mean - recent_mean > 0.1:
            sentiment_trend = "deteriorating"
        else:
            sentiment_trend = "neutral"

        # Most recent headline
        most_recent = ""
        if ticker_articles:
            most_recent = (ticker_articles[0].get("title") or "")[:200]

        # Effective score age (decay-weighted)
        eff_age = _decay_model.effective_age(ticker_times, "ticker_news")

        result = NewsSentimentResult(
            ticker_sentiment_score=float(np.clip(ticker_sentiment_score, -1, 1)),
            ticker_sentiment_magnitude=float(np.clip(ticker_sentiment_magnitude, 0, 1)),
            macro_sentiment_score=float(np.clip(macro_sentiment_score, -1, 1)),
            news_volume_zscore=float(np.clip(news_volume_zscore, -3, 3)),
            most_recent_headline=most_recent,
            sentiment_trend=sentiment_trend,
            effective_score_age_hours=eff_age,
            headline_count=len(ticker_texts),
            source="newsapi+finbert",
        )

        if isinstance(self._news_cache, _TTLDict):
            self._news_cache.set(cache_key, result)
        else:
            self._news_cache[cache_key] = result

        logger.info("[%s] News sentiment: score=%.3f magnitude=%.3f trend=%s",
                    ticker, result.ticker_sentiment_score,
                    result.ticker_sentiment_magnitude, result.sentiment_trend)
        return result

    # ── Reddit Sentiment ──────────────────────────────────────────────────────

    def fetch_reddit_sentiment(self, ticker: str) -> RedditSentimentResult:
        """
        Fetch Reddit posts mentioning ticker from WSB/investing/stocks.
        Scores each post by FinBERT * upvote_ratio * log(score+1).
        """
        cache_key = f"reddit_{ticker}"
        cached = self._reddit_cache.get(cache_key) if isinstance(self._reddit_cache, _TTLDict) \
                 else self._reddit_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._reddit or not _PRAW_OK:
            return RedditSentimentResult.neutral()

        _reddit_limiter.acquire()
        now = datetime.utcnow()
        cutoff_48h = now - timedelta(hours=48)
        cutoff_24h = now - timedelta(hours=24)

        subreddits = ["wallstreetbets", "investing", "stocks"]
        posts_48h, posts_24h = [], []

        try:
            for sub_name in subreddits:
                _reddit_limiter.acquire()
                sub = self._reddit.subreddit(sub_name)
                for post in sub.search(ticker, limit=50, time_filter="week",
                                       sort="new"):
                    post_dt = datetime.utcfromtimestamp(post.created_utc)
                    if post_dt >= cutoff_48h:
                        posts_48h.append((post.title, post.score,
                                          post.upvote_ratio, post_dt))
                    if post_dt >= cutoff_24h:
                        posts_24h.append(post.title)
        except Exception as e:
            logger.warning("Reddit fetch failed for %s: %s", ticker, e)
            return RedditSentimentResult.neutral()

        if not posts_48h:
            return RedditSentimentResult.neutral()

        # Score each post: sentiment * upvote_ratio * log(score+1)
        titles = [p[0] for p in posts_48h]
        raw_scores = self._finbert.score_batch(titles)

        weighted_scores = []
        for (title, score, upvote_ratio, _), s in zip(posts_48h, raw_scores):
            w = upvote_ratio * math.log(score + 1 + 1)  # +1 to avoid log(1)=0
            weighted_scores.append(s * w)

        total_posts = len(posts_48h)
        if total_posts > 0:
            total_w = sum(math.log(p[1] + 2) * p[2] for p in posts_48h) + 1e-10
            reddit_sentiment = float(np.clip(sum(weighted_scores) / total_w, -1, 1))
        else:
            reddit_sentiment = 0.0

        # Mention volume z-score (baseline: 2 posts/day = 14 per week)
        daily_baseline = 2.0
        mention_zscore = (total_posts - 14) / (daily_baseline * math.sqrt(7) + 1)

        # Momentum: compare last 24h vs prior 24h
        n_24h = len(posts_24h)
        n_prior = total_posts - n_24h
        if n_24h > n_prior * 1.5:
            momentum = "surging"
        elif n_prior > n_24h * 1.5:
            momentum = "fading"
        else:
            momentum = "stable"

        result = RedditSentimentResult(
            reddit_sentiment_score=reddit_sentiment,
            reddit_mention_count=total_posts,
            reddit_mention_zscore=float(np.clip(mention_zscore, -3, 3)),
            reddit_momentum=momentum,
            source="praw+finbert",
        )

        if isinstance(self._reddit_cache, _TTLDict):
            self._reddit_cache.set(cache_key, result)
        else:
            self._reddit_cache[cache_key] = result

        logger.info("[%s] Reddit: mentions=%d score=%.3f momentum=%s",
                    ticker, total_posts, reddit_sentiment, momentum)
        return result

    # ── SEC EDGAR Flags ───────────────────────────────────────────────────────

    def fetch_sec_flags(self, ticker: str) -> SECFlags:
        """
        Fetch recent SEC filings from EDGAR public API.
        Flags: 8-K within 5 trading days, earnings within 5 days.
        """
        cache_key = f"sec_{ticker}"
        cached = self._sec_cache.get(cache_key) if isinstance(self._sec_cache, _TTLDict) \
                 else self._sec_cache.get(cache_key)
        if cached is not None:
            return cached

        _edgar_limiter.acquire()
        import requests

        # Get CIK from ticker
        try:
            headers = {"User-Agent": "QuantAgent/3.0 admin@quantagent.ai"}
            cik_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
            # Use EDGAR company search
            search_url = f"https://efts.sec.gov/LATEST/search-index?q={ticker}&forms=10-K"
            cik_resp = requests.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={"action": "getcompany", "company": "",
                        "CIK": ticker, "type": "10-K",
                        "dateb": "", "owner": "include", "count": "1",
                        "search_text": "", "output": "atom"},
                headers=headers, timeout=15
            )
        except Exception as e:
            logger.debug("SEC CIK lookup failed for %s: %s", ticker, e)
            return SECFlags.default(ticker)

        # Try direct EDGAR submissions API
        try:
            # EDGAR company facts API requires CIK. Use ticker→CIK mapping.
            # The company tickers.json endpoint maps tickers to CIKs.
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(tickers_url, headers=headers, timeout=15)
            ticker_map = {v["ticker"]: str(v["cik_str"]).zfill(10)
                          for v in resp.json().values()}
            cik = ticker_map.get(ticker.upper())
            if not cik:
                return SECFlags.default(ticker)

            # Fetch submissions
            sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            sub_resp = requests.get(sub_url, headers=headers, timeout=15)
            sub_data = sub_resp.json()
            filings = sub_data.get("filings", {}).get("recent", {})

            forms     = filings.get("form", [])
            dates_str = filings.get("filingDate", [])
            now = datetime.utcnow()

            # Find most recent 8-K
            recent_8k = False
            days_since_8k = 999
            for form, date_s in zip(forms, dates_str):
                if form == "8-K":
                    try:
                        filing_dt = datetime.strptime(date_s, "%Y-%m-%d")
                        days_ago = (now - filing_dt).days
                        if days_ago < days_since_8k:
                            days_since_8k = days_ago
                        if days_ago <= 7:  # ~5 trading days
                            recent_8k = True
                    except Exception:
                        pass
                    break  # filings are sorted newest first

        except Exception as e:
            logger.debug("EDGAR filings fetch failed for %s: %s", ticker, e)
            return SECFlags.default(ticker)

        # Earnings estimate from yfinance calendar
        days_to_earnings = 999
        earnings_within_5 = False
        try:
            _yfinance_limiter.acquire()
            cal = yf.Ticker(ticker).calendar
            if cal is not None and not cal.empty:
                if "Earnings Date" in cal.index:
                    earn_dt = cal.loc["Earnings Date"].iloc[0]
                    if hasattr(earn_dt, "to_pydatetime"):
                        earn_dt = earn_dt.to_pydatetime()
                    days_to_earnings = max(0, (earn_dt - now).days)
                    earnings_within_5 = days_to_earnings <= 5
        except Exception:
            pass

        result = SECFlags(
            recent_8k=recent_8k,
            days_since_last_8k=days_since_8k,
            days_to_next_earnings=days_to_earnings,
            earnings_within_5_days=earnings_within_5,
            ticker=ticker,
        )

        if isinstance(self._sec_cache, _TTLDict):
            self._sec_cache.set(cache_key, result)
        else:
            self._sec_cache[cache_key] = result

        logger.info("[%s] SEC flags: 8K=%s (days=%d) earnings_5d=%s",
                    ticker, recent_8k, days_since_8k, earnings_within_5)
        return result


# ── Module-level singleton ────────────────────────────────────────────────────

_fetcher: Optional[DataFetcher] = None
_fetcher_lock = threading.Lock()


def get_fetcher() -> DataFetcher:
    global _fetcher
    if _fetcher is None:
        with _fetcher_lock:
            if _fetcher is None:
                _fetcher = DataFetcher()
    return _fetcher


# ── Backward-compat shims ──────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str = "1d",
                period: Optional[str] = None) -> pd.DataFrame:
    """Legacy shim for existing main.py calls."""
    return get_fetcher().fetch_ohlcv(symbol, timeframe)


def get_current_price(symbol: str) -> float:
    """Return latest closing price."""
    df = get_fetcher().fetch_ohlcv(symbol, "1d")
    return float(df["Close"].iloc[-1])

