"""Shared helper utilities for QuantAgent backend."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def direction_to_num(signal: str) -> float:
    """Map a signal string to a numeric direction {-1, 0, 1}."""
    mapping = {
        "bullish": 1.0, "uptrend": 1.0,
        "bearish": -1.0, "downtrend": -1.0,
        "neutral": 0.0, "sideways": 0.0,
    }
    return mapping.get(signal.lower(), 0.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, min_periods=period).mean()


def hurst_exponent(ts: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent using rescaled-range analysis.
    H < 0.5 → mean-reverting, H ≈ 0.5 → random walk, H > 0.5 → trending.
    """
    lags   = range(min_lag, max_lag)
    tau    = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    valid  = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return 0.5
    log_lags = np.log([v[0] for v in valid])
    log_tau  = np.log([v[1] for v in valid])
    poly     = np.polyfit(log_lags, log_tau, 1)
    return float(poly[0])


def ensure_numpy_pickle_compat() -> None:
    """
    Register NumPy import aliases used by pickles/models saved under
    different NumPy internals (e.g. numpy._core vs numpy.core).
    """
    alias_map = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
    }
    for alias_name, target_name in alias_map.items():
        if alias_name in sys.modules:
            continue
        try:
            sys.modules[alias_name] = importlib.import_module(target_name)
        except Exception:
            # Best-effort shim: if a target module is unavailable, leave it untouched.
            pass


def resolve_model_zip_path(model_save_dir: str, ticker: str = "AAPL", timeframe: str = "1d") -> str:
    """
    Resolve model .zip location across canonical and legacy directories.
    """
    ticker = str(ticker).upper().strip()
    timeframe = str(timeframe).lower().strip()
    filename = f"rl_{ticker}_{timeframe}.zip"

    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_root = os.path.abspath(os.path.join(backend_dir, ".."))

    candidates = []
    for base in [model_save_dir, os.path.join(backend_dir, "models"), os.path.join(repo_root, "models"), os.path.join(os.getcwd(), "models")]:
        p = os.path.abspath(os.path.join(base, filename))
        if p not in candidates:
            candidates.append(p)

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]
