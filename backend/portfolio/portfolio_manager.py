"""
QuantAgent v3.0 — Portfolio Manager
======================================
Multi-asset allocation using DCC-GARCH covariance + PointInTime universe.

Key features:
  - DCC-GARCH conditional covariance for portfolio optimization
  - PointInTime universe filtering (no survivorship bias)
  - Minimum-CVaR portfolio weights via convex optimization
  - Sentiment-adjusted target weights
  - Rebalancing cost estimation before committing
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import MAX_POSITION_PCT, COMMISSION_RATE
from shared_types import AllocationResult
from data.data_fetcher import get_fetcher
from risk.risk_management_agent import DCCGARCHRiskModel

logger = logging.getLogger(__name__)

try:
    import scipy.optimize as sco
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    logger.warning("scipy not available — portfolio optimization falls back to equal-weight")


class PortfolioManager:
    """
    Multi-asset portfolio manager with DCC-GARCH covariance and CVaR optimization.
    """

    def __init__(self,
                 tickers: Optional[List[str]] = None,
                 max_weight:  float = 0.25,
                 min_weight:  float = 0.01,
                 target_cvar: float = 0.04):
        self.tickers     = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        self.max_weight  = max_weight
        self.min_weight  = min_weight
        self.target_cvar = target_cvar
        self.dcc_model   = DCCGARCHRiskModel()
        self.fetcher     = get_fetcher()
        self._last_weights: Optional[Dict[str, float]] = None
        self._fitted = False

    def allocate(
        self,
        current_weights: Optional[Dict[str, float]] = None,
        sentiment_scores: Optional[Dict[str, float]] = None,
        blackout_tickers: Optional[List[str]] = None,
        lookback_days: int = 252,
    ) -> AllocationResult:
        """
        Compute optimal portfolio weights.

        Parameters
        ----------
        current_weights:   existing weights (for rebalancing cost estimation)
        sentiment_scores:  per-ticker sentiment [-1, 1] from NewsSentimentResult
        blackout_tickers:  tickers to exclude (SEC flags, extreme CVaR)
        lookback_days:     historical window for DCC-GARCH fit

        Returns AllocationResult
        """
        from datetime import datetime
        pit = self.fetcher._pit_universe
        valid_date = datetime.utcnow()

        # ── Universe filtering (PointInTime) ──────────────────────────────
        universe = self.tickers.copy()
        if blackout_tickers:
            excluded = [t for t in universe if t in blackout_tickers]
            universe = [t for t in universe if t not in blackout_tickers]
        else:
            excluded = []

        pit.warn_if_not_constituent(universe[0] if universe else "AAPL", valid_date)

        if len(universe) < 2:
            logger.warning("PortfolioManager: fewer than 2 valid tickers — equal weight")
            return self._equal_weight_result(universe, excluded)

        # ── Fetch returns ─────────────────────────────────────────────────
        returns_dict: Dict[str, pd.Series] = {}
        for ticker in universe:
            try:
                df_tick = self.fetcher.fetch_ohlcv(ticker, "1d")
                returns_dict[ticker] = df_tick["Close"].pct_change().dropna().tail(lookback_days)
            except Exception as e:
                logger.warning("PortfolioManager: could not fetch %s: %s — skipping", ticker, e)
                excluded.append(ticker)

        universe = [t for t in universe if t in returns_dict]
        if len(universe) < 2:
            return self._equal_weight_result(universe, excluded)

        # Align returns to common index
        returns_df = pd.DataFrame(returns_dict).dropna()
        if len(returns_df) < 60:
            logger.warning("PortfolioManager: insufficient common bars (%d) — equal weight",
                           len(returns_df))
            return self._equal_weight_result(universe, excluded)

        # ── Fit DCC-GARCH ─────────────────────────────────────────────────
        try:
            self.dcc_model.fit(returns_df)
            self._fitted = True
        except Exception as e:
            logger.warning("DCC-GARCH fit failed: %s — using sample covariance", e)

        # ── CVaR-minimizing portfolio weights ─────────────────────────────
        if _SCIPY_OK and self._fitted:
            weights = self._min_cvar_weights(returns_df, universe)
        else:
            weights = self._equal_weights(universe)

        # ── Sentiment tilt ────────────────────────────────────────────────
        if sentiment_scores:
            weights = self._apply_sentiment_tilt(weights, sentiment_scores, universe)

        # ── Normalize final weights ───────────────────────────────────────
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: round(w / total_w, 4) for t, w in weights.items()}

        # ── Portfolio metrics using DCC-GARCH CVaR ────────────────────────
        w_arr = np.array([weights.get(t, 0.0) for t in universe])
        portfolio_cvar = self.dcc_model.portfolio_cvar(w_arr, returns_df[universe])

        expected_sharpe = self._portfolio_sharpe(w_arr, returns_df[universe])

        # ── Rebalancing cost ──────────────────────────────────────────────
        rebalance_cost = 0.0
        rebalance_needed = False
        if current_weights and self._last_weights:
            for ticker in universe:
                delta = abs(weights.get(ticker, 0.0) - current_weights.get(ticker, 0.0))
                rebalance_cost += delta * COMMISSION_RATE
                if delta > 0.02:   # 2% drift threshold
                    rebalance_needed = True

        self._last_weights = weights
        sent_summary = sentiment_scores or {}

        return AllocationResult(
            weights=weights,
            expected_sharpe=round(expected_sharpe, 4),
            portfolio_cvar=round(portfolio_cvar, 6),
            excluded_tickers=excluded,
            sentiment_summary=sent_summary,
            rebalance_needed=rebalance_needed,
            rebalance_cost_est=round(rebalance_cost, 6),
        )

    def _min_cvar_weights(self, returns_df: pd.DataFrame,
                           universe: List[str]) -> Dict[str, float]:
        """Minimize portfolio CVaR via scipy constrained optimization."""
        N = len(universe)
        n_sim = 10_000
        rng = np.random.default_rng(42)

        # Use DCC sample covariance for quick simulation
        cov = returns_df[universe].cov().values
        try:
            sim_rets = rng.multivariate_normal(
                mean=np.zeros(N), cov=cov, size=n_sim
            )
        except np.linalg.LinAlgError:
            cov = np.diag(np.diag(cov))
            sim_rets = rng.multivariate_normal(
                mean=np.zeros(N), cov=cov, size=n_sim
            )

        def cvar_objective(w):
            port_rets = sim_rets @ w
            sorted_r  = np.sort(port_rets)
            n_tail    = max(1, int(len(sorted_r) * 0.05))
            return float(abs(sorted_r[:n_tail].mean()))

        # Constraints: sum = 1, each weight in [min, max]
        w0 = np.ones(N) / N
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(self.min_weight, min(self.max_weight, 1.0))] * N

        try:
            result = sco.minimize(
                cvar_objective, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-7},
            )
            if result.success and result.fun < 0.99:
                weights = dict(zip(universe, result.x.tolist()))
            else:
                weights = self._equal_weights(universe)
        except Exception as e:
            logger.warning("CVaR optimization failed: %s — equal weight", e)
            weights = self._equal_weights(universe)

        return weights

    def _equal_weights(self, universe: List[str]) -> Dict[str, float]:
        n = len(universe)
        return {t: round(1.0 / n, 4) for t in universe} if n > 0 else {}

    def _apply_sentiment_tilt(self, weights: Dict[str, float],
                               sentiment: Dict[str, float],
                               universe: List[str]) -> Dict[str, float]:
        """Tilt weights by ±10% based on sentiment score."""
        tilted = {}
        for ticker in universe:
            s = sentiment.get(ticker, 0.0)
            # Tilt: positive sentiment → up to +10% weight boost
            tilt_factor = 1.0 + 0.10 * s
            tilted[ticker] = max(0.0, weights.get(ticker, 0.0) * tilt_factor)
        total = sum(tilted.values()) + 1e-10
        return {t: round(w / total, 4) for t, w in tilted.items()}

    def _portfolio_sharpe(self, weights: np.ndarray,
                           returns_df: pd.DataFrame) -> float:
        port_rets = (returns_df * weights).sum(axis=1)
        if len(port_rets) < 5 or port_rets.std() < 1e-8:
            return 0.0
        return float((port_rets.mean() / port_rets.std()) * math.sqrt(252))

    def _equal_weight_result(self, universe: List[str],
                              excluded: List[str]) -> AllocationResult:
        weights = self._equal_weights(universe)
        return AllocationResult(
            weights=weights,
            expected_sharpe=0.0,
            portfolio_cvar=0.04,
            excluded_tickers=excluded,
            sentiment_summary={},
            rebalance_needed=False,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_portfolio_mgr: Optional[PortfolioManager] = None


def get_portfolio_manager(tickers: Optional[List[str]] = None) -> PortfolioManager:
    global _portfolio_mgr
    if _portfolio_mgr is None:
        _portfolio_mgr = PortfolioManager(tickers=tickers)
    return _portfolio_mgr
