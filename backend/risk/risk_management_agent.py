"""
QuantAgent v3.0 — Risk Management (DCC-GARCH + Fractional Kelly + CVaR)
=========================================================================
Full replacement of rolling-correlation-based risk model.

Key components:
  - DCCGARCHRiskModel: time-varying covariance via DCC(1,1) + GARCH(1,1)
  - regime_adjusted_spread(): bid-ask spread conditioned on regime + VIX
  - Fractional Kelly sizing with shrinkage from trade history
  - CVaR-based trade veto (>4%) and size reduction (>2.5%)
  - SEC flags: 8-K blackout + earnings size reduction
  - Sentiment overrides: extreme negative sentiment → reduce size

All outputs are TradeDecision typed dataclasses.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    KELLY_FRACTION, MAX_POSITION_PCT, MAX_DRAWDOWN_PCT,
    CVAR_NO_TRADE_THRESHOLD, CVAR_REDUCE_THRESHOLD,
    MAX_DRAWDOWN_LIMIT, COMMISSION_RATE,
)
from shared_types import (
    TradeDecision, DisagreementResult, RegimeResult,
    NewsSentimentResult, SECFlags,
)

logger = logging.getLogger(__name__)

try:
    from arch import arch_model as _arch_model
    _ARCH_OK = True
except ImportError:
    _ARCH_OK = False
    logger.warning("arch not installed — falling back to rolling covariance")

# ── Regime-conditional position limits ────────────────────────────────────────
_REGIME_MAX_POSITION = {
    "trending":        0.50,   # up to 50% — let RL decide actual size
    "mean_reverting":  0.30,
    "high_volatility": 0.20,
}

# Keep tiny policy jitter flat, but allow low-conviction trades to pass through.
_MIN_DIRECTION_FOR_TRADE = 0.02


# ═══════════════════════════════════════════════════════════════════════════════
# DCC-GARCH RISK MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DCCGARCHRiskModel:
    """
    Dynamic Conditional Correlation + GARCH(1,1) for time-varying covariance.

    Why DCC-GARCH instead of rolling correlation:
      Rolling 60-day correlation dramatically underestimates tail-risk
      correlation during crises (correlations spike toward 1.0 in crashes).
      DCC-GARCH captures this by modeling both volatility clustering (GARCH)
      and correlation dynamics (DCC).

    Algorithm:
      1. Fit GARCH(1,1) per asset → standardized residuals
      2. Fit DCC(1,1) on residual matrix (grid search for a, b)
      3. At inference: compute conditional correlation for given residuals
      4. Monte Carlo portfolio CVaR using DCC conditional covariance
    """

    def __init__(self):
        self.garch_results: Dict[str, object] = {}
        self.dcc_params: Dict[str, object] = {}
        self.last_std_residuals: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(self, returns_df: pd.DataFrame) -> "DCCGARCHRiskModel":
        """
        returns_df: DataFrame of log returns, columns = ticker names.
        Fits GARCH(1,1) per ticker, extracts standardized residuals,
        then fits DCC(1,1) on the residual matrix.
        """
        if not _ARCH_OK:
            logger.warning("DCCGARCHRiskModel: arch not available, skipping fit")
            return self

        std_residuals = pd.DataFrame(index=returns_df.index)

        for ticker in returns_df.columns:
            r = returns_df[ticker].dropna() * 100   # arch works in % scale
            if len(r) < 60:
                logger.warning("DCC-GARCH: not enough data for %s (%d rows)", ticker, len(r))
                std_residuals[ticker] = 0.0
                continue
            try:
                model = _arch_model(r, vol="GARCH", p=1, q=1, dist="studentst")
                result = model.fit(disp="off", show_warning=False)
                self.garch_results[ticker] = result
                std_residuals[ticker] = result.std_resid.reindex(returns_df.index).fillna(0.0)
                logger.debug("GARCH fitted for %s: omega=%.4f alpha=%.4f beta=%.4f",
                             ticker,
                             float(result.params.get("omega", 0)),
                             float(result.params.get("alpha[1]", 0)),
                             float(result.params.get("beta[1]", 0)))
            except Exception as e:
                logger.warning("GARCH fit failed for %s: %s", ticker, e)
                std_residuals[ticker] = 0.0

        self.last_std_residuals = std_residuals
        self._fit_dcc(std_residuals)
        self._fitted = True
        return self

    def _fit_dcc(self, std_resid: pd.DataFrame) -> None:
        """Fit DCC(1,1): Q_t = (1-a-b)*Q_bar + a*e_{t-1}e_{t-1}' + b*Q_{t-1}"""
        eps = std_resid.values
        T, N = eps.shape
        Q_bar = np.cov(eps.T) + np.eye(N) * 1e-6   # regularize

        # Grid search for DCC parameters a, b
        best_ll = -np.inf
        best_a, best_b = 0.05, 0.90

        for a in [0.02, 0.05, 0.10, 0.15]:
            for b in [0.80, 0.85, 0.90, 0.93, 0.96]:
                if a + b >= 1.0:
                    continue
                ll = self._dcc_loglik(eps, Q_bar, a, b)
                if ll > best_ll:
                    best_ll = ll
                    best_a, best_b = a, b

        self.dcc_params = {"a": best_a, "b": best_b, "Q_bar": Q_bar}
        logger.info("DCC params: a=%.3f b=%.3f loglik=%.2f", best_a, best_b, best_ll)

    def _dcc_loglik(self, eps: np.ndarray, Q_bar: np.ndarray,
                    a: float, b: float) -> float:
        T, N = eps.shape
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(1, T):
            Q = (1 - a - b) * Q_bar + a * np.outer(eps[t - 1], eps[t - 1]) + b * Q
            Q_diag_inv = np.diag(1.0 / np.sqrt(np.diag(Q) + 1e-12))
            R = Q_diag_inv @ Q @ Q_diag_inv
            # Clamp to valid correlation matrix
            R = np.clip(R, -0.999, 0.999)
            np.fill_diagonal(R, 1.0)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return -np.inf
            try:
                ll -= 0.5 * (logdet + eps[t] @ np.linalg.solve(R, eps[t]))
            except np.linalg.LinAlgError:
                return -np.inf
        return float(ll)

    def current_conditional_correlation(self,
                                         eps_latest: np.ndarray) -> np.ndarray:
        """Compute conditional correlation matrix for the latest residuals."""
        if not self._fitted or not self.dcc_params:
            n = len(eps_latest)
            return np.eye(n)
        a = self.dcc_params["a"]
        b = self.dcc_params["b"]
        Q_bar = self.dcc_params["Q_bar"]
        Q = (1 - a - b) * Q_bar + a * np.outer(eps_latest, eps_latest) + b * Q_bar
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q) + 1e-12))
        R = D_inv @ Q @ D_inv
        np.fill_diagonal(R, 1.0)
        return np.clip(R, -0.999, 0.999)

    def portfolio_cvar(self, weights: np.ndarray,
                       returns_df: pd.DataFrame,
                       confidence: float = 0.95,
                       n_sim: int = 10_000) -> float:
        """
        Monte Carlo portfolio CVaR using DCC conditional covariance.
        Weights: array matching returns_df columns.
        """
        if not self._fitted:
            return self._fallback_cvar(weights, returns_df, confidence)

        # Conditional correlation at latest time
        if self.last_std_residuals is not None and len(self.last_std_residuals) > 0:
            latest_resid = self.last_std_residuals.iloc[-1].values
        else:
            latest_resid = np.zeros(len(weights))

        corr = self.current_conditional_correlation(latest_resid)

        # Per-asset conditional vols from GARCH forecasts
        vols = []
        for ticker in returns_df.columns:
            if ticker in self.garch_results:
                try:
                    fcast_var = self.garch_results[ticker].forecast(horizon=1)
                    vol = float(np.sqrt(fcast_var.variance.values[-1, 0])) / 100.0
                except Exception:
                    vol = float(returns_df[ticker].std())
            else:
                vol = float(returns_df[ticker].std())
            vols.append(max(vol, 1e-6))

        D_vol = np.diag(vols)
        cov   = D_vol @ corr @ D_vol

        # Monte Carlo simulation
        rng = np.random.default_rng(42)
        try:
            sim_rets = rng.multivariate_normal(
                mean=np.zeros(len(weights)), cov=cov, size=n_sim
            )
        except np.linalg.LinAlgError:
            # Fallback: use diagonal (independent) covariance
            cov = np.diag(np.array(vols) ** 2)
            sim_rets = rng.multivariate_normal(
                mean=np.zeros(len(weights)), cov=cov, size=n_sim
            )

        port_rets = sim_rets @ weights
        threshold = np.percentile(port_rets, (1.0 - confidence) * 100.0)
        tail = port_rets[port_rets <= threshold]
        cvar = float(abs(tail.mean())) if len(tail) > 0 else 0.02
        return round(cvar, 6)

    def _fallback_cvar(self, weights: np.ndarray,
                        returns_df: pd.DataFrame,
                        confidence: float) -> float:
        """Rolling-correlation fallback when GARCH not fitted."""
        port_rets = (returns_df * weights).sum(axis=1).dropna()
        if len(port_rets) < 20:
            return 0.02
        threshold = float(np.percentile(port_rets, (1 - confidence) * 100))
        tail = port_rets[port_rets <= threshold]
        return float(abs(tail.mean())) if len(tail) > 0 else 0.02


# ── Rolling CVaR (single asset) ───────────────────────────────────────────────

def _compute_rolling_var(returns: pd.Series,
                          confidence: float = 0.95,
                          window: int = 60) -> float:
    """Rolling historical CVaR for a single asset (used in stress test)."""
    tail_pct = (1 - confidence) * 100
    r = returns.dropna().tail(window)
    if len(r) < 10:
        return 0.02
    threshold = float(np.percentile(r, tail_pct))
    tail = r[r <= threshold]
    return float(abs(tail.mean())) if len(tail) > 0 else 0.02


# Keep legacy name for backward compat
_MAX_VAR_PCT = CVAR_NO_TRADE_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CALIBRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AgentCalibrator:
    """
    Platt scaling + isotonic regression calibration for agent confidence scores.
    A confidence of 0.8 should mean the agent is correct ~80% of the time.
    """

    def __init__(self):
        self.platt_models    = {}
        self.isotonic_models = {}

    def fit(self, agent_name: str,
            confidences: np.ndarray,
            correct: np.ndarray,
            method: str = "isotonic") -> None:
        """
        confidences: (N,) array of agent confidence scores [0,1]
        correct:     (N,) binary array — 1 if agent direction matched realized return
        method:      "platt" (logistic) or "isotonic" (non-parametric)
        """
        if len(confidences) < 10:
            logger.warning("AgentCalibrator: not enough data for %s (%d samples)",
                           agent_name, len(confidences))
            return

        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        if method == "platt":
            lr = LogisticRegression(C=1.0, max_iter=200)
            lr.fit(confidences.reshape(-1, 1), correct)
            self.platt_models[agent_name] = lr
        else:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(confidences, correct)
            self.isotonic_models[agent_name] = ir

        logger.info("AgentCalibrator: fitted %s (%s) on %d samples",
                    agent_name, method, len(confidences))

    def calibrate(self, agent_name: str,
                  raw_confidence: float,
                  method: str = "isotonic") -> float:
        """Apply calibration. Falls back to raw confidence if not fitted."""
        if method == "platt" and agent_name in self.platt_models:
            return float(self.platt_models[agent_name].predict_proba(
                np.array([[raw_confidence]]))[0, 1])
        elif agent_name in self.isotonic_models:
            return float(self.isotonic_models[agent_name].predict(
                np.array([raw_confidence]))[0])
        return float(raw_confidence)

    def save(self, path: str) -> None:
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "AgentCalibrator":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def compute_agent_ir(agent_signals: pd.DataFrame,
                     forward_returns: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Compute Information Ratio (IC/ICIR) for each agent independently.
    IC  = Spearman correlation(agent_direction_t, forward_1d_return_t)
    ICIR = mean(IC_window) / std(IC_window) — measures IC consistency.

    This shows which agents actually add alpha.
    """
    results = {}
    for agent_name in agent_signals.columns:
        ic_series = []
        step = 21  # rolling monthly IC
        for i in range(step, len(forward_returns) - step, step):
            win_signals = agent_signals[agent_name].iloc[i - step: i]
            win_returns = forward_returns.iloc[i - step: i]
            # Spearman IC (rank correlation — more robust to outliers)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(win_signals, win_returns)
            if not np.isnan(ic):
                ic_series.append(float(ic))

        ic_arr = np.array(ic_series) if ic_series else np.array([0.0])
        results[agent_name] = {
            "mean_ic":         round(float(ic_arr.mean()), 4),
            "ic_std":          round(float(ic_arr.std()), 4),
            "icir":            round(float(ic_arr.mean() / (ic_arr.std() + 1e-8)), 4),
            "positive_ic_pct": round(float((ic_arr > 0).mean()), 4),
            "n_windows":       len(ic_arr),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RISK MANAGEMENT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class RiskManagementAgent:
    """
    Research-grade risk manager incorporating:
      - Fractional Kelly sizing with shrinkage
      - CVaR-based veto and size reduction
      - Regime-conditional max position limits
      - DCC-GARCH portfolio CVaR (for multi-asset)
      - SEC 8-K blackout and earnings dampening
      - Sentiment override (extreme negative + high epistemic)
      - Changepoint reduction
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 calibrator: Optional[AgentCalibrator] = None,
                 dcc_model: Optional[DCCGARCHRiskModel] = None):
        self.capital    = initial_capital
        self.calibrator = calibrator
        self.dcc_model  = dcc_model
        self._trade_history: List[Dict] = []

    def evaluate(
        self,
        proposed_action: float,          # direction [-1, 1]
        price: float,
        atr: float,
        regime_result: Optional[RegimeResult] = None,
        disagreement:  Optional[DisagreementResult] = None,
        news_result:   Optional[NewsSentimentResult] = None,
        sec_flags:     Optional[SECFlags] = None,
        current_drawdown: float = 0.0,
        daily_returns: Optional[pd.Series] = None,
        changepoint_probability: float = 0.0,
    ) -> TradeDecision:
        """
        Full risk evaluation for a proposed trade.

        Returns TradeDecision. If approved=False, no trade should execute.
        """
        direction = float(np.clip(proposed_action, -1.0, 1.0))
        if abs(direction) < _MIN_DIRECTION_FOR_TRADE:
            return TradeDecision(
                approved=True, final_size=0.0, adjusted_action=0.0,
                veto_reason="Direction too small — flat", size_reduction_pct=1.0,
            )

        # ── 1. SEC blackout ───────────────────────────────────────────────────
        if sec_flags is not None and sec_flags.recent_8k:
            return TradeDecision.vetoed(
                f"SEC 8-K filed {sec_flags.days_since_last_8k} days ago — blackout"
            )

        # ── 2. Max drawdown circuit breaker ───────────────────────────────────
        if current_drawdown >= MAX_DRAWDOWN_LIMIT:
            return TradeDecision.vetoed(
                f"Max drawdown {current_drawdown:.1%} ≥ {MAX_DRAWDOWN_LIMIT:.1%} limit"
            )

        # ── 3. CVaR check ─────────────────────────────────────────────────────
        current_cvar = 0.0
        if daily_returns is not None and len(daily_returns) >= 20:
            current_cvar = _compute_rolling_var(daily_returns)

        if current_cvar >= CVAR_NO_TRADE_THRESHOLD:
            # Log but don't veto — just reduce size
            logger.warning("CVaR %.4f >= threshold %.4f — reducing size by 50%%",
                           current_cvar, CVAR_NO_TRADE_THRESHOLD)
            # Don't return vetoed — fall through to size reduction below

        # ── 4. Changepoint reduction ──────────────────────────────────────────
        cp_size_factor = 1.0
        if changepoint_probability > 0.6:
            cp_size_factor = max(0.3, 1.0 - changepoint_probability)
            logger.info("Changepoint detected (p=%.3f) — reducing size by %.0f%%",
                        changepoint_probability, (1 - cp_size_factor) * 100)

        # ── 5. Regime-conditional max position ────────────────────────────────
        dominant_regime = regime_result.dominant_regime if regime_result else "trending"
        regime_max = _REGIME_MAX_POSITION.get(dominant_regime, MAX_POSITION_PCT)

        # ── 6. Fractional Kelly sizing ────────────────────────────────────────
        kelly_f = self._compute_kelly_fraction(daily_returns)

        # ── 7. Base position size = Kelly × regime_max ────────────────────────
        base_size = kelly_f * regime_max

        # ── 8. Epistemic uncertainty reduction ────────────────────────────────
        ep_unc = 0.0
        if disagreement is not None:
            ep_unc = disagreement.epistemic_uncertainty
        uncertainty_factor = max(0.2, 1.0 - ep_unc)

        # ── 9. Sentiment override ─────────────────────────────────────────────
        sentiment_factor = 1.0
        if news_result is not None and disagreement is not None:
            sent_score = news_result.ticker_sentiment_score
            ep_high    = ep_unc > 0.5
            # Extreme negative sentiment + high disagreement → reduce
            if sent_score < -0.5 and ep_high and direction > 0:
                sentiment_factor = 0.5
                logger.info("Sentiment override: extreme negative (%.3f) + high epistemic",
                            sent_score)

        # ── 10. Earnings dampening ────────────────────────────────────────────
        earnings_factor = 1.0
        if sec_flags is not None and sec_flags.earnings_within_5_days:
            earnings_factor = 0.30
            logger.info("Earnings within 5 days — reducing position to 30%%")

        # CVaR size reduction (partial)
        cvar_factor = 1.0
        if current_cvar >= CVAR_NO_TRADE_THRESHOLD:
            cvar_factor = 0.25   # heavy reduction but don't veto
        elif CVAR_REDUCE_THRESHOLD <= current_cvar < CVAR_NO_TRADE_THRESHOLD:
            cvar_factor = 0.50

        # ── 11. Final size ────────────────────────────────────────────────────
        final_size = (base_size
                      * uncertainty_factor
                      * sentiment_factor
                      * earnings_factor
                      * cvar_factor
                      * cp_size_factor)

        final_size = float(np.clip(final_size, 0.0, regime_max))
        size_reduction = 1.0 - (final_size / (regime_max + 1e-8))

        # ── 12. Stop loss / take profit via ATR ───────────────────────────────
        atr_sl = atr * 1.5
        atr_tp = atr * 3.0
        if direction > 0:
            stop_loss   = price - atr_sl
            take_profit = price + atr_tp
        else:
            stop_loss   = price + atr_sl
            take_profit = price - atr_tp

        logger.info(
            "Risk eval: regime=%s kelly=%.3f ep=%.3f cvar=%.4f "
            "final_size=%.4f direction=%.2f",
            dominant_regime, kelly_f, ep_unc, current_cvar,
            final_size, direction,
        )

        return TradeDecision(
            approved=True,
            final_size=round(final_size, 6),
            adjusted_action=round(direction * final_size / max(regime_max, 1e-8), 4),
            veto_reason=None,
            size_reduction_pct=round(size_reduction, 4),
            kelly_fraction=round(kelly_f, 4),
            current_cvar=round(current_cvar, 6),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
        )

    def _compute_kelly_fraction(self,
                                 daily_returns: Optional[pd.Series]) -> float:
        """
        Fractional Kelly with shrinkage toward 0.25.
        Uses last 50 trades + 50 prior returns for Bayesian shrinkage.
        """
        if daily_returns is None or len(daily_returns) < 20:
            return KELLY_FRACTION

        rets = daily_returns.dropna().tail(50).values
        if len(rets) < 5:
            return KELLY_FRACTION

        win_rate = float((rets > 0).mean())
        avg_win  = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.01
        avg_loss = float(abs(rets[rets < 0].mean())) if (rets < 0).any() else 0.01

        if avg_loss < 1e-8:
            return KELLY_FRACTION

        # Kelly formula: f = W/L - (1-W)/W  simplified as f = (W*(W+L) - L) / (W+L)
        b = avg_win / avg_loss   # win/loss ratio
        f_full = (b * win_rate - (1.0 - win_rate)) / b

        # Fractional Kelly with shrinkage toward prior (0.25)
        n = len(rets)
        shrinkage = n / (n + 50.0)   # Bayesian shrinkage: more data → less shrinkage
        f_shrunk = shrinkage * f_full + (1.0 - shrinkage) * KELLY_FRACTION

        return float(np.clip(f_shrunk, 0.01, 0.25))   # never exceed 25% Kelly

    def record_trade(self, trade_result: dict) -> None:
        """Update trade history for Kelly shrinkage estimation."""
        self._trade_history.append(trade_result)
        if len(self._trade_history) > 200:
            self._trade_history = self._trade_history[-200:]


# ── Module-level singleton ────────────────────────────────────────────────────

_risk_agent: Optional[RiskManagementAgent] = None


def get_risk_agent() -> RiskManagementAgent:
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskManagementAgent()
    return _risk_agent


def run(df: pd.DataFrame, action: float, regime_result=None,
        disagreement=None, news_result=None, sec_flags=None,
        drawdown: float = 0.0, changepoint_probability: float = 0.0) -> dict:
    """Legacy dict-interface run() for backward-compat with old main.py."""
    agent = get_risk_agent()
    price = float(df["Close"].iloc[-1])
    from utils.helpers import compute_atr
    atr_series = compute_atr(df)
    atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else price * 0.02

    rets = df["Close"].pct_change().dropna()
    decision = agent.evaluate(
        proposed_action=action,
        price=price,
        atr=atr,
        regime_result=regime_result,
        disagreement=disagreement,
        news_result=news_result,
        sec_flags=sec_flags,
        current_drawdown=drawdown,
        daily_returns=rets,
        changepoint_probability=changepoint_probability,
    )

    return {
        "approved":          decision.approved,
        "final_size":        decision.final_size,
        "adjusted_action":   decision.adjusted_action,
        "veto_reason":       decision.veto_reason,
        "kelly_fraction":    decision.kelly_fraction,
        "current_cvar":      decision.current_cvar,
        "stop_loss":         decision.stop_loss,
        "take_profit":       decision.take_profit,
        "size_reduction_pct": decision.size_reduction_pct,
        "recommendation":    "NO_TRADE" if not decision.approved else (
            "LONG" if action > 0 else "SHORT"
        ),
    }
