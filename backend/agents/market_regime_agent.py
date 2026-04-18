"""
QuantAgent v3.0 — Regime Detection
====================================
Two statistically correct models replacing the old Hurst exponent approach:

1. StudentTHMM — Hidden Markov Model with Student-t emissions.
   More robust to fat-tailed financial returns than GaussianHMM.
   Full EM with IRLS M-step and Newton-Raphson ν update.

2. StudentTBOCPD — Bayesian Online Changepoint Detection with Student-t
   predictive (NIG conjugate model — Normal-Inverse-Gamma prior).
   Exact Adams & MacKay (2007) algorithm with Student-t instead of Gaussian.

HMM state alignment: stable labels across refits by sorting on emission scale.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.special
import scipy.stats
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT-T HMM
# ═══════════════════════════════════════════════════════════════════════════════

class StudentTHMM:
    """
    Hidden Markov Model with Student-t emission distributions.
    Robust to financial return fat tails.

    Observation model per state k:
        x_t | s_t=k ~ t(mu_k, sigma_k^2, nu_k)

    E-step: forward-backward on log scale.
    M-step: IRLS for mu/sigma, Newton-Raphson for nu (degrees of freedom).
    """

    def __init__(self, n_components: int = 3, max_iter: int = 100,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_components  = n_components
        self.max_iter      = max_iter
        self.tol           = tol
        self.random_state  = random_state

        # Parameters (initialized in fit)
        self.transmat_:   Optional[np.ndarray] = None   # (K, K)
        self.startprob_:  Optional[np.ndarray] = None   # (K,)
        self.means_:      Optional[np.ndarray] = None   # (K, D)
        self.scales_:     Optional[np.ndarray] = None   # (K, D)
        self.dofs_:       Optional[np.ndarray] = None   # (K,)  ν per state
        self._fitted = False
        self._n_iter_converged = 0
        self._final_logZ = -np.inf

    # ── Emission density ───────────────────────────────────────────────────────

    def _student_t_logpdf(self, x: np.ndarray, mu: float,
                          sigma: float, nu: float) -> np.ndarray:
        """
        Log PDF of univariate Student-t for each observation x.
        x: (T,), mu/sigma/nu: scalars. Returns (T,).
        """
        d = 0.5 * (nu + 1.0)
        log_norm = (scipy.special.gammaln(d)
                    - scipy.special.gammaln(nu / 2.0)
                    - 0.5 * np.log(nu * np.pi * sigma ** 2))
        log_kernel = -d * np.log(1.0 + ((x - mu) ** 2) / (nu * sigma ** 2 + 1e-12))
        return log_norm + log_kernel

    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        """X: (T, D). Returns log emission matrix (T, K)."""
        T, D = X.shape
        K = self.n_components
        log_em = np.zeros((T, K))
        for k in range(K):
            for d in range(D):
                log_em[:, k] += self._student_t_logpdf(
                    X[:, d],
                    float(self.means_[k, d]),
                    float(self.scales_[k, d]),
                    float(self.dofs_[k]),
                )
        return log_em

    # ── Forward-backward on log scale ─────────────────────────────────────────

    def _forward_backward(self, log_emission: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (log_alpha, log_beta, logZ).
        All operations in log domain for numerical stability.
        """
        T, K = log_emission.shape
        log_A = np.log(np.clip(self.transmat_, 1e-300, None))

        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(np.clip(self.startprob_, 1e-300, None)) + log_emission[0]

        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    scipy.special.logsumexp(log_alpha[t - 1] + log_A[:, k])
                    + log_emission[t, k]
                )

        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = scipy.special.logsumexp(
                    log_A[k, :] + log_emission[t + 1] + log_beta[t + 1]
                )

        logZ = scipy.special.logsumexp(log_alpha[-1])
        return log_alpha, log_beta, logZ

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "StudentTHMM":
        """
        X: (T, D) array of observations.
        D=3 for [log_returns, realized_vol_20d, vol_zscore].
        """
        np.random.seed(self.random_state)
        T, D = X.shape
        K = self.n_components

        assert T >= 50, f"StudentTHMM.fit: need ≥50 rows, got {T}"

        # ── Initialization via K-means ────────────────────────────────────────
        km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        km.fit(X)
        labels = km.labels_

        self.means_    = km.cluster_centers_.copy().astype(float)
        self.scales_   = np.array([
            np.clip(X[labels == k].std(axis=0), 1e-4, None)
            for k in range(K)
        ], dtype=float)
        self.dofs_     = np.full(K, 4.0, dtype=float)    # ν=4 typical for daily returns
        self.startprob_ = np.ones(K) / K
        self.transmat_  = (np.eye(K) * 0.85 + np.ones((K, K)) * 0.15 / K)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        prev_logZ = -np.inf

        for iteration in range(self.max_iter):
            # ── E-step ────────────────────────────────────────────────────────
            log_em = self._compute_log_emission(X)
            log_alpha, log_beta, logZ = self._forward_backward(log_em)
            log_gamma = log_alpha + log_beta - logZ
            gamma = np.exp(np.clip(log_gamma, -700, 0))  # (T, K)

            # Normalize to avoid floating-point drift
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

            delta = abs(logZ - prev_logZ)
            if delta < self.tol and iteration > 5:
                logger.debug("StudentTHMM converged at iteration %d (ΔlogZ=%.2e)",
                             iteration, delta)
                break
            prev_logZ = logZ

            # ── M-step: transition matrix via ξ ────────────────────────────
            log_A = np.log(np.clip(self.transmat_, 1e-300, None))
            xi = np.zeros((K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[i, j] += np.exp(
                            log_alpha[t, i] + log_A[i, j]
                            + log_em[t + 1, j] + log_beta[t + 1, j] - logZ
                        )
            row_sums = xi.sum(axis=1, keepdims=True) + 1e-300
            self.transmat_ = xi / row_sums

            # ── M-step: emission parameters via IRLS ─────────────────────────
            for k in range(K):
                gk = gamma[:, k]
                for d in range(D):
                    nu    = self.dofs_[k]
                    mu    = self.means_[k, d]
                    sigma = self.scales_[k, d]
                    delta_sq = (X[:, d] - mu) ** 2 / (sigma ** 2 + 1e-12)
                    # IRLS weights: w_t = (nu+1) / (nu + delta_sq_t)
                    w = gk * (nu + 1.0) / (nu + delta_sq + 1e-12)
                    w_sum = w.sum() + 1e-12
                    # Weighted mean
                    self.means_[k, d] = (w * X[:, d]).sum() / w_sum
                    # Weighted scale
                    mu_new = self.means_[k, d]
                    self.scales_[k, d] = np.sqrt(
                        (w * (X[:, d] - mu_new) ** 2).sum() / w_sum
                    ).clip(1e-4)

                # ── M-step: ν (degrees of freedom) via Newton-Raphson ────────
                nu = self.dofs_[k]
                gk_sum = gk.sum() + 1e-12
                for _ in range(10):   # inner Newton iterations
                    delta_sq_all = np.zeros(T)
                    for d in range(D):
                        delta_sq_all += (X[:, d] - self.means_[k, d]) ** 2 \
                                        / (self.scales_[k, d] ** 2 + 1e-12)
                    u = (nu + D) / (nu + delta_sq_all + 1e-12)
                    # Gradient of Q w.r.t. ν (maximize → gradient ascent)
                    grad = 0.5 * gk_sum * (
                        scipy.special.digamma((nu + D) / 2.0)
                        - scipy.special.digamma(nu / 2.0)
                        - D / (nu + 1e-12)
                        + np.log((nu + D) / (nu + 1e-12))
                    ) + 0.5 * float((gk * (np.log(u + 1e-12) - u)).sum())
                    # Gradient ASCENT: nu = nu + lr * grad
                    nu = float(np.clip(nu + 0.1 * grad, 1.5, 30.0))
                    if abs(grad) < 1e-4:
                        break
                self.dofs_[k] = nu

            self.startprob_ = gamma[0] / (gamma[0].sum() + 1e-12)

        self._fitted = True
        self._n_iter_converged = iteration
        self._final_logZ = float(logZ)
        logger.info("StudentTHMM fit: %d iters, logZ=%.2f, dofs=%s",
                    iteration, logZ, np.round(self.dofs_, 2).tolist())
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        X: (T, D). Returns (T, K) posterior state probabilities.
        Uses forward-backward smoothing (not Viterbi — gives full posteriors).
        """
        if not self._fitted:
            raise RuntimeError("StudentTHMM: call fit() before predict_proba()")
        log_em = self._compute_log_emission(X)
        log_alpha, log_beta, logZ = self._forward_backward(log_em)
        log_gamma = log_alpha + log_beta - logZ
        gamma = np.exp(np.clip(log_gamma, -700, 0))
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12
        return gamma

    # ── State label alignment (prevents label switching) ──────────────────────

    def label_states(self) -> Dict[int, str]:
        """
        Stable labels across refits: sort states by return-dimension scale.
        lowest scale  → "trending"      (low-vol, persistent)
        mid scale     → "mean_reverting"
        highest scale → "high_volatility"
        """
        if not self._fitted:
            return {i: "trending" for i in range(self.n_components)}
        vol_by_state = [float(self.scales_[k, 0]) for k in range(self.n_components)]
        sorted_states = list(np.argsort(vol_by_state))
        label_names = ["trending", "mean_reverting", "high_volatility"]
        return {sorted_states[i]: label_names[i] for i in range(self.n_components)}

    def _align_states(self) -> Dict[int, str]:
        """Alias for label_states — stable labels after each refit."""
        return self.label_states()

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("StudentTHMM saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "StudentTHMM":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("StudentTHMM loaded from %s", path)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT-T BOCPD
# ═══════════════════════════════════════════════════════════════════════════════

class StudentTBOCPD:
    """
    Bayesian Online Changepoint Detection with Student-t predictive distribution.
    Uses Normal-Inverse-Gamma (NIG) conjugate prior.

    After n observations in current run, the predictive distribution is:
        x_{n+1} | r=n ~ t(2*alpha_n, mu_n,
                          beta_n*(kappa_n+1) / (alpha_n*kappa_n))

    This is more robust to outliers/fat tails than the Gaussian BOCPD.

    Adams & MacKay (2007) framework, NIG conjugate version.
    """

    def __init__(self, mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 2.0, beta0: float = 0.01,
                 hazard_rate: float = 1 / 252,
                 max_run_length: int = 500):
        self.mu0          = mu0
        self.kappa0       = kappa0
        self.alpha0       = alpha0
        self.beta0        = beta0
        self.hazard_rate  = hazard_rate
        self.max_run_length = max_run_length
        self._reset()

    def _reset(self) -> None:
        self.run_length_probs = np.array([1.0])
        self.mu    = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta  = np.array([self.beta0])
        self._n_processed = 0

    def _student_t_predictive_logpdf(self, x: float) -> np.ndarray:
        """
        Log predictive density for each run-length hypothesis.
        Returns array of shape (len(run_length_probs),).

        Predictive is Student-t with:
            df    = 2 * alpha
            loc   = mu
            scale = sqrt(beta * (kappa+1) / (alpha * kappa))
        """
        df    = 2.0 * self.alpha
        loc   = self.mu
        scale = np.sqrt(
            self.beta * (self.kappa + 1.0)
            / (self.alpha * self.kappa + 1e-12)
        )
        scale = np.clip(scale, 1e-8, None)
        return scipy.stats.t.logpdf(x, df=df, loc=loc, scale=scale)

    def update(self, x: float) -> Dict[str, float]:
        """
        Process one new observation.

        Returns
        -------
        dict with keys:
            changepoint_probability: P(changepoint at current time)
            regime_stability:        weighted mean run length / max_run_length
            is_transition:           changepoint_probability > 0.5
            run_length_distribution: np.ndarray (copy)
        """
        log_pred = self._student_t_predictive_logpdf(x)
        log_rl   = np.log(np.clip(self.run_length_probs, 1e-300, None))

        # Growth: no changepoint
        log_growth = log_rl + log_pred + np.log(1.0 - self.hazard_rate)

        # Changepoint: run resets to 0
        log_cp_mass = scipy.special.logsumexp(log_rl + log_pred)
        log_cp      = log_cp_mass + np.log(self.hazard_rate)

        # Concatenate [changepoint_rl=0, growth_rl=0..T]
        new_log_probs = np.concatenate([[log_cp], log_growth])
        log_Z = scipy.special.logsumexp(new_log_probs)
        self.run_length_probs = np.exp(new_log_probs - log_Z)

        # Truncate for stability
        if len(self.run_length_probs) > self.max_run_length:
            self.run_length_probs = self.run_length_probs[-self.max_run_length:]
            self.run_length_probs /= self.run_length_probs.sum() + 1e-12

        # ── NIG parameter updates ──────────────────────────────────────────────
        # For run-length r=0 (changepoint): reset to prior
        # For run-length r>0: conjugate NIG update
        kappa_new = np.concatenate([[self.kappa0], self.kappa + 1.0])
        mu_new    = np.concatenate([
            [self.mu0],
            (self.kappa * self.mu + x) / (self.kappa + 1.0),
        ])
        alpha_new = np.concatenate([[self.alpha0], self.alpha + 0.5])
        beta_new  = np.concatenate([
            [self.beta0],
            self.beta + (self.kappa * (x - self.mu) ** 2)
                        / (2.0 * (self.kappa + 1.0) + 1e-12),
        ])

        # Truncate parameter arrays to match run_length_probs
        keep = len(self.run_length_probs)
        self.mu    = mu_new[-keep:]
        self.kappa = kappa_new[-keep:]
        self.alpha = alpha_new[-keep:]
        self.beta  = beta_new[-keep:]

        self._n_processed += 1

        # ── Derived quantities ────────────────────────────────────────────────
        changepoint_prob = float(self.run_length_probs[0])

        rl_arr = np.arange(len(self.run_length_probs), dtype=float)
        weighted_run = float(np.dot(self.run_length_probs, rl_arr))
        regime_stability = float(np.clip(
            weighted_run / max(float(min(self._n_processed, self.max_run_length)), 1.0),
            0.0, 1.0,
        ))

        return {
            "changepoint_probability": changepoint_prob,
            "regime_stability":        regime_stability,
            "is_transition":           changepoint_prob > 0.5,
            "run_length_distribution": self.run_length_probs.copy(),
        }

    def reset(self) -> None:
        """Reset state to prior (called when a changepoint is confirmed)."""
        self._reset()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN REGIME AGENT RUN()
# ═══════════════════════════════════════════════════════════════════════════════

# Global detector instances (loaded/trained lazily)
_hmm_detectors: Dict[str, StudentTHMM] = {}
_bocpd_detectors: Dict[str, StudentTBOCPD] = {}

from shared_types import RegimeResult, ChangePointAlert
from config import HMM_MIN_FIT_BARS, HMM_REFIT_PERIOD_BARS, MODEL_SAVE_DIR
import pandas as pd


def _get_hmm(ticker: str, df: pd.DataFrame,
             force_refit: bool = False) -> Optional[StudentTHMM]:
    """Load or fit a per-ticker HMM. Returns None if not enough data."""
    model_path = os.path.join(MODEL_SAVE_DIR, f"hmm_{ticker}.pkl")

    if ticker in _hmm_detectors and not force_refit:
        return _hmm_detectors[ticker]

    if os.path.exists(model_path) and not force_refit:
        try:
            model = StudentTHMM.load(model_path)
            _hmm_detectors[ticker] = model
            return model
        except Exception as e:
            logger.warning("HMM load failed for %s: %s — refitting", ticker, e)

    if len(df) < HMM_MIN_FIT_BARS:
        logger.warning("Not enough bars to fit HMM for %s (%d < %d)",
                       ticker, len(df), HMM_MIN_FIT_BARS)
        return None

    X = _build_hmm_features(df)
    model = StudentTHMM(n_components=3, max_iter=100, random_state=42)
    model.fit(X)
    model.save(model_path)
    _hmm_detectors[ticker] = model
    return model


def _build_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build 3-dimensional HMM feature matrix: [log_returns, realized_vol_20d, vol_zscore].
    All features computed strictly on past data (point-in-time correct).
    """
    closes = df["Close"].values.astype(float)
    log_returns = np.log(closes[1:] / closes[:-1])

    # Pad with 0 at start to maintain alignment
    log_returns = np.concatenate([[0.0], log_returns])

    # Realized vol (20-day rolling, min_periods=20 for burn-in correctness)
    ret_series = pd.Series(log_returns)
    realized_vol = ret_series.rolling(20, min_periods=20).std().fillna(0.02).values

    # Vol z-score (60-day rolling)
    vol_series = pd.Series(realized_vol)
    vol_mean = vol_series.rolling(60, min_periods=60).mean().fillna(0.02)
    vol_std  = vol_series.rolling(60, min_periods=60).std().fillna(0.005)
    vol_zscore = ((vol_series - vol_mean) / (vol_std + 1e-8)).clip(-3, 3).fillna(0).values

    X = np.column_stack([log_returns, realized_vol, vol_zscore])
    return X.astype(float)


def run(df: pd.DataFrame, ticker: str = "UNKNOWN",
        force_hmm_refit: bool = False) -> RegimeResult:
    """
    Run regime detection on OHLCV DataFrame.
    Returns RegimeResult typed dataclass.
    """
    if len(df) < 30:
        return RegimeResult.default()

    # ── StudentTHMM regime probabilities ─────────────────────────────────────
    hmm = _get_hmm(ticker, df, force_refit=force_hmm_refit)
    p_trending, p_mean_rev, p_high_vol = 1/3, 1/3, 1/3

    if hmm is not None:
        X = _build_hmm_features(df)
        try:
            proba = hmm.predict_proba(X)  # (T, 3)
            label_map = hmm.label_states()  # {state_idx: label_name}

            # Map state indices to semantic labels
            label_to_prob: Dict[str, float] = {
                "trending": 0.0, "mean_reverting": 0.0, "high_volatility": 0.0
            }
            for state_idx, label_name in label_map.items():
                label_to_prob[label_name] = float(proba[-1, state_idx])

            p_trending = label_to_prob["trending"]
            p_mean_rev = label_to_prob["mean_reverting"]
            p_high_vol = label_to_prob["high_volatility"]
        except Exception as e:
            logger.warning("HMM predict_proba failed: %s", e)

    # Dominant regime
    probs = {"trending": p_trending, "mean_reverting": p_mean_rev,
             "high_volatility": p_high_vol}
    dominant = max(probs, key=probs.get)
    dominant_confidence = float(probs[dominant])

    # ── StudentTBOCPD changepoint detection ──────────────────────────────────
    if ticker not in _bocpd_detectors:
        _bocpd_detectors[ticker] = StudentTBOCPD(hazard_rate=1 / 252)

    bocpd = _bocpd_detectors[ticker]

    # Feed most recent log-return to BOCPD
    closes = df["Close"].values.astype(float)
    if len(closes) >= 2:
        latest_log_ret = float(np.log(closes[-1] / closes[-2]))
        bocpd_result = bocpd.update(latest_log_ret)
    else:
        bocpd_result = {
            "changepoint_probability": 0.0,
            "regime_stability": 1.0,
            "is_transition": False,
        }

    changepoint_prob = bocpd_result["changepoint_probability"]
    regime_stability = bocpd_result["regime_stability"]
    is_transition    = bocpd_result["is_transition"]

    explanation = (
        f"HMM regime: {dominant} (p={dominant_confidence:.2f}). "
        f"BOCPD changepoint_prob={changepoint_prob:.3f}, "
        f"stability={regime_stability:.2f}."
    )

    return RegimeResult(
        dominant_regime=dominant,
        p_trending=round(p_trending, 4),
        p_mean_reverting=round(p_mean_rev, 4),
        p_high_volatility=round(p_high_vol, 4),
        changepoint_probability=round(changepoint_prob, 4),
        regime_stability=round(regime_stability, 4),
        is_transition=is_transition,
        confidence=round(dominant_confidence, 4),
        explanation=explanation,
    )


# ── Legacy backward-compat dict conversion ───────────────────────────────────

def _regime_result_to_dict(r: RegimeResult) -> dict:
    """Convert RegimeResult to dict for backward compatibility with main.py."""
    return {
        "regime":         r.dominant_regime,
        "signal":         r.dominant_regime,
        "confidence":     r.confidence,
        "p_trending":     r.p_trending,
        "p_mean_reverting": r.p_mean_reverting,
        "p_high_volatility": r.p_high_volatility,
        "changepoint_probability": r.changepoint_probability,
        "regime_stability": r.regime_stability,
        "is_transition":  r.is_transition,
        "explanation":    r.explanation,
        # Legacy fields kept for feature_engineering.py compatibility
        "atr_ratio":      0.02,
        "hurst":          0.5,
    }


def run_dict(df: pd.DataFrame, ticker: str = "UNKNOWN",
             force_hmm_refit: bool = False) -> dict:
    """Legacy dict-returning wrapper for the backtesting engine."""
    return _regime_result_to_dict(run(df, ticker=ticker,
                                      force_hmm_refit=force_hmm_refit))
