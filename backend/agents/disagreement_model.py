"""
QuantAgent v3.0 — Deep Ensemble Disagreement Model
=====================================================
Correct epistemic/aleatoric uncertainty decomposition following:
  - Nix & Weigend (1994): learn both mean and variance
  - Lakshminarayanan et al. (2017): deep ensemble uncertainty

Each of 5 ensemble members outputs (mean, log_variance).
- epistemic_uncertainty = var(means across ensemble) → reducible
- aleatoric_uncertainty = mean(predicted variances across ensemble) → irreducible
- total_uncertainty     = epistemic + aleatoric (law of total variance)

Members are trained on bootstrap samples with XGBoost for scalability.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    logger.warning("xgboost not installed — DisagreementModel degraded")

from shared_types import DisagreementResult

STATE_DIM = 45
N_ENSEMBLE = 5


class DeepEnsembleUncertainty:
    """
    5-member XGBoost ensemble for forward-return prediction.
    Decomposes uncertainty into epistemic (model disagreement) and
    aleatoric (inherent market noise).
    """

    def __init__(self, n_ensemble: int = N_ENSEMBLE,
                 n_features: int = STATE_DIM,
                 random_state: int = 0):
        self.n_ensemble   = n_ensemble
        self.n_features   = n_features
        self.random_state = random_state
        self.models:     List       = []   # mean predictors
        self.var_models: List       = []   # log-variance predictors
        self._fitted = False
        # Running stats for normalization (updated online with new predictions)
        self._ep_mean  = 1e-6
        self._al_mean  = 1e-6

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsembleUncertainty":
        """
        X: (N, 45) features, y: (N,) forward returns.
        Trains 5 bootstrap-sampled ensemble members.
        Each member also trains a log-variance predictor on its residuals.
        """
        if not _XGB_OK:
            logger.warning("XGBoost not available — ensemble not fitted")
            self._fitted = False
            return self

        assert X.shape[1] == self.n_features, \
            f"Expected {self.n_features} features, got {X.shape[1]}"
        assert len(X) == len(y), "X and y must have same length"

        self.models.clear()
        self.var_models.clear()

        for seed in range(self.n_ensemble):
            rng = np.random.RandomState(self.random_state + seed)

            # Bootstrap sample (different subset per member)
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_b, y_b = X[idx], y[idx]

            # Step 1: fit mean predictor
            mean_model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state + seed,
                objective="reg:squarederror",
                verbosity=0,
            )
            mean_model.fit(X_b, y_b)
            self.models.append(mean_model)

            # Step 2: fit log-variance predictor on residuals
            # This gives the aleatoric (irreducible) uncertainty per point
            residuals_sq = (y_b - mean_model.predict(X_b)) ** 2
            log_var_target = np.log(residuals_sq + 1e-8)

            var_model = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state + seed + 100,
                objective="reg:squarederror",
                verbosity=0,
            )
            var_model.fit(X_b, log_var_target)
            self.var_models.append(var_model)

            logger.debug("Ensemble member %d fitted on %d bootstrap samples",
                         seed, len(X_b))

        self._fitted = True
        logger.info("DeepEnsemble fitted: %d members on %d samples",
                    self.n_ensemble, len(X))
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> DisagreementResult:
        """
        X: (1, 45) or (N, 45). Returns DisagreementResult.

        Uncertainty decomposition:
          epistemic = var(predicted_means) across 5 members
          aleatoric = mean(exp(predicted_log_var)) across 5 members
          total     = epistemic + aleatoric
        """
        if not self._fitted or not self.models:
            return DisagreementResult(
                epistemic_uncertainty=0.5,
                aleatoric_uncertainty=0.5,
                total_uncertainty=0.5,
                agent_consensus=0.0,
                dominant_signal="CONFLICTED",
            )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect predictions from each ensemble member
        means_arr    = np.array([m.predict(X) for m in self.models])     # (5, N)
        log_vars_arr = np.array([v.predict(X) for v in self.var_models]) # (5, N)
        variances    = np.exp(log_vars_arr)

        # Law of total variance decomposition
        epistemic = means_arr.var(axis=0)     # variance of means        (N,)
        aleatoric = variances.mean(axis=0)    # mean of predicted vars    (N,)
        total     = epistemic + aleatoric

        # Normalize relative to running mean (prevents explosion on first call)
        ep_mean   = max(float(epistemic.mean()), 1e-10)
        al_mean   = max(float(aleatoric.mean()), 1e-10)
        # Update running stats with EMA
        self._ep_mean = 0.9 * self._ep_mean + 0.1 * ep_mean
        self._al_mean = 0.9 * self._al_mean + 0.1 * al_mean

        def _norm(arr: np.ndarray, running_mean: float) -> float:
            raw = float(arr.mean()) / (running_mean + 1e-12)
            return float(np.clip(raw / 3.0, 0.0, 1.0))  # /3 so ≤3x mean → [0,1]

        ep_norm = _norm(epistemic, self._ep_mean)
        al_norm = _norm(aleatoric, self._al_mean)
        tot_norm = float(np.clip(ep_norm + al_norm, 0.0, 1.0))

        # Consensus direction: mean of predicted means across members and samples
        consensus = float(np.clip(means_arr.mean(), -1.0, 1.0))

        # Votes per member
        member_votes = {
            f"member_{i}": float(means_arr[i].mean())
            for i in range(self.n_ensemble)
        }

        # Dominant signal
        if consensus > 0.3:
            signal = "BUY"
        elif consensus < -0.3:
            signal = "SELL"
        elif abs(consensus) > 0.1:
            signal = "NEUTRAL"
        else:
            signal = "CONFLICTED"

        return DisagreementResult(
            epistemic_uncertainty=ep_norm,
            aleatoric_uncertainty=al_norm,
            total_uncertainty=tot_norm,
            agent_consensus=consensus,
            agent_votes=member_votes,
            dominant_signal=signal,
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DeepEnsembleUncertainty":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Legacy run() interface ────────────────────────────────────────────────────

_ensemble: Optional[DeepEnsembleUncertainty] = None


def _get_ensemble() -> Optional[DeepEnsembleUncertainty]:
    global _ensemble
    if _ensemble is not None:
        return _ensemble
    from config import MODEL_SAVE_DIR
    default_path = os.path.join(MODEL_SAVE_DIR, "ensemble_default.pkl")
    if os.path.exists(default_path):
        try:
            _ensemble = DeepEnsembleUncertainty.load(default_path)
            return _ensemble
        except Exception as e:
            logger.warning("Failed to load ensemble model: %s", e)
    return None


def _simple_disagreement(ind_r, pat_r, tre_r, reg_r) -> DisagreementResult:
    """Lightweight disagreement fallback when ensemble is not trained."""
    sig_map = {
        "bullish": 1.0, "uptrend": 1.0, "trending": 0.5,
        "bearish": -1.0, "downtrend": -1.0, "mean_reverting": -0.3,
        "neutral": 0.0, "sideways": 0.0, "high_volatility": -0.5
    }

    def _sig(r, key="signal"):
        if r is None: return 0.0
        return sig_map.get(str(r.get(key, "neutral")).lower(), 0.0)

    signals = [
        _sig(ind_r, "signal"),
        _sig(pat_r, "signal"),
        _sig(tre_r, "trend"),
        _sig(reg_r, "regime"),
    ]
    mean_sig = float(np.mean(signals))
    var_sig  = float(np.var(signals))

    # Simple proxy: variance → epistemic, no aleatoric estimate
    ep_unc = float(np.clip(var_sig * 4.0, 0, 1))
    al_unc = 0.3  # constant prior
    tot    = float(np.clip(ep_unc + al_unc, 0, 1))

    if mean_sig > 0.2:    signal = "BUY"
    elif mean_sig < -0.2: signal = "SELL"
    elif var_sig > 0.3:   signal = "CONFLICTED"
    else:                  signal = "NEUTRAL"

    return DisagreementResult(
        epistemic_uncertainty=ep_unc,
        aleatoric_uncertainty=al_unc,
        total_uncertainty=tot,
        agent_consensus=mean_sig,
        dominant_signal=signal,
    )


def run(ind_r, pat_r, tre_r, reg_r,
        feature_vector: Optional[np.ndarray] = None) -> Dict:
    """
    Legacy run() interface.
    Uses DeepEnsemble when available, falls back to simple disagreement.
    Returns dict for backward-compat with older main.py callers.
    """
    ensemble = _get_ensemble()
    if ensemble is not None and feature_vector is not None:
        result = ensemble.predict(feature_vector.reshape(1, -1))
    else:
        result = _simple_disagreement(ind_r, pat_r, tre_r, reg_r)

    # Threshold for trade recommendation
    ep   = result.epistemic_uncertainty
    tot  = result.total_uncertainty
    recommendation = "NO_TRADE" if ep > 0.6 or tot > 0.8 else "PROCEED"

    return {
        "epistemic_uncertainty":   round(ep, 4),
        "aleatoric_uncertainty":   round(result.aleatoric_uncertainty, 4),
        "total_uncertainty":       round(result.total_uncertainty, 4),
        "disagreement_score":      round(result.total_uncertainty, 4),  # legacy
        "agent_consensus":         round(result.agent_consensus, 4),
        "dominant_signal":         result.dominant_signal,
        "recommendation":          recommendation,
        "agent_votes":             result.agent_votes,
    }
