"""
QuantAgent v3.0 — Feature Pipeline
=====================================
45-dimensional state vector with strict point-in-time correctness.

ALL rolling computations use min_periods=window (NEVER partial windows).
First FEATURE_BURNIN_BARS rows return NaN — callers must drop them.

Feature index map (45 dims):
  [0-9]    Technical indicators (10)
  [10-14]  Microstructure (5)
  [15-20]  Macro overlay (6)
  [21-28]  Sentiment + alternative data (8)
  [29-31]  HMM regime probabilities (3)
  [32-33]  Disagreement scores (2)
  [34-38]  Position context (5)
  [39-40]  BOCPD changepoint signals (2)
  [41-44]  Macro regime context (4)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import FEATURE_BURNIN_BARS

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_DIM = 45
ZSCORE_WINDOW = 60        # rolling window for z-score normalization
ZSCORE_CLIP   = 3.0       # clip z-scores to [-3, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# STRICT ROLLING Z-SCORE (Section 2.1 compliance)
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_zscore_with_burnin(series: pd.Series, window: int = ZSCORE_WINDOW,
                                clip_val: float = ZSCORE_CLIP) -> pd.Series:
    """
    Point-in-time-correct rolling z-score.
    min_periods=window: NaN for first (window-1) bars. This is CORRECT.
    Downstream callers must handle NaN and use FEATURE_BURNIN_BARS.
    """
    mean = series.rolling(window, min_periods=window).mean()
    std  = series.rolling(window, min_periods=window).std()
    z    = (series - mean) / (std + 1e-8)
    return z.clip(-clip_val, clip_val)


def rolling_zscore_fill(series: pd.Series, window: int = ZSCORE_WINDOW,
                         clip_val: float = ZSCORE_CLIP) -> pd.Series:
    """Same as rolling_zscore_with_burnin but fills initial NaN with 0.0."""
    return rolling_zscore_with_burnin(series, window, clip_val).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY (staleness tracking)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureRegistry:
    """Tracks metadata and staleness for each feature dimension."""
    feature_names: List[str]
    feature_groups: Dict[str, List[int]]   # group_name → list of indices
    first_valid_idx: int = FEATURE_BURNIN_BARS
    last_update_ts: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "state_dim": len(self.feature_names),
            "first_valid_idx": self.first_valid_idx,
            "groups": {k: len(v) for k, v in self.feature_groups.items()},
            "last_update_ts": self.last_update_ts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE MONITOR (Section 2.3)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureImportanceMonitor:
    """
    Tracks permutation importance after each HMM refit.
    Warns if any feature has negative importance for 3 consecutive refits.
    """

    def __init__(self, feature_names: List[str],
                 log_path: str = "./models/feature_importance.csv"):
        self.feature_names = feature_names
        self.log_path = log_path
        self._consecutive_negative: Dict[str, int] = {}
        self._refit_count = 0

    def update(self, X: np.ndarray, hmm_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute permutation importance on held-out window.
        X: (N, 45) features, hmm_labels: (N,) HMM state labels.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance

        if len(X) < 30:
            return {}

        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, hmm_labels)
            perm = permutation_importance(clf, X, hmm_labels,
                                          n_repeats=5, random_state=42)
            importances = perm.importances_mean
        except Exception as e:
            logger.debug("Feature importance computation failed: %s", e)
            return {}

        result = {name: float(imp) for name, imp in
                  zip(self.feature_names, importances)}

        # Track consecutive negative importance
        for name, imp in result.items():
            if imp < 0:
                self._consecutive_negative[name] = \
                    self._consecutive_negative.get(name, 0) + 1
                if self._consecutive_negative[name] >= 3:
                    logger.warning(
                        "Feature '%s' has had negative permutation importance "
                        "for %d consecutive HMM refits — check data quality.",
                        name, self._consecutive_negative[name]
                    )
            else:
                self._consecutive_negative[name] = 0

        # Log to CSV
        try:
            import csv
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if self._refit_count == 0:
                    writer.writerow(["refit"] + self.feature_names)
                writer.writerow([self._refit_count] +
                                [round(result.get(n, 0), 6) for n in self.feature_names])
        except Exception:
            pass

        self._refit_count += 1
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class FeaturePipeline:
    """
    Computes the 45-dim state vector for every bar in a DataFrame.
    Returns a DataFrame of shape (T, 45) with NaN for the first
    FEATURE_BURNIN_BARS rows.

    All rolling operations use strict min_periods=window (no partial windows).
    """

    FEATURE_NAMES = [
        # Technical [0-9]
        "rsi_zscore",          # 0
        "macd_hist_zscore",    # 1
        "stoch_k_zscore",      # 2
        "roc_zscore",          # 3
        "williams_r_zscore",   # 4
        "bb_pct_b",            # 5 — raw [0,1] (no z-score needed)
        "bb_bandwidth_zscore", # 6
        "atr_close_ratio_z",   # 7
        "vpt_zscore",          # 8
        "obv_zscore",          # 9
        # Microstructure [10-14]
        "amihud_illiquidity_z",# 10
        "volume_surprise_z",   # 11
        "hl_spread_z",         # 12
        "return_autocorr_5",   # 13  raw [-1,1]
        "realized_var_ratio",  # 14  raw (detrended)
        # Macro overlay [15-20]
        "vix_zscore",          # 15
        "vix_ts_spread_z",     # 16
        "hyg_lqd_zscore",      # 17
        "dxy_momentum_z",      # 18
        "t10y2y_spread",       # 19  raw (rates)
        "credit_regime_flag",  # 20  binary
        # Sentiment [21-28]
        "ticker_sentiment_decay", # 21  decay-weighted
        "ticker_sent_magnitude",  # 22
        "macro_sentiment_z",      # 23
        "news_volume_z",          # 24
        "reddit_sentiment_z",     # 25
        "reddit_mention_z",       # 26
        "sec_earnings_flag",      # 27  binary
        "sec_8k_flag",            # 28  binary
        # HMM regime probs [29-31]
        "hmm_p_trending",      # 29
        "hmm_p_mean_reverting",# 30
        "hmm_p_high_vol",      # 31
        # Disagreement [32-33]
        "epistemic_uncertainty",  # 32
        "aleatoric_uncertainty",  # 33
        # Position context [34-38]
        "current_position",    # 34  [-1,1]
        "portfolio_cash_pct",  # 35  [0,1]
        "drawdown",            # 36  [0,1]
        "rolling_ret_5d_z",    # 37
        "rolling_ret_20d_z",   # 38
        # BOCPD [39-40]
        "bocpd_cp_prob",       # 39  [0,1]
        "bocpd_stability",     # 40  [0,1]
        # Macro regime context [41-44]
        "short_ratio_z",       # 41
        "short_pct_float_z",   # 42
        "vol_regime_flag",     # 43  0/1 — high vol when vix_z>1
        "trend_strength_z",    # 44
    ]

    def __init__(self):
        self.registry = FeatureRegistry(
            feature_names=self.FEATURE_NAMES,
            feature_groups={
                "technical":      list(range(0, 10)),
                "microstructure": list(range(10, 15)),
                "macro":          list(range(15, 21)),
                "sentiment":      list(range(21, 29)),
                "hmm_regime":     list(range(29, 32)),
                "disagreement":   list(range(32, 34)),
                "position":       list(range(34, 39)),
                "bocpd":          list(range(39, 41)),
                "macro_context":  list(range(41, 45)),
            },
            first_valid_idx=FEATURE_BURNIN_BARS,
        )
        self._importance_monitor = FeatureImportanceMonitor(self.FEATURE_NAMES)
        assert len(self.FEATURE_NAMES) == STATE_DIM, \
            f"Expected {STATE_DIM} features, got {len(self.FEATURE_NAMES)}"

    # ── Technical features ────────────────────────────────────────────────────

    def _compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features [0-9]: RSI, MACD, Stoch, ROC, Williams, BB, ATR, VPT, OBV."""
        feat = pd.DataFrame(index=df.index)
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs        = avg_gain / (avg_loss + 1e-8)
        rsi       = 100.0 - (100.0 / (1.0 + rs))
        feat["rsi_zscore"] = rolling_zscore_fill(rsi)

        # MACD histogram (12,26,9)
        ema12 = close.ewm(span=12, min_periods=12).mean()
        ema26 = close.ewm(span=26, min_periods=26).mean()
        macd  = ema12 - ema26
        macd_sig = macd.ewm(span=9, min_periods=9).mean()
        macd_hist = macd - macd_sig
        feat["macd_hist_zscore"] = rolling_zscore_fill(macd_hist)

        # Stochastic %K (14, 3)
        lo14 = low.rolling(14, min_periods=14).min()
        hi14 = high.rolling(14, min_periods=14).max()
        stoch_k = 100.0 * (close - lo14) / (hi14 - lo14 + 1e-8)
        feat["stoch_k_zscore"] = rolling_zscore_fill(stoch_k)

        # ROC (10)
        roc = close.pct_change(10)
        feat["roc_zscore"] = rolling_zscore_fill(roc)

        # Williams %R (14)
        highest_high_14 = high.rolling(14, min_periods=14).max()
        lowest_low_14   = low.rolling(14, min_periods=14).min()
        williams_r = -100.0 * (highest_high_14 - close) / \
                     (highest_high_14 - lowest_low_14 + 1e-8)
        feat["williams_r_zscore"] = rolling_zscore_fill(williams_r)

        # Bollinger Bands (20, 2σ)
        bb_mid  = close.rolling(20, min_periods=20).mean()
        bb_std  = close.rolling(20, min_periods=20).std()
        bb_up   = bb_mid + 2.0 * bb_std
        bb_lo   = bb_mid - 2.0 * bb_std
        bb_pct_b = (close - bb_lo) / (bb_up - bb_lo + 1e-8)
        bb_bw   = (bb_up - bb_lo) / (bb_mid + 1e-8)
        feat["bb_pct_b"]           = bb_pct_b.fillna(0.5).clip(0, 1)
        feat["bb_bandwidth_zscore"] = rolling_zscore_fill(bb_bw)

        # ATR/Close ratio
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, min_periods=14).mean()
        atr_ratio = atr / (close + 1e-8)
        feat["atr_close_ratio_z"] = rolling_zscore_fill(atr_ratio)

        # Volume Price Trend (VPT)
        vpt = (vol * close.pct_change()).cumsum()
        feat["vpt_zscore"] = rolling_zscore_fill(vpt)

        # OBV z-score
        obv = (vol * np.sign(close.diff())).fillna(0).cumsum()
        feat["obv_zscore"] = rolling_zscore_fill(obv)

        return feat

    # ── Microstructure features ───────────────────────────────────────────────

    def _compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features [10-14]: Amihud, volume surprise, HL spread, autocorr, var ratio."""
        feat = pd.DataFrame(index=df.index)
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # Amihud illiquidity = |ret| / dollar_volume (×10^6 for scale)
        ret_abs     = close.pct_change().abs()
        dollar_vol  = close * vol + 1e-8
        amihud      = (ret_abs / dollar_vol) * 1e6
        feat["amihud_illiquidity_z"] = rolling_zscore_fill(amihud)

        # Volume surprise z-score
        feat["volume_surprise_z"] = rolling_zscore_fill(vol)

        # HL spread (%) = (High - Low) / Close
        hl_spread = (high - low) / (close + 1e-8)
        feat["hl_spread_z"] = rolling_zscore_fill(hl_spread)

        # Return autocorrelation (lag-5, rolling 20-day)
        rets = close.pct_change()
        autocorr_5 = rets.rolling(20, min_periods=20).apply(
            lambda x: float(pd.Series(x).autocorr(lag=5)) if len(x) > 6 else 0.0,
            raw=False,
        ).fillna(0.0)
        feat["return_autocorr_5"] = autocorr_5.clip(-1, 1)

        # Realized variance ratio (short/long window) — proxy for vol clustering
        rv_5  = rets.rolling(5, min_periods=5).var()
        rv_20 = rets.rolling(20, min_periods=20).var()
        var_ratio = rv_5 / (rv_20 + 1e-12)
        feat["realized_var_ratio"] = rolling_zscore_fill(var_ratio)

        return feat

    # ── Macro overlay features ────────────────────────────────────────────────

    def _get_macro_features(self, macro_ctx: Optional[Dict]) -> Dict[str, float]:
        """Extract macro scalars from the fetched macro context dict."""
        if macro_ctx is None:
            return {k: 0.0 for k in
                    ["vix_zscore", "vix_ts_spread_z", "hyg_lqd_zscore",
                     "dxy_momentum_z", "t10y2y_spread", "credit_regime_flag"]}
        return {
            "vix_zscore":         float(macro_ctx.get("vix_zscore", 0.0)),
            "vix_ts_spread_z":    float(macro_ctx.get("vix_ts_spread", -2.0) / 10.0),
            "hyg_lqd_zscore":     float(macro_ctx.get("hyg_lqd_zscore", 0.0)),
            "dxy_momentum_z":     float(macro_ctx.get("dxy_20d_momentum", 0.0)) * 20.0,
            "t10y2y_spread":      float(macro_ctx.get("t10y2y_spread", 0.0)),
            "credit_regime_flag": float(macro_ctx.get("credit_regime_flag", 0)),
        }

    # ── Sentiment features (decay-weighted, Section 2.4) ──────────────────────

    def _get_sentiment_features(
        self,
        news_res,
        reddit_res,
        sec_flags,
    ) -> Dict[str, float]:
        """
        Sentiment features [21-28].
        Applies decay_score from DataFetcher — already computed in news_res.ticker_sentiment_score.
        """
        from shared_types import NewsSentimentResult, RedditSentimentResult, SECFlags

        # News sentiment (already decay-weighted by DataFetcher)
        ticker_sent  = float(getattr(news_res, "ticker_sentiment_score", 0.0))
        ticker_mag   = float(getattr(news_res, "ticker_sentiment_magnitude", 0.0))
        macro_sent   = float(getattr(news_res, "macro_sentiment_score", 0.0))
        news_vol_z   = float(getattr(news_res, "news_volume_zscore", 0.0))

        # Reddit sentiment
        reddit_sent  = float(getattr(reddit_res, "reddit_sentiment_score", 0.0))
        reddit_men_z = float(getattr(reddit_res, "reddit_mention_zscore", 0.0))

        # SEC flags
        earnings_flag = float(getattr(sec_flags, "earnings_within_5_days", False))
        k8_flag       = float(bool(getattr(sec_flags, "recent_8k", False)))

        return {
            "ticker_sentiment_decay": np.clip(ticker_sent, -1, 1),
            "ticker_sent_magnitude":  np.clip(ticker_mag,  0, 1),
            "macro_sentiment_z":      np.clip(macro_sent, -1, 1),
            "news_volume_z":          np.clip(news_vol_z, -3, 3),
            "reddit_sentiment_z":     np.clip(reddit_sent, -1, 1),
            "reddit_mention_z":       np.clip(reddit_men_z, -3, 3),
            "sec_earnings_flag":      earnings_flag,
            "sec_8k_flag":            k8_flag,
        }

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        macro_ctx:    Optional[Dict] = None,
        news_result   = None,
        reddit_result = None,
        sec_flags     = None,
        regime_result = None,
        disagreement  = None,
        position_state: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Compute the full 45-dim feature DataFrame for all bars in df.

        Parameters
        ----------
        df              : OHLCV DataFrame (T bars)
        ticker          : For HMM regime lookup
        macro_ctx       : Dict from DataFetcher.fetch_macro_context()
        news_result     : NewsSentimentResult (or None for neutral)
        reddit_result   : RedditSentimentResult (or None for neutral)
        sec_flags       : SECFlags (or None for defaults)
        regime_result   : RegimeResult (or None — will compute internally)
        disagreement    : DisagreementResult (or None — use defaults)
        position_state  : dict with current_position, drawdown, cash_pct

        Returns
        -------
        pd.DataFrame of shape (T, 45), NaN for first FEATURE_BURNIN_BARS rows.
        """
        from shared_types import (NewsSentimentResult, RedditSentimentResult,
                                   SECFlags, RegimeResult, DisagreementResult)

        # ── Technical features ─────────────────────────────────────────────
        tech_feat = self._compute_technical_features(df)
        micro_feat = self._compute_microstructure_features(df)

        # ── HMM regime probabilities (per-bar) ────────────────────────────
        from agents.market_regime_agent import _build_hmm_features, _get_hmm
        hmm = _get_hmm(ticker, df)
        if hmm is not None:
            try:
                X_hmm = _build_hmm_features(df)
                proba = hmm.predict_proba(X_hmm)  # (T, 3)
                label_map = hmm.label_states()
                p_trend_arr   = np.zeros(len(df))
                p_mr_arr      = np.zeros(len(df))
                p_hv_arr      = np.zeros(len(df))
                for state_idx, label_name in label_map.items():
                    if label_name == "trending":
                        p_trend_arr = proba[:, state_idx]
                    elif label_name == "mean_reverting":
                        p_mr_arr = proba[:, state_idx]
                    elif label_name == "high_volatility":
                        p_hv_arr = proba[:, state_idx]
            except Exception as e:
                logger.warning("HMM predict failed in feature pipeline: %s", e)
                p_trend_arr = np.full(len(df), 1/3)
                p_mr_arr    = np.full(len(df), 1/3)
                p_hv_arr    = np.full(len(df), 1/3)
        else:
            p_trend_arr = np.full(len(df), 1/3)
            p_mr_arr    = np.full(len(df), 1/3)
            p_hv_arr    = np.full(len(df), 1/3)

        # ── Macro context (same value broadcast across all bars) ───────────
        macro_feats = self._get_macro_features(macro_ctx)

        # ── Sentiment (same value broadcast) ──────────────────────────────
        if news_result is None:
            news_result = NewsSentimentResult.neutral()
        if reddit_result is None:
            reddit_result = RedditSentimentResult.neutral()
        if sec_flags is None:
            sec_flags = SECFlags.default(ticker)

        sent_feats = self._get_sentiment_features(news_result, reddit_result, sec_flags)

        # ── Disagreement scores ────────────────────────────────────────────
        if disagreement is None:
            ep_unc = 0.5
            al_unc = 0.5
        else:
            ep_unc = float(getattr(disagreement, "epistemic_uncertainty", 0.5))
            al_unc = float(getattr(disagreement, "aleatoric_uncertainty", 0.5))

        # ── Position context ───────────────────────────────────────────────
        if position_state is None:
            position_state = {}
        pos      = float(position_state.get("current_position", 0.0))
        cash_pct = float(position_state.get("portfolio_cash_pct", 1.0))
        drawdown = float(position_state.get("drawdown", 0.0))

        # Rolling return features
        rets      = df["Close"].pct_change()
        ret_5d_z  = rolling_zscore_fill(df["Close"].pct_change(5))
        ret_20d_z = rolling_zscore_fill(df["Close"].pct_change(20))

        # Short interest (broadcast)
        # Fetched separately if available — default 0 here
        short_ratio_z    = 0.0
        short_pct_z      = 0.0
        vol_regime_flag  = 1.0 if macro_feats.get("vix_zscore", 0) > 1.0 else 0.0

        # Trend strength (from polynomial fit quality — use rolling z-score of R²)
        # Simplified: use |ROC20| z-score as proxy for trend strength
        roc_20_abs = df["Close"].pct_change(20).abs()
        trend_strength_z_series = rolling_zscore_fill(roc_20_abs)

        # BOCPD (computed fresh for each bar — only latest value returned here;
        #         for training, the backtesting engine streams updates bar-by-bar)
        from agents.market_regime_agent import _bocpd_detectors
        bocpd_inst = _bocpd_detectors.get(ticker)
        if bocpd_inst is not None:
            bocpd_cp_prob  = float(bocpd_inst.run_length_probs[0]) \
                             if len(bocpd_inst.run_length_probs) > 0 else 0.0
            bocpd_stability = float(
                np.dot(bocpd_inst.run_length_probs,
                       np.arange(len(bocpd_inst.run_length_probs)))
            ) / max(bocpd_inst._n_processed, 1)
            bocpd_stability = min(bocpd_stability, 1.0)
        else:
            bocpd_cp_prob  = 0.0
            bocpd_stability = 1.0

        # ── Assemble full feature DataFrame ───────────────────────────────
        T = len(df)

        features = pd.DataFrame(index=df.index)

        # Technical [0-9]
        for col in tech_feat.columns:
            features[col] = tech_feat[col].values

        # Microstructure [10-14]
        for col in micro_feat.columns:
            features[col] = micro_feat[col].values

        # Macro [15-20]
        for name, val in macro_feats.items():
            features[name] = val  # broadcast constant

        # Sentiment [21-28]
        for name, val in sent_feats.items():
            features[name] = val  # broadcast constant

        # HMM regime probs [29-31]
        features["hmm_p_trending"]       = p_trend_arr
        features["hmm_p_mean_reverting"] = p_mr_arr
        features["hmm_p_high_vol"]       = p_hv_arr

        # Disagreement [32-33]
        features["epistemic_uncertainty"] = ep_unc
        features["aleatoric_uncertainty"] = al_unc

        # Position context [34-38]
        features["current_position"]  = pos
        features["portfolio_cash_pct"] = cash_pct
        features["drawdown"]           = drawdown
        features["rolling_ret_5d_z"]   = ret_5d_z.values
        features["rolling_ret_20d_z"]  = ret_20d_z.values

        # BOCPD [39-40]
        features["bocpd_cp_prob"]  = bocpd_cp_prob
        features["bocpd_stability"] = bocpd_stability

        # Macro context [41-44]
        features["short_ratio_z"]     = short_ratio_z
        features["short_pct_float_z"] = short_pct_z
        features["vol_regime_flag"]   = vol_regime_flag
        features["trend_strength_z"]  = trend_strength_z_series.values

        # ── Burn-in enforcement: NaN first FEATURE_BURNIN_BARS rows ───────
        burnin_cols = [
            "rsi_zscore", "macd_hist_zscore", "stoch_k_zscore",
            "roc_zscore", "williams_r_zscore", "bb_bandwidth_zscore",
            "atr_close_ratio_z", "vpt_zscore", "obv_zscore",
            "amihud_illiquidity_z", "volume_surprise_z", "hl_spread_z",
            "realized_var_ratio", "rolling_ret_5d_z", "rolling_ret_20d_z",
            "trend_strength_z",
        ]
        if T > FEATURE_BURNIN_BARS:
            burnin_idx = features.index[:FEATURE_BURNIN_BARS]
            for col in burnin_cols:
                if col in features.columns:
                    features.loc[burnin_idx, col] = np.nan
        else:
            for col in burnin_cols:
                if col in features.columns:
                    features[col] = np.nan

        # ── Reorder to canonical order ─────────────────────────────────────
        features = features.reindex(columns=self.FEATURE_NAMES, fill_value=0.0)

        assert features.shape[1] == STATE_DIM, \
            f"Feature dim mismatch: {features.shape[1]} != {STATE_DIM}"

        import datetime
        self.registry.last_update_ts = datetime.datetime.utcnow().isoformat()
        return features

    def get_latest_vector(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        **kwargs,
    ) -> np.ndarray:
        """
        Compute features and return ONLY the last valid row as a numpy array.
        Used for live inference. Drops NaN rows.
        """
        feat_df = self.compute(df, ticker=ticker, **kwargs)
        valid = feat_df.dropna()
        if valid.empty:
            logger.warning("No valid feature rows for %s — returning zeros", ticker)
            return np.zeros(STATE_DIM, dtype=np.float32)
        return valid.iloc[-1].values.astype(np.float32)


# ── Module-level singleton ────────────────────────────────────────────────────

_pipeline: Optional[FeaturePipeline] = None

def get_pipeline() -> FeaturePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FeaturePipeline()
    return _pipeline


# ── Backward-compat shim ──────────────────────────────────────────────────────

def build_state_vector(
    indicator_result    = None,
    pattern_result      = None,
    trend_result        = None,
    regime_result       = None,
    disagreement_score:  float = 0.0,
    drawdown:            float = 0.0,
    portfolio_cash_pct:  float = 1.0,
    price_series = None,
) -> np.ndarray:
    """Legacy shim — used by existing main.py and tests."""
    from shared_types import AgentSignal

    signal_map = {
        "bullish": 1.0, "uptrend": 1.0, "trending": 0.5,
        "bearish": -1.0, "downtrend": -1.0, "mean_reverting": -0.3,
        "neutral": 0.0, "sideways": 0.0, "high_volatility": -0.5
    }

    def get_signal(r, key):
        if r is None: return 0.0
        return signal_map.get(str(r.get(key, "neutral")).lower(), 0.0)

    def get_conf(r):
        if r is None: return 0.0
        return float(r.get("confidence", 0.0))

    state = np.zeros(STATE_DIM, dtype=np.float32)

    # Map old 16-dim fields into the first 16 dims of the new 45-dim vector
    state[29] = float(regime_result.get("p_trending",       1/3) if regime_result else 1/3)
    state[30] = float(regime_result.get("p_mean_reverting", 1/3) if regime_result else 1/3)
    state[31] = float(regime_result.get("p_high_volatility",1/3) if regime_result else 1/3)
    state[32] = float(disagreement_score)
    state[35] = float(portfolio_cash_pct)
    state[36] = float(drawdown)
    state[39] = float(regime_result.get("changepoint_probability", 0.0) if regime_result else 0.0)
    state[40] = float(regime_result.get("regime_stability", 1.0) if regime_result else 1.0)

    if price_series is not None and len(price_series) >= 20:
        rets = price_series.pct_change()
        vol  = rets.rolling(20, min_periods=20).std().iloc[-1]
        r5   = price_series.pct_change(5).iloc[-1]
        r20  = price_series.pct_change(20).iloc[-1]
        state[37] = float(np.clip(r5  / (vol * 5.0  + 1e-6), -3, 3))
        state[38] = float(np.clip(r20 / (vol * 20.0 + 1e-6), -3, 3))

    return state
