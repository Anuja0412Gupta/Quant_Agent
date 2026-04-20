"""
QuantAgent v3.0 — Training Pipeline
======================================
Full RecurrentPPO training with 5-phase curriculum, research callbacks,
and Lagrangian constraint multiplier updates.

Phases:
  0. Expert warm-up (BC on trending regime only)
  1. Curriculum stage 1 (trending + MR bars only)
  2. Curriculum stage 2 (all bars, low entropy)
  3. Fine-tuning (full regime, higher entropy for exploration)
  4. Lagrangian tightening (reduce CVaR/DD limits progressively)

Callbacks:
  - LagrangianUpdateCallback: update λ multipliers after each episode
  - ResearchMetricsCallback: log IC, regime breakdown, feature importance
  - CyclicalEntropyCallback: schedule entropy coef per phase
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    FEATURE_BURNIN_BARS, MIN_EPISODE_LENGTH, MODEL_SAVE_DIR,
    RL_TRAIN_STEPS_DAILY, CURRICULUM_STAGE_1, CURRICULUM_STAGE_2,
    RL_LEARNING_RATE, BACKTEST_INITIAL_CAPITAL
)
from utils.helpers import ensure_numpy_pickle_compat, resolve_model_zip_path

logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_OK = True
except ImportError:
    _SB3_OK = False
    logger.warning("stable_baselines3 not available — training pipeline disabled")

try:
    from sb3_contrib import RecurrentPPO
    _CONTRIB_OK = True
except ImportError:
    _CONTRIB_OK = False
    logger.warning("sb3-contrib not available — falling back to PPO")


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

if _SB3_OK:

    class LagrangianUpdateCallback(BaseCallback):
        """
        Updates Lagrangian multipliers after each episode rollout.
        Reads episode_max_dd and episode_cvar from info dict.
        """
        def __init__(self, constraint_mgr, verbose: int = 0):
            super().__init__(verbose)
            self.constraint_mgr = constraint_mgr

        def _on_step(self) -> bool:
            # Check for done environments
            for info in self.locals.get("infos", []):
                if info.get("is_success") or info.get("episode"):
                    ep_info = info.get("episode", {})
                    max_dd  = float(info.get("max_drawdown", 0.0))
                    cvar    = float(info.get("running_cvar", 0.0))
                    self.constraint_mgr.update(max_dd, cvar)
            return True

    class CyclicalEntropyCallback(BaseCallback):
        """Adjusts entropy coefficient on a cyclical cosine schedule."""
        def __init__(self, model, phase_length: int = 100_000,
                     min_ent: float = 0.001, max_ent: float = 0.05,
                     verbose: int = 0):
            super().__init__(verbose)
            self._model_ref  = model
            self.phase_length = phase_length
            self.min_ent = min_ent
            self.max_ent = max_ent

        def _on_step(self) -> bool:
            from rl.rl_policy import cyclical_entropy_coef
            step = self.num_timesteps
            ent  = cyclical_entropy_coef(step, self.phase_length,
                                          self.min_ent, self.max_ent)
            if hasattr(self._model_ref, "ent_coef"):
                self._model_ref.ent_coef = ent
            return True

    class ResearchMetricsCallback(BaseCallback):
        """
        Logs research-grade metrics every N steps:
          - Rolling Sharpe on last 252-step return buffer
          - Regime breakdown of rewards
          - Lagrangian multiplier values
        """
        def __init__(self, constraint_mgr, log_freq: int = 10_000, verbose: int = 0):
            super().__init__(verbose)
            self.constraint_mgr = constraint_mgr
            self.log_freq = log_freq
            self._return_buf: List[float] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ret = info.get("log_return", 0.0)
                if ret:
                    self._return_buf.append(float(ret))
                    if len(self._return_buf) > 252:
                        self._return_buf = self._return_buf[-252:]

            if self.num_timesteps % self.log_freq == 0 and self._return_buf:
                arr = np.array(self._return_buf)
                roll_sharpe = float((arr.mean() / (arr.std() + 1e-8)) * np.sqrt(252))
                logger.info(
                    "[Step %d] rolling_sharpe=%.3f λ_dd=%.4f λ_cvar=%.4f",
                    self.num_timesteps, roll_sharpe,
                    self.constraint_mgr.lambda_dd,
                    self.constraint_mgr.lambda_cvar,
                )
                try:
                    self.logger.record("train/rolling_sharpe", roll_sharpe)
                    self.logger.record("risk/lambda_dd",   self.constraint_mgr.lambda_dd)
                    self.logger.record("risk/lambda_cvar", self.constraint_mgr.lambda_cvar)
                except Exception:
                    pass
            return True

    class CurriculumCallback(BaseCallback):
        """Advances curriculum stage based on total training steps."""
        def __init__(self, curriculum, verbose: int = 0):
            super().__init__(verbose)
            self.curriculum = curriculum

        def _on_step(self) -> bool:
            self.curriculum.update_stage(self.num_timesteps)
            return True

else:
    # Stubs when SB3 not available
    class LagrangianUpdateCallback:
        def __init__(self, *a, **kw): pass
    class CyclicalEntropyCallback:
        def __init__(self, *a, **kw): pass
    class ResearchMetricsCallback:
        def __init__(self, *a, **kw): pass
    class CurriculumCallback:
        def __init__(self, *a, **kw): pass


# ═══════════════════════════════════════════════════════════════════════════════
# RL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    ticker:           str   = "AAPL"
    timeframe:        str   = "1d"
    total_timesteps:  int   = RL_TRAIN_STEPS_DAILY
    n_envs:           int   = 4
    learning_rate:    float = RL_LEARNING_RATE
    batch_size:       int   = 64
    n_epochs:         int   = 10
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    max_grad_norm:    float = 0.5
    seed:             int   = 42
    tensorboard_log:  str   = "./logs/tensorboard/"
    model_save_path:  str   = ""

    def __post_init__(self):
        self.ticker = str(self.ticker).upper().strip()
        self.timeframe = str(self.timeframe).lower().strip()
        if not self.model_save_path:
            self.model_save_path = os.path.join(
                MODEL_SAVE_DIR, f"rl_{self.ticker}_{self.timeframe}"
            )


class RLTrainer:
    """
    Full 5-phase training pipeline:
      Phase 0: Data loading + feature computation
      Phase 1: Expert warm-up (trending only, low steps)
      Phase 2: Curriculum stage 1 (trending + MR)
      Phase 3: Full training (all regimes, cyclical entropy)
      Phase 4: Lagrangian tightening + final fine-tune
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model   = None
        self.env     = None
        self.constraint_mgr = None
        self.curriculum     = None
        self._trained = False

    def train(self) -> None:
        """Execute full 5-phase training. Blocks until complete."""
        if not _SB3_OK or not _CONTRIB_OK:
            raise ImportError("stable_baselines3 and sb3_contrib required for training")

        cfg = self.config
        os.makedirs(cfg.model_save_path, exist_ok=True)
        os.makedirs(cfg.tensorboard_log, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        # ── Phase 0: Data + features ──────────────────────────────────────
        logger.info("Phase 0: Loading data for %s %s", cfg.ticker, cfg.timeframe)

        from data.data_fetcher import get_fetcher
        from features.feature_pipeline import get_pipeline
        from rl.rl_policy import (LagrangianConstraintManager, CurriculumScheduler)

        fetcher = get_fetcher()
        df   = fetcher.fetch_ohlcv(cfg.ticker, cfg.timeframe)
        news = fetcher.fetch_news_sentiment(cfg.ticker)
        reddit = fetcher.fetch_reddit_sentiment(cfg.ticker)
        sec    = fetcher.fetch_sec_flags(cfg.ticker)
        macro  = fetcher.fetch_macro_context()

        assert len(df) >= FEATURE_BURNIN_BARS + MIN_EPISODE_LENGTH, (
            f"Not enough historical data for {cfg.ticker}: "
            f"{len(df)} bars < {FEATURE_BURNIN_BARS + MIN_EPISODE_LENGTH} required. "
            f"Set FEATURE_BURNIN_BARS in config.py or use a longer history."
        )

        pipeline = get_pipeline()
        features_df = pipeline.compute(
            df, ticker=cfg.ticker,
            macro_ctx=macro, news_result=news,
            reddit_result=reddit, sec_flags=sec,
        )

        # Drop burn-in rows
        valid_idx = features_df.dropna().index
        features_df = features_df.loc[valid_idx].fillna(0.0)
        df_valid    = df.loc[valid_idx]

        logger.info("Features computed: %d valid bars (%.0f%% of total)",
                    len(features_df),
                    100 * len(features_df) / len(df))

        # Compute regime labels for curriculum
        from agents.market_regime_agent import run as regime_run
        regime_labels = pd.Series(
            [regime_run(df_valid.iloc[:i + 1], cfg.ticker).dominant_regime
             for i in range(0, len(df_valid), 21)],   # every 21 bars for speed
            index=df_valid.index[::21],
        ).reindex(df_valid.index, method="ffill").fillna("trending")

        # Regime probs DataFrame for curriculum
        from agents.market_regime_agent import _get_hmm, _build_hmm_features
        hmm = _get_hmm(cfg.ticker, df_valid)
        if hmm is not None:
            X_hmm   = _build_hmm_features(df_valid)
            proba   = hmm.predict_proba(X_hmm)
            label_map = hmm.label_states()
            regime_probs = pd.DataFrame(index=df_valid.index,
                                         columns=["trending", "mean_reverting", "high_volatility"])
            for state_idx, label_name in label_map.items():
                regime_probs[label_name] = proba[:, state_idx]
            regime_probs = regime_probs.astype(float)
        else:
            regime_probs = None

        # ── Phase 1: Expert warm-up (trending only) ───────────────────────
        logger.info("Phase 1: Expert warm-up (trending bars only)")
        self.constraint_mgr = LagrangianConstraintManager()
        self.curriculum     = CurriculumScheduler(regime_probs)
        self.curriculum.stage = 0  # trending only

        env = self._make_env(
            features_df, df_valid, regime_probs,
            curriculum=self.curriculum,
        )
        self.env = env

        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=512 // cfg.n_envs,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ent_coef=0.02,
            max_grad_norm=cfg.max_grad_norm,
            tensorboard_log=cfg.tensorboard_log,
            seed=cfg.seed,
            verbose=1,
        )

        callbacks = [
            LagrangianUpdateCallback(self.constraint_mgr),
            CurriculumCallback(self.curriculum),
            ResearchMetricsCallback(self.constraint_mgr),
        ]

        phase1_steps = CURRICULUM_STAGE_1 // 2
        self.model.learn(total_timesteps=phase1_steps, callback=callbacks,
                          tb_log_name=f"{cfg.ticker}_phase1")

        # ── Phase 2: Curriculum stage 1 (trending + MR) ──────────────────
        logger.info("Phase 2: Curriculum stage 1 (trending + mean-reverting)")
        self.curriculum.stage = 1
        self.model.learn(
            total_timesteps=CURRICULUM_STAGE_1,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=f"{cfg.ticker}_phase2",
        )

        # ── Phase 3: Full training with all bars + cyclical entropy ───────
        logger.info("Phase 3: Full training (all regimes)")
        self.curriculum.stage = 2
        entropy_callback = CyclicalEntropyCallback(
            self.model, phase_length=100_000, min_ent=0.001, max_ent=0.05
        )
        callbacks_full = callbacks + [entropy_callback]
        full_steps = cfg.total_timesteps - CURRICULUM_STAGE_1 - phase1_steps
        self.model.learn(
            total_timesteps=max(full_steps, 10_000),
            callback=callbacks_full,
            reset_num_timesteps=False,
            tb_log_name=f"{cfg.ticker}_phase3",
        )

        # ── Phase 4: Lagrangian tightening ────────────────────────────────
        logger.info("Phase 4: Lagrangian tightening")
        # Slightly increase multipliers for final fine-tune
        self.constraint_mgr.lambda_dd   = max(0.5, self.constraint_mgr.lambda_dd)
        self.constraint_mgr.lambda_cvar = max(0.5, self.constraint_mgr.lambda_cvar)
        self.model.learn(
            total_timesteps=min(50_000, cfg.total_timesteps // 10),
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=f"{cfg.ticker}_phase4",
        )

        # ── Save ──────────────────────────────────────────────────────────
        model_path = cfg.model_save_path
        self.model.save(model_path)
        logger.info("Model saved to %s", model_path)

        # Save Lagrangian state
        import json
        lm_path = model_path + "_lagrangian.json"
        with open(lm_path, "w") as f:
            json.dump(self.constraint_mgr.state_dict(), f, indent=2)
        logger.info("Lagrangian state saved to %s", lm_path)

        self._trained = True
        logger.info("Training complete ✓")

    def _make_env(self, features_df, df_valid, regime_probs,
                   curriculum=None, n_envs: int = 1):
        """Create vectorized TradingEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from environment.trading_env import TradingEnv

        def _make_single():
            return TradingEnv(
                features_df=features_df,
                prices_df=df_valid,
                regime_probs=regime_probs,
                constraint_mgr=self.constraint_mgr,
                curriculum=curriculum,
                initial_capital=BACKTEST_INITIAL_CAPITAL,
                episode_length=252,
                ticker=self.config.ticker,
            )

        n = self.config.n_envs
        if n == 1:
            return DummyVecEnv([_make_single])
        else:
            return DummyVecEnv([_make_single for _ in range(n)])

    def load(self, path: Optional[str] = None) -> None:
        """Load a pre-trained model."""
        if not _CONTRIB_OK:
            raise ImportError("sb3_contrib required")
        model_path = path or self.config.model_save_path
        ensure_numpy_pickle_compat()
        self.model = RecurrentPPO.load(model_path)
        logger.info("Model loaded from %s", model_path)

        # Restore Lagrangian state if available
        from rl.rl_policy import LagrangianConstraintManager
        import json
        self.constraint_mgr = LagrangianConstraintManager()
        lm_path = model_path + "_lagrangian.json"
        if os.path.exists(lm_path):
            with open(lm_path) as f:
                self.constraint_mgr.load_state_dict(json.load(f))
            logger.info("Lagrangian state restored from %s", lm_path)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from observation vector."""
        if self.model is None:
            return np.array([0.0, 0.0])
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# ── Module-level singleton ────────────────────────────────────────────────────

from typing import Optional

_trainer: Optional[RLTrainer] = None


def get_trainer(config: Optional[TrainingConfig] = None) -> RLTrainer:
    global _trainer
    if _trainer is None:
        _trainer = RLTrainer(config or TrainingConfig())
    return _trainer


def load_trained_model(ticker: str = "AAPL", timeframe: str = "1d"):
    """
    Load and return the trained RL model for inference.
    Tries RLTrainer first (local models), then direct RecurrentPPO.load()
    (Colab or externally trained models).
    """
    cfg = TrainingConfig(ticker=ticker, timeframe=timeframe)
    zip_path = resolve_model_zip_path(MODEL_SAVE_DIR, cfg.ticker, cfg.timeframe)
    ensure_numpy_pickle_compat()

    if not os.path.exists(zip_path):
        logger.warning("No trained model found at %s — using untrained policy", zip_path)
        return None

    # Try 1: Load via RLTrainer (handles Lagrangian state restoration)
    try:
        trainer = RLTrainer(cfg)
        trainer.load(zip_path)
        if trainer.model is not None:
            logger.info("Model loaded via RLTrainer: %s", zip_path)
            return trainer.model
    except Exception as e1:
        logger.warning("RLTrainer.load failed (%s), trying direct load...", e1)

    # Try 2: Direct RecurrentPPO.load() — works for Colab-trained models
    if _CONTRIB_OK:
        try:
            from sb3_contrib import RecurrentPPO
            ensure_numpy_pickle_compat()
            model = RecurrentPPO.load(zip_path)
            logger.info("Model loaded via RecurrentPPO.load(): %s", zip_path)
            return model
        except Exception as e2:
            logger.warning("RecurrentPPO.load failed: %s", e2)

    # Try 3: Fallback to standard PPO
    try:
        from stable_baselines3 import PPO
        ensure_numpy_pickle_compat()
        model = PPO.load(zip_path)
        logger.info("Model loaded via PPO.load() fallback: %s", zip_path)
        return model
    except Exception as e3:
        logger.error("All model load attempts failed for %s: %s", zip_path, e3)

    return None
