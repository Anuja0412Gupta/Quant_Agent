"""
QuantAgent v3.0 — TCN-LSTM Policy with FiLM Regime Conditioning
================================================================
Context-conditioned single policy that *learns* to handle all regimes.
Replaces the three-expert MoE approach.

Architecture:
  TCN (4 layers, dilations 1/2/4/8, receptive field = 32 steps)
  ↓ FiLM modulation from regime probabilities (state dims [29:32])
  ↓ LSTM (2 layers, 128 hidden)
  ↓ Split actor/critic heads

FiLM conditioning (Feature-wise Linear Modulation):
  Regime probs → scale + shift applied to TCN output.
  Lets the LSTM explicitly condition its feature representation on regime.

LagrangianConstraintManager:
  Enforces max_drawdown ≤ 0.20 and CVaR ≤ 0.04 via dual gradient ascent.
  Multipliers updated after each training episode, not via reward shaping.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except (ImportError, OSError) as _torch_err:
    # ImportError: torch not installed
    # OSError/WinError 1114: DLL load failed (Windows CPU/CUDA mismatch)
    _TORCH_OK = False
    logger.warning("torch not available (%s) — TCNLSTMPolicy disabled",
                   type(_torch_err).__name__)

from config import (
    LAGRANGIAN_LR, CVAR_NO_TRADE_THRESHOLD, CVAR_REDUCE_THRESHOLD,
    MAX_DRAWDOWN_LIMIT, CURRICULUM_STAGE_1, CURRICULUM_STAGE_2,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LAGRANGIAN CONSTRAINT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class LagrangianConstraintManager:
    """
    Dual gradient ascent for safety constraints.
    Enforces max_drawdown < 0.20 and CVaR(95%) < 0.04 in expectation.

    Multipliers are updated after EACH episode rollout.
    The augmented_reward() subtracts constraint violation penalties.
    """

    def __init__(self, lr_lambda: float = LAGRANGIAN_LR,
                 dd_limit: float = MAX_DRAWDOWN_LIMIT,
                 cvar_limit: float = CVAR_NO_TRADE_THRESHOLD):
        self.lr = lr_lambda
        self.dd_limit   = dd_limit
        self.cvar_limit = cvar_limit
        self.lambda_dd   = 0.0   # Lagrange multiplier: drawdown constraint
        self.lambda_cvar = 0.0   # Lagrange multiplier: CVaR constraint
        self._episode_count = 0
        self._violation_log: list = []

    def update(self, episode_max_dd: float, episode_cvar: float) -> None:
        """
        Update multipliers after one episode.
        Called from training callback after each rollout.
        """
        dd_violation   = max(0.0, episode_max_dd - self.dd_limit)
        cvar_violation = max(0.0, episode_cvar   - self.cvar_limit)

        self.lambda_dd   = max(0.0, self.lambda_dd   + self.lr * dd_violation)
        self.lambda_cvar = max(0.0, self.lambda_cvar + self.lr * cvar_violation)

        # Cap multipliers to prevent explosion
        self.lambda_dd   = min(self.lambda_dd,   100.0)
        self.lambda_cvar = min(self.lambda_cvar, 100.0)

        self._episode_count += 1
        if dd_violation > 0 or cvar_violation > 0:
            self._violation_log.append({
                "episode": self._episode_count,
                "dd_viol": round(dd_violation, 4),
                "cvar_viol": round(cvar_violation, 4),
                "lambda_dd": round(self.lambda_dd, 4),
                "lambda_cvar": round(self.lambda_cvar, 4),
            })
            logger.debug("Constraint violation: dd=%.4f cvar=%.4f → "
                         "λ_dd=%.3f λ_cvar=%.3f",
                         dd_violation, cvar_violation,
                         self.lambda_dd, self.lambda_cvar)

    def augmented_reward(self, base_reward: float,
                         step_dd: float, step_cvar: float) -> float:
        """
        Subtract Lagrangian constraint penalties from base reward.
        Called at each environment step.
        """
        dd_penalty   = self.lambda_dd   * max(0.0, step_dd   - self.dd_limit)
        cvar_penalty = self.lambda_cvar * max(0.0, step_cvar - self.cvar_limit)
        return float(base_reward - dd_penalty - cvar_penalty)

    def state_dict(self) -> dict:
        return {
            "lambda_dd":   self.lambda_dd,
            "lambda_cvar": self.lambda_cvar,
            "episode_count": self._episode_count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.lambda_dd    = float(d.get("lambda_dd",   0.0))
        self.lambda_cvar  = float(d.get("lambda_cvar", 0.0))
        self._episode_count = int(d.get("episode_count", 0))


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    Progressive training difficulty based on HMM regime probabilities.

    Stage 0 (0 – 100k steps):   Only trending bars (easy)
    Stage 1 (100k – 300k steps): Trending + mean-reverting bars (medium)
    Stage 2 (300k+ steps):       All bars including high-vol (full)
    """

    def __init__(self, regime_probabilities=None):
        """
        regime_probabilities: pd.DataFrame with columns
            [trending, mean_reverting, high_volatility], index=df.index.
        If None, all bars are valid (stage 2 only).
        """
        self.regime_probs = regime_probabilities
        self.stage   = 0
        self._stage_thresholds = [CURRICULUM_STAGE_1, CURRICULUM_STAGE_2]

    def update_stage(self, current_step: int) -> None:
        """Update stage based on total training steps."""
        if current_step >= self._stage_thresholds[1]:
            if self.stage != 2:
                self.stage = 2
                logger.info("Curriculum: advanced to Stage 2 (all bars) "
                            "at step %d", current_step)
        elif current_step >= self._stage_thresholds[0]:
            if self.stage != 1:
                self.stage = 1
                logger.info("Curriculum: advanced to Stage 1 (trending+MR) "
                            "at step %d", current_step)

    def valid_start_indices(self, df_len: int,
                            min_episode_length: int = 63) -> np.ndarray:
        """Return array of valid episode start indices for current stage."""
        import pandas as pd
        if self.regime_probs is None or self.stage == 2:
            # All indices (leave room for full episode)
            valid = np.arange(df_len - min_episode_length)
        elif self.stage == 0:
            mask = self.regime_probs["trending"] > 0.6
            valid = np.where(mask.values[:df_len])[0]
        else:  # stage 1
            mask = ((self.regime_probs["trending"] > 0.4) |
                    (self.regime_probs["mean_reverting"] > 0.4))
            valid = np.where(mask.values[:df_len])[0]

        # Filter: must be far enough from end for a full episode
        valid = valid[valid < df_len - min_episode_length]
        if len(valid) == 0:
            logger.warning("Curriculum stage %d: no valid indices, "
                           "falling back to all bars", self.stage)
            valid = np.arange(max(1, df_len - min_episode_length))
        return valid

    def sample_start(self, df_len: int,
                     min_episode_length: int = 63) -> int:
        """Sample a random valid episode start index."""
        valid = self.valid_start_indices(df_len, min_episode_length)
        assert len(valid) > 0, \
            f"Curriculum stage {self.stage}: no valid start indices"
        return int(np.random.choice(valid))


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLICAL ENTROPY COEFFICIENT
# ═══════════════════════════════════════════════════════════════════════════════

def cyclical_entropy_coef(step: int,
                           phase_length: int = 100_000,
                           min_ent: float = 0.001,
                           max_ent: float = 0.05) -> float:
    """
    Entropy oscillates on a cosine schedule within each phase.
    Prevents premature convergence to flat policies.
    """
    phase_progress = (step % phase_length) / phase_length
    ent = min_ent + 0.5 * (max_ent - min_ent) * (1.0 + math.cos(math.pi * phase_progress))
    return float(ent)


# ═══════════════════════════════════════════════════════════════════════════════
# TCN-LSTM POLICY WITH FILM CONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:

    class CausalConv1d(nn.Module):
        """1D causal convolution with dilation (no future leakage)."""
        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, dilation: int):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation,
                padding=self.padding,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T)
            out = self.conv(x)
            return out[:, :, :-self.padding] if self.padding > 0 else out

    class TCNBlock(nn.Module):
        """Temporal Convolutional Network block with residual connection."""
        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, dilation: int, dropout: float = 0.1):
            super().__init__()
            self.conv1 = CausalConv1d(in_channels, out_channels,
                                       kernel_size, dilation)
            self.norm1 = nn.GroupNorm(1, out_channels)   # group=1 = LayerNorm over C
            self.conv2 = CausalConv1d(out_channels, out_channels,
                                       kernel_size, dilation)
            self.norm2 = nn.GroupNorm(1, out_channels)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.GELU()

            # Residual projection if input/output dims differ
            self.residual_proj = (nn.Conv1d(in_channels, out_channels, 1)
                                  if in_channels != out_channels else nn.Identity())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.residual_proj(x)
            out = self.act(self.norm1(self.conv1(x)))
            out = self.dropout(out)
            out = self.act(self.norm2(self.conv2(out)))
            out = self.dropout(out)
            return out + residual

    class TCNLSTMPolicy(nn.Module):
        """
        TCN encoder → FiLM regime conditioning → LSTM → actor/critic heads.

        Receptive field of TCN = 2^4 * 2 = 32 steps ≈ 6.5 trading weeks.
        FiLM: scale + shift TCN output by linear projection of regime probs
              (state dims [29:32]).
        """

        def __init__(self, obs_dim: int = 45,
                     tcn_channels: int = 64,
                     tcn_layers: int = 4,
                     lstm_hidden: int = 128,
                     lstm_layers: int = 2,
                     action_dim: int = 2,
                     dropout: float = 0.1):
            super().__init__()
            self.obs_dim       = obs_dim
            self.tcn_channels  = tcn_channels
            self.lstm_hidden   = lstm_hidden
            self.action_dim    = action_dim

            # TCN: dilations [1, 2, 4, 8]
            self.tcn_blocks = nn.ModuleList()
            in_ch = obs_dim
            for i in range(tcn_layers):
                dilation = 2 ** i
                self.tcn_blocks.append(
                    TCNBlock(in_ch, tcn_channels, kernel_size=3,
                             dilation=dilation, dropout=dropout)
                )
                in_ch = tcn_channels

            # FiLM conditioning: regime probs (dim 3) → scale + shift over TCN output
            # Output size: 2 * tcn_channels (scale and shift)
            self.film_layer = nn.Sequential(
                nn.Linear(3, 64), nn.GELU(),
                nn.Linear(64, 2 * tcn_channels),
            )

            # LSTM on top of modulated TCN features
            self.lstm = nn.LSTM(
                input_size=tcn_channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )

            # Actor head
            self.actor = nn.Sequential(
                nn.Linear(lstm_hidden, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, action_dim),
            )

            # Critic head (separate from actor for PPO stability)
            self.critic = nn.Sequential(
                nn.Linear(lstm_hidden, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

            # Log std for continuous action distribution
            self.log_std = nn.Parameter(
                torch.zeros(action_dim) - 1.0   # init: std ≈ 0.37
            )

        def forward(self,
                    obs_sequence: "torch.Tensor",
                    hidden: Optional[Tuple] = None
                    ) -> Tuple["torch.Tensor", "torch.Tensor", Tuple]:
            """
            obs_sequence: (B, seq_len, obs_dim) OR (B, obs_dim) for single step.
            Returns: (action_mean, value, new_hidden)
            """
            # Handle both 2D and 3D input
            if obs_sequence.dim() == 2:
                obs_sequence = obs_sequence.unsqueeze(1)  # (B, 1, D)

            B, T, D = obs_sequence.shape

            # ── TCN forward (treat time as 1D signal) ─────────────────────
            x = obs_sequence.permute(0, 2, 1)   # (B, D, T)
            for block in self.tcn_blocks:
                x = block(x)
            tcn_out = x.permute(0, 2, 1)         # (B, T, tcn_channels)

            # ── FiLM modulation from regime probabilities ──────────────────
            regime_probs = obs_sequence[:, :, 29:32]   # (B, T, 3)
            film_params  = self.film_layer(regime_probs)  # (B, T, 2*C)
            scale = film_params[:, :, :self.tcn_channels]   # (B, T, C)
            shift = film_params[:, :, self.tcn_channels:]   # (B, T, C)
            tcn_out = tcn_out * (1.0 + scale) + shift       # FiLM modulation

            # ── LSTM ────────────────────────────────────────────────────────
            lstm_out, new_hidden = self.lstm(tcn_out, hidden)
            last_out = lstm_out[:, -1, :]   # (B, lstm_hidden)

            # ── Heads ────────────────────────────────────────────────────────
            action_mean = self.actor(last_out)
            value       = self.critic(last_out)

            return action_mean, value, new_hidden

        def get_action_and_value(
            self,
            obs: "torch.Tensor",
            hidden: Optional[Tuple] = None,
            action: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor",
                   "torch.Tensor", Tuple]:
            """
            Used by PPO training loop.
            Returns: (action, log_prob, entropy, value, new_hidden)
            """
            action_mean, value, new_hidden = self.forward(obs, hidden)

            std = torch.exp(self.log_std.clamp(-5, 2))
            dist = torch.distributions.Normal(action_mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(-1)
            entropy  = dist.entropy().sum(-1)

            return action, log_prob, entropy, value.squeeze(-1), new_hidden

        def get_value(self,
                      obs: "torch.Tensor",
                      hidden: Optional[Tuple] = None
                      ) -> "torch.Tensor":
            _, value, _ = self.forward(obs, hidden)
            return value.squeeze(-1)

else:
    # Stub when torch not available
    class TCNLSTMPolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch not available — TCNLSTMPolicy requires PyTorch")

    class CausalConv1d:  # type: ignore
        pass

    class TCNBlock:  # type: ignore
        pass
