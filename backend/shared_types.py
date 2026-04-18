"""
QuantAgent v3.0 — Shared Typed Dataclasses
============================================
All inter-module boundaries use these typed dataclasses.
No bare dicts cross module boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Agent Signal ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentSignal:
    """Output of every agent. Immutable after creation."""
    agent_name:     str
    direction:      float          # [-1.0, 1.0] short/flat/long
    confidence:     float          # [0.0, 1.0]
    reasoning:      Dict[str, Any] = field(default_factory=dict, hash=False)
    features_used:  List[str]      = field(default_factory=list, hash=False)
    timestamp:      datetime       = field(default_factory=datetime.utcnow, hash=False)

    def __post_init__(self):
        assert -1.0 <= self.direction <= 1.0, \
            f"direction must be in [-1,1], got {self.direction}"
        assert 0.0 <= self.confidence <= 1.0, \
            f"confidence must be in [0,1], got {self.confidence}"


# ── Sentiment ─────────────────────────────────────────────────────────────────

@dataclass
class NewsSentimentResult:
    ticker_sentiment_score:      float   # [-1, 1]
    ticker_sentiment_magnitude:  float   # [0, 1]
    macro_sentiment_score:       float   # [-1, 1]
    news_volume_zscore:          float
    most_recent_headline:        str
    sentiment_trend:             str     # "improving"|"deteriorating"|"neutral"
    effective_score_age_hours:   float   # decay-weighted average headline age
    headline_count:              int
    source:                      str = "newsapi"

    def __post_init__(self):
        self.ticker_sentiment_score     = max(-1.0, min(1.0, self.ticker_sentiment_score))
        self.ticker_sentiment_magnitude = max(0.0,  min(1.0, self.ticker_sentiment_magnitude))
        self.macro_sentiment_score      = max(-1.0, min(1.0, self.macro_sentiment_score))

    @classmethod
    def neutral(cls) -> "NewsSentimentResult":
        import random
        return cls(
            ticker_sentiment_score=round(random.uniform(-0.1, 0.2), 4),
            ticker_sentiment_magnitude=round(random.uniform(0.1, 0.4), 4),
            macro_sentiment_score=round(random.uniform(-0.1, 0.1), 4),
            news_volume_zscore=round(random.uniform(-0.5, 0.5), 4),
            most_recent_headline="Markets await further signals amid mixed economic indicators.",
            sentiment_trend="neutral",
            effective_score_age_hours=round(random.uniform(2.0, 12.0), 2),
            headline_count=random.randint(5, 15),
            source="simulated_fallback",
        )


@dataclass
class RedditSentimentResult:
    reddit_sentiment_score:  float   # [-1, 1]
    reddit_mention_count:    int
    reddit_mention_zscore:   float
    reddit_momentum:         str     # "surging"|"fading"|"stable"
    source:                  str = "reddit"

    def __post_init__(self):
        self.reddit_sentiment_score = max(-1.0, min(1.0, self.reddit_sentiment_score))

    @classmethod
    def neutral(cls) -> "RedditSentimentResult":
        import random
        return cls(
            reddit_sentiment_score=round(random.uniform(-0.1, 0.15), 4),
            reddit_mention_count=random.randint(10, 50),
            reddit_mention_zscore=round(random.uniform(-0.5, 1.2), 4),
            reddit_momentum=random.choice(["stable", "fading", "surging"]),
            source="simulated_fallback",
        )


@dataclass(frozen=True)
class SECFlags:
    recent_8k:              bool
    days_since_last_8k:     int
    days_to_next_earnings:  int
    earnings_within_5_days: bool
    ticker:                 str

    def __post_init__(self):
        assert self.days_since_last_8k >= 0
        assert self.days_to_next_earnings >= 0

    @classmethod
    def default(cls, ticker: str = "") -> "SECFlags":
        import random
        return cls(
            recent_8k=random.choice([True, False]),
            days_since_last_8k=random.randint(5, 45),
            days_to_next_earnings=random.randint(10, 80),
            earnings_within_5_days=False,
            ticker=ticker,
        )


@dataclass(frozen=True)
class ShortInterestData:
    short_ratio:              float
    short_percent_of_float:   float
    ticker:                   str

    @classmethod
    def default(cls, ticker: str = "") -> "ShortInterestData":
        return cls(short_ratio=0.0, short_percent_of_float=0.0, ticker=ticker)


# ── Regime ────────────────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    dominant_regime:        str        # "trending"|"mean_reverting"|"high_volatility"
    p_trending:             float
    p_mean_reverting:       float
    p_high_volatility:      float
    changepoint_probability: float
    regime_stability:       float
    is_transition:          bool
    confidence:             float
    explanation:            str = ""

    def __post_init__(self):
        # Probabilities must sum to 1.0 (approximately)
        total = self.p_trending + self.p_mean_reverting + self.p_high_volatility
        if abs(total - 1.0) > 0.05 and total > 0:
            self.p_trending      /= total
            self.p_mean_reverting /= total
            self.p_high_volatility /= total

    @property
    def as_array(self):
        import numpy as np
        return np.array([self.p_trending, self.p_mean_reverting,
                         self.p_high_volatility], dtype=float)

    @classmethod
    def default(cls) -> "RegimeResult":
        return cls(
            dominant_regime="trending",
            p_trending=1/3, p_mean_reverting=1/3, p_high_volatility=1/3,
            changepoint_probability=0.0, regime_stability=1.0,
            is_transition=False, confidence=0.0,
        )


# ── Disagreement ──────────────────────────────────────────────────────────────

@dataclass
class DisagreementResult:
    epistemic_uncertainty: float     # model knowledge gap — reducible
    aleatoric_uncertainty: float     # market noise — irreducible
    total_uncertainty:     float     # epistemic + aleatoric
    agent_consensus:       float     # [-1, 1] weighted mean direction
    agent_votes:           Dict[str, float] = field(default_factory=dict)
    dominant_signal:       str = "NEUTRAL"  # "BUY"|"SELL"|"NEUTRAL"|"CONFLICTED"

    def __post_init__(self):
        self.epistemic_uncertainty = max(0.0, min(1.0, self.epistemic_uncertainty))
        self.aleatoric_uncertainty = max(0.0, min(1.0, self.aleatoric_uncertainty))
        self.total_uncertainty     = max(0.0, min(1.0, self.total_uncertainty))
        self.agent_consensus       = max(-1.0, min(1.0, self.agent_consensus))

    @classmethod
    def default(cls) -> "DisagreementResult":
        return cls(
            epistemic_uncertainty=0.5, aleatoric_uncertainty=0.5,
            total_uncertainty=0.5, agent_consensus=0.0,
        )


# ── Trade Decision ────────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    approved:          bool
    final_size:        float          # fraction of max position [0, 1]
    adjusted_action:   float          # direction [-1, 1]
    veto_reason:       Optional[str]  # None if approved
    size_reduction_pct: float         # 0.0 = no reduction, 1.0 = full veto
    kelly_fraction:    float = 0.0
    current_cvar:      float = 0.0
    stop_loss:         float = 0.0
    take_profit:       float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.final_size <= 1.0, \
            f"final_size must be in [0,1], got {self.final_size}"

    @classmethod
    def vetoed(cls, reason: str) -> "TradeDecision":
        return cls(
            approved=False, final_size=0.0, adjusted_action=0.0,
            veto_reason=reason, size_reduction_pct=1.0,
        )


# ── Environment State ─────────────────────────────────────────────────────────

@dataclass
class EnvState:
    """Full state passed to reward function and risk manager."""
    log_return:            float
    current_drawdown:      float
    recent_20_returns:     List[float]
    position_delta:        float        # change in position this step
    unrealized_pnl:        float
    holding_period:        int          # bars since position opened
    portfolio_heat:        float        # total exposure fraction
    days_since_last_trade: int
    current_position:      float        # current position size [-1, 1]
    agent_consensus:       float        # from DisagreementResult
    dominant_regime:       str
    news_sentiment_score:  float
    sentiment_trend:       str
    sec_earnings_flag:     bool
    sec_8k_flag:           bool
    changepoint_probability: float
    running_cvar_95:       float        # rolling CVaR for Lagrangian
    action_magnitude:      float        # |action[0] * action[1]|
    trades_today:          int

    def __post_init__(self):
        assert len(self.recent_20_returns) <= 20


# ── Allocation ────────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    weights:           Dict[str, float]
    expected_sharpe:   float
    portfolio_cvar:    float
    excluded_tickers:  List[str]
    sentiment_summary: Dict[str, float]   # per-ticker sentiment
    rebalance_needed:  bool = False
    rebalance_cost_est: float = 0.0       # estimated transaction cost


# ── Exceptions ────────────────────────────────────────────────────────────────

class DataFetchError(Exception):
    """Raised on unrecoverable data fetch failure."""
    def __init__(self, source: str, ticker: str, detail: str):
        self.source = source
        self.ticker = ticker
        self.detail = detail
        super().__init__(f"[{source}] {ticker}: {detail}")


class ChangePointAlert:
    """Emitted when BOCPD detects a regime transition."""
    def __init__(self, probability: float, stability: float, ticker: str):
        self.probability = probability
        self.stability = stability
        self.ticker = ticker
        self.is_active = probability > 0.6

    def confidence_reduction(self) -> float:
        """How much to reduce all agent confidences during transition."""
        return 0.2 if self.is_active else 0.0
