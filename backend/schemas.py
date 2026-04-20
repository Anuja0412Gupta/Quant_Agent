"""
QuantAgent v3.0 — Pydantic API Schemas
=========================================
All FastAPI request/response models. Typed at the boundary.
No bare dicts cross the HTTP boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ── Request Schemas ───────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=4)

class LoginRequest(BaseModel):
    username: str
    password: str

class AnalyzeRequest(BaseModel):
    symbol:    str = Field("AAPL", description="Ticker symbol")
    timeframe: str = Field("1d", description="'1d' or '1h'")
    include_sentiment: bool = Field(True, description="Include news/Reddit sentiment")
    include_macro:     bool = Field(True, description="Include macro context")

    @validator("symbol")
    def symbol_upper(cls, v):
        return v.upper().strip()

    @validator("timeframe")
    def valid_timeframe(cls, v):
        assert v in ("1d", "1h", "15m"), "timeframe must be '1d', '1h', or '15m'"
        return v


class BacktestRequest(BaseModel):
    symbol:      str   = Field("AAPL")
    timeframe:   str   = Field("1d")
    period:      str   = Field("1y")
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    initial_capital: float = Field(100_000.0, ge=1000.0)
    use_walk_forward: bool = Field(True)
    n_folds:     int   = Field(4, ge=1, le=20)

    @validator("symbol")
    def symbol_upper(cls, v):
        return v.upper().strip()


class TradeRequest(BaseModel):
    symbol:     str
    action:     str  = Field(..., description="'BUY' | 'SELL' | 'HOLD'")
    quantity:   float = Field(..., gt=0)
    price:      Optional[float] = None
    notes:      Optional[str]  = None

    @validator("action")
    def valid_action(cls, v):
        v = v.upper()
        assert v in ("BUY", "SELL", "HOLD"), "action must be BUY, SELL, or HOLD"
        return v


class TrainRequest(BaseModel):
    symbol:    str   = Field("AAPL")
    timeframe: str   = Field("1d")
    total_timesteps: int = Field(500_000, ge=10_000)

class AblationRequest(BaseModel):
    symbol:    str   = Field("AAPL")
    timeframe: str   = Field("1d")
    period:    str   = Field("2y")


# ── Common sub-schemas ────────────────────────────────────────────────────────

class RegimeResultSchema(BaseModel):
    dominant_regime:        str
    p_trending:             float
    p_mean_reverting:       float
    p_high_volatility:      float
    changepoint_probability: float
    regime_stability:       float
    is_transition:          bool
    confidence:             float
    explanation:            Optional[str] = None


class SentimentSchema(BaseModel):
    ticker_sentiment_score:      float
    ticker_sentiment_magnitude:  float
    macro_sentiment_score:       float
    news_volume_zscore:          float
    most_recent_headline:        str
    sentiment_trend:             str
    effective_score_age_hours:   float
    headline_count:              int
    source:                      str

    # Reddit extras
    reddit_sentiment_score:  Optional[float] = 0.0
    reddit_mention_count:    Optional[int]   = 0
    reddit_mention_zscore:   Optional[float] = 0.0
    reddit_momentum:         Optional[str]   = "stable"


class SECFlagsSchema(BaseModel):
    recent_8k:              bool
    days_since_last_8k:     int
    days_to_next_earnings:  int
    earnings_within_5_days: bool


class DisagreementSchema(BaseModel):
    epistemic_uncertainty:  float
    aleatoric_uncertainty:  float
    total_uncertainty:      float
    agent_consensus:        float
    dominant_signal:        str
    recommendation:         str


class TradeDecisionSchema(BaseModel):
    approved:          bool
    final_size:        float
    adjusted_action:   float
    veto_reason:       Optional[str]
    size_reduction_pct: float
    kelly_fraction:    float
    current_cvar:      float
    stop_loss:         float
    take_profit:       float


class MacroContextSchema(BaseModel):
    vix_level:           float
    vix9d_level:         float
    vix_ts_spread:       float
    vix_zscore:          float
    hyg_lqd_ratio:       float
    hyg_lqd_zscore:      float
    credit_regime_flag:  int
    dxy_20d_momentum:    float
    t10y2y_spread:       float
    fed_funds_rate:      float
    hy_credit_spread:    float
    consumer_sentiment:  float
    freshness_ts:        str


class OHLCVBar(BaseModel):
    timestamp: str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float


class AgentSignalSchema(BaseModel):
    agent_name:  str
    direction:   float
    confidence:  float
    reasoning:   Dict[str, Any] = {}
    signal:      str = "neutral"


class RLWeightsSchema(BaseModel):
    rl_action:          float
    gate_value:         float
    effective_action:   float
    effective_position: float
    disagreement_score: float
    active_regime:      str
    regime_confidence:  float
    direction:          str


# ── Response Schemas ──────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    symbol:       str
    timeframe:    str
    timestamp:    str
    current_price: float
    price_change_pct: float

    regime:       RegimeResultSchema
    sentiment:    Optional[SentimentSchema] = None
    sec_flags:    Optional[SECFlagsSchema]  = None
    disagreement: DisagreementSchema
    trade_decision: TradeDecisionSchema
    macro_context: Optional[MacroContextSchema] = None

    agent_signals: List[AgentSignalSchema] = []
    ohlcv_bars:    List[OHLCVBar] = []
    rl_weights:    Optional[RLWeightsSchema] = None

    # Research metrics
    feature_dim:   int = 45
    burnin_bars:   int = 252
    model_version: str = "3.0"


class BacktestResultSchema(BaseModel):
    symbol:      str
    timeframe:   str
    period:      str
    start_date:  str
    end_date:    str

    overall_metrics: Dict[str, float]
    regime_metrics:  Dict[str, Dict[str, float]]  # per-regime breakdown
    walk_forward_summary: Optional[Dict] = None
    stress_test_results:  Optional[Dict] = None

    drawdown_curve:   List[float] = []
    equity_curve:     List[float] = []
    return_series:    List[float] = []
    timestamps:       List[str]   = []

    attribution: Optional[Dict] = None   # BHB attribution
    model_version: str = "3.0"


class TradeResponse(BaseModel):
    success:    bool
    trade_id:   str
    symbol:     str
    action:     str
    quantity:   float
    price:      float
    timestamp:  str
    portfolio_value: float
    cash:        float
    message:    Optional[str] = None


class PortfolioSchema(BaseModel):
    total_value:    float
    cash:           float
    invested_value: float
    unrealized_pnl: float
    realized_pnl:   float
    positions:      List[Dict[str, Any]] = []
    trade_history:  List[Dict[str, Any]] = []
    allocation:     Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status:          str
    version:         str = "3.0"
    timestamp:       str
    components:      Dict[str, str]
    data_freshness:  Optional[str] = None
    model_loaded:    bool = False
    feature_dim:     int  = 45
    burnin_bars:     int  = 252


class AlmgrenChrissScheduleSchema(BaseModel):
    n_bars:               int
    total_shares:         int
    estimated_cost_bps:   float
    participation_rate:   float
    trades_pct_per_bar:   List[float]
