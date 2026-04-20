"""
QuantAgent v3.0 Configuration
==============================
Central settings for all agents, risk management, and RL controller.
Research-grade constants with burn-in enforcement and typed boundaries.
"""

import os

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv optional

# ── Data ─────────────────────────────────────────────────────────────────────
DEFAULT_SYMBOL = "AAPL"
DEFAULT_TIMEFRAME = "1d"
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]

TIMEFRAME_PERIODS = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "1h":  "730d",
    "1d":  "5y",
}

# ── Indicator Agent ───────────────────────────────────────────────────────────
RSI_PERIOD          = 14
RSI_OVERBOUGHT      = 70
RSI_OVERSOLD        = 30
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
STOCH_K             = 14
STOCH_D             = 3
ROC_PERIOD          = 10
WILLIAMS_PERIOD     = 14

# ── Trend Agent ───────────────────────────────────────────────────────────────
TREND_POLY_DEGREE   = 2
TREND_LOOKBACK      = 50
SUPPORT_RESIST_WINDOW = 20

# ── Market Regime Agent ───────────────────────────────────────────────────────
ATR_PERIOD          = 14
REGIME_VARIANCE_WINDOW = 20
HURST_MIN_LAGS      = 2
HURST_MAX_LAGS      = 20

# ── Decision Agent ────────────────────────────────────────────────────────────
DIRECTION_MAP       = {"bullish": 1, "uptrend": 1, "trending": 0.5,
                       "bearish": -1, "downtrend": -1,
                       "mean_reverting": -0.3,
                       "neutral": 0, "sideways": 0,
                       "high_volatility": -0.5, "low_volatility": 0.2}
ATR_SL_MULTIPLIER   = 1.5
ATR_TP_MULTIPLIER   = 3.0
BUY_THRESHOLD       = 0.15
SELL_THRESHOLD      = -0.15

# ── Disagreement Model ────────────────────────────────────────────────────────
DISAGREEMENT_THRESHOLD = 0.04   # variance above this → reduce / skip
POSITION_REDUCE_FACTOR = 0.5

# ── Risk Management ───────────────────────────────────────────────────────────
KELLY_FRACTION      = 0.25      # fractional Kelly
MAX_POSITION_PCT    = 0.10      # 10 % of portfolio per trade
MAX_DRAWDOWN_PCT    = 0.20      # 20 % max drawdown limit
PORTFOLIO_VALUE     = 100_000   # default virtual portfolio ($)

# ── RL Meta-Controller ────────────────────────────────────────────────────────
RL_MODEL_PATH       = "models/rl_model"
RL_TIMESTEPS        = 50_000
RL_LEARNING_RATE    = 3e-4
N_AGENTS            = 4         # indicator, pattern, trend, regime

# ── Self-Critique ─────────────────────────────────────────────────────────────
CRITIQUE_WEIGHTS_FILE = "models/agent_weights.json"
DEFAULT_AGENT_WEIGHTS = {
    "indicator": 0.30,
    "pattern":   0.25,
    "trend":     0.25,
    "regime":    0.20,
}

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_INITIAL_CAPITAL = 100_000
BACKTEST_COMMISSION      = 0.001   # 0.1 %

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST  = "0.0.0.0"
API_PORT  = int(os.environ.get("PORT", 8000))   # Railway injects $PORT

_vercel = os.environ.get("VERCEL_FRONTEND_URL", "")   # set this in Railway
CORS_ORIGINS = [
    "*",                           # allow all origins (safe for public read-only API)
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    *([_vercel] if _vercel else []),   # your Vercel URL injected at runtime
]

# ── v3.0 Research-Grade Constants ─────────────────────────────────────────────

# Burn-in: minimum bars before any feature or training is valid.
# 252 = 1 trading year ensures all 60-bar rolling windows are fully saturated.
FEATURE_BURNIN_BARS     = 252
MIN_EPISODE_LENGTH      = 63      # min bars in a training episode (~1 quarter)
HMM_MIN_FIT_BARS        = 252
HMM_REFIT_PERIOD_BARS   = 63

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_SAVE_DIR          = os.environ.get("MODEL_SAVE_DIR",  "./models")
FINBERT_CACHE_DIR       = os.environ.get("FINBERT_CACHE_DIR", "./models/finbert")
YFINANCE_TZ_CACHE_DIR   = os.environ.get("YFINANCE_TZ_CACHE_DIR", "./models/yfinance_tz")
PAPER_PORTFOLIO_PATH    = os.environ.get("PAPER_PORTFOLIO_PATH",
                                         "./data/paper_portfolio.json")

# ── External API Keys (loaded from .env) ──────────────────────────────────────
FRED_API_KEY            = os.environ.get("FRED_API_KEY", "")
NEWS_API_KEY            = os.environ.get("NEWS_API_KEY", "")
REDDIT_CLIENT_ID        = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET    = os.environ.get("REDDIT_CLIENT_SECRET", "")

# ── Database ──────────────────────────────────────────────────────────────────
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://smartqueue_admin:%40Ashok123@cluster0.ayenjp1.mongodb.net/smart_queue?retryWrites=true&w=majority&appName=Cluster0")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "quant_agent")

# ── Sentiment decay half-lives (hours) ────────────────────────────────────────
SENTIMENT_HALF_LIVES    = {
    "earnings_news":  4,
    "ticker_news":   24,
    "reddit":        12,
    "macro_news":    48,
}

# ── Execution model ───────────────────────────────────────────────────────────
SLIPPAGE_DAILY          = 0.0005
SLIPPAGE_INTRADAY       = 0.001
COMMISSION_RATE         = 0.001
MAX_PARTICIPATION_RATE  = 0.10

# ── RL training ───────────────────────────────────────────────────────────────
RL_TRAIN_STEPS_DAILY    = 500_000   # was 25_000 — severely undertrained
RL_TRAIN_STEPS_INTRA    = 200_000
CURRICULUM_STAGE_1      = 100_000   # was 5_000
CURRICULUM_STAGE_2      = 300_000   # was 15_000

# ── Risk limits ───────────────────────────────────────────────────────────────
CVAR_NO_TRADE_THRESHOLD = 0.03    # was 0.04 — tighter CVaR improves Sharpe
CVAR_REDUCE_THRESHOLD   = 0.02    # was 0.025
MAX_DRAWDOWN_LIMIT      = 0.15    # was 0.20 — force model to be less risky
LAGRANGIAN_LR           = 0.02    # was 0.01 — faster constraint adaptation

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SYMBOL    = "AAPL"
DEFAULT_TIMEFRAME = "1d"
MAX_DRAWDOWN_PCT  = MAX_DRAWDOWN_LIMIT   # alias used by risk_management_agent

# ── Legacy aliases ────────────────────────────────────────────────────────────
RL_MODEL_PATH    = os.path.join(MODEL_SAVE_DIR, "rl_model")
RL_TIMESTEPS     = RL_TRAIN_STEPS_DAILY
RL_LEARNING_RATE = 3e-4
