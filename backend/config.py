"""
QuantAgent Configuration
========================
Central settings for all agents, risk management, and RL controller.
"""

import os

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
