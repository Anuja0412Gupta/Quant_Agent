# QuantAgent ⚡

**A full-stack, multi-agent AI trading system** that continuously improves with reinforcement learning.

---

## Architecture

```
backend/
├── agents/
│   ├── indicator_agent.py       — RSI, MACD, Stochastic, ROC, Williams %R
│   ├── pattern_agent.py         — Double Top/Bottom, H&S, Flag, Triangle
│   ├── trend_agent.py           — Polynomial regression trendlines
│   ├── market_regime_agent.py   — ATR, Hurst exponent, rolling variance
│   ├── decision_agent.py        — Confidence-weighted fusion
│   ├── disagreement_model.py    — Signal variance gating
│   ├── risk_management_agent.py — Fractional Kelly + vol-adjusted sizing
│   ├── rl_meta_controller.py    — PPO meta-controller (Stable Baselines3)
│   └── self_critique_agent.py   — Post-trade weight updates
├── backtesting/
│   └── backtesting_engine.py    — Full bar-by-bar simulation
├── data/
│   └── data_fetcher.py          — yfinance OHLCV loader
├── utils/helpers.py
├── config.py
├── main.py                      — FastAPI REST API
└── requirements.txt

frontend/
└── src/
    ├── components/
    │   ├── CandlestickChart.jsx  — TradingView lightweight-charts
    │   ├── AgentSignals.jsx      — Per-agent signal cards
    │   ├── TradeDecision.jsx     — LONG/SHORT/NO_TRADE display
    │   └── BacktestPanel.jsx     — Metrics + equity curve
    └── App.jsx                   — Main dashboard
```

---

## Quick Start

### 1. Backend

```powershell
cd QUANT\backend
pip install -r requirements.txt
python main.py
# API running at http://localhost:8000
```

### 2. Frontend

```powershell
cd QUANT\frontend
npm run dev
# Dashboard at http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/analyze/{symbol}?timeframe=1d` | Full agent pipeline |
| `GET` | `/backtest/{symbol}?timeframe=1d&period=5y` | Historical backtest |
| `GET` | `/regime/{symbol}` | Market regime only |
| `GET` | `/agents/weights` | Current agent weights |
| `POST` | `/agents/weights` | Override weights |
| `POST` | `/critique` | Self-critique after trade |
| `POST` | `/rl/train` | Trigger RL training |

---

## Supported Timeframes & Symbols

| Timeframe | Period |
|-----------|--------|
| `1m` | 7 days |
| `5m`, `15m` | 60 days |
| `1h` | 730 days |
| `1d` | 5 years |

Any Yahoo Finance ticker: `AAPL`, `MSFT`, `BTC-USD`, `RELIANCE.NS`, etc.

---

## RL Meta-Controller

The PPO agent learns which agent to trust more based on:
- **State**: indicator/pattern/trend/regime signals + volatility + disagreement + drawdown
- **Action**: agent weights + position size multiplier + trade/skip decision
- **Reward**: risk-adjusted return (Sharpe − drawdown penalty)

Trigger training via: `POST /rl/train` with `{ "timesteps": 50000 }`
