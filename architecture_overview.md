# ⚡ QuantAgent platform Architecture Deep-Dive

Welcome to the **QuantAgent v2.0 Architecture Guide**. This document explores the entire full-stack project, analyzing how the multi-agent system, the RL meta-controller, the FastAPI backend, and the React frontend tie together to build a research-grade algorithmic trading platform.

## 🏗 High-Level System Architecture

At its core, **QuantAgent** is a system that tries to replicate how a quantitative hedge fund team works. It utilizes multiple "expert" agents (technical analysts, risk managers, market regime detectors) who independently score the current market. These agents' outputs are fed into a Reinforcement Learning (RL) "Meta-Controller" (the portfolio manager) which makes the final executive decision. 

Everything is glued together by a Python FastAPI backend and visualized on a React + Vite frontend.

---

## 🧠 The Backend AI Pipeline

When you hit "Analyze" for a ticker (e.g. `AAPL`), the backend executes `_run_full_pipeline` in `main.py`. The data is pulled and flows through the agents sequentially:

### 1. Feature Engineering & Environment Agents
* **`indicator_agent.py`**: Computes standard technical momentum oscillators like RSI, MACD, Stochastic, ROC, and Williams %R. Returns bullish/bearish scores.
* **`trend_agent.py`**: Measures directional strength using polynomial regression fitting over multiple lookback windows and computes dynamic support/resistance zones.
* **`pattern_agent.py`**: Scans the OHLCV candlestick data for classic reversal or continuation patterns (e.g., Doji, Engulfing).
* **`market_regime_agent.py`**: A critically important agent. It uses the **Hurst Exponent** and **ATR (Average True Range)** to classify the market environment into one of three regimes:
   1. `trending` (Hurst > 0.55)
   2. `mean_reverting` (Hurst < 0.45)
   3. `high_volatility` (ATR indicates panic/high variance)

### 2. The Disagreement Model (`disagreement_model.py`)
This is the "Uncertainty Engine". It takes the suggestions from the four agents above and calculates the statistical variance between their predictions. 
* *High Disagreement*: The technical agent says BUY, but the trend agent says SELL.
* *Low Disagreement*: All agents uniformly agree to BUY.
This outputs a `disagreement_score` used to gauge prediction confidence.

### 3. The RL Meta-Controller (`rl_meta_controller.py`)
This is the "Brain" of v2.0. It's built using `stable_baselines3` (PPO algorithms). It features two advanced architectural choices:
* **Mixture of Experts (MoE)**: It doesn't use one monolithic neural network. Instead, it has three separate PPO policies designed for the three specific market regimes. It selects the active policy based on the `market_regime_agent`'s classification.
* **Bayesian Disagreement Gating**: The RL agent outputs a continuous action (LONG/SHORT intensity) and a position size. However, if the `disagreement_model` detects high systemic uncertainty, it automatically *shrinks (gates)* both the action confidence and position size to protect capital. 

### 4. Risk Gatekeeper (`risk_management_agent.py`)
No matter what the RL brain says, this agent has final veto power. It computes fractional Kelly criteria, Max Drawdown constraints, and Value-at-Risk (VaR). If VaR exceeds safe thresholds (e.g. > 4%), it forces a `NO_TRADE` or reduces sizing.

---

## 🔥 Backend Operational Features

Beyond simple analysis, the backend exposes endpoints for sophisticated quant workflows:
1. **Walk-Forward Validation (`/rl/walk_forward`)**: Rather than standard single-pass backtesting, it trains the RL model on window A, tests on window B, then rolls the windows forward. It’s asynchronous to prevent blocking the UI.
2. **Stress Testing (`/stress_test`)**: Takes the chosen stock and deliberately simulates trading during its **worst historical 30-day crash** to see if the Risk Manager agent adequately prevented ruin.
3. **Reward Ablation (`/rl/ablation`)**: Runs 5 different variations of the RL reward function (e.g. standard Sharpe vs. Sortino vs. penalizing drawdowns heavily) to compare performance.
4. **Paper Trading (`/paper_trade`)**: Maintains a persistent `paper_portfolio.json` ledger. Simulates live execution, tracks unrealized/realized P&L, and provides live AI recommendations (ADD/HOLD/REDUCE/SELL) on open holdings.

---

## 🖥 The Frontend React Architecture

The frontend is a single-page application built with Vite, React, and Vanilla CSS (`index.css` for highly customized, vibrant, modern "glassmorphism" styling).

### State Management (`store.js`)
It relies on **Zustand**—a lightweight, fast state manager—to hold the globally selected `ticker`, `timeframe`, and the massive JSON responses returned by the backend. This prevents prop-drilling across the 5 complex dashboard tabs.

### The Dashboard Tabs Structure
* `App.jsx` handles the global header, symbol input, timeframe selection, and tab routing.
* **🔍 Analyze Tab**: Renders the immediate outputs of a single pipeline run.
    * `CandlestickChart.jsx`: Recharts-based price graph.
    * `AgentSignals.jsx`: Visualizes technical agent confidences.
    * `StateVector.jsx` & `DisagreementHeatmap.jsx`: Shows the raw 16-D tensor fed to the RL neural network.
    * `SHAPPanel.jsx`: Explainable AI (XAI) feature importance attribution.
* **🧠 RL Brain Tab**: Inspects the internal neurology of the RL models. Visualizes the continuous reward policy, entropy decay curves (exploration vs exploitation), and the Multi-Expert regime shifts.
* **📊 Compare Tab**: Submits 2-5 ticker symbols simultaneously. Compares their metrics side-by-side and uses an inverse-volatility + RL-confidence weighting to suggest an optimal Portfolio allocation.
* **📈 Backtest Tab**: Where Walk-Forward cross-validation and historical Stress Tests are executed and equity charts are analyzed.
* **💼 Paper Trade Tab (`PaperTradeTab.jsx`)**: The interactive dashboard to execute mock trades (BUY/SELL), manage the $100k virtual portfolio, view total portfolio heat, and poll the AI for active trade management.

---

## 🚀 The End-To-End "Inside Out" Lifecycle

Here is what happens inside the codebase when you type "AAPL" and click "Analyze":
1. **[UI]** `App.jsx` calls `handleAnalyze()` and updates state via Zustand (`setLoading(true)`). 
2. **[API]** `axios.get('/analyze/AAPL?timeframe=1d')` hits `main.py` in the backend.
3. **[Data]** `data_fetcher.py` downloads historical OHLCV data using `yfinance`.
4. **[Inference]** Sequential execution: `indicator` → `pattern` → `trend` → `regime`.
5. **[Disagreement]** `disagreement_model.py` maps variance.
6. **[RL Brain]** `rl_meta_controller.py` compiles the state vector, checks the regime (e.g., "Trending"), loads the Trending PPO model, evaluates the state, and outputs `action = 0.85` (Strong LONG).
7. **[Risk]** `risk_management_agent.py` caps the trade size due to high market volatility to a max allocation of 6%.
8. **[Response]** Everything is bundled into a massive JSON payload carrying historical price arrays, weights, textual reasonings, and action commands, returned to the Frontend.
9. **[UI]** Zustand receives the payload. React triggers a re-render. `AgentSignals.jsx`, `SHAPPanel.jsx` and `CandlestickChart.jsx` parse the JSON to build their respective visualizations. 

This is the beauty of the platform: A modular, highly scalable Quant AI system disguised behind a slick interface.
