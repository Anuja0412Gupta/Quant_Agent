# QuantAgent — Architecture & Project Structure Overview

QuantAgent is an advanced, research-grade algorithmic trading platform. It relies on a rigorous micro-agent architecture running entirely on a Python FastAPI backend, coupled with a responsive, glassmorphic React front-end dashboard that allows developers and traders to visualize the pipeline's internal logic.

## 1. High-Level Architecture Flow

The backend processes data through a sophisticated pipeline before presenting actionable decisions to the React frontend.

```mermaid
flowchart TD
  subgraph External Data Sources
    YF[Yahoo Finance]
    NAPI[NewsAPI & PRAW Reddit]
    FRED[FRED Macro Data]
  end

  subgraph Backend Backend (FastAPI / Python)
    DF[Data Fetcher Pipeline]
    FP[Feature Engineering 45-Dim]
    
    subgraph Multi-Agent System
      Regime[Market Regime Agent HMM]
      Ens[DeepEnsemble Models]
      RL[RL Meta-Controller TCN+LSTM]
      Critic[Self-Critique Veto]
    end
    
    DF --> FP
    FP --> Regime
    FP --> Ens
    Regime --> RL
    Ens --> RL
    RL --> Critic
  end
  
  subgraph Frontend (React / Vite)
    App[App.jsx Main Application]
    Dash[Analyze Tab]
    PTR[Paper Trade Execution]
    RLBrain[RL Brain Insights]
  end

  YF --> DF
  NAPI --> DF
  FRED --> DF
  Critic --> |FastAPI JSON Response| App
  App --> Dash
  App --> PTR
  App --> RLBrain
```

---

## 2. Backend Project Structure (`/backend`)

The Python backend encapsulates the data processing, reinforcement learning models, API servers, and trading engine logic.

### API & Entry Points
- `main.py`: The entry point for the FastAPI server. Registers all critical endpoints (e.g., `/analyze`, `/price`, `/portfolio/trade`, `/backtest`).
- `config.py`: Centralized environment configurations globally defining variables (e.g., feature burn-in length, slippage bounds, commission percentages).
- `shared_types.py` & `schemas.py`: Heavily utilized Pydantic validation schemas defining exactly what JSON shapes the frontend should expect and enforcing strict typing.

### `/agents` (The Brain Trust)
Contains the atomic micro-models used to calculate specific properties of an asset before it reaches final RL gating.
- `market_regime_agent.py`: Uses a Hidden Markov Model (HMM) running parallel to Bayesian Online Changepoint Detection (BOCPD) to label the market into "Trending", "Mean Reverting", or "High Volatility".
- `indicator_agent.py`: Calculates technical trading indicator formulas.
- `reward_function.py`: Governs the crucial reward shaping mathematics for RL, including penalties for profound drawdowns, unnecessary volatility, and transaction friction.
- `self_critique_agent.py`: A final fail-safe mechanism that uses SHAP Values and conditional threshold gating to scale down sizing or fully veto dangerous AI trades.

### `/data` & `/features` (Ingestion & Processing)
- `data_fetcher.py`: Standardizes data connections retrieving prices, SEC 8K/Earnings dates, Reddit sentiment density, and FRED macroscopic data.
- `feature_pipeline.py`: A massive preprocessing pipeline expanding standard OHLCV prices into a fully standardized, 45-dimensional matrix devoid of look-ahead bias spanning sentiment indicators and momentum metrics. 

### `/environment` & `/training` (Reinforcement Learning)
- `trading_env.py`: Builds a custom Gymnasium environment defining how an autonomous agent is penalized and rewarded when allocating fractional components of its capital through historical timeseries.
- `trainer.py`: Drives the Stable-Baselines3 execution loop. Runs through progressive Curriculum stages forcing actors to master simple Trending models before battling High Volatility scenarios.
- `/rl` modules contain the `RecurrentPPO` architectures and Lagrangian constraint frameworks.

### `/backtest` & `/portfolio` (Verification)
- `backtesting_engine.py`: Runs a continuous walk-forward (OOS) fold test producing stress-tested Sharpe, Sortino ratios, and peak-to-trough Drawdowns.
- `portfolio_manager.py`: Drives quantitative weight allocations utilizing DCC-GARCH models to scale portfolio capital efficiently across multiple uncorrelated equity tickers.

---

## 3. Frontend Project Structure (`/frontend`)

The frontend is a Vite-powered React application with an emphasis on "Glassmorphism," utilizing modern visualizations bridging complex quantitative algorithms to user-friendly outputs.

### Application Core
- `App.jsx`: The top-level React component handling primary navigation, caching overarching ticker selections (`AAPL`), state variables, and housing the `/analyze` API execution.
- `index.css`: Defines the global design variables, modern gradient aesthetics, layout breakpoints, and UI animations.
- `store.js`: Potential centralized state (e.g., Zustand or Redux layer) to cache cross-tab analytics if needed.

### `/components` (The UI Dashboards)
Every complex subset of the dashboard has been segmented into modular panels for ease-of-maintenance.
- **Visual Analytics:**
  - `CandlestickChart.jsx`, `SHAPPanel.jsx`, `RegimePanel.jsx`: Components rendering dense timeseries charts and visualizations analyzing agent reasoning via Recharts libraries.
- **Tabs / Workspaces:**
  - `AnalyzeTab`: Aggregates the visual analytics into a comprehensive overview of a given symbol.
  - `PaperTradeTab.jsx`: Simulates live trading functionality. Allows the user to simulate PnL executing BUY/SELL orders connected to a local internal simulated brokerage via the backend.
  - `RLBrainTab.jsx`: A unique workspace allowing developers to actively train new RL systems and run dynamic Reward Ablation Studies to measure penalty impacts via interactive BarCharts.
  - `AISuggestionsTab.jsx`: An accessibility-driven UI piece extracting dense machine outputs (`gate_values`, `disagreements`) and parsing them into simple, human readable plain-English sentiment ("The Agent recommends holding Apple because...").
  - `BacktestTab.jsx` & `WalkForwardPanel.jsx`: Visualizes automated Walk-Forward verification matrices determining system survivability across rolling historical testing epochs.
