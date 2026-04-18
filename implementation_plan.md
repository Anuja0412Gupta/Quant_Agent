# QuantAgent — Production-Grade RL Platform Upgrade

## Overview

This plan upgrades QuantAgent from a weight-adjusting meta-controller into a **production-grade, research-backed RL-driven trading platform**. RL becomes the *primary decision-maker* (not just weight adjuster). The system becomes statistically valid, fully explainable, and research-interview-ready.

**Full data flow (new):**
```
Agents → Feature Vector → RL Policy (regime-selected MoE) → Disagreement Gate → Risk Agent (VaR) → Trade Execution
```

---

## User Review Required

> [!IMPORTANT]
> **Alpaca / Zerodha API keys** — Paper trading (Upgrade 15) requires API credentials. We can stub these out as optional and enable them via environment variables; no blocker to ship without them initially.

> [!IMPORTANT]
> **Training compute** — Walk-forward validation (8–10 folds × 3 regime policies) is CPU-intensive. On the current setup, this may take 5–15 min. We can run this as a background job endpoint (`POST /rl/walk_forward`) so the UI doesn't block.

> [!WARNING]
> **Breaking API changes** — Several existing endpoints will gain new response fields. The frontend will require new tabs (RL Brain, Compare, Live Demo) and significant state restructuring. Zustand will be introduced for shared state. This is a significant rewrite of the React frontend.

> [!WARNING]
> **`self_critique_agent.py`** — This agent currently adjusts weights manually (it's redundant after the RL redesign). It will be demoted to a logging-only diagnostic tool rather than deleted, preserving backward compatibility.

---

## Proposed Changes

### PHASE 1 — Backend: Core RL Redesign

#### [MODIFY] [rl_meta_controller.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/agents/rl_meta_controller.py)

**Biggest change in the entire upgrade.** RL is redesigned from weight-adjuster to primary portfolio decision-maker.

- **Action space**: Continuous `[-1, +1]` (position size: `-1` = full short, `0` = neutral, `+1` = full long) + optional risk factor
- **State vector (expanded, ~14–16 dims)**: All agent signals + confidences, ATR, Hurst, disagreement score (continuous), rolling portfolio state (cash, drawdown)
- **Normalization**: All inputs Z-score or MinMax normalized point-in-time only (NO full-dataset normalization — lookahead bias prevention)
- **Output**: `rl_action ∈ [-1, +1]`, `position_size ∈ [0, 1]`, `risk_factor`
- **Disagreement gating**: `effective_action = rl_action × (1 − α × disagreement_score)` (Bayesian RL "uncertainty-aware position sizing")
- **MoE (Mixture of Experts)**: Train separate PPO policies per regime (trending, mean-reverting, high-volatility); MarketRegimeAgent selects the active policy

---

#### [NEW] [reward_function.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/agents/reward_function.py)

QUANT-level reward function (publishable-grade):

```
reward = return − λ₁×drawdown_penalty − λ₂×volatility_penalty − λ₃×transaction_cost − λ₄×overtrading_penalty
```

- Real fee structures: Zerodha brokerage `0.03%` + STT + volume-based slippage
- RL must learn cost-reduction *emergently* from reward — not via hard rules
- **Reward ablation study**: Run 5 variants removing one penalty term at a time; store results for comparison bar chart in RL Brain tab

---

#### [NEW] [feature_engineering.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/agents/feature_engineering.py)

Centralized, point-in-time-correct feature pipeline:

- Removes correlated/redundant features
- All normalization is rolling-window (Z-score or MinMax) using only past data
- State vector: agent signals + confidences, ATR volatility, disagreement score, Hurst, rolling returns, portfolio state
- Explicit docstring: "NO full-dataset normalization used anywhere in this pipeline" (interview proof)

---

#### [MODIFY] [disagreement_model.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/agents/disagreement_model.py)

- Extend from binary (HIGH/LOW) to **continuous scaling**
- Output `disagreement_score ∈ [0, 1]` (continuous, not just threshold-based)
- Feeds into `effective_action = rl_action × (1 − α × disagreement_score)`
- Label in UI: "Uncertainty-Aware Position Sizing (Bayesian RL)"

---

#### [MODIFY] [risk_management_agent.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/agents/risk_management_agent.py)

Complete risk layer upgrade:
- Stop-loss + trailing stop
- Position limits
- **VaR computation** (rolling 95th percentile of returns)
- **Hard override** of unsafe RL actions before execution
- Volatility-based exposure scaling
- `current_var`, `stop_alert` fields added to output

---

#### [MODIFY] [backtesting_engine.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/backtesting/backtesting_engine.py)

Complete walk-forward validation:
- Rolling 3-month train / 1-month test across 8–10 windows
- Per-fold OOS Sharpe ratio + performance distribution
- Mean ± std of all metrics across folds
- Fix: switch to fractional commission model (Zerodha: 0.03% brokerage + STT)
- Return `walk_forward_folds` array in response
- Add Sortino ratio, CAGR, per-ablation metrics

---

#### [MODIFY] [main.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/main.py)

New endpoints:
- `GET /compare?stocks=AAPL,TSLA,NVDA` — Multi-stock comparison (returns/vol/Sharpe/MDD/win rate/beta)
- `POST /rl/walk_forward` — Trigger walk-forward validation (background task)
- `POST /rl/ablation` — Run reward ablation experiment
- `GET /rl/brain` — RL policy state: action, regime, disagreement gate value, reward curve, SHAP
- `POST /portfolio/build` — Demo Portfolio Builder (2–5 stocks, RL suggests weights + reasoning)
- `GET /portfolio/compare` — Portfolio comparison (equity curves for RL / equal-weight / buy-and-hold)
- `POST /stress_test` — March 2020 crash replay (step-by-step 30-day simulation)
- `POST /paper_trade` — Alpaca/Zerodha paper trade (optional, env-var gated)

---

### PHASE 2 — Backend: Explainability

#### [NEW] [shap_explainer.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/agents/shap_explainer.py)

- SHAP values per agent per decision
- Wraps RL policy forward pass with `shap.KernelExplainer`
- Returns `shap_values` dict: which agent features drove the RL action
- Used in Analyze tab + RL Brain tab

---

### PHASE 3 — Frontend: Full 5-Tab Redesign

Current frontend: 1 single-page app with basic chart + signals.
New frontend: **5 tabs with Zustand shared state**, completely rebuilt.

#### [NEW] store.js — Zustand Global State

Shared state: ticker, timeframe, analysis, backtest, rl_brain, portfolio, folds

---

#### TAB 1 — Analyze Tab

**[MODIFY] [AgentSignals.jsx](file:///c:/Users/Anuja/Desktop/QUANT/frontend/src/components/AgentSignals.jsx)**
- Add per-agent confidence scores
- Full agent signal display (RSI, MACD, trend, regime)

**[NEW] StateVector.jsx** — Full RL input display: normalized signals, ATR, Hurst, disagreement, portfolio state

**[NEW] DisagreementHeatmap.jsx** — Color-coded matrix showing which agents conflict (feeds disagreement gating formula)

**[NEW] WhatIfPanel.jsx** — "Perturbation-based attribution" panel
- Flip one agent signal → re-run RL forward pass → show how allocation changes
- Label: "What If" (NOT "causal counterfactual")

**[NEW] SHAPPanel.jsx** — SHAP values per agent per decision; shows which agents drove the RL action

---

#### TAB 2 — RL Brain Tab

**[NEW] RLBrainTab.jsx**

- **Action Output**: Live gauge showing `rl_action ∈ [-1, +1]`, `position_size ∈ [0, 1]`, risk factor
- **Reward Curve**: Training history — rolling reward, entropy, policy loss over time
- **Reward Ablation Chart** *(KEY showstopper)*: Bar chart comparing Sharpe across 5 ablation variants (remove each penalty term one at a time). Target: "Without drawdown penalty, Sharpe 1.8→ drops. With it: 1.8→2.4 during March 2020."
- **Disagreement Gate**: Live visualization: `effective_action = rl_action × (1 − α × disagreement_score)`; Continuous scaling display
- **Policy Reasoning**: Natural language trace of RL decision logic + active regime policy
- **Regime Overlay**: Which PPO policy is active (trending/mean-reverting/high-volatility) with confidence + background color

---

#### TAB 3 — Compare Tab

**[NEW] CompareTab.jsx**

- **Multi-Stock Table**: Side-by-side returns (multi-timeframe), volatility, Sharpe, max drawdown, win rate, beta vs Nifty/S&P
- **Correlation Matrix**: Pairwise heatmap + portfolio diversification score
- **AI Allocation** *(NOVEL)*: RL runs on each stock → outputs allocation weights with confidence + reasoning narrative. API: `/compare?stocks=AAPL,TSLA,NVDA`
- **Demo Portfolio Builder** *(KEY)*: User selects 2–5 stocks, RL suggests weights, user can adjust, system simulates portfolio equity curve vs equal-weight baseline. Shows live Sharpe, MDD, CAGR, win rate. Beta vs Index — rolling beta, alpha, information ratio vs benchmark (Nifty/S&P)

---

#### TAB 4 — Backtest Tab

**[NEW] BacktestTab.jsx** (replaces `BacktestPanel.jsx`)

- **Equity Curve**: Three overlaid curves — RL portfolio, static-weights, buy-and-hold; action timeline overlaid on price
- **Walk-Forward Folds** *(KEY)*: Rolling 3-month train / 1-month test. Plot OOS Sharpe per fold + performance distribution. Mean ± std of all metrics. Directly kills the overfitting objection
- **Metrics Table**: Per-fold AND per-ablation variant: Sharpe, Sortino, MDD, CAGR, win rate
- **Rolling Sharpe**: Rolling Sharpe with background color = active regime. Shows strategy consistency over time
- **Crash Stress Test** *(KEY SHOWSTOPPER)*: "Stress Test: March 2020" button. Step-by-step 30-day replay showing: when VaR override fired, how drawdown cap limited loss, RL vs buy-and-hold equity curve

---

#### TAB 5 — Live Demo Tab

**[NEW] LiveDemoTab.jsx**

- **Step-by-Step Replay**: Sequential flow visualization: Input stock(s) → Agent outputs + confidence → Normalized RL state vector → RL action + allocation % + policy reasoning → Risk gate decision → Trade execution
- **Portfolio Comparison** *(PRIMARY SHOWSTOPPER)*: During tick-level replay, three live equity curves update simultaneously: RL portfolio, equal-weight, buy-and-hold. Side-by-side metrics table updating each tick
- **Live Portfolio**: Real-time state — open positions, unrealized P&L, total exposure, cash
- **Risk Dashboard**: Live VaR gauge, active stop-loss levels, current drawdown vs drawdown cap, stop alerts
- **Alpaca Paper Trading** (optional, env-var gated): Alpaca/Zerodha API with latency handling, retry logic, failure recovery. Real order feed optional

---

#### [MODIFY] [App.jsx](file:///c:/Users/Anuja/Desktop/QUANT/frontend/src/App.jsx)

- Replace single-page rendering with 5-tab navigation
- Integrate Zustand store
- Shared header with ticker/timeframe input

#### [MODIFY] [index.css](file:///c:/Users/Anuja/Desktop/QUANT/frontend/src/index.css)

- Design system expansion: tab navigation styles, gauge components, heatmap cells, regime color overlays, 3-curve equity chart styles

---

### PHASE 4 — Dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/Anuja/Desktop/QUANT/backend/requirements.txt)

Add:
- `shap` — SHAP explainability
- `alpaca-trade-api` — Paper trading (optional)
- `scipy` — Statistical tests for walk-forward validation

#### [MODIFY] package.json (frontend)

Add:
- `zustand` — Global shared state
- `recharts` or `chart.js` — For reward curve, ablation bar chart, rolling Sharpe (lightweight, already likely available)

---

## My Suggested Improvements (Beyond the Spec)

> [!TIP]
> **1. Point-in-Time Lookahead Audit Script** — Add `feature_audit.py` that automatically checks every feature computation for lookahead bias. Run this at startup. This is a single-function addition but extremely valuable for interviews ("we have automated lookahead protection").

> [!TIP]
> **2. Confidence Interval on Sharpe** — In walk-forward results, add bootstrapped 95% CI on mean Sharpe across folds. One extra `scipy.stats.bootstrap` call, but makes the results publishable-grade.

> [!TIP]
> **3. Policy Entropy Monitoring** — Track PPO policy entropy over training. Collapsing entropy = policy is becoming deterministic / overfit. Show this in the RL Brain reward curve chart as a second axis.

> [!TIP]
> **4. Regime Detection Confidence Threshold** — Only switch to a regime-specific policy if `regime_confidence > 0.6`. Otherwise use a fallback "general" policy. Prevents thrashing between regimes on ambiguous days.

> [!TIP]
> **5. Transaction Cost Calibration** — Add a `cost_calibration.py` utility that lets users input their actual broker fee structure (Zerodha/Alpaca/IBKR) and it auto-tunes the `λ₃` penalty weight. Makes the platform genuinely broker-agnostic.

---

## Verification Plan

### Automated Tests
- `POST /rl/ablation` → validate 5 ablation results in response
- `POST /rl/walk_forward` → validate 8+ folds in response, OOS Sharpe per fold
- `GET /compare?stocks=AAPL,TSLA,NVDA` → validate weights sum to ~1.0, correlation matrix dimensions
- `POST /stress_test` → validate March 2020 replay has 30 steps, VaR events marked
- Feature audit: run `feature_audit.py` on full pipeline — zero lookahead violations expected

### Manual Verification (Browser)
- RL Brain tab: Ablation bar chart loads and shows Sharpe improvement with drawdown penalty
- Backtest tab: Walk-forward folds chart shows OOS distribution
- Live Demo tab: Three equity curves update simultaneously during tick replay
- Analyze tab: SHAP values and What-If panel respond to agent signal flips
- Compare tab: Portfolio Builder simulates equity curve vs equal-weight baseline

---

## Implementation Phases & Priority Order

| Phase | Components | Priority |
|-------|-----------|----------|
| 1 | `feature_engineering.py`, `reward_function.py`, RL redesign | 🔴 Critical |
| 2 | Walk-forward validation, risk agent VaR, disagreement gating | 🔴 Critical |
| 3 | New API endpoints (`/compare`, `/rl/brain`, `/stress_test`, `/portfolio`) | 🟠 High |
| 4 | SHAP explainer | 🟠 High |
| 5 | Frontend: Zustand + 5-tab layout + RL Brain tab + Backtest tab | 🟠 High |
| 6 | Frontend: Compare tab + Live Demo tab + Analyze tab enhancements | 🟡 Medium |
| 7 | Alpaca paper trading integration | 🟢 Optional |
| 8 | Suggested improvements (audit script, CI on Sharpe, entropy monitoring) | 🟢 Optional |
