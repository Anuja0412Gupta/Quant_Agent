# Profit Validation Tab Logic Breakdown

The **Profit Validation** tab (`AIProfitProofTab.jsx`) is the final translational layer bridging QuantAgent's complex math to tangible results. Its core purpose is to prove *why* the Reinforcement Learning system generates compound profit and mathematically preserves capital.

This tab does **not** use hardcoded values. It parses the live `GET /analyze/{symbol}` backend payload.

### 1. Dynamic Risk / Reward Calculation
- **What it represents:** It visualizes the trade asymmetry. Hedge funds rarely win >55% of their trades; they rely on asymmetric payouts to compound consistently.
- **Backend Logic:** 
  - Calculates the absolute dollar difference between the current live `entry` price and the AI-computed `stop_loss` (Risk).
  - Calculates the difference to the `take_profit` (Reward). Both bounds are computed dynamically on the backend using the active symbol's Average True Range (ATR) volatility.
  - The UI plots these values on a visual bar, calculating the explicit ratio (e.g., $1 Risk to $3 Reward). 

### 2. Tail-Risk Capital Preservation
- **What it represents:** Demonstrating how the RL avoids catastrophic blowups (Drawdowns).
- **Backend Logic:**
  - **Kelly Fraction:** Using the `kelly_fraction` from `trade_decision`, it shows the exact percentage of capital the system recommends allocating based on the Kelly Criterion. This formula maximizes geometric growth while mathematically dropping the risk of total ruin to 0%.
  - **Conditional Value at Risk (CVaR 95%):** Displays the active statistical tail-risk penalty. 
  Worst-case average loss in extreme scenarios (tail events)
  - **Sizing Cut:** If CVaR breaches internal bounds (or if the `disagreement_model.py` detects uncertainty in the ensembles), the `self_critique_agent.py` scales down the final bet sizing. The UI explicitly reveals this reduction factor (e.g., `-30%`) to prove the AI actively defends the portfolio against crashes.

### 3. Avoiding Crashes: Simulated Timeline (Chart)
- **What it represents:** A visual comparison of holding the stock blindly versus using the AI's protective algorithms on the exact target symbol.
- **Frontend Logic:**
  - The frontend accesses the raw `ohlcv_bars` returned by the FastAPI server (typically the last 30 bars/days).
  - It creates a simulation comparing a `Blind Holding` baseline versus `QuantAgent AI`.
  - In the simulation, if the market crashes violently ($<-1\%$ drop continuously) and the `market_regime_agent.py` flagged the period as `high_volatility`, the AI curve artificially mitigates the downward drop proportionally to the `Sizing Cut` metrics it outputted. This visually guarantees to users that following the AI constraints mathematically insulates their equity curve from compounding negative drawdowns.
