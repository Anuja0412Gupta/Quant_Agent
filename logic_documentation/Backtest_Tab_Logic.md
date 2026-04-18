# Backtest & Walk-Forward Validation Logic Breakdown
The **Backtest** tab (`BacktestTab.jsx`) determines if the RL model actually holds predictive trading power in the real world or if it just "memorized" past charts (Overfitting).

### 1. Core Metrics Summary
- **Backend Logic (`backtesting_engine.py`):** The system injects a simulated `$100,000` capital and replays the entire RL stepping logic.
- **Displayed Values:** Standard institutional measurements.
  - *Sharpe Ratio:* Return per unit of volatility.
  Higher = better risk-adjusted returns
  - *Sortino Ratio:* Similar to Sharpe, but only penalizes *downside* volatility.
  - *Calmar Ratio:* Calculates return relative to the Maximum Drawdown (peak-to-trough drop).

### 2. Equity Curves Overlaid Graph
- Charts the historical account balance of three different portfolios over identically matched timelines:
  - *Static Agent:* A basic rule-based strategy.
  - *Buy & Hold:* Holding the asset entirely without trading to map market baseline.
  - *RL Agent:* Displays the algorithm's capability to outperform baseline growth.

### 3. Walk-Forward OOS Folds (The Crux of System Integrity)
- **What it represents:** Proof that the model isn't "overfitted" to the training data.
- **Backend Logic:** Deep learning models can easily memorize historical charts. **Walk-Forward Analysis** fixes this. 
  - The algorithm actively segments historical timelines into "Folds".
  - It trains on Fold #1 (e.g. 2020), then immediately tests on unseen out-of-sample data in Fold #2 (2021). Then it shifts: trains on 2021, tests on 2022. 
- **The Graph:** Displays the Out-of-Sample (OOS) Sharpe generated for each individual unseen fold. If the Sharpe remains positively stable across multiple distinct folds in the bar chart, the dataset has successfully generalized and the logic is verified for actual trading usage.

### 4. Extreme Historical Scenario Stress Testing
- **Backend Logic:** Sometimes basic historical testing isn't extreme enough. The `stress_test_results` object within the `/backtest` API payload dynamically simulates flash crashes.
- It injects arbitrary multi-sigma volatility shocks and artificially multiplies backend transaction slippage (proxy spreads) up to 5x. By checking the simulated CAGR drop-off under extreme stress, researchers know strictly what total ruin boundaries apply.
