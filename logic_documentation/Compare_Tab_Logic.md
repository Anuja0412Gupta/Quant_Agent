# Compare Tab Logic Breakdown
The **Compare** tab (`CompareTab.jsx`) scales the system from analyzing single assets to determining portfolio allocation weights across multiple uncorrelated tickers simultaneously.

### 1. DCC-GARCH Allocation Matrix
- **What it represents:** Recommends exactly what percentage of cash should go into which stocks.
- **Backend Logic (`portfolio_manager.py`):** 
  - Standard algorithmic portfolios use "Markowitz Mean-Variance Optimization." However, volatility is not static.
  - The backend runs a **Dynamic Conditional Correlation (DCC-GARCH)** model. Unlike standard correlation constants, DCC-GARCH understands that during a massive market crash, historical correlations trend toward 1.0 as panic sets in universally.
  - The model calculates time-varying covariance matrices across the user's selected ticker basket and mathematically isolates optimal weights minimizing Conditional Value at Risk (CVaR).

### 2. Sentiment Tilts
- **What it is:** Pure math is not enough; sometimes News sentiment overshadows asset history.
- **Backend Logic:** The system retrieves real-time FinBERT scores for the selected basket via `data_fetcher.py`. Positive sentiment scores are mathematically transformed into a scalar that tilts the finalized GARCH weights up or down, effectively creating an allocation explicitly balancing both risk variances and public social hype.

### 3. Displayed Outputs
- **Expected Sharpe & CVaR:** Values indicating what risk metrics this specific basket represents.
- **Rebalance Cost:** Changing your portfolio costs money. The backend simulates the transaction fees explicitly required (slippage and percentage bases) to reach the new weights, presenting it in basis points (bps) to the user.
