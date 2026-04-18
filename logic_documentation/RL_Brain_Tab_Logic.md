# RL Brain Tab Logic Breakdown
The **RL Brain** tab (`RLBrainTab.jsx`) maps directly to the active Reinforcement Learning loops happening inside Stable-Baselines3 (`rl_meta_controller.py`). It is the ultimate dashboard for quantitative researchers monitoring the "AI's mind."

### 1. The Action Gauge
- **What it represents:** The immediate mathematically-calculated trade size and direction vector (ranging from -1.0 to 1.0).
A number between -1 and +1
+1 → Full LONG (buy aggressively)
-1 → Full SHORT (sell aggressively)
0 → Do nothing (stay in cash)
- **Backend Logic:** The RL neural network (TCN-LSTM) consumes the 45-dimensional feature space and outputs a continuous variable. Strong floats near 1.0 indicate aggressive LONG positions, while -1.0 means aggressive SHORTs. Values near 0.0 indicate holding CASH (Flat). 

### 2. Disagreement Gate & Active Regime
- **What it represents:** Explaining how the raw RL action is reduced or choked out by external uncertainty.
- **Backend Logic:** The effective action sent to the brokerage is computed as: `Raw RL Action * Disagreement Multiplier`. 
- **Active Regime Policy:** Recharts boxes visually isolate whether the RL considers the environment `Trending` or `High_Volatility`. This directs which "head" of the Mixture-of-Experts policy the model queries.

Financial markets are non-stationary, so a single policy often fails across different conditions. By splitting the policy into regime-specific experts, we reduce policy interference and allow each expert to learn more efficiently. This leads to better generalization and more robust risk-adjusted returns.

### 3. Reward Curve Visualization
- **Backend Logic:** Polled continuously during backend training modes via `GET /rl/brain`. The Reward Curve tracks the historical PnL-adjusted return minus penalty subtractions over total training timesteps. If this curve diverges entirely downwards, the model is unstable.

### 4. Reward Ablation Study (The Core Statistical Proof)
- **What it represents:** Evaluates the necessity of every penalty mathematically imposed into the system.
- **Backend Logic (`reward_function.py`):** The RL is given rewards for Risk-Adjusted Returns. It is mathematically penalized computationally for Drawdowns ($-\lambda_1 dd^2$), High Volatility ($-\lambda_2 excess\_vol$), Transaction Costs ($-\lambda_3 friction$), and Rapid Overtrading ($-\lambda_4 churn$).
- **The Graph:** The backend processes an entire historical asset series 5 concurrent times. On each pass, it completely disables a single penalty ($\lambda = 0$). By charting the resulting Sharpe ratio, researchers can visually confirm that *all* penalties are required for maximizing stable PnL, as disabling any penalty results in a systemic drop in the final Sharpe ratio bar graphic.
