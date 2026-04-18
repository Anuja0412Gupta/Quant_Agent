# QuantAgent Detailed Code Explanation

Here is a detailed, block-by-block explanation of the core files in the QuantAgent project.

## 1. [backend/config.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/config.py) (The Settings File)
This file acts as the central control panel for the entire backend. Whenever an agent needs a default number, it looks here. This makes it easy to tweak the AI's behavior without hunting through multiple files.

## 2. [backend/main.py](file:///c:/Users/Anuja/Desktop/QUANT/backend/main.py) (The API Server)
This is the gateway connecting the backend Python logic to the frontend website. It uses **FastAPI** to create web endpoints. It coordinates fetching data, running agents, and returning JSON.

---

## 3. The Specialized Analysts (`backend/agents/`)

These files act as individual trading experts looking at the same stock chart.
- **`indicator_agent.py`**: Computes pure momentum math (RSI, MACD) to see if an asset is overbought or oversold.
- **`pattern_agent.py`**: Scans the chart visually for geometric shapes (Double Tops, Bullish Flags) that historically predict price movements.
- **`trend_agent.py`**: Uses polynomial regression math to draw smooth curves and define if the overall market is in an uptrend or downtrend.
- **`market_regime_agent.py`**: Determines if the market is trending nicely or chopping sideways randomly by calculating the Hurst Exponent and Volatility ranges.

---

## 4. The Brains and Supervisors (`backend/agents/`)

These agents don't look at the stock chart directly; they look at the analysts' opinions and manage the risk.

### `decision_agent.py` & `disagreement_model.py`
```python
# Decision Agent
score  += agent_weight * agent_confidence * agent_direction

# Disagreement Model
disagreement_index = float(np.var(weighted_signals))
```
**Explanation:** The `decision_agent` gathers all outputs from the analysts and creates a weighted final score. If the score is high enough, it votes LONG. 
The `disagreement_model` measures how wildly the analysts' opinions vary. If half scream LONG and half scream SHORT, the mathematical variance spikes, and this script acts as a circuit breaker, ordering the system to skip the trade entirely.

### `risk_management_agent.py`
```python
kelly_shares = kelly_value / last_price
vol_shares, vol_pct = _volatility_position(atr, last_price, portfolio_value)
chosen_shares = min(kelly_shares, vol_shares)
```
**Explanation:** Once the decision to trade is made, this script calculates the exact number of shares to buy. It calculates two numbers:
1. The **Fractional Kelly Criterion** (a formula maximizing compound growth based on win rate).
2. The **Volatility size** (shrinking the bet if the market is currently moving wildly).
It takes the smaller of the two to be safe.

### `self_critique_agent.py` & `rl_meta_controller.py`
```python
# Critique Agent
correctness = _agent_was_correct(pred, trade_result)
delta       = _LEARNING_RATE * correctness * pnl_scale

# RL Meta Controller
reward = (pnl_pct / max(volatility, 1e-6)) - (drawdown * 2.0)
```
**Explanation:** This is the Machine Learning loop. After a trade finishes, the **Critique agent** looks back to see which analysts predicted the result correctly and boosts their "trust weights". 
The **RL Meta Controller** uses Proximal Policy Optimization (PPO), a heavy AI model, to observe the market state and dynamically spit out new weights and position multipliers to maximize rewards (profits minus the drawdowns).

---

## 5. Core Infrastructure (`backend/`)

- **`data_fetcher.py`**: Uses Python's `requests` library to directly hit Yahoo Finance's hidden v8 API. This downloads the actual OHLCV (Open, High, Low, Close, Volume) data. 
- **`backtesting_engine.py`**: A time machine. It loops over years of historical Yahoo data, feeding it bar-by-bar to the agents exactly as it would live. It then mathematically charts every simulated trade to output the bot's historical Win Rate, Max Drawdown, and Sharpe Ratio.

---

## 6. The User Dashboard (`frontend/src/`)

This is the React web app you see when you open the UI.

### `App.jsx`
```javascript
const { data } = await axios.get(`${API}/analyze/${ticker.toUpperCase()}`);
```
**Explanation:** The main layout page. It handles the input bar where you type 'AAPL', coordinates sending that request to the python server, and shows loading spinners while the AI thinks.

### `CandlestickChart.jsx` (Not shown previously, but core)
**Explanation:** Renders the actual interactive stock chart using the Lightweight Charts library by TradingView.

### `TradeDecision.jsx`
```javascript
<span className="score-value">{(confidence * 100)?.toFixed(1)}%</span>
```
**Explanation:** This component displays the final massive "LONG / SHORT" badge on your screen. It takes the output from the `decision_agent` and `risk_management_agent` to print out your exact entry price, stop-loss, take-profit, and how many shares to buy.

### `BacktestPanel.jsx`
```javascript
<MetricTile label="Total Return" value={`${(total_return * 100).toFixed(2)}%`} />
```
**Explanation:** This draws the historical backtesting results. It uses a graph library called `recharts` to plot the "Equity Curve" (a line showing your money growing or shrinking over time) and renders tiles for your simulated Win Rate and Max Drawdown.
