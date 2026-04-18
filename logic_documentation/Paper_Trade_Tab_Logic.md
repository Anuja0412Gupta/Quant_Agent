# Paper Trade Execution Tab Logic Breakdown
The **Paper Trade** tab (`PaperTradeTab.jsx`) acts as a simulated Local Broker. It validates trading behavior before connecting actual API keys to platforms like Alpaca or Zerodha. 

### 1. Market Value & Unrealized PnL Constraints
- **What it represents:** The fluctuating valuation of the stocks currently held relative to active price points.
- **Frontend Logic:**
  - On render, the React frontend pings the ultra-fast backend `GET /price/{symbol}` cache endpoints across every unique ticker identified inside the user's portfolio holding array.
  - The live price is measured against the historical `avg_cost` execution price to synthesize an `Unrealized PnL` and an overall floating Market Value. 
  - To bypass aggressive internet browser GET caching mechanisms, a UNIX Timestamp parameter (`?_t=${Date.now()}`) overrides browser limitations ensuring Live Prices continuously stream accurate fractional differentials.

### 2. Trade Execution Pipeline 
- **Backend Validation (`main.py`):**
  - Purchasing logic triggers `POST /portfolio/trade`. The backend checks available `portfolio["cash"]` limits, prohibiting leveraged buys.
  - Upon successful validation, the backend mathematically recalculates `avg_cost` (Dollar Cost Averaging) across the new volume injected into the ledger, then dynamically stores these representations inside a persistent external `out.json` file storage buffer.
  - A unique transactional `trade_id` is assigned and appended to the visible "Trade History" ledger.
  - When actively selling, the engine zeroes out the `avg_cost` and formally locks integer discrepancies directly into `portfolio["realized_pnl"]`, officially realizing the gains/losses into pure cash.
