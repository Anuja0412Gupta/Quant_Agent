# Live Demo Tab Logic Breakdown
The **Live Demo** tab (`LiveDemoTab.jsx`) is a specialized React component built primarily to showcase how the frontend components visibly react to incoming streaming metrics when connected to a WebSocket or continuous polling setup.

### 1. Simulated Ticker Replay Mechanism
- **Frontend Logic:** Instead of maintaining a complex, constantly open financial data firehose stream (which would cost significant bandwidth and API limits), the component queries the standard `GET /analyze` backend route once.
- Once the historical array of OHLCV bars and associated signals are downloaded to the client browser, a JavaScript `setInterval` timer triggers (typically every ~1000ms).
- This timer mathematically truncates the larger array and injects the array slices sequentially into the downstream Recharts rendering views.

### 2. State Mutation Mocking
- **Visual Responsiveness:** As the artificial tick occurs, the component artificially mutates randomized perturbations onto the `agent_signals` logic and the VIX indices, rendering visual demonstrations of the gauges spinning and the text updating, verifying the sub-components are functionally decoupled to receive live changes over time.
