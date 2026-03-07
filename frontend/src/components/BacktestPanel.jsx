import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

/**
 * BacktestPanel
 * Displays backtest metrics tiles and an equity curve chart.
 */
function MetricTile({ label, value, colorClass }) {
  return (
    <div className="metric-tile">
      <div className={`mt-value ${colorClass}`}>{value}</div>
      <div className="mt-label">{label}</div>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1d2235', border: '1px solid rgba(99,120,255,0.2)', borderRadius: 8, padding: '8px 12px' }}>
      <p style={{ color: '#647091', fontSize: 11, marginBottom: 4 }}>Bar #{label}</p>
      <p style={{ color: '#00e5a0', fontSize: 13, fontFamily: 'JetBrains Mono', fontWeight: 600 }}>
        ${payload[0].value?.toLocaleString()}
      </p>
    </div>
  );
};

export default function BacktestPanel({ data }) {
  if (!data) return null;
  const { metrics, equity_curve, symbol, timeframe, period } = data;
  if (!metrics) return null;

  const {
    total_return, win_rate, sharpe_ratio,
    max_drawdown, profit_factor, n_trades, final_capital,
  } = metrics;

  const equityData = (equity_curve || []).map((v, i) => ({ bar: i, equity: v }));

  // Downsample for performance
  const step     = Math.max(1, Math.floor(equityData.length / 200));
  const sampled  = equityData.filter((_, i) => i % step === 0);

  return (
    <div>
      <div className="section-label">Backtest Results — {symbol} {timeframe} ({period})</div>
      <div className="card">
        <div className="metrics-grid">
          <MetricTile
            label="Total Return"
            value={`${(total_return * 100).toFixed(2)}%`}
            colorClass={total_return >= 0 ? 'mt-positive' : 'mt-negative'}
          />
          <MetricTile label="Win Rate"      value={`${(win_rate * 100).toFixed(1)}%`}   colorClass="mt-neutral" />
          <MetricTile
            label="Sharpe Ratio"
            value={sharpe_ratio?.toFixed(2)}
            colorClass={sharpe_ratio >= 1 ? 'mt-positive' : sharpe_ratio >= 0 ? 'mt-amber' : 'mt-negative'}
          />
          <MetricTile
            label="Max Drawdown"
            value={`${(max_drawdown * 100).toFixed(2)}%`}
            colorClass="mt-negative"
          />
          <MetricTile
            label="Profit Factor"
            value={profit_factor?.toFixed(2)}
            colorClass={profit_factor >= 1.5 ? 'mt-positive' : profit_factor >= 1 ? 'mt-amber' : 'mt-negative'}
          />
          <MetricTile label="Trades"        value={n_trades}                              colorClass="mt-neutral" />
          <MetricTile label="Final Capital" value={`$${final_capital?.toLocaleString()}`} colorClass="mt-positive" />
        </div>

        <div className="card-title" style={{ marginBottom: 12 }}>Equity Curve</div>
        <div className="equity-chart-wrapper">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sampled}>
              <CartesianGrid stroke="rgba(99,120,255,0.06)" />
              <XAxis
                dataKey="bar"
                tick={{ fill: '#647091', fontSize: 10 }}
                axisLine={{ stroke: 'rgba(99,120,255,0.15)' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#647091', fontSize: 10 }}
                axisLine={{ stroke: 'rgba(99,120,255,0.15)' }}
                tickLine={false}
                tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                width={58}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="equity"
                stroke="#4f7dff"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#4f7dff' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
