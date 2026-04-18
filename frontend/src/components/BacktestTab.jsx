import { useState, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * BacktestTab — Full Walk-Forward Validated Backtesting
 * ======================================================
 * - Three overlaid equity curves (RL, static, buy-and-hold)
 * - Walk-forward folds chart (OOS Sharpe per fold) — KEY
 * - Metrics table: per-fold + aggregate
 * - Crash Stress Test (30-day replay) — KEY SHOWSTOPPER
 */
export default function BacktestTab() {
  const [ticker,    setTicker]    = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1d');
  const [backtest,  setBacktest]  = useState(null);
  const [loading,   setLoading]   = useState(false);

  const fetchBacktest = useCallback(async () => {
    setLoading(true);
    try {
      const period = { '1m': '7d', '5m': '60d', '15m': '60d', '1h': '730d', '1d': '5y' }[timeframe] || '5y';
      const { data } = await axios.post(`${API}/backtest`, {
        symbol: ticker.toUpperCase(), timeframe, period
      }, { timeout: 300_000 });
      setBacktest(data);
    } catch (e) {
      console.error('Backtest failed:', e.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, timeframe]);

  const fetchStress = useCallback(async () => {
    // Stress test is already computed and returned in the /backtest endpoint
    // We just map it directly in the UI render.
  }, []);

  // Equity curve data
  const equityData = backtest ? (backtest.equity_curve || []).map((v, i) => ({ bar: i, value: v })) : [];

  // Walk-forward folds data
  const foldData = backtest?.walk_forward_folds || [];
  const wfSummary = backtest?.walk_forward_summary || {};
  const stress = backtest?.stress_test_results || null;
  const stLoading = loading;

  // Stress test data
  const stressSteps = stress?.steps || [];
  const stressSummary = stress?.summary || {};

  const m = backtest?.overall_metrics || {};

  const metricCard = (label, value, color) => (
    <div key={label} style={{ padding: '12px 16px', borderRadius: 10,
                              background: 'rgba(255,255,255,0.03)',
                              border: '1px solid rgba(255,255,255,0.07)',
                              display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 11, color: '#647091' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 800, fontFamily: 'monospace', color: color || '#e2e8f0' }}>
        {value}
      </div>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Controls */}
      <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span className="card-title" style={{ marginRight: 8 }}>📊 Backtest</span>
        <input id="backtest-ticker-input" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
               style={{ width: 90, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: 8, padding: '7px 12px', color: '#fff', fontSize: 13 }} />
        <select id="backtest-tf-select" value={timeframe} onChange={e => setTimeframe(e.target.value)}
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                         borderRadius: 8, padding: '7px 12px', color: '#fff', fontSize: 13 }}>
          {['1m','5m','15m','1h','1d'].map(tf => <option key={tf} value={tf}>{tf}</option>)}
        </select>
        <button id="run-backtest-btn" className="btn-analyze" onClick={fetchBacktest} disabled={loading} style={{ marginLeft: 'auto' }}>
          {loading ? 'Running…' : '▶ Run Backtest & Stress Test'}
        </button>
      </div>

      {/* Metrics Summary */}
      {backtest && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {metricCard('Annualized Ret', `${((m.annualized_ret || 0) * 100).toFixed(2)}%`, (m.annualized_ret || 0) >= 0 ? '#68d391' : '#fc8181')}
          {metricCard('Sharpe Ratio',   m.sharpe?.toFixed(4),                 m.sharpe >= 1 ? '#68d391' : '#f6e05e')}
          {metricCard('Sortino Ratio',  m.sortino?.toFixed(4),                '#63b3ed')}
          {metricCard('Calmar Ratio',   m.calmar?.toFixed(4),                 '#68d391')}
          {metricCard('Max Drawdown',   `${((m.max_drawdown || 0) * 100).toFixed(2)}%`,    '#fc8181')}
          {metricCard('Win Rate',       `${((m.win_rate || 0) * 100).toFixed(1)}%`,         '#f6e05e')}
          {metricCard('Profit Factor',  m.profit_factor?.toFixed(3),                '#a78bfa')}
          {metricCard('Trades',         m.n_trades,                                  '#a0aec0')}
        </div>
      )}

      {/* Equity Curve */}
      {equityData.length > 0 && (
        <div className="card">
          <div className="card-header"><span className="card-title">📈 Equity Curve</span></div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={equityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="bar" tick={{ fill: '#647091', fontSize: 10 }} />
              <YAxis tick={{ fill: '#647091', fontSize: 10 }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
              <Tooltip contentStyle={{ background: 'rgba(26,27,58,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                       formatter={v => [`$${v?.toFixed(0)}`]} />
              <ReferenceLine y={100_000} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="value" stroke="#6366f1" dot={false} strokeWidth={2.5} name="RL Portfolio" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Walk-Forward Folds — KEY */}
      {foldData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">🔄 Walk-Forward OOS Sharpe</span>
            <div style={{ fontSize: 11, color: '#8b9fc0' }}>
              Mean: <span style={{ color: '#68d391', fontWeight: 700 }}>{wfSummary.mean_test_sharpe?.toFixed(4)}</span>
              &nbsp;±&nbsp;{wfSummary.std_test_sharpe?.toFixed(4)}
              &nbsp;|&nbsp;Worst Fold: {wfSummary.worst_test_sharpe?.toFixed(3)}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={foldData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="fold" tick={{ fill: '#647091', fontSize: 10 }} label={{ value: 'Fold', position: 'insideBottom', fill: '#647091', fontSize: 10 }} />
              <YAxis tick={{ fill: '#647091', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: 'rgba(26,27,58,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                       formatter={(v, n) => [v?.toFixed(4), n]} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
              <Bar dataKey="test_sharpe"  fill="#6366f1" radius={[4,4,0,0]} name="OOS Sharpe" />
            </BarChart>
          </ResponsiveContainer>
          {/* Fold table */}
          <div style={{ overflowX: 'auto', marginTop: 12 }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'separate', borderSpacing: '0 3px' }}>
              <thead>
                <tr style={{ color: '#647091' }}>
                  {['Fold','Train Bars','Test Bars','Sharpe','Sortino','CVaR(95%)','Drawdown'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '4px 8px' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {foldData.map(f => (
                  <tr key={f.fold} style={{ background: 'rgba(255,255,255,0.02)' }}>
                    <td style={{ padding: '5px 8px', color: '#e2e8f0', fontWeight: 700 }}>{f.fold}</td>
                    <td style={{ padding: '5px 8px', color: '#8b9fc0' }}>{f.train_bars}</td>
                    <td style={{ padding: '5px 8px', color: '#8b9fc0' }}>{f.test_bars}</td>
                    <td style={{ padding: '5px 8px', fontFamily: 'monospace', color: f.test_sharpe >= 0 ? '#68d391' : '#fc8181' }}>
                      {f.test_sharpe?.toFixed(4)}
                    </td>
                    <td style={{ padding: '5px 8px', fontFamily: 'monospace', color: '#63b3ed' }}>{f.test_sortino?.toFixed(4)}</td>
                    <td style={{ padding: '5px 8px', fontFamily: 'monospace', color: '#fc8181' }}>
                      {(f.test_cvar * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '5px 8px', fontFamily: 'monospace', color: '#fc8181' }}>
                      {(f.test_maxdd * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Historical Stress Tests */}
      {stress && Object.keys(stress).length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">⚡ Extreme Historical Scenarios</span>
            <span style={{ fontSize: 11, color: '#647091' }}>
              Vol shocks & widened proxy spreads applied to strategy PnL
            </span>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            {Object.entries(stress).map(([key, sc]) => (
              <div key={key} style={{
                padding: 12, borderRadius: 10, background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.07)',
              }}>
                <div style={{ color: '#e2e8f0', fontWeight: 700, marginBottom: 6, fontSize: 12 }}>{sc.scenario || key}</div>
                
                <div style={{ display: 'flex', gap: 16 }}>
                  <div>
                    <div style={{ fontSize: 10, color: '#647091' }}>Stressed CAGR</div>
                    <div style={{ fontSize: 16, fontWeight: 800, color: (sc.annualized_ret || 0) >= 0 ? '#68d391' : '#fc8181', fontFamily: 'monospace' }}>
                      {((sc.annualized_ret || 0) * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: '#647091' }}>Max DD</div>
                    <div style={{ fontSize: 16, fontWeight: 800, color: '#fc8181', fontFamily: 'monospace' }}>
                      {((sc.max_drawdown || 0) * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
                
                <div style={{ fontSize: 11, color: '#8b9fc0', marginTop: 8 }}>
                  Spread Multiplier: <span style={{ color: '#f6e05e', fontWeight: 700 }}>{sc.spread_multiplier}x</span>
                  &nbsp;|&nbsp; Sharpe: <span style={{ color: '#63b3ed', fontWeight: 700 }}>{sc.sharpe?.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
