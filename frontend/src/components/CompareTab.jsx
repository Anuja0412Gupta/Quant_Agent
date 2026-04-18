import { useState, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, ResponsiveContainer, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
  ScatterChart, Scatter, Cell,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * CompareTab — Multi-Stock Comparison + Portfolio Builder
 * ========================================================
 * - Side-by-side metrics table
 * - Correlation matrix heatmap
 * - AI allocation weights (RL-driven)
 * - Demo Portfolio Builder: user selects stocks, RL suggests weights,
 *   shows equity curve vs equal-weight baseline
 */
export default function CompareTab() {
  const [stocks,      setStocks]      = useState('AAPL,MSFT,NVDA');
  const [compareData, setCompareData] = useState(null);
  const [portfolio,   setPortfolio]   = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [ptLoading,   setPtLoading]   = useState(false);

  const fetchCompare = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API}/compare`, {
        params: { stocks, timeframe: '1d', period: '1y' },
        timeout: 120_000,
      });
      setCompareData(data);
    } catch (e) {
      console.error('Compare fetch failed:', e.message);
    } finally {
      setLoading(false);
    }
  }, [stocks]);

  const fetchPortfolio = useCallback(async () => {
    setPtLoading(true);
    try {
      const tickers = stocks.split(',').map(s => s.trim().toUpperCase());
      const { data } = await axios.post(`${API}/portfolio/build`, {
        stocks: tickers, timeframe: '1d',
      }, { timeout: 120_000 });
      setPortfolio(data);
    } catch (e) {
      console.error('Portfolio fetch failed:', e.message);
    } finally {
      setPtLoading(false);
    }
  }, [stocks]);

  // Portfolio equity curve
  const equityData = portfolio ? (portfolio.equity_rl || []).map((v, i) => ({
    day: i,
    RL:         v,
    EqualWeight: portfolio.equity_equal?.[i],
    BuyHold:     portfolio.equity_bh?.[i],
  })) : [];

  // Correlation matrix cells
  const corr = compareData?.correlation || {};
  const corrTickers = Object.keys(corr);

  const corrColor = (val) => {
    if (val === undefined || val === null) return 'rgba(255,255,255,0.05)';
    const abs = Math.abs(val);
    if (abs > 0.7) return val > 0 ? 'rgba(104,211,145,0.4)' : 'rgba(252,129,129,0.4)';
    if (abs > 0.4) return val > 0 ? 'rgba(104,211,145,0.2)' : 'rgba(252,129,129,0.2)';
    return 'rgba(246,224,94,0.1)';
  };

  const dirIcon  = d => ({ LONG: '▲', SHORT: '▼', FLAT: '—' }[d] ?? '—');
  const dirColor = d => ({ LONG: '#68d391', SHORT: '#fc8181', FLAT: '#f6e05e' }[d] ?? '#a0aec0');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Controls */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">🔍 Multi-Stock Compare</span>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            id="compare-stocks-input"
            value={stocks}
            onChange={e => setStocks(e.target.value)}
            placeholder="AAPL,TSLA,NVDA"
            style={{ flex: 1, minWidth: 200, background: 'rgba(255,255,255,0.05)',
                     border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
                     padding: '8px 14px', color: '#fff', fontSize: 13 }}
          />
          <button id="run-compare-btn" className="btn-analyze" onClick={fetchCompare} disabled={loading}>
            {loading ? 'Analyzing…' : '▶ Compare'}
          </button>
          <button id="build-portfolio-btn" className="btn-backtest" onClick={fetchPortfolio} disabled={ptLoading}>
            {ptLoading ? 'Building…' : '📦 Build Portfolio'}
          </button>
        </div>
      </div>

      {/* Comparison Table */}
      {compareData?.stocks?.length > 0 && (
        <div className="card">
          <div className="card-header"><span className="card-title">📊 Stock Metrics</span></div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '0 4px', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#647091', fontSize: 11 }}>
                  {['Symbol', 'Return (1Y)', 'Volatility', 'Sharpe', 'Max DD', 'Win Rate', 'RL Signal', 'RL Weight'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 12px', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {compareData.stocks.map(s => {
                  if (s.error) return (
                    <tr key={s.symbol}>
                      <td colSpan={8} style={{ color: '#fc8181', padding: '8px 12px', fontSize: 12 }}>
                        {s.symbol}: {s.error}
                      </td>
                    </tr>
                  );
                  const alloc = compareData.rl_allocation?.find(a => a.symbol === s.symbol);
                  return (
                    <tr key={s.symbol} style={{
                      background: 'rgba(255,255,255,0.025)',
                      borderRadius: 8, transition: 'background 0.2s',
                    }}>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: '#e2e8f0' }}>{s.symbol}</td>
                      <td style={{ padding: '10px 12px', color: s.ann_return >= 0 ? '#68d391' : '#fc8181' }}>
                        {(s.ann_return * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '10px 12px', color: '#a0aec0' }}>{(s.ann_vol * 100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 12px', color: s.sharpe >= 1 ? '#68d391' : s.sharpe >= 0 ? '#f6e05e' : '#fc8181' }}>
                        {s.sharpe?.toFixed(3)}
                      </td>
                      <td style={{ padding: '10px 12px', color: '#fc8181' }}>{(s.max_drawdown * 100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 12px', color: '#a0aec0' }}>{(s.win_rate * 100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: dirColor(s.direction) }}>
                        {dirIcon(s.direction)} {s.rl_action?.toFixed(3)}
                      </td>
                      <td style={{ padding: '10px 12px', color: '#c3dafe', fontFamily: 'monospace' }}>
                        {alloc ? ((alloc.rl_weight || 0) * 100).toFixed(1) + '%' : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Correlation Matrix */}
      {corrTickers.length > 0 && (
        <div className="card">
          <div className="card-header"><span className="card-title">🔗 Correlation Matrix</span></div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'separate', borderSpacing: 3, fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ color: '#647091', fontSize: 11, padding: 6 }}></th>
                  {corrTickers.map(t => (
                    <th key={t} style={{ color: '#8b9fc0', padding: '4px 8px', fontSize: 11 }}>{t}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corrTickers.map(row => (
                  <tr key={row}>
                    <td style={{ color: '#8b9fc0', padding: '4px 8px', fontSize: 11, fontWeight: 600 }}>{row}</td>
                    {corrTickers.map(col => {
                      const val = corr[row]?.[col];
                      return (
                        <td key={col} style={{
                          padding: '8px 14px', textAlign: 'center', borderRadius: 6,
                          background: corrColor(val), fontFamily: 'monospace',
                          fontSize: 12, color: '#e2e8f0',
                        }}>
                          {val !== undefined ? val.toFixed(2) : '—'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Portfolio Builder */}
      {portfolio && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">📦 Demo Portfolio Builder</span>
            <span style={{ fontSize: 11, color: '#647091' }}>RL-Weighted vs Equal-Weight vs Buy &amp; Hold</span>
          </div>
          {/* Weight badges */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
            {Object.entries(portfolio.weights || {}).map(([ticker, w]) => (
              <div key={ticker} style={{
                padding: '4px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.3)',
                color: '#c3dafe',
              }}>
                {ticker}: {(w * 100).toFixed(1)}%
              </div>
            ))}
          </div>
          {/* Equity curves */}
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={equityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="day" tick={{ fill: '#647091', fontSize: 10 }} />
              <YAxis tick={{ fill: '#647091', fontSize: 10 }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
              <Tooltip
                contentStyle={{ background: 'rgba(26,27,58,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                formatter={v => [`$${v?.toFixed(0)}`]}
              />
              <Legend wrapperStyle={{ color: '#a0aec0', fontSize: 12 }} />
              <Line type="monotone" dataKey="RL"          stroke="#6366f1" dot={false} strokeWidth={2.5} />
              <Line type="monotone" dataKey="EqualWeight" stroke="#68d391" dot={false} strokeWidth={1.5} strokeDasharray="4 4" />
              <Line type="monotone" dataKey="BuyHold"     stroke="#f6e05e" dot={false} strokeWidth={1.5} strokeDasharray="2 4" />
            </LineChart>
          </ResponsiveContainer>
          {/* Metrics row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 16 }}>
            {[
              { label: 'RL Portfolio', m: portfolio.metrics_rl,    color: '#6366f1' },
              { label: 'Equal Weight', m: portfolio.metrics_equal, color: '#68d391' },
              { label: 'Buy & Hold',   m: portfolio.metrics_bh,    color: '#f6e05e' },
            ].map(({ label, m, color }) => m && (
              <div key={label} style={{
                padding: 12, borderRadius: 10,
                background: `${color}11`, border: `1px solid ${color}33`,
              }}>
                <div style={{ fontSize: 11, color, fontWeight: 700, marginBottom: 8 }}>{label}</div>
                {[
                  { k: 'total_return', label: 'Return',  fmt: v => `${(v*100).toFixed(2)}%` },
                  { k: 'sharpe',       label: 'Sharpe',  fmt: v => v?.toFixed(3) },
                  { k: 'sortino',      label: 'Sortino', fmt: v => v?.toFixed(3) },
                  { k: 'max_dd',       label: 'Max DD',  fmt: v => `${(v*100).toFixed(2)}%` },
                ].map(({ k, label: ml, fmt }) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: '#8b9fc0' }}>{ml}</span>
                    <span style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{fmt(m[k])}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
