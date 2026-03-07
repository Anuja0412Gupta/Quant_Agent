import { useState, useCallback } from 'react';
import axios from 'axios';
import CandlestickChart from './components/CandlestickChart';
import AgentSignals     from './components/AgentSignals';
import TradeDecision    from './components/TradeDecision';
import BacktestPanel    from './components/BacktestPanel';
import './index.css';

const API = import.meta.env.VITE_API_URL || 'https://quant-agent-backend-pcmt.onrender.com';
const TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d'];

function RLWeightsCard({ weights }) {
  if (!weights?.weights) return null;
  const entries = Object.entries(weights.weights);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-header">
        <span className="card-title">🤖 RL Agent Weights</span>
        <span style={{ fontSize: 11, color: '#647091' }}>PPO Policy</span>
      </div>
      <div className="rl-weights">
        {entries.map(([name, val]) => (
          <div className="rl-weight-row" key={name}>
            <span className="rl-weight-name">{name}</span>
            <div className="rl-weight-bar-track">
              <div className="rl-weight-bar-fill" style={{ width: `${val * 100}%` }} />
            </div>
            <span className="rl-weight-val">{(val * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 12, fontSize: 11, color: '#3a4466' }}>
        Position Multiplier: {(weights.position_multiplier * 100).toFixed(0)}%
        &nbsp;·&nbsp;
        {weights.should_trade ? '✅ RL recommends trading' : '⏸️ RL suggests abstaining'}
      </div>
    </div>
  );
}

export default function App() {
  const [ticker,    setTicker]    = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1d');
  const [analysis,  setAnalysis]  = useState(null);
  const [backtest,  setBacktest]  = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [btLoading, setBtLoading] = useState(false);
  const [error,     setError]     = useState(null);

  const handleAnalyze = useCallback(async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.get(`${API}/analyze/${ticker.toUpperCase()}`, {
        params: { timeframe },
        timeout: 60_000,
      });
      setAnalysis(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }, [ticker, timeframe]);

  const handleBacktest = useCallback(async () => {
    if (!ticker.trim()) return;
    setBtLoading(true);
    setError(null);
    try {
      const period = { '1m': '7d', '5m': '60d', '15m': '60d', '1h': '730d', '1d': '5y' }[timeframe] || '5y';
      const { data } = await axios.get(`${API}/backtest/${ticker.toUpperCase()}`, {
        params: { timeframe, period },
        timeout: 180_000,
      });
      setBacktest(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Backtest failed');
    } finally {
      setBtLoading(false);
    }
  }, [ticker, timeframe]);

  const handleKey = (e) => { if (e.key === 'Enter') handleAnalyze(); };

  const lastPrice = analysis?.ohlcv?.at(-1)?.close;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">⚡</span>
          <span className="logo-gradient">QuantAgent</span>
        </div>
        <div className="header-controls">
          <div className="statusbar">
            <div className={`status-dot ${loading || btLoading ? 'loading' : ''}`} />
            <span>{loading ? 'Analyzing…' : btLoading ? 'Backtesting…' : 'Ready'}</span>
          </div>
          <input
            id="ticker-input"
            className="ticker-input"
            value={ticker}
            onChange={e => setTicker(e.target.value.toUpperCase())}
            onKeyDown={handleKey}
            placeholder="AAPL"
            maxLength={10}
          />
          <select
            id="timeframe-select"
            className="tf-select"
            value={timeframe}
            onChange={e => setTimeframe(e.target.value)}
          >
            {TIMEFRAMES.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
          <button
            id="analyze-btn"
            className="btn-analyze"
            onClick={handleAnalyze}
            disabled={loading || btLoading}
          >
            {loading ? 'Analyzing…' : '▶ Analyze'}
          </button>
          <button
            id="backtest-btn"
            className="btn-backtest"
            onClick={handleBacktest}
            disabled={loading || btLoading}
          >
            {btLoading ? 'Running…' : '📊 Backtest'}
          </button>
        </div>
      </header>

      <main className="main-content">
        {error && (
          <div className="error-message">⚠️ {error}</div>
        )}

        {!analysis && !loading && (
          <div className="empty-state">
            <div className="empty-icon">⚡</div>
            <div className="empty-text">Enter a ticker and click Analyze</div>
            <div className="empty-sub">Supports NYSE, NASDAQ, NSE (e.g. AAPL, MSFT, RELIANCE.NS)</div>
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <div className="spinner" />
            <div className="loading-text">Running all agents on {ticker}…</div>
          </div>
        )}

        {analysis && !loading && (
          <>
            {/* Symbol info */}
            <div className="symbol-info">
              <span className="symbol-badge">{analysis.symbol}</span>
              <span className="price-badge">${lastPrice?.toFixed(2)}</span>
              <span style={{ color: '#647091', fontSize: 13 }}>{analysis.timeframe}</span>
            </div>

            {/* Candlestick chart */}
            <div className="chart-card">
              <div className="card-header">
                <span className="card-title">📉 Price Chart</span>
                <span style={{ fontSize: 11, color: '#647091' }}>
                  {analysis.ohlcv?.length} bars
                </span>
              </div>
              <div className="chart-container">
                <CandlestickChart data={analysis.ohlcv} />
              </div>
            </div>

            {/* Agent signals grid */}
            <AgentSignals data={analysis} />

            {/* Decision + RL weights */}
            <div className="bottom-grid">
              <div>
                <TradeDecision data={analysis} />
                <RLWeightsCard weights={analysis.rl_weights} />
              </div>

              {/* Backtest panel or trigger prompt */}
              <div>
                {backtest ? (
                  <BacktestPanel data={backtest} />
                ) : (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div className="empty-state">
                      <div className="empty-icon">📊</div>
                      <div className="empty-text">Run a backtest to see historical performance</div>
                      <button
                        className="btn-analyze"
                        onClick={handleBacktest}
                        disabled={btLoading}
                        style={{ marginTop: 12 }}
                      >
                        {btLoading ? 'Running…' : '📊 Run Backtest'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Standalone backtest panel if no analysis loaded */}
        {!analysis && backtest && !btLoading && (
          <BacktestPanel data={backtest} />
        )}

        {btLoading && (
          <div className="loading-state">
            <div className="spinner" />
            <div className="loading-text">Backtesting {ticker} — simulating {timeframe} bars…</div>
          </div>
        )}
      </main>
    </div>
  );
}
