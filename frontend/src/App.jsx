/**
 * v3.0 App.jsx — Full Research-Grade UI
 * ========================================
 * Analyze tab now shows: RegimePanel, SentimentPanel, MacroPanel,
 * the OHLCV chart, agent signals, trade decision, and SHAP.
 * All v3.0 API fields wired through.
 */
import { useState, useCallback } from 'react';
import axios from 'axios';
import CandlestickChart    from './components/CandlestickChart';
import AgentSignals        from './components/AgentSignals';
import TradeDecision       from './components/TradeDecision';
import StateVector         from './components/StateVector';
import DisagreementHeatmap from './components/DisagreementHeatmap';
import SHAPPanel           from './components/SHAPPanel';
import RegimePanel         from './components/RegimePanel';
import SentimentPanel      from './components/SentimentPanel';
import MacroPanel          from './components/MacroPanel';
import WalkForwardPanel    from './components/WalkForwardPanel';
import RLBrainTab          from './components/RLBrainTab';
import CompareTab          from './components/CompareTab';
import BacktestTab         from './components/BacktestTab';
import LiveDemoTab         from './components/LiveDemoTab';
import PaperTradeTab       from './components/PaperTradeTab';
import AISuggestionsTab    from './components/AISuggestionsTab';
import AIProfitProofTab    from './components/AIProfitProofTab';
import './index.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TIMEFRAMES = ['1d', '1h', '15m'];
const TABS = [
  { id: 'analyze',  label: '🔍 Analyze',     desc: 'Full Research Analysis' },
  { id: 'suggest',  label: '💡 AI Suggestions', desc: 'Plain-English Explanation' },
  { id: 'profit',   label: '💰 Profit Validation', desc: 'Proof of Returns' },
  { id: 'rl',       label: '🧠 RL Brain',    desc: 'TCN-LSTM + Lagrangian' },
  { id: 'compare',  label: '📊 Compare',     desc: 'Multi-Stock + DCC-GARCH' },
  { id: 'backtest', label: '📈 Backtest',    desc: 'Walk-Forward + Stress' },
  { id: 'live',     label: '🎬 Live Demo',   desc: 'Tick Replay' },
  { id: 'paper',    label: '💼 Paper Trade', desc: 'Simulate P&L' },
];

export default function App() {
  const [userId, setUserId] = useState(localStorage.getItem('quant_user_id') || '');
  const [loginInput, setLoginInput] = useState('');
  const [passwordInput, setPasswordInput] = useState('');
  const [authMode, setAuthMode] = useState('login');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const [activeTab, setActiveTab] = useState('analyze');
  const [ticker,    setTicker]    = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1d');
  const [analysis,  setAnalysis]  = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [searchTimeout, setSearchTimeout] = useState(null);

  const handleSearchChange = (e) => {
    const val = e.target.value.toUpperCase();
    setTicker(val);
    setShowSuggestions(true);
    
    if (searchTimeout) clearTimeout(searchTimeout);
    
    if (val.length < 2) {
      setSuggestions([]);
      return;
    }

    setSearchTimeout(setTimeout(async () => {
      try {
        const { data } = await axios.get(`${API}/search`, { params: { q: val } });
        setSuggestions(data || []);
      } catch (err) {
        console.error("Search error", err);
      }
    }, 300));
  };

  const handleSelectSuggestion = (sym) => {
    setTicker(sym);
    setShowSuggestions(false);
    setTimeout(() => document.getElementById('analyze-btn')?.click(), 100);
  };

  const handleAnalyze = useCallback(async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.get(`${API}/analyze/${ticker.toUpperCase()}`, {
        params: { timeframe, include_sentiment: true, include_macro: true },
        timeout: 120_000,
      });
      setAnalysis(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }, [ticker, timeframe]);

  const handleKey = (e) => { if (e.key === 'Enter') handleAnalyze(); };

  const handleAuth = async (e) => {
    e.preventDefault();
    if (!loginInput.trim() || !passwordInput.trim()) return;
    setAuthLoading(true);
    setAuthError('');
    try {
      if (authMode === 'signup') {
        await axios.post(`${API}/auth/signup`, { username: loginInput.trim(), password: passwordInput });
        setUserId(loginInput.trim());
        localStorage.setItem('quant_user_id', loginInput.trim());
      } else {
        await axios.post(`${API}/auth/login`, { username: loginInput.trim(), password: passwordInput });
        setUserId(loginInput.trim());
        localStorage.setItem('quant_user_id', loginInput.trim());
      }
    } catch (err) {
      setAuthError(err.response?.data?.detail || err.message || 'Authentication failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setUserId('');
    setLoginInput('');
    setPasswordInput('');
    localStorage.removeItem('quant_user_id');
  };

  if (!userId) {
    return (
      <div className="app" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div style={{ padding: 40, background: 'var(--bg-3)', borderRadius: 16, textAlign: 'center', border: '1px solid var(--border)', maxWidth: 400, width: '100%' }}>
          <div className="logo" style={{ marginBottom: 24, justifyContent: 'center' }}>
            <span className="logo-icon">⚡</span>
            <span className="logo-gradient">QuantAgent</span>
          </div>
          
          <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
             <button 
                type="button" 
                onClick={() => { setAuthMode('login'); setAuthError(''); }} 
                style={{ flex: 1, padding: '8px', background: authMode === 'login' ? 'var(--accent-blue)' : 'var(--bg-card)', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
             >Sign In</button>
             <button 
                type="button" 
                onClick={() => { setAuthMode('signup'); setAuthError(''); }} 
                style={{ flex: 1, padding: '8px', background: authMode === 'signup' ? 'var(--accent-blue)' : 'var(--bg-card)', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
             >Create Account</button>
          </div>

          <form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <input 
              value={loginInput} onChange={e => setLoginInput(e.target.value)} 
              placeholder="Username" className="ticker-input" style={{ width: '100%', padding: '12px', fontSize: 16, textAlign: 'center' }} 
              required 
            />
            <input 
              type="password"
              value={passwordInput} onChange={e => setPasswordInput(e.target.value)} 
              placeholder="Password" className="ticker-input" style={{ width: '100%', padding: '12px', fontSize: 16, textAlign: 'center' }} 
              required minLength={4}
            />
            {authError && <div style={{ color: '#ff6b6b', fontSize: 13, fontWeight: 'bold' }}>{authError}</div>}
            
            <button type="submit" className="btn-analyze" style={{ padding: '12px', width: '100%', fontSize: 16 }} disabled={authLoading}>
              {authLoading ? 'Authenticating...' : (authMode === 'login' ? 'Secure Login' : 'Register Account')}
            </button>
          </form>
        </div>
      </div>
    );
  }

  const price      = analysis?.current_price;
  const changeSign = (analysis?.price_change_pct ?? 0) >= 0 ? '+' : '';
  const changePct  = analysis?.price_change_pct;

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">⚡</span>
          <span className="logo-gradient">QuantAgent</span>
          <span className="logo-version">v3.0 · Research Grade</span>
          <span className="logo-dims">45-dim · HMM-BOCPD · TCN-LSTM</span>
        </div>
        <div className="header-controls">
          <div className="statusbar">
            <div className={`status-dot ${loading ? 'loading' : ''}`} />
            <span>{loading ? 'Analyzing…' : 'Ready'}</span>
            <span style={{ marginLeft: 16, borderLeft: '1px solid var(--border)', paddingLeft: 16, color: 'var(--text-1)' }}>👤 {userId}</span>
            <button onClick={handleLogout} style={{ background: 'transparent', border: 'none', color: '#ff6b6b', cursor: 'pointer', marginLeft: 8, fontSize: 11, fontWeight: 'bold' }}>Logout</button>
          </div>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <input
              id="ticker-input"
              className="ticker-input"
              style={{ fontSize: '18px', padding: '12px 16px', width: '250px', fontWeight: 'bold' }}
              value={ticker}
              onChange={handleSearchChange}
              onKeyDown={handleKey}
              onFocus={() => { if(ticker.length >= 2) setShowSuggestions(true); }}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              placeholder="Search Apple, NVDA..."
              autoComplete="off"
            />
            {showSuggestions && suggestions.length > 0 && (
              <div style={{
                position: 'absolute', top: '100%', left: 0, width: '100%',
                background: 'var(--bg-card)', border: '1px solid var(--border)',
                borderRadius: '8px', marginTop: '4px', zIndex: 999, overflow: 'hidden',
                boxShadow: '0 10px 25px rgba(0,0,0,0.5)'
              }}>
                {suggestions.map((s, idx) => (
                  <div
                    key={idx}
                    onClick={() => handleSelectSuggestion(s.symbol)}
                    style={{
                      padding: '10px 16px', cursor: 'pointer', borderBottom: '1px solid var(--border)',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                    }}
                    onMouseEnter={(e) => e.target.style.background = 'var(--bg-3)'}
                    onMouseLeave={(e) => e.target.style.background = 'transparent'}
                  >
                    <span style={{ fontWeight: 'bold', color: 'var(--text-1)' }}>{s.symbol}</span>
                    <span style={{ fontSize: '12px', color: 'var(--text-2)', maxWidth: '140px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.longname}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
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
            disabled={loading}
          >
            {loading ? 'Analyzing…' : '▶ Analyze'}
          </button>
        </div>
      </header>

      {/* ── Tab Navigation ──────────────────────────────────────────────── */}
      <nav className="tab-nav">
        {TABS.map(tab => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-label">{tab.label}</span>
            <span className="tab-desc">{tab.desc}</span>
          </button>
        ))}
      </nav>

      {/* ── Main Content ────────────────────────────────────────────────── */}
      <main className="main-content">
        {error && <div className="error-message">⚠️ {error}</div>}

        {/* ── TAB 1: ANALYZE ──────────────────────────────────────────── */}
        {activeTab === 'analyze' && (
          <>
            {!analysis && !loading && (
              <div className="empty-state">
                <div className="empty-icon">⚡</div>
                <div className="empty-text">Enter a ticker and click Analyze</div>
                <div className="empty-sub">
                  45-dim features · HMM regime · FinBERT sentiment · TCN-LSTM RL action
                </div>
                <div className="empty-sub" style={{ color: '#4ade80', marginTop: 8 }}>
                  252-bar burn-in enforced · No look-ahead bias
                </div>
              </div>
            )}
            {loading && (
              <div className="loading-state">
                <div className="spinner" />
                <div className="loading-text">
                  Running HMM · BOCPD · FinBERT · RL policy on {ticker}…
                </div>
              </div>
            )}
            {analysis && !loading && (
              <>
                {/* Symbol price bar */}
                <div className="symbol-info">
                  <span className="symbol-badge">{analysis.symbol}</span>
                  {price && (
                    <span className="price-badge">${price.toFixed(2)}</span>
                  )}
                  {changePct != null && (
                    <span className={`change-badge ${changePct >= 0 ? 'pos' : 'neg'}`}>
                      {changeSign}{changePct.toFixed(2)}%
                    </span>
                  )}
                  <span style={{ color: '#647091', fontSize: 13 }}>{analysis.timeframe}</span>
                  <span className="model-badge">v{analysis.model_version} · {analysis.feature_dim}D · {analysis.burnin_bars}bar burn-in</span>
                </div>

                {/* Row 1: Regime + Sentiment */}
                <div className="analyze-row-2col">
                  <RegimePanel regime={analysis.regime} />
                  <SentimentPanel
                    sentiment={analysis.sentiment}
                    secFlags={analysis.sec_flags}
                  />
                </div>

                {/* Row 2: Chart */}
                <div className="chart-card">
                  <div className="card-header">
                    <span className="card-title">📉 Price Chart</span>
                    <span style={{ fontSize: 11, color: '#647091' }}>
                      {analysis.ohlcv_bars?.length} bars
                    </span>
                  </div>
                  <div className="chart-container">
                    <CandlestickChart data={analysis.ohlcv_bars} />
                  </div>
                </div>

                {/* Row 3: Macro */}
                {analysis.macro_context && (
                  <MacroPanel macro={analysis.macro_context} />
                )}

                {/* Row 4: Agent signals + Trade Decision */}
                <div className="analyze-row-2col">
                  <div>
                    <AgentSignals data={analysis} />
                    <DisagreementHeatmap analysis={analysis} />
                  </div>
                  <div>
                    <TradeDecision data={analysis} />
                    <StateVector analysis={analysis} />
                  </div>
                </div>

                {/* Row 5: SHAP */}
                <SHAPPanel shap={analysis?.shap} />
              </>
            )}
          </>
        )}

        {/* ── TAB 1.5: AI SUGGESTIONS ─────────────────────────────── */}
        {activeTab === 'suggest' && <AISuggestionsTab analysis={analysis} />}

        {/* ── TAB 1.7: PROFIT PROOF ───────────────────────────────── */}
        {activeTab === 'profit' && <AIProfitProofTab analysis={analysis} />}

        {/* ── TAB 2: RL BRAIN ─────────────────────────────────────── */}
        {activeTab === 'rl' && <RLBrainTab analysis={analysis} />}

        {/* ── TAB 3: COMPARE ──────────────────────────────────────── */}
        {activeTab === 'compare' && <CompareTab userId={userId} />}

        {/* ── TAB 4: BACKTEST ─────────────────────────────────────── */}
        {activeTab === 'backtest' && <BacktestTab />}

        {/* ── TAB 5: LIVE DEMO ────────────────────────────────────── */}
        {activeTab === 'live' && <LiveDemoTab />}

        {/* ── TAB 6: PAPER TRADE ──────────────────────────────────── */}
        {activeTab === 'paper' && <PaperTradeTab currentTicker={ticker} userId={userId} />}
      </main>
    </div>
  );
}
