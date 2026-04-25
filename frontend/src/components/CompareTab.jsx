import { useState, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
  LineChart, Line, ResponsiveContainer, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PALETTE = ['#00e5a0', '#60a5fa', '#f59e0b', '#f472b6', '#a78bfa', '#34d399'];

const dirCfg = {
  LONG:  { color: '#00e5a0', icon: '▲', bg: 'rgba(0,229,160,0.1)',  border: 'rgba(0,229,160,0.3)'  },
  SHORT: { color: '#ff4f72', icon: '▼', bg: 'rgba(255,79,114,0.1)', border: 'rgba(255,79,114,0.3)' },
  FLAT:  { color: '#ffb830', icon: '—', bg: 'rgba(255,184,48,0.1)', border: 'rgba(255,184,48,0.3)' },
};

function corrColor(val) {
  if (val === undefined || val === null) return 'rgba(255,255,255,0.03)';
  const abs = Math.abs(val);
  if (abs > 0.8) return val > 0 ? 'rgba(0,229,160,0.45)'  : 'rgba(255,79,114,0.45)';
  if (abs > 0.5) return val > 0 ? 'rgba(0,229,160,0.25)'  : 'rgba(255,79,114,0.25)';
  if (abs > 0.3) return val > 0 ? 'rgba(0,229,160,0.12)'  : 'rgba(255,79,114,0.12)';
  return 'rgba(255,255,255,0.04)';
}

function StatCell({ label, value, color = '#f1f5f9' }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '14px 16px' }}>
      <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: 'var(--mono)' }}>{value}</div>
    </div>
  );
}

export default function CompareTab({ onTradeTicker }) {
  const [stocks,      setStocks]      = useState('AAPL,MSFT,NVDA');
  const [compareData, setCompareData] = useState(null);
  const [portfolio,   setPortfolio]   = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [ptLoading,   setPtLoading]   = useState(false);

  const fetchCompare = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API}/compare`, { params: { stocks, timeframe: '1d', period: '1y' }, timeout: 120_000 });
      setCompareData(data);
    } catch (e) { console.error('Compare failed:', e.message); }
    finally { setLoading(false); }
  }, [stocks]);

  const fetchPortfolio = useCallback(async () => {
    setPtLoading(true);
    try {
      const tickers = stocks.split(',').map(s => s.trim().toUpperCase());
      const { data } = await axios.post(`${API}/portfolio/build`, { stocks: tickers, timeframe: '1d' }, { timeout: 120_000 });
      setPortfolio(data);
    } catch (e) { console.error('Portfolio failed:', e.message); }
    finally { setPtLoading(false); }
  }, [stocks]);

  const corr        = compareData?.correlation || {};
  const corrTickers = Object.keys(corr);
  const equityData  = portfolio ? (portfolio.equity_rl || []).map((v, i) => ({
    day: i, RL: v,
    EqualWeight: portfolio.equity_equal?.[i],
    BuyHold:     portfolio.equity_bh?.[i],
  })) : [];

  // ── Best Trade Scorer ────────────────────────────────────────────────
  // Composite score using: Sharpe (40%), RL action strength (30%),
  // return (15%), win rate (10%), drawdown penalty (5%)
  const bestTrade = useMemo(() => {
    const valid = (compareData?.stocks || []).filter(s => !s.error && s.direction !== 'FLAT');
    if (!valid.length) return null;

    const scored = valid.map(s => {
      const alloc = compareData.rl_allocation?.find(a => a.symbol === s.symbol);
      const sharpeScore  = Math.min(1, Math.max(0, (s.sharpe || 0) / 3));          // 0-3 sharpe → 0-1
      const actionScore  = Math.min(1, Math.abs(s.rl_action || 0));                // 0-1
      const returnScore  = Math.min(1, Math.max(0, (s.ann_return || 0) + 0.3) / 0.6); // normalised
      const winScore     = Math.min(1, (s.win_rate || 0.5));                        // 0-1
      const ddPenalty    = Math.min(1, Math.abs(s.max_drawdown || 0) * 2);         // 0-1 (lower is better)
      const rlWeight     = (alloc?.rl_weight || 0);                                 // portfolio weight

      const composite =
        sharpeScore * 0.35 +
        actionScore * 0.30 +
        returnScore * 0.15 +
        winScore    * 0.10 +
        rlWeight    * 0.05 +
        (1 - ddPenalty) * 0.05;

      return { ...s, composite, sharpeScore, actionScore, alloc };
    });

    scored.sort((a, b) => b.composite - a.composite);
    return scored[0];
  }, [compareData]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* ── Search Bar ─────────────────────────────────────────── */}
      <div className="card">
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>
          📊 MULTI-STOCK COMPARISON ENGINE
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, position: 'relative', minWidth: 200 }}>
            <input
              id="compare-stocks-input"
              value={stocks}
              onChange={e => setStocks(e.target.value.toUpperCase())}
              placeholder="AAPL, TSLA, NVDA, MSFT"
              style={{
                width: '100%', background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10,
                padding: '12px 16px', color: '#f1f5f9', fontSize: 14,
                outline: 'none',
              }}
            />
            <div style={{ fontSize: 11, color: '#475569', marginTop: 6 }}>
              Enter comma-separated tickers · 1-year daily data
            </div>
          </div>
          <button id="run-compare-btn" disabled={loading} onClick={fetchCompare} style={{
            padding: '12px 28px', borderRadius: 10, border: 'none',
            background: loading ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            color: loading ? '#64748b' : '#fff', fontSize: 14, fontWeight: 700, cursor: 'pointer',
          }}>
            {loading ? '⏳ Analyzing…' : '▶ Compare'}
          </button>
          <button id="build-portfolio-btn" disabled={ptLoading} onClick={fetchPortfolio} style={{
            padding: '12px 28px', borderRadius: 10, border: 'none',
            background: ptLoading ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg, #00e5a0, #059669)',
            color: ptLoading ? '#64748b' : '#000', fontSize: 14, fontWeight: 700, cursor: 'pointer',
          }}>
            {ptLoading ? '⏳ Building…' : '📦 Build Portfolio'}
          </button>
        </div>
      </div>

      {/* ── Best Trade Recommendation Banner ───────────────────── */}
      {bestTrade && (() => {
        const dc    = dirCfg[bestTrade.direction] || dirCfg.FLAT;
        const score = Math.round(bestTrade.composite * 100);
        const alloc = compareData?.rl_allocation?.find(a => a.symbol === bestTrade.symbol);
        return (
          <div style={{
            background: `linear-gradient(135deg, ${dc.color}12 0%, rgba(10,15,35,0) 100%)`,
            border: `2px solid ${dc.color}50`,
            borderRadius: 18,
            padding: '28px 32px',
            display: 'grid',
            gridTemplateColumns: '1fr auto',
            gap: 32,
            alignItems: 'center',
          }}>
            <div>
              <div style={{ fontSize: 11, color: dc.color, fontWeight: 800, letterSpacing: 2, marginBottom: 12 }}>
                ⚡ AI BEST TRADE OPPORTUNITY
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 18 }}>
                <div style={{ fontSize: 52, fontWeight: 900, color: dc.color, fontFamily: 'var(--mono)', filter: `drop-shadow(0 0 14px ${dc.color})` }}>
                  {bestTrade.symbol}
                </div>
                <div style={{ padding: '6px 18px', borderRadius: 24, background: dc.bg, border: `1px solid ${dc.border}`, color: dc.color, fontSize: 16, fontWeight: 800 }}>
                  {dc.icon} {bestTrade.direction}
                </div>
              </div>
              {/* Score breakdown */}
              <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 16 }}>
                {[
                  { label: 'COMPOSITE SCORE', value: `${score}/100`, color: score > 70 ? '#00e5a0' : score > 50 ? '#ffb830' : '#ff4f72' },
                  { label: 'SHARPE RATIO',    value: bestTrade.sharpe?.toFixed(2), color: bestTrade.sharpe >= 1 ? '#00e5a0' : '#ffb830' },
                  { label: '1Y RETURN',       value: `${(bestTrade.ann_return * 100).toFixed(1)}%`, color: bestTrade.ann_return >= 0 ? '#00e5a0' : '#ff4f72' },
                  { label: 'WIN RATE',        value: `${(bestTrade.win_rate * 100).toFixed(1)}%`, color: '#94a3b8' },
                  { label: 'RL WEIGHT',       value: alloc ? `${((alloc.rl_weight || 0) * 100).toFixed(1)}%` : '—', color: dc.color },
                ].map(({ label, value, color }) => (
                  <div key={label}>
                    <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color, fontFamily: 'var(--mono)' }}>{value || '—'}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 13, color: '#64748b', lineHeight: 1.8 }}>
                Ranked #1 from <strong style={{ color: '#94a3b8' }}>{(compareData?.stocks || []).filter(s => !s.error).length} stocks</strong> using
                a composite model: Sharpe (35%) · RL Action (30%) · 1Y Return (15%) · Win Rate (10%) · Drawdown profile (10%).
                RL model signals a{' '}
                <strong style={{ color: dc.color }}>{bestTrade.direction}</strong> with
                action strength <strong style={{ color: dc.color }}>{(Math.abs(bestTrade.rl_action || 0) * 100).toFixed(1)}%</strong>.
              </div>
            </div>
            {/* CTA */}
            <div style={{ textAlign: 'center', flexShrink: 0, minWidth: 160 }}>
              <div style={{ fontSize: 64, fontWeight: 900, fontFamily: 'var(--mono)', color: dc.color, lineHeight: 1, filter: `drop-shadow(0 0 16px ${dc.color})` }}>
                {score}
              </div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 24, letterSpacing: 1 }}>AI SCORE</div>
              <button
                onClick={() => onTradeTicker && onTradeTicker(bestTrade.symbol)}
                style={{
                  padding: '14px 28px', borderRadius: 12, border: 'none',
                  background: `linear-gradient(135deg, ${dc.color}ee, ${dc.color}99)`,
                  color: bestTrade.direction === 'LONG' ? '#000' : '#fff',
                  fontSize: 15, fontWeight: 800, cursor: 'pointer', width: '100%',
                  boxShadow: `0 4px 20px ${dc.color}40`,
                  transition: 'transform 0.2s, box-shadow 0.2s',
                }}
                onMouseEnter={e => { e.currentTarget.style.transform = 'scale(1.05)'; e.currentTarget.style.boxShadow = `0 8px 30px ${dc.color}60`; }}
                onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)';    e.currentTarget.style.boxShadow = `0 4px 20px ${dc.color}40`; }}
              >
                💼 Trade {bestTrade.symbol} Now
              </button>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 8 }}>→ Opens Paper Trade tab</div>
            </div>
          </div>
        );
      })()}

      {/* ── Per-Stock Signal Cards ──────────────────────────────── */}
      {compareData?.stocks?.length > 0 && (() => {
        const valid = compareData.stocks.filter(s => !s.error);
        return (
          <>
            {/* Signal Cards */}
            <div>
              <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 14 }}>
                AI SIGNALS & METRICS
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 20 }}>
                {compareData.stocks.map((s, idx) => {
                  if (s.error) return (
                    <div key={s.symbol} style={{ padding: 16, background: 'rgba(255,79,114,0.08)', border: '1px solid rgba(255,79,114,0.2)', borderRadius: 12, color: '#ff4f72', fontSize: 13 }}>
                      ❌ {s.symbol}: {s.error}
                    </div>
                  );
                  const alloc  = compareData.rl_allocation?.find(a => a.symbol === s.symbol);
                  const dir    = (s.direction || 'FLAT');
                  const dc     = dirCfg[dir] || dirCfg.FLAT;
                  const color  = PALETTE[idx % PALETTE.length];
                  return (
                    <div key={s.symbol} style={{
                      background: 'var(--bg-card)', border: `1px solid ${dc.border}`,
                      borderRadius: 14, padding: '20px', display: 'flex', flexDirection: 'column', gap: 12,
                    }}>
                      {/* Header */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontSize: 20, fontWeight: 900, color }}>
                          {s.symbol}
                        </div>
                        <div style={{ padding: '4px 12px', borderRadius: 20, background: dc.bg, border: `1px solid ${dc.border}`, color: dc.color, fontSize: 12, fontWeight: 800 }}>
                          {dc.icon} {dir}
                        </div>
                      </div>

                      {/* Key stats */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                        {[
                          { label: '1Y Return',   value: `${(s.ann_return * 100).toFixed(1)}%`, color: s.ann_return >= 0 ? '#00e5a0' : '#ff4f72' },
                          { label: 'Sharpe',      value: s.sharpe?.toFixed(2) ?? '—', color: s.sharpe >= 1 ? '#00e5a0' : s.sharpe >= 0 ? '#ffb830' : '#ff4f72' },
                          { label: 'Volatility',  value: `${(s.ann_vol * 100).toFixed(1)}%`, color: '#94a3b8' },
                          { label: 'Max Drawdown',value: `${(s.max_drawdown * 100).toFixed(1)}%`, color: '#ff4f72' },
                          { label: 'Win Rate',    value: `${(s.win_rate * 100).toFixed(1)}%`, color: '#94a3b8' },
                          { label: 'RL Weight',   value: alloc ? `${((alloc.rl_weight || 0) * 100).toFixed(1)}%` : '—', color },
                        ].map(({ label, value, color: c }) => (
                          <div key={label} style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 8, padding: '8px 10px' }}>
                            <div style={{ fontSize: 10, color: '#64748b', fontWeight: 700, marginBottom: 3 }}>{label}</div>
                            <div style={{ fontSize: 15, fontWeight: 800, fontFamily: 'var(--mono)', color: c }}>{value}</div>
                          </div>
                        ))}
                      </div>

                      {/* RL action */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 10px', background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
                        <span style={{ fontSize: 11, color: '#64748b' }}>RL ACTION</span>
                        <span style={{ fontSize: 14, fontFamily: 'var(--mono)', fontWeight: 800, color: dc.color }}>
                          {s.rl_action > 0 ? '+' : ''}{s.rl_action?.toFixed(4) ?? '—'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* ── Correlation Matrix ────────────────────────────── */}
            {corrTickers.length > 0 && (
              <div className="card">
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>
                  🔗 CORRELATION MATRIX
                </div>
                <div style={{ fontSize: 13, color: '#64748b', marginBottom: 16, lineHeight: 1.6 }}>
                  <span style={{ color: '#00e5a0' }}>■</span> High positive correlation &nbsp;
                  <span style={{ color: '#ff4f72' }}>■</span> High negative correlation &nbsp;
                  <span style={{ color: '#94a3b8' }}>■</span> Low correlation (diversification benefit)
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ borderCollapse: 'separate', borderSpacing: 4, fontSize: 13 }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '6px 12px', color: '#475569' }}></th>
                        {corrTickers.map(t => (
                          <th key={t} style={{ padding: '8px 14px', color: '#94a3b8', fontWeight: 700, fontSize: 13 }}>{t}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {corrTickers.map(row => (
                        <tr key={row}>
                          <td style={{ padding: '8px 14px', color: '#94a3b8', fontWeight: 700, fontSize: 13 }}>{row}</td>
                          {corrTickers.map(col => {
                            const val = corr[row]?.[col];
                            const isDiag = row === col;
                            return (
                              <td key={col} style={{
                                padding: '12px 16px', textAlign: 'center', borderRadius: 8,
                                background: isDiag ? 'rgba(255,255,255,0.06)' : corrColor(val),
                                fontFamily: 'var(--mono)', fontSize: 14, color: isDiag ? '#64748b' : '#f1f5f9',
                                fontWeight: isDiag ? 400 : 700,
                              }}>
                                {isDiag ? '—' : val !== undefined ? val.toFixed(2) : '?'}
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
          </>
        );
      })()}

      {/* ── Portfolio Builder Results ───────────────────────────── */}
      {portfolio && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Weight pills */}
          <div className="card">
            <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>
              📦 RL-OPTIMIZED PORTFOLIO WEIGHTS
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
              {Object.entries(portfolio.weights || {}).map(([ticker, w], idx) => {
                const color = PALETTE[idx % PALETTE.length];
                return (
                  <div key={ticker} style={{
                    padding: '10px 20px', borderRadius: 24, fontWeight: 700,
                    background: `${color}18`, border: `1px solid ${color}40`,
                    color, display: 'flex', alignItems: 'center', gap: 8, fontSize: 15,
                  }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: color }} />
                    {ticker}
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 18 }}>
                      {(w * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Metrics comparison */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 24 }}>
              {[
                { label: '🤖 RL Portfolio',  m: portfolio.metrics_rl,    color: '#6366f1' },
                { label: '⚖️ Equal Weight',   m: portfolio.metrics_equal, color: '#00e5a0' },
                { label: '📈 Buy & Hold',     m: portfolio.metrics_bh,    color: '#f59e0b' },
              ].map(({ label, m, color }) => m && (
                <div key={label} style={{
                  background: `${color}0a`, border: `1px solid ${color}30`,
                  borderRadius: 14, padding: '18px 20px',
                }}>
                  <div style={{ fontSize: 13, color, fontWeight: 800, marginBottom: 14 }}>{label}</div>
                  {[
                    { k: 'total_return', label: 'Total Return', fmt: v => `${(v * 100).toFixed(2)}%`, color: v => v >= 0 ? '#00e5a0' : '#ff4f72' },
                    { k: 'sharpe',       label: 'Sharpe Ratio', fmt: v => v?.toFixed(3), color: v => v >= 1 ? '#00e5a0' : v >= 0 ? '#ffb830' : '#ff4f72' },
                    { k: 'sortino',      label: 'Sortino Ratio',fmt: v => v?.toFixed(3), color: () => '#94a3b8' },
                    { k: 'max_dd',       label: 'Max Drawdown', fmt: v => `${(v * 100).toFixed(2)}%`, color: () => '#ff4f72' },
                  ].map(({ k, label: ml, fmt, color: vc }) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <span style={{ fontSize: 12, color: '#64748b' }}>{ml}</span>
                      <span style={{ fontSize: 15, color: vc(m[k]), fontFamily: 'var(--mono)', fontWeight: 700 }}>{fmt(m[k])}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            {/* Equity curve */}
            {equityData.length > 0 && (
              <>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
                  Portfolio equity curves (starting $10,000) — shows how RL weighting compares vs naive strategies
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={equityData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="day" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} label={{ value: 'Trading Day', position: 'insideBottom', fill: '#475569', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip
                      contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 13 }}
                      formatter={v => [`$${Number(v).toLocaleString()}`]}
                    />
                    <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 13, paddingTop: 16 }} />
                    <Line type="monotone" dataKey="RL"          stroke="#6366f1" dot={false} strokeWidth={2.5} name="RL Portfolio" />
                    <Line type="monotone" dataKey="EqualWeight" stroke="#00e5a0" dot={false} strokeWidth={1.5} strokeDasharray="5 3" name="Equal Weight" />
                    <Line type="monotone" dataKey="BuyHold"     stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="2 4" name="Buy & Hold" />
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!compareData && !portfolio && (
        <div style={{
          textAlign: 'center', padding: '80px 24px',
          color: '#475569', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16
        }}>
          <div style={{ fontSize: 48 }}>📊</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#64748b' }}>Multi-Stock Comparison</div>
          <div style={{ fontSize: 14, color: '#374151', maxWidth: 420, lineHeight: 1.7 }}>
            Enter tickers separated by commas and click <strong style={{ color: '#94a3b8' }}>Compare</strong> to analyze signals,
            metrics, and correlations across multiple stocks simultaneously.
          </div>
        </div>
      )}
    </div>
  );
}
