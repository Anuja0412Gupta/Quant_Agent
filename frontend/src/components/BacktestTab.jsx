import { useState, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── helpers ─────────────────────────────────────────────────────────────────
function pct(v, dp = 2)  { return `${((v || 0) * 100).toFixed(dp)}%`; }
function fixed(v, dp = 3){ return v != null ? Number(v).toFixed(dp) : '—'; }

function MetricTile({ label, value, color = '#f1f5f9', sub }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: 12, padding: '16px 18px',
    }}>
      <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 800, color, fontFamily: 'var(--mono)' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

const STRESS_COLORS = {
  'COVID Crash':     '#ff4f72',
  'Dotcom Bust':     '#f59e0b',
  'GFC 2008':        '#fb923c',
  'Flash Crash':     '#a78bfa',
  'Rate Shock 2022': '#60a5fa',
};

export default function BacktestTab() {
  const [ticker,      setTicker]      = useState('AAPL');
  const [timeframe,   setTimeframe]   = useState('1d');
  const [backtest,    setBacktest]    = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);
  const [progressMsg, setProgressMsg] = useState('');
  const [elapsed,     setElapsed]     = useState(0);

  const fetchBacktest = useCallback(async () => {
    setLoading(true); setError(null); setBacktest(null);
    setProgressMsg('Starting backtest job…'); setElapsed(0);

    // Tick elapsed seconds while running
    const startTs = Date.now();
    const ticker$ = setInterval(() => setElapsed(Math.round((Date.now() - startTs) / 1000)), 1000);

    try {
      const period = { '1m': '7d', '5m': '60d', '15m': '60d', '1h': '730d', '1d': '5y' }[timeframe] || '5y';

      // 1️⃣  Fire-and-forget — backend returns job_id in < 1 second
      const { data: job } = await axios.post(`${API}/backtest`, {
        symbol: ticker.toUpperCase(), timeframe, period,
      }, { timeout: 15_000 });

      if (!job.job_id) throw new Error(job.detail || 'No job_id returned');
      setProgressMsg(job.message || 'Job queued…');

      // 2️⃣  Poll status every 5 seconds until done / error
      await new Promise((resolve, reject) => {
        const poll = setInterval(async () => {
          try {
            const { data: status } = await axios.get(
              `${API}/backtest/status/${job.job_id}`, { timeout: 10_000 }
            );
            setProgressMsg(status.progress_msg || status.status);

            if (status.status === 'done') {
              clearInterval(poll);
              setBacktest(status.result);
              resolve();
            } else if (status.status === 'error') {
              clearInterval(poll);
              reject(new Error(status.error || 'Backtest failed'));
            }
          } catch (pollErr) {
            // Network blip — keep polling
            console.warn('Poll blip:', pollErr.message);
          }
        }, 5_000);
      });

    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Backtest failed');
    } finally {
      clearInterval(ticker$);
      setLoading(false);
    }
  }, [ticker, timeframe]);

  const m       = backtest?.overall_metrics   || {};
  const folds   = backtest?.walk_forward_folds || [];
  const wfSum   = backtest?.walk_forward_summary || {};
  const stress  = backtest?.stress_test_results;

  const equityData = (backtest?.equity_curve || []).map((v, i) => ({ bar: i, value: v }));

  const returnColor = (m.annualized_ret || 0) >= 0 ? '#00e5a0' : '#ff4f72';
  const sharpeColor = (m.sharpe || 0) >= 1 ? '#00e5a0' : (m.sharpe || 0) >= 0 ? '#ffb830' : '#ff4f72';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* ── Controls ─────────────────────────────────────────────── */}
      <div className="card">
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>
          📈 WALK-FORWARD BACKTEST ENGINE
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>TICKER</div>
            <input
              id="backtest-ticker-input"
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              style={{
                width: 100, background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10,
                padding: '10px 14px', color: '#f1f5f9', fontSize: 14,
              }}
            />
          </div>
          <div>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>TIMEFRAME</div>
            <select
              id="backtest-tf-select"
              value={timeframe}
              onChange={e => setTimeframe(e.target.value)}
              style={{
                background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 10, padding: '10px 14px', color: '#f1f5f9', fontSize: 14,
              }}
            >
              {['1m','5m','15m','1h','1d'].map(tf => <option key={tf} value={tf}>{tf}</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#475569', marginBottom: 6 }}>
              Includes walk-forward folds + 5 historical stress scenarios
            </div>
            <button
              id="run-backtest-btn"
              disabled={loading}
              onClick={fetchBacktest}
              style={{
                padding: '12px 28px', borderRadius: 10, border: 'none',
                background: loading ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                color: loading ? '#64748b' : '#fff', fontSize: 14, fontWeight: 800, cursor: 'pointer',
                boxShadow: loading ? 'none' : '0 4px 20px rgba(99,102,241,0.35)',
              }}
            >
              {loading ? '⏳ Running…' : '▶ Run Backtest & Stress Test'}
            </button>
          </div>
        </div>
        {error && (
          <div style={{ marginTop: 14, padding: '12px 16px', background: 'rgba(255,79,114,0.08)', border: '1px solid rgba(255,79,114,0.3)', borderRadius: 10, color: '#ff4f72', fontSize: 13 }}>
            ❌ {error}
          </div>
        )}
      </div>

      {/* ── Loading state ─────────────────────────── */}
      {loading && (
        <div style={{ background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 14, padding: '32px 28px', textAlign: 'center' }}>
          <div style={{ fontSize: 36, marginBottom: 16 }}>⚙️</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#a5b4fc', marginBottom: 10 }}>
            Running Walk-Forward Backtest
          </div>
          {/* Live progress bar — cycles through 6 stages */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
            {['Fetching data', 'Features', 'HMM fit', 'Loading model', 'Walk-forward folds', 'Stress test'].map((stage, idx) => {
              const stages = ['Fetching','Comput','Fitting','Loading','Running','stress'];
              const active  = stages.some(s => progressMsg.toLowerCase().includes(s.toLowerCase()) && idx === stages.findIndex(s2 => progressMsg.toLowerCase().includes(s2.toLowerCase())));
              const done    = stages.findIndex(s => progressMsg.toLowerCase().includes(s.toLowerCase())) > idx;
              return (
                <div key={stage} style={{
                  padding: '4px 12px', borderRadius: 20, fontSize: 11, fontWeight: 700,
                  background: done ? 'rgba(0,229,160,0.15)' : active ? 'rgba(99,102,241,0.25)' : 'rgba(255,255,255,0.04)',
                  border: `1px solid ${done ? 'rgba(0,229,160,0.4)' : active ? 'rgba(99,102,241,0.5)' : 'rgba(255,255,255,0.08)'}`,
                  color:   done ? '#00e5a0' : active ? '#a5b4fc' : '#475569',
                  transition: 'all 0.4s ease',
                }}>
                  {done ? '✓ ' : active ? '⚡ ' : ''}{stage}
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 13, color: '#64748b', fontFamily: 'var(--mono)', marginBottom: 8 }}>
            {progressMsg}
          </div>
          <div style={{ fontSize: 12, color: '#374151' }}>
            ⏱ {elapsed}s elapsed — typically 5–10 min on cloud
          </div>
        </div>
      )}

      {/* ── Summary Metrics ──────────────────────────────────────── */}
      {backtest && !loading && (
        <>
          <div>
            <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 14 }}>
              OVERALL PERFORMANCE METRICS
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
              <MetricTile label="ANNUALIZED RETURN" value={pct(m.annualized_ret)} color={returnColor}
                sub={(m.annualized_ret || 0) >= 0 ? '✓ Positive CAGR' : '⚠ Negative CAGR'} />
              <MetricTile label="SHARPE RATIO" value={fixed(m.sharpe, 3)} color={sharpeColor}
                sub={m.sharpe >= 1 ? '✓ Excellent risk-adj.' : m.sharpe >= 0 ? 'Acceptable' : '⚠ Below zero'} />
              <MetricTile label="SORTINO RATIO" value={fixed(m.sortino, 3)} color="#60a5fa"
                sub="Downside deviation adjusted" />
              <MetricTile label="CALMAR RATIO" value={fixed(m.calmar, 3)} color="#a78bfa"
                sub="Return / Max Drawdown" />
              <MetricTile label="MAX DRAWDOWN" value={pct(m.max_drawdown)} color="#ff4f72"
                sub="Worst peak-to-trough" />
              <MetricTile label="WIN RATE" value={pct(m.win_rate, 1)} color="#ffb830"
                sub={`${m.n_trades || 0} total trades`} />
              <MetricTile label="PROFIT FACTOR" value={fixed(m.profit_factor, 3)} color="#00e5a0"
                sub="Gross profit / gross loss" />
              <MetricTile label="TOTAL TRADES" value={m.n_trades ?? '—'} color="#94a3b8"
                sub="Executed positions" />
            </div>
          </div>

          {/* ── Equity Curve ─────────────────────────────────────── */}
          {equityData.length > 0 && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>📈 PORTFOLIO EQUITY CURVE</div>
                  <div style={{ fontSize: 13, color: '#94a3b8' }}>Starting capital $100,000 — RL managed positions</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>FINAL VALUE</div>
                  <div style={{ fontSize: 22, fontWeight: 800, fontFamily: 'var(--mono)', color: returnColor }}>
                    ${(equityData.at(-1)?.value ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={equityData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
                  <XAxis dataKey="bar" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false}
                    tickFormatter={v => `Bar ${v}`} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false}
                    tickFormatter={v => `$${(v/1000).toFixed(0)}k`} width={70} />
                  <Tooltip
                    contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 13 }}
                    formatter={v => [`$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'Portfolio Value']}
                  />
                  <ReferenceLine y={100_000} stroke="rgba(255,255,255,0.12)" strokeDasharray="5 4" label={{ value: 'Start $100k', fill: '#475569', fontSize: 11 }} />
                  <Line type="monotone" dataKey="value" stroke="#6366f1" dot={false} strokeWidth={2.5} name="RL Portfolio" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ── Walk-Forward Folds ────────────────────────────────── */}
          {folds.length > 0 && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>🔄 WALK-FORWARD OUT-OF-SAMPLE SHARPE</div>
                  <div style={{ fontSize: 13, color: '#94a3b8' }}>
                    Model trained on each fold's in-sample data, tested on unseen out-of-sample period.
                    Consistent positive OOS Sharpe proves the strategy truly generalizes.
                  </div>
                </div>
                <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 24 }}>
                  <div style={{ display: 'flex', gap: 24 }}>
                    {[
                      { label: 'MEAN OOS SHARPE', value: fixed(wfSum.mean_test_sharpe, 3), color: (wfSum.mean_test_sharpe || 0) >= 0 ? '#00e5a0' : '#ff4f72' },
                      { label: 'STD DEV', value: `±${fixed(wfSum.std_test_sharpe, 3)}`, color: '#94a3b8' },
                      { label: 'WORST FOLD', value: fixed(wfSum.worst_test_sharpe, 3), color: '#ff4f72' },
                    ].map(({ label, value, color }) => (
                      <div key={label} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}>{label}</div>
                        <div style={{ fontSize: 20, fontWeight: 800, fontFamily: 'var(--mono)', color }}>{value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={folds} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
                  <XAxis dataKey="fold" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false}
                    tickFormatter={v => `Fold ${v}`} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 13 }}
                    formatter={(v, n) => [v?.toFixed(4), n]}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
                  <Bar dataKey="test_sharpe" name="OOS Sharpe" radius={[6,6,0,0]}>
                    {folds.map((f, i) => (
                      <Cell key={i} fill={(f.test_sharpe || 0) >= 0 ? '#6366f1' : '#ff4f72'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* Fold table */}
              <div style={{ overflowX: 'auto', marginTop: 20 }}>
                <table style={{ width: '100%', fontSize: 13, borderCollapse: 'separate', borderSpacing: '0 6px' }}>
                  <thead>
                    <tr>
                      {['Fold', 'Train Bars', 'Test Bars', 'OOS Sharpe', 'OOS Sortino', 'CVaR (95%)', 'Max Drawdown'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '6px 14px', fontSize: 11, color: '#64748b', fontWeight: 700 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {folds.map(f => (
                      <tr key={f.fold} style={{ background: 'rgba(255,255,255,0.02)' }}>
                        <td style={{ padding: '10px 14px', fontWeight: 800, color: '#f1f5f9', fontSize: 14 }}>#{f.fold}</td>
                        <td style={{ padding: '10px 14px', color: '#64748b', fontFamily: 'var(--mono)' }}>{f.train_bars}</td>
                        <td style={{ padding: '10px 14px', color: '#64748b', fontFamily: 'var(--mono)' }}>{f.test_bars}</td>
                        <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', fontWeight: 700,
                          color: (f.test_sharpe || 0) >= 0 ? '#00e5a0' : '#ff4f72' }}>
                          {fixed(f.test_sharpe, 4)}
                        </td>
                        <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', color: '#60a5fa' }}>
                          {fixed(f.test_sortino, 4)}
                        </td>
                        <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', color: '#ff4f72' }}>
                          {pct(f.test_cvar)}
                        </td>
                        <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', color: '#ff4f72' }}>
                          {pct(f.test_maxdd)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Stress Test Scenarios ────────────────────────────── */}
          {stress && Object.keys(stress).length > 0 && (
            <div className="card">
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>
                  ⚡ EXTREME HISTORICAL STRESS SCENARIOS
                </div>
                <div style={{ fontSize: 13, color: '#94a3b8' }}>
                  Volatility shocks & widened spreads applied to strategy P&amp;L — tests survivability under real crisis conditions.
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
                {Object.entries(stress).map(([key, sc]) => {
                  const scenarioColor = STRESS_COLORS[sc.scenario || key] || '#94a3b8';
                  const retColor      = (sc.annualized_ret || 0) >= 0 ? '#00e5a0' : '#ff4f72';
                  return (
                    <div key={key} style={{
                      background: `${scenarioColor}08`,
                      border: `1px solid ${scenarioColor}30`,
                      borderRadius: 14,
                      padding: '20px',
                    }}>
                      <div style={{ fontSize: 12, fontWeight: 800, color: scenarioColor, marginBottom: 16 }}>
                        {sc.scenario || key}
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
                        <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: '10px 12px' }}>
                          <div style={{ fontSize: 10, color: '#64748b', marginBottom: 4 }}>STRESSED CAGR</div>
                          <div style={{ fontSize: 20, fontWeight: 800, fontFamily: 'var(--mono)', color: retColor }}>
                            {pct(sc.annualized_ret)}
                          </div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: '10px 12px' }}>
                          <div style={{ fontSize: 10, color: '#64748b', marginBottom: 4 }}>MAX DRAWDOWN</div>
                          <div style={{ fontSize: 20, fontWeight: 800, fontFamily: 'var(--mono)', color: '#ff4f72' }}>
                            {pct(sc.max_drawdown)}
                          </div>
                        </div>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, color: '#64748b' }}>
                        <span>Sharpe: <strong style={{ color: '#60a5fa' }}>{sc.sharpe?.toFixed(2) ?? '—'}</strong></span>
                        <span>Spread: <strong style={{ color: '#ffb830' }}>{sc.spread_multiplier}×</strong></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Empty state ──────────────────────────────────────────── */}
      {!backtest && !loading && (
        <div style={{ textAlign: 'center', padding: '80px 24px', color: '#475569', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
          <div style={{ fontSize: 48 }}>📈</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#64748b' }}>Walk-Forward Backtesting</div>
          <div style={{ fontSize: 14, color: '#374151', maxWidth: 480, lineHeight: 1.8 }}>
            Select a ticker and timeframe, then click <strong style={{ color: '#94a3b8' }}>Run Backtest</strong>.
            The engine performs walk-forward cross-validation to prove real out-of-sample performance,
            then stress-tests the strategy against 5 historical market crises.
          </div>
        </div>
      )}
    </div>
  );
}
