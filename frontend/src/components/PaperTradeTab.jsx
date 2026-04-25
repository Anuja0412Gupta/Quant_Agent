import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const INITIAL_CAPITAL = 100000;

const fmtUSD = (v, dp = 2) =>
  `$${Number(v || 0).toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
const fmtPct = (v, dp = 2, show = true) =>
  show ? `${v >= 0 ? '+' : ''}${Number(v || 0).toFixed(dp)}%` : `${Number(v || 0).toFixed(dp)}%`;

export default function PaperTradeTab({ currentTicker, userId }) {
  const [portfolio, setPortfolio] = useState(null);
  const [history,   setHistory]   = useState([]);
  const [tradeSymbol, setTradeSymbol] = useState(currentTicker || 'AAPL');
  const [tradeAction, setTradeAction] = useState('BUY');
  const [tradeShares, setTradeShares] = useState(1);
  const [quotePrice,  setQuotePrice]  = useState(null);
  const [quoteLoading,setQuoteLoading]= useState(false);
  const [isTrading,   setIsTrading]   = useState(false);
  const [tradeAnalysis, setTradeAnalysis]     = useState(null);
  const [analysisLoading,setAnalysisLoading]  = useState(false);
  const [rlApplied,   setRlApplied]   = useState(false);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);
  const [toast,       setToast]       = useState(null); // {msg, type}

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3200);
  };

  const selectedPosition = portfolio?.positions?.find(p => p.symbol === tradeSymbol);
  const maxSellable  = Number(selectedPosition?.shares || 0);
  const tradeQty     = Number(tradeShares || 0);
  const orderValue   = quotePrice != null ? tradeQty * quotePrice : null;
  const invalidSell  = tradeAction === 'SELL' && tradeQty > maxSellable;

  // ── hydrate portfolio with live prices ─────────────────────────────────────
  const hydratePortfolio = useCallback(async (raw) => {
    const posObj  = raw?.positions || {};
    const symbols = Object.keys(posObj);
    const prices  = {};
    await Promise.all(symbols.map(async sym => {
      try {
        const { data } = await axios.get(`${API}/price/${sym}?_t=${Date.now()}`, { timeout: 10000 });
        prices[sym] = Number(data?.current_price || 0);
      } catch { prices[sym] = Number(posObj[sym]?.avg_price || 0); }
    }));

    const positions = symbols.map(sym => {
      const p             = posObj[sym] || {};
      const shares        = Number(p.quantity || 0);
      const avg_cost      = Number(p.avg_price || 0);
      const live_price    = Number(prices[sym] || avg_cost);
      const market_value  = shares * live_price;
      const unrealized_pnl= (live_price - avg_cost) * shares;
      const unrealized_pct= avg_cost > 0 ? ((live_price - avg_cost) / avg_cost) * 100 : 0;
      return { symbol: sym, shares, avg_cost, live_price, market_value, unrealized_pnl, unrealized_pct };
    });

    const total_market_value   = positions.reduce((a, p) => a + p.market_value, 0);
    const total_unrealized_pnl = positions.reduce((a, p) => a + p.unrealized_pnl, 0);
    const total_value          = Number(raw?.cash || 0) + total_market_value;
    const total_return_pct     = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100;

    setPortfolio({
      cash: Number(raw?.cash || 0),
      total_market_value,
      total_value,
      total_return_pct,
      realized_pnl:   Number(raw?.realized_pnl || 0),
      unrealized_pnl: total_unrealized_pnl,
      positions,
    });

    const hist = Array.isArray(raw?.trade_history) ? [...raw.trade_history].reverse() : [];
    setHistory(hist.map(t => ({
      id: t.trade_id,
      timestamp:   t.timestamp,
      action:      t.action,
      symbol:      t.symbol,
      shares:      Number(t.quantity || 0),
      price:       Number(t.price || 0),
      trade_value: Number(t.value || 0),
    })));
  }, []);

  const fetchPortfolio = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API}/portfolio`, { params: { user_id: userId, _t: Date.now() } });
      await hydratePortfolio(data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load portfolio');
    } finally { setLoading(false); }
  }, [hydratePortfolio]);

  useEffect(() => {
    fetchPortfolio();
    const id = setInterval(fetchPortfolio, 10000);
    return () => clearInterval(id);
  }, [fetchPortfolio]);

  useEffect(() => {
    if (currentTicker && currentTicker !== tradeSymbol) setTradeSymbol(currentTicker);
  }, [currentTicker]);

  useEffect(() => {
    let mounted = true;
    const sym = String(tradeSymbol || '').trim().toUpperCase();
    if (!sym) { setQuotePrice(null); setTradeAnalysis(null); return; }

    setQuoteLoading(true);
    axios.get(`${API}/price/${sym}`, { timeout: 10000 })
      .then(({ data }) => { if (mounted) setQuotePrice(Number(data?.current_price || 0)); })
      .catch(() => { if (mounted) setQuotePrice(null); })
      .finally(() => { if (mounted) setQuoteLoading(false); });

    setAnalysisLoading(true);
    axios.get(`${API}/analyze/${sym}`, { params: { timeframe: '1d' }, timeout: 60000 })
      .then(({ data }) => { if (mounted) setTradeAnalysis(data); })
      .catch(() => { if (mounted) setTradeAnalysis(null); })
      .finally(() => { if (mounted) setAnalysisLoading(false); });

    return () => { mounted = false; };
  }, [tradeSymbol]);

  const quickSell = (symbol, fraction) => {
    const pos = portfolio?.positions?.find(p => p.symbol === symbol);
    if (!pos) return;
    const qty = Number((pos.shares * fraction).toFixed(4));
    if (qty <= 0) return;
    setTradeSymbol(symbol); setTradeAction('SELL'); setTradeShares(qty); setRlApplied(false);
  };

  const applyRLSuggestion = () => {
    if (!tradeAnalysis || !portfolio) return;
    const dir = tradeAnalysis?.rl_weights?.direction;
    if (!dir || dir === 'FLAT') return;
    const action  = dir === 'LONG' ? 'BUY' : 'SELL';
    const effPos  = Number(tradeAnalysis?.rl_weights?.effective_position || 0);
    const dollars = (portfolio.total_value || 100000) * effPos;
    const price   = quotePrice || 1;
    const shares  = Math.max(0.0001, parseFloat((dollars / price).toFixed(4)));
    setTradeAction(action); setTradeShares(shares); setRlApplied(true);
  };

  const handleTrade = async (e) => {
    e.preventDefault();
    if (!tradeSymbol || tradeQty <= 0) return;
    if (invalidSell) { setError(`Cannot sell ${tradeQty} — only ${maxSellable.toFixed(4)} held.`); return; }
    setIsTrading(true); setError(null);
    try {
      await axios.post(`${API}/portfolio/trade`, {
        symbol: tradeSymbol, action: tradeAction, quantity: parseFloat(tradeShares),
      }, { params: { user_id: userId } });
      await fetchPortfolio();
      setTradeShares(1); setRlApplied(false);
      showToast(`${tradeAction} ${tradeQty} × ${tradeSymbol} executed ✓`, 'success');
    } catch (err) {
      const msg = err.response?.data?.detail || 'Trade failed';
      setError(msg);
      showToast(msg, 'error');
    } finally { setIsTrading(false); }
  };

  if (loading && !portfolio) {
    return (
      <div style={{ textAlign: 'center', padding: '80px 24px', color: '#64748b' }}>
        <div className="spinner" style={{ margin: '0 auto 16px' }} />
        <div style={{ fontSize: 16 }}>Loading Paper Portfolio…</div>
      </div>
    );
  }

  // ── derived display vars ────────────────────────────────────────────────────
  const returnColor = (portfolio?.total_return_pct || 0) >= 0 ? '#00e5a0' : '#ff4f72';

  const dir          = tradeAnalysis?.rl_weights?.direction;
  const dirColor     = dir === 'LONG' ? '#00e5a0' : dir === 'SHORT' ? '#ff4f72' : '#ffb830';
  const effPos       = Number(tradeAnalysis?.rl_weights?.effective_position || 0);
  const targetDollar = (portfolio?.total_value || 100000) * effPos;
  const price        = quotePrice || 0;
  const sugShares    = price > 0 ? Math.max(0.0001, parseFloat((targetDollar / price).toFixed(4))) : 0;
  const canTrade     = dir && dir !== 'FLAT' && price > 0 && sugShares > 0;
  const existingPos  = portfolio?.positions?.find(p => p.symbol === tradeSymbol);
  const estPnL       = dir === 'SHORT' && existingPos?.avg_cost > 0 && price > 0
    ? (price - existingPos.avg_cost) * sugShares : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Toast notification */}
      {toast && (
        <div style={{
          position: 'fixed', top: 24, right: 24, zIndex: 9999,
          padding: '14px 22px', borderRadius: 12, fontWeight: 700, fontSize: 14,
          background: toast.type === 'success' ? 'rgba(0,229,160,0.15)' : 'rgba(255,79,114,0.15)',
          border: `1px solid ${toast.type === 'success' ? '#00e5a0' : '#ff4f72'}`,
          color: toast.type === 'success' ? '#00e5a0' : '#ff4f72',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          animation: 'fadeIn 0.2s ease',
        }}>
          {toast.type === 'success' ? '✅' : '❌'} {toast.msg}
        </div>
      )}

      {/* Error bar */}
      {error && (
        <div style={{ padding: '12px 18px', background: 'rgba(255,79,114,0.08)', border: '1px solid rgba(255,79,114,0.3)', borderRadius: 10, color: '#ff4f72', fontSize: 13 }}>
          ⚠️ {error}
        </div>
      )}

      {/* ── Portfolio Summary Tiles ─────────────────────────────── */}
      {portfolio && (() => {
        const tiles = [
          {
            icon: '💰', label: 'TOTAL VALUE',
            value: fmtUSD(portfolio.total_value),
            color: '#60a5fa',
            sub: `Started at $${INITIAL_CAPITAL.toLocaleString()}`,
          },
          {
            icon: '🏦', label: 'AVAILABLE CASH',
            value: fmtUSD(portfolio.cash),
            color: '#00e5a0',
            sub: `${((portfolio.cash / portfolio.total_value) * 100).toFixed(1)}% of portfolio`,
          },
          {
            icon: '📦', label: 'MARKET VALUE',
            value: fmtUSD(portfolio.total_market_value),
            color: '#a78bfa',
            sub: `${portfolio.positions?.length || 0} open positions`,
          },
          {
            icon: portfolio.total_return_pct >= 0 ? '📈' : '📉', label: 'TOTAL RETURN',
            value: fmtPct(portfolio.total_return_pct),
            color: returnColor,
            sub: `vs $${INITIAL_CAPITAL.toLocaleString()} capital`,
          },
          {
            icon: portfolio.unrealized_pnl >= 0 ? '✅' : '⚠️', label: 'UNREALIZED P&L',
            value: (portfolio.unrealized_pnl >= 0 ? '+' : '') + fmtUSD(portfolio.unrealized_pnl),
            color: portfolio.unrealized_pnl >= 0 ? '#00e5a0' : '#ff4f72',
            sub: 'Open positions only',
          },
          {
            icon: '🏁', label: 'REALIZED P&L',
            value: (portfolio.realized_pnl >= 0 ? '+' : '') + fmtUSD(portfolio.realized_pnl),
            color: portfolio.realized_pnl >= 0 ? '#00e5a0' : '#ff4f72',
            sub: 'Closed trades',
          },
        ];
        return (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
            {tiles.map(({ icon, label, value, color, sub }) => (
              <div key={label} style={{
                background: `linear-gradient(145deg, ${color}12 0%, rgba(10,15,35,0.6) 100%)`,
                border: `1.5px solid ${color}40`,
                borderRadius: 14,
                padding: '20px 18px',
                position: 'relative',
                overflow: 'hidden',
                transition: 'transform 0.2s, box-shadow 0.2s',
                cursor: 'default',
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = `0 8px 28px ${color}25`; }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
              >
                {/* Glow orb in corner */}
                <div style={{
                  position: 'absolute', top: -20, right: -20,
                  width: 80, height: 80, borderRadius: '50%',
                  background: color, opacity: 0.07, filter: 'blur(20px)',
                  pointerEvents: 'none',
                }} />
                <div style={{ fontSize: 20, marginBottom: 8 }}>{icon}</div>
                <div style={{ fontSize: 10, color: color, fontWeight: 800, letterSpacing: 1.5, marginBottom: 8, opacity: 0.85 }}>
                  {label}
                </div>
                <div style={{
                  fontSize: 22, fontWeight: 900, fontFamily: 'var(--mono)',
                  color, lineHeight: 1.1, marginBottom: 8,
                  textShadow: `0 0 20px ${color}50`,
                }}>
                  {value}
                </div>
                <div style={{ fontSize: 11, color: '#475569' }}>{sub}</div>
              </div>
            ))}
          </div>
        );
      })()}

      {/* ── AI Suggestion Banner ─────────────────────────────────── */}
      <div style={{
        background: `linear-gradient(135deg, ${dirColor}10 0%, rgba(10,15,35,0) 100%)`,
        border: `1.5px solid ${dirColor}40`,
        borderRadius: 16, padding: '24px 28px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 24 }}>
          {/* Left: ticker + signal */}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 12 }}>
              💡 AI TRADING SUGGESTION
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 14 }}>
              <div style={{ position: 'relative' }}>
                <input
                  value={tradeSymbol}
                  onChange={e => setTradeSymbol(e.target.value.toUpperCase())}
                  placeholder="Ticker"
                  maxLength={10}
                  style={{
                    fontSize: 28, fontWeight: 900, fontFamily: 'var(--mono)',
                    color: '#f1f5f9', background: 'transparent',
                    border: 'none', borderBottom: '2px solid rgba(255,255,255,0.12)',
                    width: 140, outline: 'none', paddingBottom: 4,
                  }}
                />
                {quoteLoading && <div className="spinner" style={{ width: 12, height: 12, borderWidth: 2, position: 'absolute', right: 0, top: 8 }} />}
              </div>
              {dir && (
                <div style={{ padding: '6px 18px', borderRadius: 24, fontWeight: 800, fontSize: 15,
                  background: `${dirColor}18`, border: `1px solid ${dirColor}50`, color: dirColor }}>
                  {dir === 'LONG' ? '▲ BUY' : dir === 'SHORT' ? '▼ SELL' : '⚖️ HOLD'}
                </div>
              )}
              {analysisLoading && <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
            </div>

            {/* Metric pills */}
            {tradeAnalysis && !analysisLoading && (
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
                {[
                  { label: 'RL ACTION',     value: Number(tradeAnalysis?.rl_weights?.rl_action || 0).toFixed(4), color: dirColor },
                  { label: 'EFFECTIVE',     value: Number(tradeAnalysis?.rl_weights?.effective_action || 0).toFixed(4), color: '#60a5fa' },
                  { label: 'GATE',          value: `${(Number(tradeAnalysis?.rl_weights?.gate_value || 1) * 100).toFixed(1)}%`, color: '#94a3b8' },
                  { label: 'POSITION',      value: `${(effPos * 100).toFixed(1)}%`, color: dirColor },
                  { label: 'LIVE PRICE',    value: price > 0 ? fmtUSD(price) : '—', color: '#f1f5f9' },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 8, padding: '6px 12px',
                  }}>
                    <div style={{ fontSize: 10, color: '#475569', marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 14, fontFamily: 'var(--mono)', fontWeight: 700, color }}>{value}</div>
                  </div>
                ))}
              </div>
            )}

            {/* Pre-trade preview */}
            {canTrade && (
              <div style={{
                background: dir === 'LONG' ? 'rgba(0,229,160,0.06)' : 'rgba(255,79,114,0.06)',
                border: dir === 'LONG' ? '1px solid rgba(0,229,160,0.2)' : '1px solid rgba(255,79,114,0.2)',
                borderRadius: 10, padding: '14px 16px', marginBottom: 12,
              }}>
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, marginBottom: 10 }}>📊 PRE-TRADE PREVIEW</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                  {[
                    { label: 'SUGGESTED SHARES', value: sugShares.toFixed(4) },
                    { label: 'ORDER VALUE',       value: fmtUSD(sugShares * price) },
                    { label: 'ENTRY PRICE',       value: fmtUSD(price) },
                    dir === 'LONG'
                      ? { label: 'CASH AFTER',  value: fmtUSD((portfolio?.cash || 0) - sugShares * price),
                          color: (portfolio?.cash || 0) >= sugShares * price ? '#00e5a0' : '#ff4f72' }
                      : estPnL !== null
                        ? { label: 'EST. REALIZED P&L', value: (estPnL >= 0 ? '+' : '') + fmtUSD(estPnL),
                            color: estPnL >= 0 ? '#00e5a0' : '#ff4f72' }
                        : { label: 'CASH AFTER',  value: '+' + fmtUSD(sugShares * price), color: '#00e5a0' },
                  ].map(({ label, value, color = '#f1f5f9' }) => (
                    <div key={label} style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 8, padding: '10px 12px' }}>
                      <div style={{ fontSize: 10, color: '#64748b', fontWeight: 700, marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 15, fontWeight: 800, fontFamily: 'var(--mono)', color }}>{value}</div>
                    </div>
                  ))}
                </div>
                {dir === 'LONG' && (portfolio?.cash || 0) < sugShares * price && (
                  <div style={{ marginTop: 10, fontSize: 12, color: '#f59e0b', fontWeight: 600 }}>
                    ⚠️ Insufficient cash — reduce share count or sell a position first.
                  </div>
                )}
                {rlApplied && (
                  <div style={{ marginTop: 10, fontSize: 12, color: '#00e5a0', fontWeight: 600 }}>
                    ✅ Order form pre-filled with RL values — review below then Submit.
                  </div>
                )}
              </div>
            )}

            {tradeAnalysis && (
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.7 }}>
                Regime: <strong style={{ color: '#94a3b8' }}>{tradeAnalysis?.regime?.regime || tradeAnalysis?.regime?.dominant_regime || '—'}</strong>
                {(tradeAnalysis?.disagreement?.total_uncertainty || 0) > 0.5
                  ? ' · ⚠ High uncertainty — position size automatically reduced for safety.'
                  : ' · ✓ Stable conditions, agents in agreement.'}
              </div>
            )}
          </div>

          {/* Right: CTA button */}
          <div style={{ flexShrink: 0, textAlign: 'center' }}>
            <button
              type="button"
              onClick={applyRLSuggestion}
              disabled={!canTrade}
              style={{
                padding: '14px 28px', borderRadius: 12, border: 'none',
                background: canTrade
                  ? `linear-gradient(135deg, ${dirColor}ee, ${dirColor}88)`
                  : 'rgba(255,255,255,0.05)',
                color:  canTrade ? (dir === 'LONG' ? '#000' : '#fff') : '#475569',
                fontSize: 14, fontWeight: 800, cursor: canTrade ? 'pointer' : 'not-allowed',
                boxShadow: canTrade ? `0 4px 20px ${dirColor}40` : 'none',
                transition: 'transform 0.2s, box-shadow 0.2s',
                minWidth: 200,
              }}
              onMouseEnter={e => { if (canTrade) { e.currentTarget.style.transform = 'scale(1.04)'; e.currentTarget.style.boxShadow = `0 8px 28px ${dirColor}60`; }}}
              onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; e.currentTarget.style.boxShadow = canTrade ? `0 4px 20px ${dirColor}40` : 'none'; }}
            >
              {canTrade ? `⚡ Trade with RL Suggestion` : dir === 'FLAT' ? '🛑 RL Suggests FLAT' : '⏳ Loading…'}
            </button>
            {canTrade && (
              <div style={{ fontSize: 11, color: '#475569', marginTop: 8 }}>
                Fills order form below
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Order Form | Open Positions ──────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 24 }}>

        {/* Order Form */}
        <div className="card">
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 20 }}>
            EXECUTE ORDER
          </div>
          <form onSubmit={handleTrade} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div>
              <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, marginBottom: 6 }}>SYMBOL</div>
              <input
                value={tradeSymbol}
                onChange={e => setTradeSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                maxLength={10}
                required
                style={{
                  width: '100%', background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10,
                  padding: '11px 14px', color: '#f1f5f9', fontSize: 14, boxSizing: 'border-box',
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, marginBottom: 6 }}>ACTION</div>
                <select
                  value={tradeAction}
                  onChange={e => setTradeAction(e.target.value)}
                  style={{
                    width: '100%', background: 'rgba(255,255,255,0.04)',
                    border: `1px solid ${tradeAction === 'BUY' ? 'rgba(0,229,160,0.3)' : 'rgba(255,79,114,0.3)'}`,
                    borderRadius: 10, padding: '11px 14px',
                    color: tradeAction === 'BUY' ? '#00e5a0' : '#ff4f72',
                    fontSize: 14, fontWeight: 700,
                  }}
                >
                  <option value="BUY">▲ BUY</option>
                  <option value="SELL">▼ SELL</option>
                </select>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, marginBottom: 6 }}>SHARES</div>
                <input
                  type="number"
                  value={tradeShares}
                  onChange={e => setTradeShares(e.target.value)}
                  min="0.0001"
                  step="any"
                  required
                  style={{
                    width: '100%', background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10,
                    padding: '11px 14px', color: '#f1f5f9', fontSize: 14, boxSizing: 'border-box',
                  }}
                />
              </div>
            </div>

            {/* Order summary */}
            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 10, padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { label: 'Live Price',    value: quoteLoading ? 'Fetching…' : price > 0 ? fmtUSD(price) : '—' },
                { label: 'Order Value',   value: orderValue != null ? fmtUSD(orderValue) : '—' },
                ...(tradeAction === 'SELL' ? [{ label: 'Max Sellable', value: `${maxSellable.toFixed(4)} shares` }] : []),
              ].map(({ label, value }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: '#64748b' }}>{label}</span>
                  <span style={{ color: '#f1f5f9', fontFamily: 'var(--mono)', fontWeight: 600 }}>{value}</span>
                </div>
              ))}
              {invalidSell && (
                <div style={{ color: '#ff4f72', fontSize: 12, fontWeight: 600, marginTop: 4 }}>
                  ⚠ Exceeds holdings ({maxSellable.toFixed(4)} shares max)
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={isTrading || invalidSell || tradeQty <= 0}
              style={{
                padding: '13px', borderRadius: 10, border: 'none',
                background: isTrading || invalidSell || tradeQty <= 0
                  ? 'rgba(255,255,255,0.05)'
                  : tradeAction === 'BUY'
                    ? 'linear-gradient(135deg, #00e5a0, #059669)'
                    : 'linear-gradient(135deg, #ff4f72, #dc2626)',
                color: isTrading || invalidSell || tradeQty <= 0 ? '#475569' : '#fff',
                fontSize: 15, fontWeight: 800, cursor: 'pointer',
                boxShadow: tradeAction === 'BUY' ? '0 4px 16px rgba(0,229,160,0.3)' : '0 4px 16px rgba(255,79,114,0.3)',
                transition: 'transform 0.15s',
              }}
              onMouseEnter={e => { if (!isTrading && !invalidSell) e.currentTarget.style.transform = 'scale(1.02)'; }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; }}
            >
              {isTrading ? '⏳ Executing…' : `Submit ${tradeAction} Order`}
            </button>
          </form>
        </div>

        {/* Open Positions */}
        <div className="card">
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 20 }}>
            OPEN POSITIONS · {portfolio?.positions?.length || 0} holdings
          </div>
          {!portfolio?.positions?.length ? (
            <div style={{ textAlign: 'center', padding: '48px 0', color: '#475569' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📭</div>
              <div style={{ fontSize: 14 }}>No open positions</div>
              <div style={{ fontSize: 12, marginTop: 6, color: '#374151' }}>Use the order form to start trading</div>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '0 8px', fontSize: 13 }}>
                <thead>
                  <tr>
                    {['Symbol', 'Shares', 'Avg Cost', 'Live Price', 'Market Val', 'Unrealized P&L', 'Quick Actions'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '4px 14px', fontSize: 11, color: '#64748b', fontWeight: 700 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map(pos => (
                    <tr key={pos.symbol} style={{ background: 'rgba(255,255,255,0.02)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 900, color: '#f1f5f9', fontFamily: 'var(--mono)', fontSize: 15 }}>
                        {pos.symbol}
                      </td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--mono)', color: '#94a3b8' }}>{pos.shares.toFixed(4)}</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--mono)', color: '#94a3b8' }}>{fmtUSD(pos.avg_cost)}</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--mono)', color: '#f1f5f9', fontWeight: 700 }}>{fmtUSD(pos.live_price)}</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--mono)', color: '#94a3b8' }}>{fmtUSD(pos.market_value)}</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--mono)', fontWeight: 700,
                        color: pos.unrealized_pnl >= 0 ? '#00e5a0' : '#ff4f72' }}>
                        {pos.unrealized_pnl >= 0 ? '+' : ''}{fmtUSD(pos.unrealized_pnl)}
                        <span style={{ fontSize: 11, marginLeft: 6, opacity: 0.8 }}>
                          ({fmtPct(pos.unrealized_pct)})
                        </span>
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        <div style={{ display: 'flex', gap: 6 }}>
                          {[['25%', 0.25], ['50%', 0.5], ['All', 1]].map(([label, frac]) => (
                            <button key={label} type="button" onClick={() => quickSell(pos.symbol, frac)}
                              style={{
                                padding: '5px 10px', borderRadius: 8, border: `1px solid ${frac === 1 ? 'rgba(255,79,114,0.4)' : 'rgba(255,255,255,0.12)'}`,
                                background: frac === 1 ? 'rgba(255,79,114,0.1)' : 'rgba(255,255,255,0.04)',
                                color: frac === 1 ? '#ff4f72' : '#94a3b8',
                                fontSize: 12, fontWeight: 700, cursor: 'pointer',
                              }}>
                              Sell {label}
                            </button>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ── Trade History ─────────────────────────────────────────── */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5 }}>TRADE HISTORY</div>
          <div style={{ fontSize: 12, color: '#475569' }}>{history.length} executions</div>
        </div>
        {!history.length ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: '#475569' }}>
            <div style={{ fontSize: 28, marginBottom: 10 }}>📜</div>
            <div style={{ fontSize: 14 }}>No trades yet — execute your first order above</div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto', maxHeight: 320, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'separate', borderSpacing: '0 6px' }}>
              <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-card)', zIndex: 1 }}>
                <tr>
                  {['Date / Time', 'Action', 'Symbol', 'Shares', 'Price', 'Total Value'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 14px', fontSize: 11, color: '#64748b', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((t, i) => (
                  <tr key={t.id || i} style={{ background: 'rgba(255,255,255,0.02)' }}>
                    <td style={{ padding: '10px 14px', color: '#64748b' }}>{new Date(t.timestamp).toLocaleString()}</td>
                    <td style={{ padding: '10px 14px', fontWeight: 800,
                      color: t.action === 'BUY' ? '#00e5a0' : '#ff4f72' }}>
                      {t.action === 'BUY' ? '▲' : '▼'} {t.action}
                    </td>
                    <td style={{ padding: '10px 14px', fontWeight: 900, fontFamily: 'var(--mono)', color: '#f1f5f9' }}>{t.symbol}</td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', color: '#94a3b8' }}>{Number(t.shares).toFixed(4)}</td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', color: '#94a3b8' }}>{fmtUSD(t.price)}</td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--mono)', color: '#f1f5f9', fontWeight: 700 }}>{fmtUSD(t.trade_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
