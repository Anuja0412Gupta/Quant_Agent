import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const INITIAL_CAPITAL = 100000;

export default function PaperTradeTab({ currentTicker }) {
  const [portfolio, setPortfolio] = useState(null);
  const [history, setHistory] = useState([]);

  const [tradeSymbol, setTradeSymbol] = useState(currentTicker || 'AAPL');
  const [tradeAction, setTradeAction] = useState('BUY');
  const [tradeShares, setTradeShares] = useState(1);
  const [quotePrice, setQuotePrice] = useState(null);
  const [quoteLoading, setQuoteLoading] = useState(false);
  const [isTrading, setIsTrading] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const selectedPosition = portfolio?.positions?.find((p) => p.symbol === tradeSymbol);
  const maxSellable = Number(selectedPosition?.shares || 0);
  const tradeQty = Number(tradeShares || 0);
  const orderValue = quotePrice != null ? tradeQty * quotePrice : null;
  const invalidSellQty = tradeAction === 'SELL' && tradeQty > maxSellable;

  const hydratePortfolio = useCallback(async (raw) => {
    const positionsObj = raw?.positions || {};
    const symbols = Object.keys(positionsObj);

    const prices = {};
    await Promise.all(
      symbols.map(async (sym) => {
        try {
          const { data } = await axios.get(`${API}/price/${sym}?_t=${Date.now()}`, {
            timeout: 10000,
          });
          prices[sym] = Number(data?.current_price || 0);
        } catch {
          prices[sym] = Number(positionsObj[sym]?.avg_price || 0);
        }
      })
    );

    const positions = symbols.map((sym) => {
      const p = positionsObj[sym] || {};
      const shares = Number(p.quantity || 0);
      const avg_cost = Number(p.avg_price || 0);
      const live_price = Number(prices[sym] || avg_cost);
      const market_value = shares * live_price;
      const unrealized_pnl = (live_price - avg_cost) * shares;
      const unrealized_pct = avg_cost > 0 ? ((live_price - avg_cost) / avg_cost) * 100 : 0;
      return {
        symbol: sym,
        shares,
        avg_cost,
        live_price,
        market_value,
        unrealized_pnl,
        unrealized_pct,
      };
    });

    const total_market_value = positions.reduce((acc, p) => acc + p.market_value, 0);
    const total_unrealized_pnl = positions.reduce((acc, p) => acc + p.unrealized_pnl, 0);
    const total_value = Number(raw?.cash || 0) + total_market_value;
    const total_return_pct = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100;

    setPortfolio({
      cash: Number(raw?.cash || 0),
      total_market_value,
      total_value,
      total_return_pct,
      realized_pnl: Number(raw?.realized_pnl || 0),
      unrealized_pnl: total_unrealized_pnl,
      positions,
    });

    const hist = Array.isArray(raw?.trade_history) ? [...raw.trade_history].reverse() : [];
    setHistory(hist.map((t) => ({
      id: t.trade_id,
      timestamp: t.timestamp,
      action: t.action,
      symbol: t.symbol,
      shares: Number(t.quantity || 0),
      price: Number(t.price || 0),
      trade_value: Number(t.value || 0),
      realized_pnl: 0,
    })));
  }, []);

  const fetchPortfolio = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API}/portfolio?_t=${Date.now()}`);
      await hydratePortfolio(data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load portfolio');
    } finally {
      setLoading(false);
    }
  }, [hydratePortfolio]);

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 10000);
    return () => clearInterval(interval);
  }, [fetchPortfolio]);

  useEffect(() => {
    if (currentTicker && currentTicker !== tradeSymbol) {
      setTradeSymbol(currentTicker);
    }
  }, [currentTicker, tradeSymbol]);

  useEffect(() => {
    let mounted = true;
    const sym = String(tradeSymbol || '').trim().toUpperCase();
    if (!sym) {
      setQuotePrice(null);
      return;
    }
    setQuoteLoading(true);
    axios.get(`${API}/price/${sym}`, { timeout: 10000 })
      .then(({ data }) => {
        if (mounted) setQuotePrice(Number(data?.current_price || 0));
      })
      .catch(() => {
        if (mounted) setQuotePrice(null);
      })
      .finally(() => {
        if (mounted) setQuoteLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [tradeSymbol]);

  const quickSell = (symbol, fraction) => {
    const pos = portfolio?.positions?.find((p) => p.symbol === symbol);
    if (!pos) return;
    const qty = Number((pos.shares * fraction).toFixed(4));
    if (qty <= 0) return;
    setTradeSymbol(symbol);
    setTradeAction('SELL');
    setTradeShares(qty);
  };

  const handleTrade = async (e) => {
    e.preventDefault();
    if (!tradeSymbol || Number(tradeShares) <= 0) return;
    if (invalidSellQty) {
      setError(`Cannot sell ${tradeQty} shares. You only hold ${maxSellable.toFixed(4)} ${tradeSymbol}.`);
      return;
    }

    setIsTrading(true);
    setError(null);
    try {
      await axios.post(`${API}/portfolio/trade`, {
        symbol: tradeSymbol,
        action: tradeAction,
        quantity: parseFloat(tradeShares),
      });
      await fetchPortfolio();
      setTradeShares(1);
    } catch (err) {
      setError(err.response?.data?.detail || 'Trade failed');
    } finally {
      setIsTrading(false);
    }
  };

  if (loading && !portfolio) {
    return (
      <div className="loading-state">
        <div className="spinner" />
        <div className="loading-text">Loading Paper Portfolio...</div>
      </div>
    );
  }

  return (
    <div className="paper-trade-tab" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {error && <div className="error-message">Warning: {error}</div>}

      {portfolio && (
        <div className="metrics-grid">
          <div className="metric-tile">
            <div className="mt-label">Available Cash</div>
            <div className="mt-value" style={{ color: '#68d391' }}>
              ${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-tile">
            <div className="mt-label">Market Value</div>
            <div className="mt-value">
              ${portfolio.total_market_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-tile">
            <div className="mt-label">Total Portfolio Value</div>
            <div className="mt-value" style={{ color: '#4f7dff' }}>
              ${portfolio.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-tile">
            <div className="mt-label">Total Return</div>
            <div className={`mt-value ${portfolio.total_return_pct >= 0 ? 'mt-positive' : 'mt-negative'}`}>
              {portfolio.total_return_pct > 0 ? '+' : ''}{portfolio.total_return_pct.toFixed(2)}%
            </div>
          </div>
          <div className="metric-tile">
            <div className="mt-label">Unrealized PnL</div>
            <div className={`mt-value ${portfolio.unrealized_pnl >= 0 ? 'mt-positive' : 'mt-negative'}`}>
              {portfolio.unrealized_pnl >= 0 ? '+' : ''}${portfolio.unrealized_pnl.toFixed(2)}
            </div>
          </div>
          <div className="metric-tile">
            <div className="mt-label">Realized PnL</div>
            <div className={`mt-value ${portfolio.realized_pnl >= 0 ? 'mt-positive' : 'mt-negative'}`}>
              {portfolio.realized_pnl >= 0 ? '+' : ''}${portfolio.realized_pnl.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 2fr', gap: 24 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Execute Trade</span>
          </div>
          <form onSubmit={handleTrade} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <label style={{ fontSize: 12, color: 'var(--text-2)', textTransform: 'uppercase' }}>Symbol</label>
              <input
                className="ticker-input"
                style={{ width: '100%' }}
                value={tradeSymbol}
                onChange={(e) => setTradeSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                maxLength={10}
                required
              />
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <label style={{ fontSize: 12, color: 'var(--text-2)', textTransform: 'uppercase' }}>Action</label>
                <select className="tf-select" style={{ width: '100%' }} value={tradeAction} onChange={(e) => setTradeAction(e.target.value)}>
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <label style={{ fontSize: 12, color: 'var(--text-2)', textTransform: 'uppercase' }}>Shares</label>
                <input
                  type="number"
                  className="ticker-input"
                  style={{ width: '100%' }}
                  value={tradeShares}
                  onChange={(e) => setTradeShares(e.target.value)}
                  min="0.0001"
                  step="any"
                  required
                />
              </div>
            </div>

            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-2)' }}>
                <span>Live Price</span>
                <span style={{ fontFamily: 'var(--mono)', color: 'var(--text-1)' }}>
                  {quoteLoading ? 'Loading...' : (quotePrice != null ? `$${quotePrice.toFixed(2)}` : '-')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-2)', marginTop: 6 }}>
                <span>Estimated Order Value</span>
                <span style={{ fontFamily: 'var(--mono)', color: 'var(--text-1)' }}>
                  {orderValue != null ? `$${orderValue.toFixed(2)}` : '-'}
                </span>
              </div>
              {tradeAction === 'SELL' && (
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: invalidSellQty ? '#ff6b6b' : 'var(--text-2)', marginTop: 6 }}>
                  <span>Sellable Shares</span>
                  <span style={{ fontFamily: 'var(--mono)' }}>{maxSellable.toFixed(4)}</span>
                </div>
              )}
            </div>

            <button
              type="submit"
              className="btn-analyze"
              style={{ padding: '12px', marginTop: 8 }}
              disabled={isTrading || invalidSellQty || tradeQty <= 0}
            >
              {isTrading ? 'Executing...' : `Submit ${tradeAction} Order`}
            </button>
          </form>
        </div>

        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header" style={{ marginBottom: 20 }}>
            <span className="card-title">Open Positions</span>
          </div>

          {portfolio?.positions?.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 150 }}>
              <div className="empty-text">No open positions.</div>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-2)' }}>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Symbol</th>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Shares</th>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Avg Cost</th>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Live Price</th>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Market Val</th>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Unrealized</th>
                    <th style={{ padding: '8px 12px', fontWeight: 600 }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio?.positions.map((pos) => (
                    <tr key={pos.symbol} style={{ borderBottom: '1px solid var(--bg-3)' }}>
                      <td style={{ padding: '12px', fontWeight: 700, fontFamily: 'var(--mono)' }}>{pos.symbol}</td>
                      <td style={{ padding: '12px', fontFamily: 'var(--mono)' }}>{pos.shares.toFixed(4)}</td>
                      <td style={{ padding: '12px', fontFamily: 'var(--mono)' }}>${pos.avg_cost.toFixed(2)}</td>
                      <td style={{ padding: '12px', fontFamily: 'var(--mono)' }}>${pos.live_price.toFixed(2)}</td>
                      <td style={{ padding: '12px', fontFamily: 'var(--mono)' }}>${pos.market_value.toFixed(2)}</td>
                      <td style={{ padding: '12px', fontFamily: 'var(--mono)', color: pos.unrealized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {pos.unrealized_pnl >= 0 ? '+' : ''}{pos.unrealized_pnl.toFixed(2)} ({pos.unrealized_pct.toFixed(2)}%)
                      </td>
                      <td style={{ padding: '12px', minWidth: 200 }}>
                        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                          <button type="button" className="btn-backtest" style={{ padding: '4px 8px', fontSize: 11 }} onClick={() => quickSell(pos.symbol, 0.25)}>
                            Sell 25%
                          </button>
                          <button type="button" className="btn-backtest" style={{ padding: '4px 8px', fontSize: 11 }} onClick={() => quickSell(pos.symbol, 0.5)}>
                            Sell 50%
                          </button>
                          <button type="button" className="btn-backtest" style={{ padding: '4px 8px', fontSize: 11, borderColor: '#ff6b6b', color: '#ff6b6b' }} onClick={() => quickSell(pos.symbol, 1)}>
                            Sell All
                          </button>
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

      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Trades</span>
          <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{history.length} trades</span>
        </div>

        {history.length === 0 ? (
          <div className="empty-state" style={{ minHeight: 100 }}>
            <div className="empty-text">No trade history yet.</div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-card)', zIndex: 1 }}>
                <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-2)' }}>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Date/Time</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Action</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Symbol</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Shares</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Price</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Total Value</th>
                </tr>
              </thead>
              <tbody>
                {history.map((t, idx) => (
                  <tr key={t.id || idx} style={{ borderBottom: '1px solid var(--bg-3)' }}>
                    <td style={{ padding: '10px 12px', color: 'var(--text-1)' }}>{new Date(t.timestamp).toLocaleString()}</td>
                    <td style={{ padding: '10px 12px', fontWeight: 700, color: t.action === 'BUY' ? 'var(--accent-green)' : 'var(--accent-red)' }}>{t.action}</td>
                    <td style={{ padding: '10px 12px', fontWeight: 700, fontFamily: 'var(--mono)' }}>{t.symbol}</td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)' }}>{t.shares}</td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)' }}>${Number(t.price).toFixed(2)}</td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--mono)' }}>${Number(t.trade_value).toFixed(2)}</td>
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
