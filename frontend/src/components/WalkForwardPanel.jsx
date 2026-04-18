/**
 * WalkForwardPanel v3.0
 * =======================
 * Displays per-fold walk-forward results, fold comparison chart,
 * and overall walk-forward efficiency score.
 */
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';

const COLORS = { positive: '#4ade80', negative: '#f87171' };

function FoldSummaryRow({ fold }) {
  const sharp = fold.test_sharpe ?? 0;
  return (
    <tr className={`fold-row ${sharp > 0 ? 'positive-fold' : 'negative-fold'}`}>
      <td>Fold {fold.fold + 1}</td>
      <td>{fold.train_bars}d train / {fold.test_bars}d test</td>
      <td style={{ color: sharp > 0 ? '#4ade80' : '#f87171' }}>{sharp.toFixed(3)}</td>
      <td>{(fold.test_sortino ?? 0).toFixed(3)}</td>
      <td style={{ color: (fold.test_maxdd ?? 0) > 0.15 ? '#f87171' : '#94a3b8' }}>
        {((fold.test_maxdd ?? 0) * 100).toFixed(1)}%
      </td>
      <td>{((fold.test_cvar ?? 0) * 100).toFixed(2)}%</td>
    </tr>
  );
}

export default function WalkForwardPanel({ folds, summary }) {
  const [activeTab, setActiveTab] = useState('folds');

  if (!folds || folds.length === 0) {
    return (
      <div className="card wf-card">
        <div className="card-header">
          <span className="card-icon">📊</span>
          <h3>Walk-Forward Analysis</h3>
        </div>
        <div className="no-data">
          Train a model and run backtest to see walk-forward results.
          <div className="no-data-sub">Enforces {folds?.burnin_bars ?? 252}-bar burn-in</div>
        </div>
      </div>
    );
  }

  const chartData = folds.map((f, i) => ({
    name:    `F${i + 1}`,
    sharpe:  f.test_sharpe ?? 0,
    maxdd:   -((f.test_maxdd ?? 0) * 100),
    cvar:    -((f.test_cvar ?? 0) * 100),
  }));

  const wfe = summary?.walk_forward_efficiency ?? 0;

  return (
    <div className="card wf-card">
      <div className="card-header">
        <span className="card-icon">📊</span>
        <h3>Walk-Forward Analysis</h3>
        <div className="wfe-badge" style={{
          color: wfe > 0.6 ? '#4ade80' : wfe > 0.3 ? '#facc15' : '#f87171'
        }}>
          WFE: {(wfe * 100).toFixed(0)}%
        </div>
      </div>

      {/* Summary row */}
      {summary && (
        <div className="wf-summary-row">
          <div className="wf-summary-item">
            <span className="wf-summary-label">Avg Test Sharpe</span>
            <span className="wf-summary-val" style={{
              color: (summary.mean_test_sharpe ?? 0) > 0 ? '#4ade80' : '#f87171'
            }}>
              {(summary.mean_test_sharpe ?? 0).toFixed(3)}
            </span>
          </div>
          <div className="wf-summary-item">
            <span className="wf-summary-label">Worst Fold</span>
            <span className="wf-summary-val" style={{ color: '#f87171' }}>
              {(summary.worst_test_sharpe ?? 0).toFixed(3)}
            </span>
          </div>
          <div className="wf-summary-item">
            <span className="wf-summary-label">Positive Folds</span>
            <span className="wf-summary-val">
              {((summary.pct_positive_folds ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
          <div className="wf-summary-item">
            <span className="wf-summary-label">Avg Max DD</span>
            <span className="wf-summary-val" style={{ color: '#facc15' }}>
              {((summary.mean_test_maxdd ?? 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Tab selector */}
      <div className="wf-tabs">
        {['folds', 'chart', 'regimes'].map(tab => (
          <button
            key={tab}
            className={`wf-tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'folds' ? '📋 Fold Table' : tab === 'chart' ? '📈 Sharpe Chart' : '🎯 Regimes'}
          </button>
        ))}
      </div>

      {/* Fold table */}
      {activeTab === 'folds' && (
        <div className="fold-table-wrap">
          <table className="fold-table">
            <thead>
              <tr>
                <th>Fold</th><th>Window</th><th>Sharpe</th>
                <th>Sortino</th><th>Max DD</th><th>CVaR 95%</th>
              </tr>
            </thead>
            <tbody>
              {folds.map((f, i) => <FoldSummaryRow key={i} fold={f} />)}
            </tbody>
          </table>
        </div>
      )}

      {/* Sharpe chart */}
      {activeTab === 'chart' && (
        <div className="wf-chart">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Bar dataKey="sharpe" name="Test Sharpe" radius={[4, 4, 0, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={d.sharpe > 0 ? COLORS.positive : COLORS.negative} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Regime breakdown */}
      {activeTab === 'regimes' && (
        <div className="regime-breakdown">
          {folds.map((f, i) => (
            <div key={i} className="regime-fold">
              <div className="regime-fold-header">Fold {i + 1} regime breakdown</div>
              {f.regime_breakdown && Object.entries(f.regime_breakdown).map(([reg, metrics]) => (
                <div key={reg} className="regime-fold-row">
                  <span className="regime-name">{reg}</span>
                  <span className="regime-sharpe" style={{
                    color: (metrics.sharpe ?? 0) > 0 ? '#4ade80' : '#f87171'
                  }}>
                    Sharpe: {(metrics.sharpe ?? 0).toFixed(3)}
                  </span>
                  <span className="regime-n">{metrics.n_bars ?? 0} bars</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
