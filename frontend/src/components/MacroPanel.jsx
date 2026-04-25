/**
 * MacroPanel v4.0 — Clean, easy-to-read macro data tiles
 */
import React from 'react';

function StatTile({ label, value, unit = '', color, badges = [] }) {
  const display = typeof value === 'number'
    ? `${value.toFixed(2)}${unit}`
    : (value ?? '—');

  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)',
      borderRadius: 10,
      padding: '12px 14px',
      border: '1px solid rgba(255,255,255,0.05)',
    }}>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 800, color: color || '#f1f5f9', fontFamily: 'var(--mono)' }}>
        {display}
      </div>
      {badges.map((b, i) => (
        <div key={i} style={{ marginTop: 6, fontSize: 11, color: b.color || '#64748b' }}>{b.text}</div>
      ))}
    </div>
  );
}

export default function MacroPanel({ macro }) {
  if (!macro) {
    return (
      <div className="card" style={{ minHeight: 150, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#475569' }}>🌍 Run analysis to see macro data</span>
      </div>
    );
  }

  const vixColor = macro.vix_level > 30 ? '#ff4f72' : macro.vix_level > 20 ? '#ffb830' : '#00e5a0';
  const yieldColor = macro.t10y2y_spread < 0 ? '#ff4f72' : macro.t10y2y_spread < 0.5 ? '#ffb830' : '#00e5a0';
  const dxyColor = macro.dxy_20d_momentum > 0.02 ? '#ff4f72' : macro.dxy_20d_momentum < -0.02 ? '#00e5a0' : '#94a3b8';
  const creditRisk = macro.hyg_lqd_zscore < -1.5;

  return (
    <div className="card">
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>MACRO CONTEXT</div>
        {macro.freshness_ts && (
          <div style={{ fontSize: 12, color: '#475569' }}>
            Updated {new Date(macro.freshness_ts).toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Volatility */}
      <div style={{ fontSize: 12, color: '#64748b', fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>VOLATILITY</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 20 }}>
        <StatTile label="VIX" value={macro.vix_level} color={vixColor}
          badges={macro.vix_level > 30 ? [{ text: '⚠ Fear Elevated', color: '#ff4f72' }] : []} />
        <StatTile label="VIX 9D" value={macro.vix9d_level} color={vixColor} />
        <StatTile label="Term Spread" value={macro.vix_ts_spread}
          color={macro.vix_ts_spread < 0 ? '#ff4f72' : '#00e5a0'}
          badges={[{ text: macro.vix_ts_spread < 0 ? 'Backwardation' : 'Contango', color: macro.vix_ts_spread < 0 ? '#ff4f72' : '#64748b' }]} />
        <StatTile label="VIX Z-Score" value={macro.vix_zscore} unit="σ" color={vixColor} />
      </div>

      {/* Credit */}
      <div style={{ fontSize: 12, color: '#64748b', fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>CREDIT MARKETS</div>
      {creditRisk && (
        <div style={{ padding: '10px 14px', background: 'rgba(255,79,114,0.1)', border: '1px solid rgba(255,79,114,0.3)', borderRadius: 10, fontSize: 13, color: '#ff4f72', marginBottom: 12, fontWeight: 600 }}>
          ⚠ Credit stress detected — HYG/LQD z-score at {macro.hyg_lqd_zscore?.toFixed(2)}σ
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 20 }}>
        <StatTile label="HYG/LQD Ratio" value={macro.hyg_lqd_ratio}
          color={creditRisk ? '#ff4f72' : '#00e5a0'} />
        <StatTile label="HYG Z-Score" value={macro.hyg_lqd_zscore} unit="σ"
          color={creditRisk ? '#ff4f72' : '#94a3b8'} />
        <StatTile label="HY Spread" value={macro.hy_credit_spread} unit=" bps"
          color={macro.hy_credit_spread > 500 ? '#ff4f72' : '#94a3b8'} />
      </div>

      {/* Rates & FX */}
      <div style={{ fontSize: 12, color: '#64748b', fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>RATES & FX</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
        <StatTile label="10Y−2Y Spread" value={macro.t10y2y_spread} unit="%" color={yieldColor}
          badges={[{ text: macro.t10y2y_spread < 0 ? '⚠ Inverted' : 'Normal', color: yieldColor }]} />
        <StatTile label="Fed Funds" value={macro.fed_funds_rate} unit="%" color="#94a3b8" />
        <StatTile label="DXY Momentum" value={(macro.dxy_20d_momentum || 0) * 100} unit="%" color={dxyColor}
          badges={[{ text: macro.dxy_20d_momentum > 0.02 ? '$ Strong (risk-off)' : macro.dxy_20d_momentum < -0.02 ? '$ Weak (risk-on)' : '$ Neutral', color: dxyColor }]} />
        <StatTile label="Consumer Sent." value={macro.consumer_sentiment} color="#94a3b8" />
      </div>
    </div>
  );
}
