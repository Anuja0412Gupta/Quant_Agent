/**
 * RegimePanel v4.0 — Clean, highly readable market regime display
 */
import React from 'react';

const REGIME_CFG = {
  trending:        { fill: '#00e5a0', label: 'Trending',        icon: '📈', desc: 'Momentum / directional move' },
  mean_reverting:  { fill: '#60a5fa', label: 'Mean-Reverting',  icon: '↔️', desc: 'Price oscillating around a mean' },
  high_volatility: { fill: '#ff4f72', label: 'High Volatility', icon: '⚡', desc: 'Elevated risk, wider swings' },
};

function ProbBar({ label, icon, prob, fill, isDominant }) {
  const pct = (prob * 100).toFixed(1);
  return (
    <div style={{
      padding: '12px 16px',
      background: isDominant ? `${fill}14` : 'rgba(255,255,255,0.03)',
      borderRadius: 10,
      border: isDominant ? `1px solid ${fill}40` : '1px solid rgba(255,255,255,0.05)',
      transition: 'all 0.3s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 18 }}>{icon}</span>
          <span style={{ color: isDominant ? fill : '#94a3b8', fontWeight: isDominant ? 700 : 400, fontSize: 14 }}>
            {label}
          </span>
          {isDominant && (
            <span style={{ fontSize: 10, background: `${fill}22`, color: fill, padding: '2px 8px', borderRadius: 20, fontWeight: 700, letterSpacing: 1 }}>
              DOMINANT
            </span>
          )}
        </div>
        <span style={{ color: isDominant ? fill : '#64748b', fontFamily: 'var(--mono)', fontSize: 15, fontWeight: 700 }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: fill, borderRadius: 3, transition: 'width 0.8s ease' }} />
      </div>
    </div>
  );
}

export default function RegimePanel({ regime }) {
  if (!regime) {
    return (
      <div className="card" style={{ minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#475569' }}>📡 Run analysis to detect market regime</span>
      </div>
    );
  }

  const dominant = regime.dominant_regime || regime.regime || 'trending';
  const cfg      = REGIME_CFG[dominant] || { fill: '#94a3b8', label: dominant, icon: '?', desc: '' };
  const stability    = regime.regime_stability ?? 1.0;
  const cpProb       = regime.changepoint_probability ?? 0.0;
  const confidence   = regime.confidence ?? 0.0;

  const bars = [
    { key: 'trending',        prob: regime.p_trending        || 0 },
    { key: 'mean_reverting',  prob: regime.p_mean_reverting  || 0 },
    { key: 'high_volatility', prob: regime.p_high_volatility || 0 },
  ];

  return (
    <div className="card">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>
            MARKET REGIME
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, color: cfg.fill, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>{cfg.icon}</span> {cfg.label.toUpperCase()}
          </div>
          <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>{cfg.desc}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>HMM Confidence</div>
          <div style={{ fontSize: 28, fontFamily: 'var(--mono)', fontWeight: 800, color: cfg.fill }}>
            {(confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Probability bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 }}>
        {bars.map(b => {
          const c = REGIME_CFG[b.key] || { fill: '#94a3b8', label: b.key, icon: '?' };
          return (
            <ProbBar key={b.key} label={c.label} icon={c.icon}
              prob={b.prob} fill={c.fill} isDominant={b.key === dominant} />
          );
        })}
      </div>

      {/* 2-stat footer */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '12px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>STABILITY</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: stability > 0.6 ? '#00e5a0' : stability > 0.3 ? '#ffb830' : '#ff4f72' }}>
            {(stability * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '12px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{cpProb > 0.6 ? '⚠ CHANGE RISK' : 'CHANGE PROB'}</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: cpProb > 0.6 ? '#ff4f72' : cpProb > 0.3 ? '#ffb830' : '#00e5a0' }}>
            {(cpProb * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {regime.explanation && (
        <div style={{ marginTop: 16, padding: '12px 16px', background: 'rgba(255,255,255,0.03)', borderRadius: 10, fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>
          {regime.explanation}
        </div>
      )}
    </div>
  );
}
