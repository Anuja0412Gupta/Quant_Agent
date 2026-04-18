/**
 * RegimePanel v3.0 — HMM Regime + BOCPD Changepoint
 * =====================================================
 * Animated regime probability bars, changepoint probability meter,
 * and regime stability indicator.
 */
import React from 'react';

const REGIME_COLORS = {
  trending:        { fill: '#4ade80', label: 'Trending',        icon: '📈' },
  mean_reverting:  { fill: '#60a5fa', label: 'Mean-Reverting',  icon: '↔️' },
  high_volatility: { fill: '#f87171', label: 'High Volatility', icon: '⚡' },
};

function RegimeBar({ regime, probability, isDominant }) {
  const cfg   = REGIME_COLORS[regime] || { fill: '#94a3b8', label: regime, icon: '?' };
  const width = Math.round(probability * 100);

  return (
    <div className={`regime-bar-wrap ${isDominant ? 'dominant' : ''}`}>
      <div className="regime-bar-header">
        <span className="regime-icon">{cfg.icon}</span>
        <span className="regime-label">{cfg.label}</span>
        {isDominant && <span className="regime-dominant-tag">DOMINANT</span>}
        <span className="regime-prob" style={{ color: cfg.fill }}>
          {(probability * 100).toFixed(1)}%
        </span>
      </div>
      <div className="regime-track">
        <div
          className="regime-fill"
          style={{ width: `${width}%`, background: cfg.fill }}
        />
      </div>
    </div>
  );
}

function ChangePointMeter({ probability }) {
  const pct   = Math.round(probability * 100);
  const color = probability > 0.6 ? '#f87171'
              : probability > 0.3 ? '#facc15'
              : '#4ade80';
  return (
    <div className="cp-meter">
      <div className="cp-meter-header">
        <span>BOCPD Changepoint Probability</span>
        <span style={{ color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div className="cp-track">
        <div
          className="cp-fill"
          style={{
            width:      `${pct}%`,
            background: `linear-gradient(90deg, #4ade80, ${color})`,
          }}
        />
      </div>
      {probability > 0.6 && (
        <div className="cp-alert">⚠ Regime transition likely — size reduced</div>
      )}
    </div>
  );
}

export default function RegimePanel({ regime }) {
  if (!regime) {
    return (
      <div className="card regime-card">
        <div className="card-header">
          <span className="card-icon">🎯</span>
          <h3>Market Regime</h3>
        </div>
        <div className="no-data">Run analysis to see regime detection</div>
      </div>
    );
  }

  const dominant = regime.dominant_regime || regime.regime || 'trending';
  const cfg      = REGIME_COLORS[dominant] || { fill: '#94a3b8', label: dominant, icon: '?' };

  const bars = [
    { regime: 'trending',        prob: regime.p_trending       || 0 },
    { regime: 'mean_reverting',  prob: regime.p_mean_reverting || 0 },
    { regime: 'high_volatility', prob: regime.p_high_volatility || 0 },
  ];

  const stability    = regime.regime_stability       ?? 1.0;
  const cpProb       = regime.changepoint_probability ?? 0.0;
  const isTransition = regime.is_transition           ?? false;
  const confidence   = regime.confidence              ?? 0.0;

  return (
    <div className="card regime-card">
      <div className="card-header">
        <span className="card-icon">🎯</span>
        <h3>Market Regime</h3>
        <div className="regime-badge" style={{ color: cfg.fill, borderColor: cfg.fill }}>
          {cfg.icon} {cfg.label}
        </div>
      </div>

      {/* Regime probability bars */}
      <div className="regime-bars">
        {bars.map(b => (
          <RegimeBar
            key={b.regime}
            regime={b.regime}
            probability={b.prob}
            isDominant={b.regime === dominant}
          />
        ))}
      </div>

      {/* HMM confidence */}
      <div className="regime-meta">
        <div className="meta-item">
          <span className="meta-label">HMM Confidence</span>
          <span className="meta-val" style={{ color: cfg.fill }}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Regime Stability</span>
          <span className="meta-val" style={{
            color: stability > 0.6 ? '#4ade80' : stability > 0.3 ? '#facc15' : '#f87171'
          }}>
            {(stability * 100).toFixed(0)}%
          </span>
        </div>
        {isTransition && (
          <div className="meta-item">
            <span className="transition-tag">⚡ IN TRANSITION</span>
          </div>
        )}
      </div>

      {/* BOCPD changepoint meter */}
      <ChangePointMeter probability={cpProb} />

      {/* Explanation */}
      {regime.explanation && (
        <div className="regime-explanation">{regime.explanation}</div>
      )}
    </div>
  );
}
