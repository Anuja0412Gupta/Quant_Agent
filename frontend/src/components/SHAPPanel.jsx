/**
 * SHAPPanel — SHAP feature attributions
 * ======================================
 * Visualizes which agent features drove the RL action.
 * Populated from /analyze/{symbol} response (shap key),
 * or from a dedicated SHAP fetch if available.
 */
export default function SHAPPanel({ shap }) {
  if (!shap) return (
    <div className="card" style={{ marginTop: 16, opacity: 0.6 }}>
      <div className="card-header">
        <span className="card-title">📊 SHAP Attribution</span>
        <span style={{ fontSize: 11, color: '#647091' }}>Loading…</span>
      </div>
      <div style={{ padding: '24px', textAlign: 'center', color: '#647091', fontSize: 13 }}>
        SHAP values not yet computed. Trigger an analysis to see feature attributions.
      </div>
    </div>
  );

  const { shap_values = {}, top_features = [], action = 0, method = '—', active_regime = '—' } = shap;

  const maxAbs = Math.max(...Object.values(shap_values).map(Math.abs), 0.001);

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-header">
        <span className="card-title">📊 SHAP Attribution</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: '#8b9fc0' }}>{method}</span>
          <span style={{ fontSize: 11, color: '#647091' }}>Regime: {active_regime}</span>
        </div>
      </div>

      <div style={{ marginBottom: 10, padding: '8px 12px', borderRadius: 8,
                    background: 'rgba(255,255,255,0.04)', fontSize: 12, color: '#a0aec0' }}>
        RL Action = <span style={{ color: action >= 0 ? '#68d391' : '#fc8181', fontWeight: 700, fontFamily: 'monospace' }}>
          {action.toFixed(4)}
        </span>
        &nbsp;{'  '}↑ longer bars = stronger influence on this action
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {Object.entries(shap_values)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .slice(0, 10)
          .map(([feat, val]) => {
            const pct   = (Math.abs(val) / maxAbs) * 100;
            const pos   = val >= 0;
            const label = feat.replace(/_/g, ' ');
            return (
              <div key={feat} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 150, fontSize: 11, color: '#8b9fc0', textAlign: 'right',
                               textTransform: 'capitalize', flexShrink: 0 }}>
                  {label}
                </span>
                <div style={{ flex: 1, height: 16, background: 'rgba(255,255,255,0.05)', borderRadius: 4,
                               position: 'relative', overflow: 'hidden' }}>
                  <div style={{
                    position: 'absolute',
                    [pos ? 'left' : 'right']: '50%',
                    width: `${pct / 2}%`,
                    height: '100%',
                    background: pos
                      ? 'linear-gradient(90deg, rgba(104,211,145,0.7), rgba(104,211,145,0.3))'
                      : 'linear-gradient(270deg, rgba(252,129,129,0.7), rgba(252,129,129,0.3))',
                    borderRadius: pos ? '0 4px 4px 0' : '4px 0 0 4px',
                    transition: 'width 0.5s ease',
                  }} />
                  <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0,
                                 width: 1, background: 'rgba(255,255,255,0.15)' }} />
                </div>
                <span style={{ width: 60, fontSize: 11, fontFamily: 'monospace', textAlign: 'left',
                                color: pos ? '#68d391' : '#fc8181', flexShrink: 0 }}>
                  {val >= 0 ? '+' : ''}{val.toFixed(4)}
                </span>
              </div>
            );
          })}
      </div>

      {top_features.length > 0 && (
        <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8,
                      background: 'rgba(99,102,241,0.1)', fontSize: 11, color: '#a0aec0' }}>
          <strong style={{ color: '#c3dafe' }}>Top driver:</strong>{' '}
          <span style={{ color: top_features[0]?.direction === 'positive' ? '#68d391' : '#fc8181' }}>
            {top_features[0]?.feature}
          </span>{' '}
          ({top_features[0]?.direction === 'positive' ? '↑' : '↓'} action by {Math.abs(top_features[0]?.shap ?? 0).toFixed(4)})
        </div>
      )}
    </div>
  );
}
