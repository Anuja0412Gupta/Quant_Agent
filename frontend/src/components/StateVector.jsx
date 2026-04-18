import { useMemo } from 'react';

/**
 * StateVector - Full RL input display
 * Shows core features fed into the RL state vector
 */
export default function StateVector({ analysis }) {
  if (!analysis) return null;

  const rl = analysis.rl_weights || {};
  const dis = analysis.disagreement || {};
  const reg = analysis.regime || {};

  const list = Array.isArray(analysis.agent_signals) ? analysis.agent_signals : [];
  const byName = Object.fromEntries(
    list.map((a) => [String(a?.agent_name || '').toLowerCase(), a])
  );

  const indicator = byName.indicator || {};
  const pattern = byName.pattern || {};
  const trend = byName.trend || {};

  const features = useMemo(() => [
    { label: 'Indicator Signal', value: indicator.signal ?? indicator.reasoning?.signal ?? '-', numeric: false },
    { label: 'Indicator Conf', value: indicator.confidence ?? indicator.reasoning?.confidence, numeric: true },
    { label: 'Pattern Signal', value: pattern.signal ?? pattern.reasoning?.signal ?? '-', numeric: false },
    { label: 'Pattern Conf', value: pattern.confidence ?? pattern.reasoning?.confidence, numeric: true },
    { label: 'Trend Signal', value: trend.signal ?? trend.reasoning?.trend ?? '-', numeric: false },
    { label: 'Trend Conf', value: trend.confidence ?? trend.reasoning?.confidence, numeric: true },
    { label: 'Regime', value: reg.regime ?? reg.signal ?? '-', numeric: false },
    { label: 'Regime Conf', value: reg.confidence, numeric: true },
    { label: 'ATR Ratio', value: reg.atr_ratio, numeric: true },
    { label: 'Disagreement Score', value: dis.disagreement_score ?? dis.total_uncertainty, numeric: true },
    { label: 'Hurst Exponent', value: reg.hurst, numeric: true },
    { label: 'RL Action (raw)', value: rl.rl_action, numeric: true },
    { label: 'Effective Action', value: rl.effective_action, numeric: true },
    { label: 'Gate Value', value: rl.gate_value, numeric: true },
    { label: 'Effective Position', value: rl.effective_position, numeric: true },
    { label: 'Active Regime Policy', value: rl.active_regime ?? '-', numeric: false },
  ], [indicator, pattern, trend, reg, dis, rl]);

  const getColor = (val, numeric) => {
    if (!numeric) return '#a0aec0';
    if (val === undefined || val === null) return '#a0aec0';
    if (val > 0.3) return '#68d391';
    if (val < -0.3) return '#fc8181';
    return '#f6e05e';
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-header">
        <span className="card-title">RL State Vector</span>
        <span style={{ fontSize: 11, color: '#647091' }}>16 features - point-in-time normalized</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 16px', padding: '12px 0' }}>
        {features.map(({ label, value, numeric }) => (
          <div
            key={label}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '5px 8px',
              borderRadius: 6,
              background: 'rgba(255,255,255,0.03)',
            }}
          >
            <span style={{ fontSize: 11, color: '#8b9fc0' }}>{label}</span>
            <span
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: getColor(value, numeric),
                fontFamily: numeric ? 'monospace' : 'inherit',
              }}
            >
              {numeric
                ? (value !== undefined && value !== null ? Number(value).toFixed(4) : '-')
                : (value || '-')}
            </span>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11, color: '#3a4466', marginTop: 4, textAlign: 'center' }}>
        Rolling-window features only (no full-dataset leakage)
      </div>
    </div>
  );
}
