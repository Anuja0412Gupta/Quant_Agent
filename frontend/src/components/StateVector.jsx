import { useMemo } from 'react';

/**
 * StateVector v4.0 — Premium RL feature inspector
 */
export default function StateVector({ analysis }) {
  if (!analysis) return null;

  const rl  = analysis.rl_weights  || {};
  const dis = analysis.disagreement || {};
  const reg = analysis.regime       || {};

  const list   = Array.isArray(analysis.agent_signals) ? analysis.agent_signals : [];
  const byName = Object.fromEntries(list.map(a => [String(a?.agent_name || '').toLowerCase(), a]));
  const indicator = byName.indicator || {};
  const pattern   = byName.pattern   || {};
  const trend     = byName.trend     || {};

  // Group features into sections for readability
  const sections = useMemo(() => [
    {
      title: '🤖 RL Action Output',
      color: '#60a5fa',
      features: [
        { label: 'Raw RL Action',      value: rl.rl_action,         numeric: true,  desc: 'Unscaled model output' },
        { label: 'Gate Value',         value: rl.gate_value,        numeric: true,  desc: 'Risk gate multiplier' },
        { label: 'Effective Action',   value: rl.effective_action,  numeric: true,  desc: 'Post-gate action' },
        { label: 'Effective Position', value: rl.effective_position,numeric: true,  desc: 'Final position %' },
        { label: 'Regime Policy',      value: rl.active_regime,     numeric: false, desc: 'Active RL sub-policy' },
        { label: 'Direction',          value: rl.direction,         numeric: false, desc: 'Buy / Sell / Flat' },
      ],
    },
    {
      title: '🧠 Agent Inputs',
      color: '#a78bfa',
      features: [
        { label: 'Indicator Signal',   value: indicator.signal ?? indicator.reasoning?.signal, numeric: false },
        { label: 'Indicator Conf',     value: indicator.confidence ?? indicator.reasoning?.confidence, numeric: true },
        { label: 'Pattern Signal',     value: pattern.signal ?? pattern.reasoning?.signal,   numeric: false },
        { label: 'Pattern Conf',       value: pattern.confidence ?? pattern.reasoning?.confidence, numeric: true },
        { label: 'Trend Direction',    value: trend.signal ?? trend.reasoning?.trend,         numeric: false },
        { label: 'Trend Conf',         value: trend.confidence ?? trend.reasoning?.confidence, numeric: true },
      ],
    },
    {
      title: '📡 Market State',
      color: '#00e5a0',
      features: [
        { label: 'Regime',             value: reg.regime ?? reg.signal,  numeric: false },
        { label: 'Regime Confidence',  value: reg.confidence,            numeric: true  },
        { label: 'Hurst Exponent',     value: reg.hurst,                 numeric: true,  desc: '>0.5 trending, <0.5 mean-rev' },
        { label: 'ATR Ratio',          value: reg.atr_ratio,             numeric: true,  desc: 'Volatility / price' },
        { label: 'Disagreement Score', value: dis.disagreement_score ?? dis.total_uncertainty, numeric: true },
        { label: 'Agent Consensus',    value: dis.agent_consensus,       numeric: true  },
      ],
    },
  ], [rl, dis, reg, indicator, pattern, trend]);

  function formatVal(value, numeric) {
    if (value === undefined || value === null || value === '') return '—';
    if (!numeric) return String(value).toUpperCase();
    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);
    // Show more precision for tiny numbers
    if (Math.abs(n) < 0.001) return n.toExponential(2);
    if (Math.abs(n) < 0.1)   return n.toFixed(4);
    return n.toFixed(3);
  }

  function valColor(value, numeric, label) {
    if (!numeric || value === undefined || value === null) {
      // For direction / signal strings
      const s = String(value || '').toUpperCase();
      if (['BUY', 'BULLISH', 'UPTREND', 'TRENDING'].includes(s)) return '#00e5a0';
      if (['SELL', 'BEARISH', 'DOWNTREND'].includes(s)) return '#ff4f72';
      if (['NEUTRAL', 'HOLD', 'FLAT'].includes(s)) return '#ffb830';
      return '#94a3b8';
    }
    const n = Number(value);
    if (!Number.isFinite(n)) return '#94a3b8';
    if (n > 0.05)  return '#00e5a0';
    if (n < -0.05) return '#ff4f72';
    return '#f1f5f9';
  }

  return (
    <div className="card">
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>RL STATE VECTOR</div>
        <div style={{ fontSize: 12, color: '#475569' }}>{analysis.feature_dim || 16} features · point-in-time · no lookahead</div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {sections.map(section => (
          <div key={section.title}>
            {/* Section header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <div style={{ width: 3, height: 16, background: section.color, borderRadius: 2 }} />
              <span style={{ fontSize: 12, fontWeight: 700, color: section.color }}>{section.title}</span>
            </div>

            {/* Feature rows */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {section.features.map(({ label, value, numeric, desc }) => (
                <div key={label} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.04)',
                  borderRadius: 8, padding: '9px 12px',
                }}>
                  <div>
                    <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>{label}</div>
                    {desc && <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>{desc}</div>}
                  </div>
                  <span style={{
                    fontSize: 13, fontWeight: 700,
                    color: valColor(value, numeric, label),
                    fontFamily: numeric ? 'var(--mono)' : 'inherit',
                    textAlign: 'right',
                  }}>
                    {formatVal(value, numeric)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16, padding: '10px 14px', background: 'rgba(255,255,255,0.02)', borderRadius: 8, fontSize: 11, color: '#475569', textAlign: 'center' }}>
        🔒 Rolling-window features only · No full-dataset leakage · Burn-in: {analysis.burnin_bars || 252} bars
      </div>
    </div>
  );
}
