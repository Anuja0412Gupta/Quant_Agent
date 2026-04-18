/**
 * DisagreementHeatmap
 * Color-coded matrix showing agent signal alignment / conflict.
 */

const DIRECTION_MAP = {
  bullish: 1,
  uptrend: 1,
  trending: 0.5,
  bearish: -1,
  downtrend: -1,
  mean_reverting: -0.3,
  neutral: 0,
  sideways: 0,
  high_volatility: -0.5,
  low_volatility: 0.2,
};

export default function DisagreementHeatmap({ analysis }) {
  if (!analysis) return null;

  const dis = analysis.disagreement || {};
  const agents = ['indicator', 'pattern', 'trend', 'regime'];

  const fromList = Array.isArray(analysis.agent_signals) ? analysis.agent_signals : [];
  const listMap = Object.fromEntries(
    fromList.map((a) => [String(a?.agent_name || '').toLowerCase(), a])
  );

  const signals = { ...(dis.agent_signals || {}) };
  const confs = { ...(dis.agent_confidences || {}) };

  agents.forEach((a) => {
    if (signals[a] == null) {
      const rawLabel = a === 'regime'
        ? (analysis.regime?.regime || analysis.regime?.signal)
        : (listMap[a]?.signal || listMap[a]?.reasoning?.signal || listMap[a]?.reasoning?.trend);
      const key = String(rawLabel || '').toLowerCase();
      signals[a] = DIRECTION_MAP[key] ?? 0;
    }
    if (confs[a] == null) {
      confs[a] = a === 'regime'
        ? (analysis.regime?.confidence ?? 0)
        : (listMap[a]?.confidence ?? listMap[a]?.reasoning?.confidence ?? 0);
    }
  });

  const score = dis.disagreement_score ?? dis.total_uncertainty ?? 0;
  const gateVal = 1.0 - 0.8 * score;

  const sigColor = (val) => {
    if (val === undefined) return '#2d3561';
    if (val > 0.3) return 'rgba(72,187,120,0.35)';
    if (val < -0.3) return 'rgba(245,101,101,0.35)';
    return 'rgba(246,224,94,0.2)';
  };

  const confColor = (val) => {
    if (val === undefined) return '#2d3561';
    const pct = val * 100;
    if (pct > 70) return 'rgba(72,187,120,0.5)';
    if (pct > 40) return 'rgba(246,224,94,0.35)';
    return 'rgba(245,101,101,0.3)';
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-header">
        <span className="card-title">Disagreement Heatmap</span>
        <span
          style={{
            fontSize: 11,
            padding: '2px 8px',
            borderRadius: 10,
            background:
              score > 0.6
                ? 'rgba(245,101,101,0.2)'
                : score > 0.3
                  ? 'rgba(246,224,94,0.2)'
                  : 'rgba(72,187,120,0.2)',
            color: score > 0.6 ? '#fc8181' : score > 0.3 ? '#f6e05e' : '#68d391',
          }}
        >
          Score: {score.toFixed(3)}
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '3px', fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', color: '#647091', paddingBottom: 8, fontSize: 11 }}>Agent</th>
              {agents.map((a) => (
                <th key={a} style={{ color: '#8b9fc0', textTransform: 'capitalize', fontSize: 11, padding: '4px 8px' }}>
                  {a}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ color: '#8b9fc0', fontSize: 11, paddingRight: 12 }}>Signal</td>
              {agents.map((a) => (
                <td
                  key={a}
                  style={{
                    textAlign: 'center',
                    padding: '6px 10px',
                    borderRadius: 6,
                    background: sigColor(signals[a]),
                    color: '#fff',
                    fontWeight: 600,
                    fontFamily: 'monospace',
                  }}
                >
                  {signals[a] !== undefined ? Number(signals[a]).toFixed(2) : '-'}
                </td>
              ))}
            </tr>
            <tr>
              <td style={{ color: '#8b9fc0', fontSize: 11, paddingTop: 4 }}>Confidence</td>
              {agents.map((a) => (
                <td
                  key={a}
                  style={{
                    textAlign: 'center',
                    padding: '6px 10px',
                    borderRadius: 6,
                    background: confColor(confs[a]),
                    color: '#fff',
                    fontWeight: 600,
                    fontFamily: 'monospace',
                  }}
                >
                  {confs[a] !== undefined ? `${(Number(confs[a]) * 100).toFixed(0)}%` : '-'}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <div
        style={{
          marginTop: 12,
          padding: '8px 12px',
          borderRadius: 8,
          background: 'rgba(99,102,241,0.1)',
          border: '1px solid rgba(99,102,241,0.2)',
        }}
      >
        <div style={{ fontSize: 11, color: '#a0aec0', marginBottom: 4 }}>RL Bayesian Gate Formula</div>
        <div style={{ fontFamily: 'monospace', fontSize: 13, color: '#c3dafe' }}>
          effective_action = rl_action x (1 - 0.8 x {score.toFixed(3)}) ={' '}
          <span style={{ color: '#68d391', fontWeight: 700 }}>rl_action x {gateVal.toFixed(3)}</span>
        </div>
        <div style={{ fontSize: 10, color: '#647091', marginTop: 4 }}>
          {dis.recommendation === 'NO_TRADE'
            ? 'High disagreement - trading blocked'
            : dis.recommendation === 'REDUCE'
              ? 'Moderate disagreement - position reduced'
              : 'Low disagreement - full RL strength'}
        </div>
      </div>
    </div>
  );
}
