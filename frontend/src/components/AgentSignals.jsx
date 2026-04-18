/**
 * AgentSignals
 * Displays a card for each agent: signal badge, confidence bar, and explanation.
 */

function ConfidenceBar({ value }) {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  const pct = Math.round(safe * 100);
  const hue = pct > 66 ? '#00e5a0' : pct > 33 ? '#ffb830' : '#ff4f72';
  return (
    <div className="confidence-bar-wrapper">
      <div className="confidence-label">
        <span>Confidence</span>
        <strong>{pct}%</strong>
      </div>
      <div className="confidence-bar-track">
        <div
          className="confidence-bar-fill"
          style={{ width: `${pct}%`, background: hue }}
        />
      </div>
    </div>
  );
}

function IndicatorValues({ values }) {
  if (!values || Object.keys(values).length === 0) return null;
  return (
    <div className="indicator-grid">
      {Object.entries(values).map(([k, v]) => (
        <div className="indicator-item" key={k}>
          <span className="ik">{k.toUpperCase()}</span>
          <span className="iv">{typeof v === 'number' ? v.toFixed(2) : v}</span>
        </div>
      ))}
    </div>
  );
}

function AgentCard({ title, signal, confidence, explanation, extra }) {
  const sigClass = `signal-${String(signal || '').toLowerCase().replace(/ /g, '_')}`;
  return (
    <div className="agent-card">
      <div className="agent-name">{title}</div>
      <div className={`agent-signal ${sigClass}`}>{signal || '-'}</div>
      <ConfidenceBar value={confidence || 0} />
      {extra}
      <p className="agent-explanation">{explanation || ''}</p>
    </div>
  );
}

export default function AgentSignals({ data }) {
  if (!data) return null;

  const disagreement = data.disagreement || {};
  const regime = data.regime || {};

  const list = Array.isArray(data.agent_signals) ? data.agent_signals : [];
  const byName = Object.fromEntries(
    list.map((a) => [String(a?.agent_name || '').toLowerCase(), a])
  );

  const indicator = byName.indicator?.reasoning || {};
  const pattern = byName.pattern?.reasoning || {};
  const trend = byName.trend?.reasoning || {};

  const indicatorConfidence = byName.indicator?.confidence ?? indicator?.confidence ?? 0;
  const patternConfidence = byName.pattern?.confidence ?? pattern?.confidence ?? 0;
  const trendConfidence = byName.trend?.confidence ?? trend?.confidence ?? 0;
  const disagreementScore = disagreement?.disagreement_score ?? disagreement?.total_uncertainty ?? 0;

  return (
    <div>
      <div className="section-label">Agent Signals</div>
      <div className="agents-grid">
        <AgentCard
          title="Indicator Agent"
          signal={indicator?.signal}
          confidence={indicatorConfidence}
          explanation={indicator?.explanation}
          extra={<IndicatorValues values={indicator?.indicator_values} />}
        />

        <AgentCard
          title="Pattern Agent"
          signal={pattern?.signal}
          confidence={patternConfidence}
          explanation={pattern?.pattern ? `${pattern.pattern} - ${pattern?.explanation || ''}` : pattern?.explanation}
        />

        <AgentCard
          title="Trend Agent"
          signal={trend?.trend}
          confidence={trendConfidence}
          explanation={trend?.explanation}
          extra={
            <div className="indicator-grid">
              <div className="indicator-item"><span className="ik">SLOPE</span><span className="iv">{trend?.slope?.toFixed?.(4) ?? '-'}</span></div>
              <div className="indicator-item"><span className="ik">STRENGTH</span><span className="iv">{trend?.strength != null ? `${(trend.strength * 100).toFixed(1)}%` : '-'}</span></div>
              <div className="indicator-item"><span className="ik">SUPPORT</span><span className="iv">{trend?.support?.toFixed?.(2) ?? '-'}</span></div>
              <div className="indicator-item"><span className="ik">RESIST</span><span className="iv">{trend?.resistance?.toFixed?.(2) ?? '-'}</span></div>
            </div>
          }
        />

        <AgentCard
          title="Market Regime"
          signal={regime?.regime || regime?.signal}
          confidence={regime?.confidence}
          explanation={regime?.explanation}
          extra={
            <div className="indicator-grid">
              <div className="indicator-item"><span className="ik">HURST</span><span className="iv">{regime?.hurst?.toFixed?.(3) ?? '-'}</span></div>
              <div className="indicator-item"><span className="ik">ATR%</span><span className="iv">{regime?.atr_ratio != null ? `${(regime.atr_ratio * 100).toFixed(2)}%` : '-'}</span></div>
            </div>
          }
        />

        {!!Object.keys(disagreement).length && (
          <div className="agent-card">
            <div className="agent-name">Disagreement</div>
            <div className="disagree-row">
              <span className="disagree-index" style={{ color: disagreementScore > 0.6 ? '#ff4f72' : '#00e5a0' }}>
                {Number.isFinite(disagreementScore) ? disagreementScore.toFixed(4) : '-'}
              </span>
              <span className={`rec-badge rec-${disagreement.recommendation}`}>
                {disagreement.recommendation || 'PROCEED'}
              </span>
            </div>
            <ConfidenceBar value={Math.min(1, Math.max(0, disagreementScore))} />
            <p className="agent-explanation">
              Consensus: {disagreement.dominant_signal || 'NEUTRAL'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
