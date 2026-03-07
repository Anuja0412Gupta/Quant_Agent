/**
 * AgentSignals
 * Displays a card for each agent: signal badge, confidence bar, and explanation.
 */

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
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
  const sigClass = `signal-${signal?.toLowerCase().replace(/ /g, '_')}`;
  return (
    <div className="agent-card">
      <div className="agent-name">{title}</div>
      <div className={`agent-signal ${sigClass}`}>{signal || '—'}</div>
      <ConfidenceBar value={confidence || 0} />
      {extra}
      <p className="agent-explanation">{explanation}</p>
    </div>
  );
}

export default function AgentSignals({ data }) {
  if (!data) return null;
  const { indicator, pattern, trend, regime, disagreement } = data;

  return (
    <div>
      <div className="section-label">Agent Signals</div>
      <div className="agents-grid">
        <AgentCard
          title="📊 Indicator Agent"
          signal={indicator?.signal}
          confidence={indicator?.confidence}
          explanation={indicator?.explanation}
          extra={<IndicatorValues values={indicator?.indicator_values} />}
        />
        <AgentCard
          title="🕯️ Pattern Agent"
          signal={pattern?.signal}
          confidence={pattern?.confidence}
          explanation={`${pattern?.pattern} — ${pattern?.explanation}`}
        />
        <AgentCard
          title="📈 Trend Agent"
          signal={trend?.trend}
          confidence={trend?.confidence}
          explanation={trend?.explanation}
          extra={
            <div className="indicator-grid">
              <div className="indicator-item"><span className="ik">SLOPE</span><span className="iv">{trend?.slope?.toFixed(4)}</span></div>
              <div className="indicator-item"><span className="ik">STRENGTH</span><span className="iv">{(trend?.strength * 100)?.toFixed(1)}%</span></div>
              <div className="indicator-item"><span className="ik">SUPPORT</span><span className="iv">{trend?.support?.toFixed(2)}</span></div>
              <div className="indicator-item"><span className="ik">RESIST</span><span className="iv">{trend?.resistance?.toFixed(2)}</span></div>
            </div>
          }
        />
        <AgentCard
          title="🌊 Market Regime"
          signal={regime?.regime}
          confidence={regime?.confidence}
          explanation={regime?.explanation}
          extra={
            <div className="indicator-grid">
              <div className="indicator-item"><span className="ik">HURST</span><span className="iv">{regime?.hurst?.toFixed(3)}</span></div>
              <div className="indicator-item"><span className="ik">ATR%</span><span className="iv">{((regime?.atr_ratio || 0) * 100).toFixed(2)}%</span></div>
            </div>
          }
        />

        {/* Disagreement card */}
        {disagreement && (
          <div className="agent-card">
            <div className="agent-name">⚖️ Disagreement</div>
            <div className="disagree-row">
              <span className="disagree-index" style={{ color: disagreement.high_disagreement ? '#ff4f72' : '#00e5a0' }}>
                {disagreement.disagreement_index?.toFixed(4)}
              </span>
              <span className={`rec-badge rec-${disagreement.recommendation}`}>
                {disagreement.recommendation}
              </span>
            </div>
            <ConfidenceBar value={Math.min(disagreement.disagreement_index * 25, 1)} />
            <p className="agent-explanation">{disagreement.explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}
