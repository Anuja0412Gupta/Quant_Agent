/**
 * TradeDecision
 * Displays the final trade decision: action badge, entry/SL/TP, score, reasoning.
 */
export default function TradeDecision({ data }) {
  if (!data) return null;
  const { decision, risk } = data;
  if (!decision) return null;

  const { action, entry, stop_loss, take_profit, confidence, score, reasoning } = decision;

  return (
    <div>
      <div className="section-label">Trade Decision</div>
      <div className={`decision-card ${action}`}>
        <div className="action-badge">
          <div className="action-dot" />
          {action}
        </div>

        <div className="price-grid">
          <div className="price-item entry">
            <div className="pi-label">Entry</div>
            <div className="pi-value">${entry?.toFixed(2)}</div>
          </div>
          <div className="price-item sl">
            <div className="pi-label">Stop Loss</div>
            <div className="pi-value">${stop_loss?.toFixed(2)}</div>
          </div>
          <div className="price-item tp">
            <div className="pi-label">Take Profit</div>
            <div className="pi-value">${take_profit?.toFixed(2)}</div>
          </div>
        </div>

        <div className="score-row">
          <span className="score-label">Fusion Score</span>
          <span className="score-value" style={{ color: score > 0 ? '#00e5a0' : score < 0 ? '#ff4f72' : '#ffb830' }}>
            {score > 0 ? '+' : ''}{score?.toFixed(4)}
          </span>
          <span className="score-label" style={{ marginLeft: 'auto' }}>Confidence</span>
          <span className="score-value" style={{ color: '#4f7dff' }}>{(confidence * 100)?.toFixed(1)}%</span>
        </div>

        <div className="reasoning-text">{reasoning}</div>

        {/* Risk sizing summary */}
        {risk && (
          <div style={{ marginTop: 16 }}>
            <div className="card-title" style={{ marginBottom: 10 }}>Position Sizing</div>
            <div className="risk-stat"><span className="rs-label">Shares</span><span className="rs-value">{risk.shares}</span></div>
            <div className="risk-stat"><span className="rs-label">Position Value</span><span className="rs-value">${risk.position_value?.toLocaleString()}</span></div>
            <div className="risk-stat"><span className="rs-label">Portfolio %</span><span className="rs-value">{(risk.position_pct * 100)?.toFixed(2)}%</span></div>
            <div className="risk-stat"><span className="rs-label">Kelly Fraction</span><span className="rs-value">{(risk.kelly_fraction * 100)?.toFixed(2)}%</span></div>
          </div>
        )}
      </div>
    </div>
  );
}
