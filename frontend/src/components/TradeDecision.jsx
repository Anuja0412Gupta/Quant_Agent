/**
 * TradeDecision
 * Displays the final trade decision: action badge, entry/SL/TP, score, reasoning.
 */
export default function TradeDecision({ data }) {
  if (!data) return null;
  const { trade_decision, rl_weights, current_price } = data;
  if (!trade_decision) return null;

  // Use rl_weights direction/action for the primary signal (more accurate)
  const rlAction = rl_weights?.rl_action ?? trade_decision.adjusted_action ?? 0;
  const effectiveAction = rl_weights?.effective_action ?? rlAction;
  const action = effectiveAction > 0.02 ? "BUY" : (effectiveAction < -0.02 ? "SELL" : "HOLD");
  const entry = current_price;
  const stop_loss = trade_decision.stop_loss;
  const take_profit = trade_decision.take_profit;
  const score = rlAction;

  // position_pct = final_size (fraction 0→1, e.g. 0.05 = 5% of portfolio)
  const positionPct = (trade_decision.final_size ?? 0) * 100;
  const effectivePosPct = (rl_weights?.effective_position ?? 0) * 100;
  
  // Calculate confidence inversely proportional to disagreement
  const confidence = rl_weights ? (1 - (rl_weights.disagreement_score ?? 0)) : 0.5;
  
  // Veto reason from rl_weights (most up-to-date) or trade_decision
  const vetoReason = rl_weights?.veto_reason || trade_decision.veto_reason;
  const reasoning = vetoReason
    ? `Veto: ${vetoReason}`
    : (trade_decision.approved !== false
        ? "Self-Critique Engine approved trade parameters. CVaR within limits."
        : "Trade sizing restricted due to conditional value at risk limits.");

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
          <span className="score-label">RL Action Score</span>
          <span className="score-value" style={{ color: score > 0 ? '#00e5a0' : score < 0 ? '#ff4f72' : '#ffb830' }}>
            {score > 0 ? '+' : ''}{score?.toFixed(4)}
          </span>
          <span className="score-label" style={{ marginLeft: 'auto' }}>Confidence</span>
          <span className="score-value" style={{ color: '#4f7dff' }}>{(confidence * 100)?.toFixed(1)}%</span>
        </div>

        <div className="reasoning-text">{reasoning}</div>

        {/* Risk sizing summary */}
        <div style={{ marginTop: 16 }}>
          <div className="card-title" style={{ marginBottom: 10 }}>Position Sizing</div>
          <div className="risk-stat">
            <span className="rs-label">RL Effective Position</span>
            <span className="rs-value" style={{ color: effectivePosPct > 0 ? '#00e5a0' : '#fc8181' }}>
              {effectivePosPct.toFixed(2)}%
            </span>
          </div>
          <div className="risk-stat">
            <span className="rs-label">Risk-Adjusted Size</span>
            <span className="rs-value">{positionPct.toFixed(2)}% of portfolio</span>
          </div>
          <div className="risk-stat">
            <span className="rs-label">Sizing Cut</span>
            <span className="rs-value" style={{ color: trade_decision.size_reduction_pct > 0 ? '#ff4f72' : '#00e5a0' }}>
              -{(trade_decision.size_reduction_pct * 100)?.toFixed(1)}%
            </span>
          </div>
          <div className="risk-stat">
            <span className="rs-label">Kelly Fraction</span>
            <span className="rs-value">{(trade_decision.kelly_fraction * 100)?.toFixed(2)}%</span>
          </div>
          <div className="risk-stat">
            <span className="rs-label">CVaR (95%)</span>
            <span className="rs-value" style={{ color: (trade_decision.current_cvar ?? 0) > 0.03 ? '#fc8181' : '#68d391' }}>
              {((trade_decision.current_cvar ?? 0) * 100)?.toFixed(2)}%
            </span>
          </div>
          {rl_weights?.risk_veto && (
            <div className="risk-stat" style={{ marginTop: 6 }}>
              <span className="rs-label" style={{ color: '#fc8181' }}>⚠ Risk Veto Active</span>
              <span className="rs-value" style={{ color: '#fc8181', fontSize: 10 }}>{vetoReason}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

