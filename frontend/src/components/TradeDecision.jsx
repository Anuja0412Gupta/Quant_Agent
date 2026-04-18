/**
 * TradeDecision
 * Displays the final trade decision: action badge, entry/SL/TP, score, reasoning.
 */
export default function TradeDecision({ data }) {
  if (!data) return null;
  const { trade_decision, rl_weights, current_price } = data;
  if (!trade_decision) return null;

  const action = trade_decision.adjusted_action > 0.05 ? "BUY" : (trade_decision.adjusted_action < -0.05 ? "SELL" : "HOLD");
  const entry = current_price;
  const stop_loss = trade_decision.stop_loss;
  const take_profit = trade_decision.take_profit;
  const score = trade_decision.adjusted_action;
  
  // Calculate confidence inversely proportional to disagreement
  const confidence = rl_weights ? (1 - rl_weights.disagreement_score) : 0.5;
  const reasoning = trade_decision.veto_reason ? `Veto: ${trade_decision.veto_reason}` : (trade_decision.approved ? "Self-Critique Engine approved trade parameters. CVaR well within limits." : "Trade sizing restricted due to conditional value at risk limits.");

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
          <div className="risk-stat"><span className="rs-label">Adjusted Size</span><span className="rs-value">{trade_decision.final_size.toFixed(4)} shares</span></div>
          <div className="risk-stat"><span className="rs-label">Sizing Cut</span><span className="rs-value" style={{ color: trade_decision.size_reduction_pct > 0 ? '#ff4f72' : '#00e5a0' }}>-{(trade_decision.size_reduction_pct * 100)?.toFixed(1)}%</span></div>
          <div className="risk-stat"><span className="rs-label">Kelly Fraction</span><span className="rs-value">{(trade_decision.kelly_fraction * 100)?.toFixed(2)}%</span></div>
          <div className="risk-stat"><span className="rs-label">CVaR (95%)</span><span className="rs-value">{(trade_decision.current_cvar * 100)?.toFixed(2)}%</span></div>
        </div>
      </div>
    </div>
  );
}
