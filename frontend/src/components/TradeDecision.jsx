import React from 'react';

/**
 * TradeDecision
 * Displays the final trade decision: action badge, entry/SL/TP, score, reasoning.
 */
export default function TradeDecision({ data }) {
  if (!data) return null;
  const { trade_decision, rl_weights, current_price, disagreement } = data;
  if (!trade_decision) return null;

  // Use rl_weights direction/action for the primary signal (more accurate)
  const rlAction = rl_weights?.rl_action ?? trade_decision.adjusted_action ?? 0;
  const effectiveAction = rl_weights?.effective_action ?? rlAction;
  
  // Decide Action explicitly based on clear boundaries
  const action = effectiveAction > 0.01 ? "BUY" : (effectiveAction < -0.01 ? "SELL" : "HOLD");
  
  const entry = current_price;
  const stop_loss = trade_decision.stop_loss;
  const take_profit = trade_decision.take_profit;

  // position_pct = final_size (fraction 0→1, e.g. 0.05 = 5% of portfolio)
  const positionPct = (trade_decision.final_size ?? 0) * 100;
  const effectivePosPct = (rl_weights?.effective_position ?? 0) * 100;
  
  // Use pure agent consensus as confidence (Avoiding the 0% penalty of external noise variables)
  const consensus = disagreement?.agent_consensus ?? 0.5;
  const confidencePercent = (consensus * 100).toFixed(1);
  
  // Veto reason from rl_weights or trade_decision
  const vetoReason = rl_weights?.veto_reason || trade_decision.veto_reason;
  const reasoning = vetoReason
    ? `Veto: ${vetoReason}`
    : (trade_decision.approved !== false
        ? "Self-Critique Engine approved trade parameters. CVaR within limits."
        : "Trade sizing restricted due to conditional value at risk limits.");

  // Styling
  const colorMap = {
      BUY: { color: "#00e5a0", icon: "🚀", bg: "rgba(0, 229, 160, 0.1)", border: "rgba(0, 229, 160, 0.4)" },
      SELL: { color: "#ff4f72", icon: "🔻", bg: "rgba(255, 79, 114, 0.1)", border: "rgba(255, 79, 114, 0.4)" },
      HOLD: { color: "#ffb830", icon: "⚖️", bg: "rgba(255, 184, 48, 0.1)", border: "rgba(255, 184, 48, 0.4)" }
  };
  const theme = colorMap[action];

  return (
    <div className="card" style={{ 
      border: `2px solid ${theme.border}`,
      background: `linear-gradient(180deg, ${theme.bg} 0%, var(--bg-card) 100%)`,
      boxShadow: `0 0 40px ${theme.bg}`
    }}>
      <div className="card-header" style={{ marginBottom: 12 }}>
        <span className="card-title" style={{ color: theme.color, fontSize: 16 }}>⚡ Primary Directive</span>
        <span style={{ color: 'var(--text-3)', fontSize: 12, fontFamily: 'var(--mono)' }}>v3.0 Engine</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <div style={{ fontSize: 48, filter: `drop-shadow(0 0 10px ${theme.color})` }}>{theme.icon}</div>
        <div>
           <div style={{ fontSize: 42, fontWeight: 900, fontFamily: 'var(--mono)', color: theme.color, letterSpacing: -1, lineHeight: 1 }}>{action}</div>
           <div style={{ fontSize: 13, color: 'var(--text-1)', fontWeight: 600, marginTop: 4 }}>
             RECOMMENDED ACTION
           </div>
        </div>
      </div>

      <div className="price-grid" style={{ marginBottom: 24 }}>
        <div className="price-item entry" style={{ background: 'rgba(0,0,0,0.2)' }}>
          <div className="pi-label">Market Entry</div>
          <div className="pi-value" style={{ color: '#fff' }}>${entry?.toFixed(2)}</div>
        </div>
        <div className="price-item sl" style={{ background: 'rgba(0,0,0,0.2)' }}>
          <div className="pi-label">Stop Loss</div>
          <div className="pi-value">${stop_loss?.toFixed(2)}</div>
        </div>
        <div className="price-item tp" style={{ background: 'rgba(0,0,0,0.2)' }}>
          <div className="pi-label">Take Profit</div>
          <div className="pi-value">${take_profit?.toFixed(2)}</div>
        </div>
      </div>

      <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 12, padding: 16, marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
             <span style={{ color: 'var(--text-2)', fontSize: 12, fontWeight: 600 }}>MODEL CONSENSUS</span>
             <span style={{ color: '#fff', fontFamily: 'var(--mono)', fontWeight: 700 }}>{confidencePercent}%</span>
          </div>
          <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
             <div style={{ width: `${confidencePercent}%`, height: '100%', background: theme.color, borderRadius: 3 }} />
          </div>
      </div>

      <div className="reasoning-text" style={{ background: 'rgba(0,0,0,0.2)', fontSize: 13, color: 'var(--text-1)' }}>
        {reasoning}
      </div>

      {/* Risk sizing summary */}
      <div style={{ marginTop: 24, paddingTop: 16, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="card-title" style={{ marginBottom: 16, fontSize: 11 }}>Risk Management Limits</div>
        
        <div className="risk-stat">
          <span className="rs-label">Base RL Sizing</span>
          <span className="rs-value" style={{ color: effectivePosPct > 0 ? '#00e5a0' : '#fc8181' }}>
            {effectivePosPct.toFixed(2)}%
          </span>
        </div>
        
        <div className="risk-stat">
          <span className="rs-label">Drawdown/VaR Penalties</span>
          <span className="rs-value" style={{ color: trade_decision.size_reduction_pct > 0 ? '#ff4f72' : '#00e5a0' }}>
            -{(trade_decision.size_reduction_pct * 100)?.toFixed(1)}%
          </span>
        </div>
        
        <div className="risk-stat" style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 8px', margin: '4px -8px', borderRadius: 8 }}>
          <span className="rs-label" style={{ fontWeight: 600, color: '#fff' }}>FINAL EXECUTABLE SIZE</span>
          <span className="rs-value" style={{ fontSize: 16, color: theme.color }}>{positionPct.toFixed(2)}%</span>
        </div>

      </div>
    </div>
  );
}
