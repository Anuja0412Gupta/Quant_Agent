/* c:\Users\Anuja\Desktop\QUANT\frontend\src\components\AISuggestionsTab.jsx */
import React from 'react';

export default function AISuggestionsTab({ analysis }) {
  if (!analysis) {
    return (
      <div className="empty-state">
        <div className="empty-icon">💡</div>
        <div className="empty-text">No analysis available.</div>
        <div className="empty-sub">Please Search for a stock and Analyze first.</div>
      </div>
    );
  }

  // Extract variables
  const ticker = analysis.symbol;
  const currentPrice = analysis.current_price;
  const regime = analysis.regime?.regime || analysis.active_regime || 'Unknown';
  const regimeConf = analysis.regime?.confidence || 0;
  const sentimentScore = analysis.sentiment?.ticker_sentiment_score || 0;
  const action = analysis.rl_weights?.direction || 'FLAT';
  
  // Math logic extraction
  const rlActionWeight = typeof analysis.rl_weights?.rl_action === 'number' ? analysis.rl_weights.rl_action : 0;
  const actionIntensity = Math.abs(rlActionWeight);
  const positionScale = (analysis.rl_weights?.effective_position * 100 || 0).toFixed(1);
  const signalAgreement = analysis.disagreement?.agent_consensus || (1 - (analysis.disagreement?.total_uncertainty || 0));

  // Labels
  function getSentimentWord(score) {
    if (score >= 0.25) return { word: "Positive", color: "#4ade80", icon: "☀️" };
    if (score >= 0.1) return { word: "Slightly Positive", color: "#86efac", icon: "🌤️" };
    if (score <= -0.25) return { word: "Negative", color: "#f87171", icon: "⛈️" };
    if (score <= -0.1) return { word: "Slightly Negative", color: "#fca5a5", icon: "🌧️" };
    return { word: "Neutral", color: "#94a3b8", icon: "☁️" };
  }
  
  const sentiment = getSentimentWord(sentimentScore);
  
  const getRegimeDetails = (r) => {
    if (r === 'trending') return { icon: "📈", desc: "The stock is riding a strong momentum wave.", color: "#4ade80" };
    if (r === 'mean_reverting') return { icon: "↕️", desc: "The stock is bouncing sideways in a range.", color: "#60a5fa" };
    if (r === 'high_volatility') return { icon: "🌪️", desc: "Turbulent, unpredictable price swings.", color: "#f87171" };
    return { icon: "📊", desc: "Standard market conditions.", color: "#94a3b8" };
  };
  const regimeData = getRegimeDetails(regime);

  // Verdict config
  let verdictColor = "var(--text-1)";
  let verdictLabel = "HOLD / FLAT";
  let verdictIcon = "⚖️";
  if (action === 'LONG' || action === 'BUY') {
    verdictColor = "#00e5a0";
    verdictLabel = "BUY POSITION";
    verdictIcon = "🚀";
  } else if (action === 'SHORT' || action === 'SELL') {
    verdictColor = "#ff4f72";
    verdictLabel = "SELL POSITION";
    verdictIcon = "🔻";
  }

  return (
    <div className="suggestion-dashboard">
      <div className="suggestion-header">
        <div className="sh-left">
          <h2 className="sh-title">AI Trading Plan for {ticker}</h2>
          <p className="sh-subtitle">Simplified breakdown of the agent's logic for human review.</p>
        </div>
        <div className="sh-price">${currentPrice?.toFixed(2)}</div>
      </div>

      <div className="verdict-banner glass-panel" style={{ borderLeft: `6px solid ${verdictColor}` }}>
        <div className="verdict-icon">{verdictIcon}</div>
        <div className="verdict-content">
          <div className="verdict-label">FINAL VERDICT</div>
          <div className="verdict-action" style={{ color: verdictColor }}>{verdictLabel}</div>
          <p className="verdict-desc">
            The AI recommends {action === 'FLAT' ? 'staying completely out of the market' : <span style={{color: verdictColor}}>{verdictLabel}</span>} for {ticker}. 
            It has sized the portfolio allocation to exactly <strong style={{color: '#fff'}}>{positionScale}%</strong> based on current mathematical risk parameters.
          </p>
        </div>
      </div>

      <h3 className="section-title mt-32 mb-16">The 3 Pillars of this Decision</h3>
      
      <div className="pillars-grid">
        {/* Pillar 1: Regime */}
        <div className="pillar-card glass-panel">
          <div className="pillar-header">
            <span className="pillar-emoji">{regimeData.icon}</span>
            <span className="pillar-title">Market Regime</span>
          </div>
          <div className="pillar-value" style={{ color: regimeData.color }}>
            {regime.replace('_', ' ').toUpperCase()}
          </div>
          <div className="progress-bar-container">
            <div className="progress-label"><span>AI Confidence</span> <span>{(regimeConf * 100).toFixed(0)}%</span></div>
            <div className="progress-track"><div className="progress-fill" style={{ width: `${(regimeConf * 100).toFixed(0)}%`, background: regimeData.color }}></div></div>
          </div>
          <p className="pillar-desc">{regimeData.desc}</p>
        </div>

        {/* Pillar 2: Sentiment */}
        <div className="pillar-card glass-panel">
          <div className="pillar-header">
            <span className="pillar-emoji">{sentiment.icon}</span>
            <span className="pillar-title">News Sentiment</span>
          </div>
          <div className="pillar-value" style={{ color: sentiment.color }}>
            {sentiment.word.toUpperCase()}
          </div>
          <div className="progress-bar-container">
             <div className="progress-label"><span>Polarity Score</span> <span>{sentimentScore.toFixed(2)}</span></div>
             <div className="progress-track">
               <div className="progress-fill" style={{ width: `${Math.max(5, Math.abs(sentimentScore * 100))}%`, background: sentiment.color }}></div>
             </div>
          </div>
          <p className="pillar-desc">
            {sentimentScore > 0 ? 'Positive headlines act as a tailwind.' : sentimentScore < 0 ? 'Negative news acts as a headwind.' : 'News media is largely un-opinionated right now.'}
          </p>
        </div>

        {/* Pillar 3: AI Agreement */}
        <div className="pillar-card glass-panel">
          <div className="pillar-header">
            <span className="pillar-emoji">🤝</span>
            <span className="pillar-title">Model Agreement</span>
          </div>
          <div className="pillar-value" style={{ color: signalAgreement > 0.6 ? '#4ade80' : '#fbbf24' }}>
            {signalAgreement > 0.6 ? 'STRONG ALIGNMENT' : 'CONFLICTING SIGNALS'}
          </div>
          <div className="progress-bar-container">
            <div className="progress-label"><span>Consensus Matrix</span> <span>{(signalAgreement * 100).toFixed(0)}%</span></div>
            <div className="progress-track">
               <div className="progress-fill" style={{ width: `${(signalAgreement * 100).toFixed(0)}%`, background: signalAgreement > 0.6 ? '#4ade80' : '#fbbf24' }}></div>
            </div>
          </div>
          <p className="pillar-desc">
            {signalAgreement > 0.6 
              ? 'Most technical indicators and sub-agents agree on the direction.' 
              : 'Different agents are disagreeing. The AI has reduced sizing to match this risk.'}
          </p>
        </div>
      </div>
    </div>
  );
}
