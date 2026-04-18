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
  const regime = analysis.regime?.dominant_regime || 'Unknown';
  const regimeConf = analysis.regime?.confidence || 0;
  const sentimentScore = analysis.sentiment?.ticker_sentiment_score || 0;
  const newsSentimentWord = getSentimentWord(sentimentScore);
  const action = analysis.rl_weights?.direction || 'NO TRADE';
  const rlActionWeight = typeof analysis.rl_weights?.rl_action === 'number' ? analysis.rl_weights.rl_action : 0;
  const actionIntensity = Math.abs(rlActionWeight);
  const signalAgreement = 1 - (analysis.disagreement?.total_uncertainty || 0);

  // Generate plain-english paragraphs
  const getActionIntensityText = (val) => {
    if (val > 0.75) return "very strongly recommends";
    if (val > 0.4) return "recommends";
    if (val > 0.15) return "weakly suggests";
    return "has no strong suggestion to";
  };

  const getConfidenceLevel = (val) => {
    if (val > 0.8) return "High Confidence";
    if (val > 0.5) return "Medium Confidence";
    return "Low Confidence";
  };

  function getSentimentWord(score) {
    if (score >= 0.25) return "Positive";
    if (score >= 0.1) return "Slightly Positive";
    if (score <= -0.25) return "Negative";
    if (score <= -0.1) return "Slightly Negative";
    return "Neutral";
  }

  const recommendationColor =
    action === 'LONG' ? 'var(--accent-green)' : action === 'SHORT' ? 'var(--accent-red)' : 'var(--text-1)';

  return (
    <div className="ai-suggestions-tab" style={{ maxWidth: 800, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="card" style={{ padding: 32 }}>
        <h2 style={{ marginBottom: 8, color: 'var(--text-1)' }}>
          AI Trading Suggestion for <span style={{ color: 'var(--accent-blue)' }}>{ticker}</span>
        </h2>
        <p style={{ fontSize: 16, color: 'var(--text-2)', lineHeight: 1.6, marginBottom: 24 }}>
          This page breaks down exactly why the AI made its decision, skipping the complex math.
        </p>

        <div style={{ background: 'var(--bg-3)', padding: 24, borderRadius: 12, border: '1px solid var(--border)' }}>
          <h3 style={{ marginBottom: 16, fontSize: 20 }}>
            Conclusion: <span style={{ color: recommendationColor, fontWeight: 'bold' }}>{action.replace('LONG', 'BUY')}</span>
          </h3>

          <p style={{ fontSize: 16, lineHeight: 1.8, color: 'var(--text-1)' }}>
             The AI <strong>{getActionIntensityText(actionIntensity)}</strong> taking a <strong>{action.replace('LONG', 'BUY')}</strong> position on {ticker} at the current price of ${currentPrice?.toFixed(2)}. 
             This decision is rated with <strong>{getConfidenceLevel(signalAgreement)}</strong> because the underlying signals {(signalAgreement > 0.6) ? 'mostly agree with each other' : 'are somewhat conflicting'}.
          </p>
        </div>
      </div>

      <div className="card" style={{ padding: 32 }}>
        <h3 style={{ marginBottom: 16, fontSize: 18, color: 'var(--text-1)' }}>How did it reach this conclusion?</h3>
        
        <ul style={{ listStyleType: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <li style={{ background: 'rgba(255,255,255,0.03)', padding: 16, borderRadius: 8 }}>
            <strong style={{ display: 'block', fontSize: 15, marginBottom: 4, color: 'var(--accent-blue)' }}>1. Market Environment (Regime)</strong>
            <span style={{ color: 'var(--text-2)', fontSize: 15, lineHeight: 1.5 }}>
              The AI identified the current market environment for {ticker} as <strong>{regime}</strong> (with {(regimeConf * 100).toFixed(0)}% certainty). 
              {regime === 'trending' && ' This means the stock is currently riding a wave, and the AI prefers to follow the trend rather than bet against it.'}
              {regime === 'mean_reverting' && ' This means the stock is moving sideways within a range, so the AI will try to buy low and sell high within that specific range.'}
              {regime === 'volatile' && ' This means the stock is experiencing high turbulence, causing the AI to be more cautious and reduce trade sizes to protect your capital.'}
            </span>
          </li>

          <li style={{ background: 'rgba(255,255,255,0.03)', padding: 16, borderRadius: 8 }}>
            <strong style={{ display: 'block', fontSize: 15, marginBottom: 4, color: 'var(--accent-green)' }}>2. News & Social Sentiment</strong>
            <span style={{ color: 'var(--text-2)', fontSize: 15, lineHeight: 1.5 }}>
              By analyzing thousands of recent news headlines and Reddit discussions, the AI calculated that the overall public sentiment is <strong>{newsSentimentWord}</strong>. 
              {newsSentimentWord.includes('Positive') && ' Good news usually provides a tailwind that supports a Buy decision.'}
              {newsSentimentWord.includes('Negative') && ' Bad news creates a headwind, leading the AI to be cautious or recommend Selling.'}
              {newsSentimentWord.includes('Neutral') && ' The news isn\'t heavily swaying the stock right now, so the AI is relying mostly on price patterns.'}
            </span>
          </li>

          <li style={{ background: 'rgba(255,255,255,0.03)', padding: 16, borderRadius: 8 }}>
            <strong style={{ display: 'block', fontSize: 15, marginBottom: 4, color: 'var(--accent-purple)' }}>3. Risk & Protection</strong>
            <span style={{ color: 'var(--text-2)', fontSize: 15, lineHeight: 1.5 }}>
              Even if a trade looks great, the AI scales the trade size to protect your money. Based on recent volatility and disagreement among the AI's internal sub-agents, it decided to use a <strong>position scaling of {(analysis.rl_weights?.effective_position * 100 || 0).toFixed(1)}%</strong> for this trade.
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
}
