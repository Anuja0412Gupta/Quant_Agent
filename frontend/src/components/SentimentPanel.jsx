/**
 * SentimentPanel v3.0
 * ========================
 * Displays decay-weighted FinBERT news sentiment, Reddit crowd sentiment,
 * and SEC filing flags.
 */
import React from 'react';

const trend_icons = {
  improving:    { icon: '↗', color: '#4ade80' },
  deteriorating:{ icon: '↘', color: '#f87171' },
  neutral:      { icon: '→', color: '#94a3b8' },
};

function ScoreBar({ value, label }) {
  // value in [-1, 1]
  const pct    = Math.round(((value + 1) / 2) * 100);
  const color  = value > 0.2 ? '#4ade80' : value < -0.2 ? '#f87171' : '#facc15';
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-header">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-value" style={{ color }}>
          {value >= 0 ? '+' : ''}{value.toFixed(3)}
        </span>
      </div>
      <div className="score-bar-track">
        <div className="score-bar-center" />
        <div
          className="score-bar-fill"
          style={{
            width:      `${Math.abs(value) * 50}%`,
            left:       value >= 0 ? '50%' : `${50 - Math.abs(value) * 50}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
}

function Badge({ label, active, color = '#4ade80' }) {
  return (
    <span
      className="badge"
      style={{
        background: active ? `${color}22` : 'rgba(255,255,255,0.05)',
        borderColor: active ? color : 'rgba(255,255,255,0.1)',
        color:       active ? color : '#64748b',
      }}
    >
      {active ? '●' : '○'} {label}
    </span>
  );
}

export default function SentimentPanel({ sentiment, secFlags }) {
  if (!sentiment) {
    return (
      <div className="card sentiment-card">
        <div className="card-header">
          <span className="card-icon">📰</span>
          <h3>Market Sentiment</h3>
        </div>
        <div className="no-data">Run analysis to see sentiment data</div>
      </div>
    );
  }

  const trendInfo = trend_icons[sentiment.sentiment_trend] || trend_icons.neutral;

  return (
    <div className="card sentiment-card">
      <div className="card-header">
        <span className="card-icon">📰</span>
        <h3>Market Sentiment</h3>
        <div className="sentiment-trend" style={{ color: trendInfo.color }}>
          {trendInfo.icon} {sentiment.sentiment_trend}
        </div>
      </div>

      {/* Headline */}
      {sentiment.most_recent_headline && (
        <div className="headline-box">
          <span className="headline-label">Latest Headline</span>
          <p className="headline-text">"{sentiment.most_recent_headline}"</p>
          {sentiment.effective_score_age_hours !== undefined && (
            <span className="headline-age">
              Effective age: {sentiment.effective_score_age_hours.toFixed(1)}h (decay-weighted)
            </span>
          )}
        </div>
      )}

      {/* Score bars */}
      <div className="score-bars">
        <ScoreBar value={sentiment.ticker_sentiment_score}  label="Ticker Sentiment (FinBERT)" />
        <ScoreBar value={sentiment.macro_sentiment_score}   label="Macro Sentiment" />
        {sentiment.reddit_sentiment_score !== undefined && (
          <ScoreBar value={sentiment.reddit_sentiment_score} label="Reddit Crowd Sentiment" />
        )}
      </div>

      {/* Stats row */}
      <div className="sentiment-stats">
        <div className="stat-pill">
          <span className="stat-label">Headlines</span>
          <span className="stat-val">{sentiment.headline_count || 0}</span>
        </div>
        <div className="stat-pill">
          <span className="stat-label">Vol Z-Score</span>
          <span className="stat-val">{(sentiment.news_volume_zscore || 0).toFixed(2)}σ</span>
        </div>
        {sentiment.reddit_mention_count !== undefined && (
          <div className="stat-pill">
            <span className="stat-label">Reddit Mentions</span>
            <span className="stat-val">{sentiment.reddit_mention_count}</span>
          </div>
        )}
        {sentiment.reddit_momentum && (
          <div className="stat-pill">
            <span className="stat-label">Reddit Trend</span>
            <span className="stat-val"
              style={{ color: sentiment.reddit_momentum === 'surging' ? '#4ade80'
                            : sentiment.reddit_momentum === 'fading'  ? '#f87171' : '#94a3b8' }}>
              {sentiment.reddit_momentum}
            </span>
          </div>
        )}
      </div>

      {/* Magnitude bar */}
      <div className="magnitude-wrap">
        <span className="stat-label">Sentiment Magnitude</span>
        <div className="magnitude-track">
          <div
            className="magnitude-fill"
            style={{ width: `${(sentiment.ticker_sentiment_magnitude || 0) * 100}%` }}
          />
        </div>
        <span className="magnitude-val">
          {((sentiment.ticker_sentiment_magnitude || 0) * 100).toFixed(0)}%
        </span>
      </div>

      {/* Source */}
      <div className="source-label">Source: {sentiment.source || 'newsapi+finbert'}</div>

      {/* SEC Flags */}
      {secFlags && (
        <div className="sec-flags">
          <span className="sec-title">SEC Flags</span>
          <div className="badges-row">
            <Badge label="8-K Filed"        active={secFlags.recent_8k}              color="#f97316" />
            <Badge label="Earnings ≤5d"     active={secFlags.earnings_within_5_days} color="#f87171" />
          </div>
          <div className="sec-details">
            {secFlags.days_since_last_8k < 999 && (
              <span>Last 8-K: {secFlags.days_since_last_8k}d ago</span>
            )}
            {secFlags.days_to_next_earnings < 999 && (
              <span>Next earnings: {secFlags.days_to_next_earnings}d</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
