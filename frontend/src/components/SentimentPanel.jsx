/**
 * SentimentPanel v4.0 — Clean, readable news & Reddit sentiment dashboard
 */
import React from 'react';

function ScoreGauge({ value, label }) {
  // value in [-1, 1]
  const color = value > 0.15 ? '#00e5a0' : value < -0.15 ? '#ff4f72' : '#ffb830';
  const word  = value > 0.15 ? 'Positive' : value < -0.15 ? 'Negative' : 'Neutral';
  const pct   = Math.abs(value) * 100;

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: '#94a3b8' }}>{label}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, fontWeight: 700, background: `${color}22`, color, padding: '2px 8px', borderRadius: 20 }}>
            {word}
          </span>
          <span style={{ fontSize: 14, fontFamily: 'var(--mono)', fontWeight: 700, color }}>
            {value >= 0 ? '+' : ''}{value.toFixed(3)}
          </span>
        </div>
      </div>
      <div style={{ height: 8, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: color, borderRadius: 4,
          marginLeft: value >= 0 ? '50%' : `${50 - pct}%`,
          transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  );
}

export default function SentimentPanel({ sentiment, secFlags }) {
  if (!sentiment) {
    return (
      <div className="card" style={{ minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#475569' }}>📰 Run analysis to see sentiment data</span>
      </div>
    );
  }

  const score = sentiment.ticker_sentiment_score ?? 0;
  const mainColor = score > 0.15 ? '#00e5a0' : score < -0.15 ? '#ff4f72' : '#ffb830';
  const mainWord  = score > 0.15 ? 'BULLISH' : score < -0.15 ? 'BEARISH' : 'NEUTRAL';
  const trend     = sentiment.sentiment_trend || 'neutral';
  const trendIcon = trend === 'improving' ? '↗' : trend === 'deteriorating' ? '↘' : '→';
  const trendColor = trend === 'improving' ? '#00e5a0' : trend === 'deteriorating' ? '#ff4f72' : '#94a3b8';

  return (
    <div className="card">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>NEWS SENTIMENT</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: mainColor }}>{mainWord}</div>
          <div style={{ fontSize: 13, color: trendColor, marginTop: 2 }}>
            {trendIcon} Trend {trend}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>MAGNITUDE</div>
          <div style={{ fontSize: 28, fontFamily: 'var(--mono)', fontWeight: 800, color: mainColor }}>
            {((sentiment.ticker_sentiment_magnitude || 0) * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Latest headline */}
      {sentiment.most_recent_headline && (
        <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 10, padding: '12px 14px', marginBottom: 16, borderLeft: `3px solid ${mainColor}` }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 600 }}>LATEST HEADLINE</div>
          <p style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.6, margin: 0 }}>
            "{sentiment.most_recent_headline}"
          </p>
          {sentiment.effective_score_age_hours !== undefined && (
            <div style={{ fontSize: 11, color: '#475569', marginTop: 6 }}>
              📅 Effective age: {sentiment.effective_score_age_hours.toFixed(1)}h (decay-weighted)
            </div>
          )}
        </div>
      )}

      {/* Score gauges */}
      <ScoreGauge value={sentiment.ticker_sentiment_score ?? 0}  label="Ticker (FinBERT)" />
      <ScoreGauge value={sentiment.macro_sentiment_score   ?? 0} label="Macro Sentiment" />
      {sentiment.reddit_sentiment_score !== undefined && (
        <ScoreGauge value={sentiment.reddit_sentiment_score} label="Reddit Crowd" />
      )}

      {/* Quick stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 4 }}>
        {[
          { label: 'Headlines', val: sentiment.headline_count || 0 },
          { label: 'Vol Z-Score', val: `${(sentiment.news_volume_zscore || 0).toFixed(2)}σ` },
          { label: 'Reddit', val: sentiment.reddit_mention_count ?? '—' },
        ].map(s => (
          <div key={s.label} style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '10px 12px', textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#f1f5f9', fontFamily: 'var(--mono)' }}>{s.val}</div>
          </div>
        ))}
      </div>

      {/* SEC flags */}
      {secFlags && (secFlags.recent_8k || secFlags.earnings_within_5_days) && (
        <div style={{ marginTop: 16, padding: '12px 14px', background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.3)', borderRadius: 10 }}>
          <div style={{ fontSize: 11, color: '#f97316', fontWeight: 700, marginBottom: 8 }}>🚨 SEC FLAGS</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {secFlags.recent_8k && (
              <span style={{ fontSize: 12, background: 'rgba(249,115,22,0.15)', color: '#f97316', padding: '4px 10px', borderRadius: 20, fontWeight: 600 }}>
                8-K Filed ({secFlags.days_since_last_8k}d ago)
              </span>
            )}
            {secFlags.earnings_within_5_days && (
              <span style={{ fontSize: 12, background: 'rgba(248,113,113,0.15)', color: '#f87171', padding: '4px 10px', borderRadius: 20, fontWeight: 600 }}>
                ⚠ Earnings in {secFlags.days_to_next_earnings}d
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
