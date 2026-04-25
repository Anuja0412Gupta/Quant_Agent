/**
 * AgentSignals v4.0 — Premium agent signal dashboard
 */
import React from 'react';

const SIGNAL_CFG = {
  BUY:           { color: '#00e5a0', icon: '🚀', bg: 'rgba(0,229,160,0.1)',  border: 'rgba(0,229,160,0.3)'  },
  BULLISH:       { color: '#00e5a0', icon: '📈', bg: 'rgba(0,229,160,0.1)',  border: 'rgba(0,229,160,0.3)'  },
  TRENDING:      { color: '#00e5a0', icon: '📈', bg: 'rgba(0,229,160,0.1)',  border: 'rgba(0,229,160,0.3)'  },
  trending:      { color: '#00e5a0', icon: '📈', bg: 'rgba(0,229,160,0.1)',  border: 'rgba(0,229,160,0.3)'  },
  SELL:          { color: '#ff4f72', icon: '🔻', bg: 'rgba(255,79,114,0.1)', border: 'rgba(255,79,114,0.3)' },
  BEARISH:       { color: '#ff4f72', icon: '📉', bg: 'rgba(255,79,114,0.1)', border: 'rgba(255,79,114,0.3)' },
  UPTREND:       { color: '#00e5a0', icon: '↗️', bg: 'rgba(0,229,160,0.08)', border: 'rgba(0,229,160,0.25)' },
  DOWNTREND:     { color: '#ff4f72', icon: '↘️', bg: 'rgba(255,79,114,0.08)', border: 'rgba(255,79,114,0.25)' },
  NEUTRAL:       { color: '#ffb830', icon: '⚖️', bg: 'rgba(255,184,48,0.1)', border: 'rgba(255,184,48,0.3)' },
  HOLD:          { color: '#ffb830', icon: '⚖️', bg: 'rgba(255,184,48,0.1)', border: 'rgba(255,184,48,0.3)' },
  FLAT:          { color: '#64748b', icon: '➖', bg: 'rgba(100,116,139,0.1)', border: 'rgba(100,116,139,0.3)' },
  HIGH_VOLATILITY:{ color: '#ff4f72', icon: '⚡', bg: 'rgba(255,79,114,0.1)', border: 'rgba(255,79,114,0.3)' },
  mean_reverting:{ color: '#60a5fa', icon: '↔️', bg: 'rgba(96,165,250,0.1)', border: 'rgba(96,165,250,0.3)'  },
};

function getSignalCfg(signal) {
  const key = String(signal || '').toUpperCase().replace(/ /g, '_');
  return SIGNAL_CFG[key] || SIGNAL_CFG[String(signal || '').toLowerCase()] || {
    color: '#64748b', icon: '❓', bg: 'rgba(100,116,139,0.1)', border: 'rgba(100,116,139,0.3)'
  };
}

function AgentCard({ icon, title, signal, confidence, explanation, extraRows = [] }) {
  const cfg = getSignalCfg(signal);
  const pct = Math.round(Math.max(0, Math.min(1, confidence || 0)) * 100);
  const confColor = pct > 66 ? '#00e5a0' : pct > 33 ? '#ffb830' : '#ff4f72';

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: `1px solid ${cfg.border}`,
      borderRadius: 14,
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 20 }}>{icon}</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.5 }}>{title}</span>
        </div>
        {signal && (
          <div style={{
            padding: '4px 12px', borderRadius: 20,
            background: cfg.bg, border: `1px solid ${cfg.border}`,
            color: cfg.color, fontSize: 12, fontWeight: 800, letterSpacing: 1,
          }}>
            {cfg.icon} {String(signal).toUpperCase()}
          </div>
        )}
      </div>

      {/* Confidence bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12, color: '#64748b' }}>
          <span>Confidence</span>
          <span style={{ color: confColor, fontWeight: 700, fontFamily: 'var(--mono)' }}>{pct}%</span>
        </div>
        <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: confColor, borderRadius: 3, transition: 'width 0.8s ease' }} />
        </div>
      </div>

      {/* Extra key-value rows */}
      {extraRows.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6 }}>
          {extraRows.map(({ label, value }) => (
            <div key={label} style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 8, padding: '8px 10px' }}>
              <div style={{ fontSize: 10, color: '#64748b', fontWeight: 700, marginBottom: 3 }}>{label}</div>
              <div style={{ fontSize: 13, color: '#f1f5f9', fontFamily: 'var(--mono)', fontWeight: 600 }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Explanation */}
      {explanation && (
        <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6, margin: 0, borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 10 }}>
          {explanation}
        </p>
      )}
    </div>
  );
}

export default function AgentSignals({ data }) {
  if (!data) return null;

  const disagreement = data.disagreement || {};
  const regime       = data.regime || {};
  const list         = Array.isArray(data.agent_signals) ? data.agent_signals : [];
  const byName       = Object.fromEntries(list.map(a => [String(a?.agent_name || '').toLowerCase(), a]));

  const indicator = byName.indicator?.reasoning || {};
  const pattern   = byName.pattern?.reasoning   || {};
  const trend     = byName.trend?.reasoning     || {};

  const indicatorConf   = byName.indicator?.confidence ?? 0;
  const patternConf     = byName.pattern?.confidence   ?? 0;
  const trendConf       = byName.trend?.confidence     ?? 0;
  const disagreeScore   = disagreement?.disagreement_score ?? disagreement?.total_uncertainty ?? 0;
  const agentConsensus  = disagreement?.agent_consensus ?? 0;

  const ConsensusColor = disagreeScore > 0.7 ? '#ff4f72' : disagreeScore > 0.4 ? '#ffb830' : '#00e5a0';

  return (
    <div className="card" style={{ background: 'transparent', border: 'none', padding: 0, boxShadow: 'none' }}>
      <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>
        MULTI-AGENT SIGNAL CONSENSUS
      </div>

      {/* Agreement banner */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: `rgba(${disagreeScore > 0.7 ? '255,79,114' : disagreeScore > 0.4 ? '255,184,48' : '0,229,160'},0.08)`,
        border: `1px solid ${ConsensusColor}40`,
        borderRadius: 12, padding: '14px 20px', marginBottom: 20, gap: 16,
      }}>
        <div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>DOMINANT SIGNAL</div>
          <div style={{ fontSize: 22, fontWeight: 800, color: ConsensusColor }}>
            {disagreement.dominant_signal || 'NEUTRAL'}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>AGENT CONSENSUS</div>
          <div style={{ fontSize: 22, fontWeight: 800, fontFamily: 'var(--mono)', color: ConsensusColor }}>
            {(agentConsensus * 100).toFixed(0)}%
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>DISAGREEMENT</div>
          <div style={{ fontSize: 22, fontWeight: 800, fontFamily: 'var(--mono)', color: ConsensusColor }}>
            {(disagreeScore * 100).toFixed(0)}%
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>RECOMMENDATION</div>
          <div style={{ fontSize: 16, fontWeight: 800, color: ConsensusColor, padding: '4px 12px', background: `${ConsensusColor}22`, borderRadius: 20 }}>
            {disagreement.recommendation || 'PROCEED'}
          </div>
        </div>
      </div>

      {/* Agent cards grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
        <AgentCard
          icon="📊" title="Indicator Agent"
          signal={indicator?.signal}
          confidence={indicatorConf}
          explanation={indicator?.explanation}
          extraRows={Object.entries(indicator?.indicator_values || {}).slice(0, 4).map(([k, v]) => ({
            label: k.toUpperCase(),
            value: typeof v === 'number' ? v.toFixed(3) : String(v),
          }))}
        />

        <AgentCard
          icon="🕯️" title="Pattern Agent"
          signal={pattern?.signal}
          confidence={patternConf}
          explanation={pattern?.explanation}
          extraRows={pattern?.pattern ? [{ label: 'PATTERN', value: pattern.pattern }] : []}
        />

        <AgentCard
          icon="📐" title="Trend Agent"
          signal={trend?.trend}
          confidence={trendConf}
          explanation={trend?.explanation}
          extraRows={[
            { label: 'SLOPE', value: trend?.slope?.toFixed?.(4) ?? '-' },
            { label: 'STRENGTH', value: trend?.strength != null ? `${(trend.strength * 100).toFixed(1)}%` : '-' },
            { label: 'SUPPORT', value: trend?.support?.toFixed?.(2) ?? '-' },
            { label: 'RESIST', value: trend?.resistance?.toFixed?.(2) ?? '-' },
          ].filter(r => r.value !== '-')}
        />

        <AgentCard
          icon="🎯" title="Regime Agent"
          signal={regime?.regime || regime?.signal}
          confidence={regime?.confidence}
          explanation={regime?.explanation}
          extraRows={[
            { label: 'HURST', value: regime?.hurst?.toFixed?.(3) ?? '-' },
            { label: 'ATR %', value: regime?.atr_ratio != null ? `${(regime.atr_ratio * 100).toFixed(2)}%` : '-' },
          ].filter(r => r.value !== '-')}
        />
      </div>
    </div>
  );
}
