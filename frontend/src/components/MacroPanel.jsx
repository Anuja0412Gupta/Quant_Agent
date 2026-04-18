/**
 * MacroPanel v3.0
 * =================
 * Displays macro context: VIX regime, credit spread, DXY momentum,
 * yield curve spread, and Fed funds rate.
 */
import React from 'react';

function MacroStat({ label, value, unit = '', color, description }) {
  return (
    <div className="macro-stat">
      <div className="macro-stat-label">{label}</div>
      <div className="macro-stat-value" style={{ color }}>
        {typeof value === 'number' ? value.toFixed(2) : value}{unit}
      </div>
      {description && <div className="macro-stat-desc">{description}</div>}
    </div>
  );
}

function CreditAlert({ ratio, threshold = 0 }) {
  const isRisk = ratio < threshold;
  return (
    <div className={`credit-alert ${isRisk ? 'alert-danger' : 'alert-ok'}`}>
      <span>{isRisk ? '⚠ Credit stress detected' : '✓ Credit stable'}</span>
    </div>
  );
}

export default function MacroPanel({ macro }) {
  if (!macro) {
    return (
      <div className="card macro-card">
        <div className="card-header">
          <span className="card-icon">🌍</span>
          <h3>Macro Context</h3>
        </div>
        <div className="no-data">Run analysis to see macro data</div>
      </div>
    );
  }

  const vixColor = macro.vix_level > 30 ? '#f87171'
                 : macro.vix_level > 20 ? '#facc15'
                 : '#4ade80';

  const yieldColor = macro.t10y2y_spread < 0 ? '#f87171'
                   : macro.t10y2y_spread < 0.5 ? '#facc15'
                   : '#4ade80';

  const dxyColor = macro.dxy_20d_momentum > 0.02  ? '#f87171'   // strong dollar bad for risk
                 : macro.dxy_20d_momentum < -0.02 ? '#4ade80'   // weak dollar good
                 : '#94a3b8';

  return (
    <div className="card macro-card">
      <div className="card-header">
        <span className="card-icon">🌍</span>
        <h3>Macro Context</h3>
        {macro.freshness_ts && (
          <span className="macro-freshness">
            Updated: {new Date(macro.freshness_ts).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* VIX section */}
      <div className="macro-section">
        <div className="macro-section-title">Volatility Term Structure</div>
        <div className="macro-stats-row">
          <MacroStat label="VIX"      value={macro.vix_level}      color={vixColor} description="1M realized vol" />
          <MacroStat label="VIX 9D"   value={macro.vix9d_level}    color={vixColor} description="Short-term vol" />
          <MacroStat label="TS Spread" value={macro.vix_ts_spread}  unit="" color={
            macro.vix_ts_spread < 0 ? '#f87171' : '#4ade80'
          } description="VIX9D−VIX (backwardation<0)" />
          <MacroStat label="VIX Z"    value={macro.vix_zscore}     unit="σ" color={vixColor} />
        </div>
      </div>

      {/* Credit section */}
      <div className="macro-section">
        <div className="macro-section-title">Credit Markets</div>
        <div className="macro-stats-row">
          <MacroStat label="HYG/LQD"    value={macro.hyg_lqd_ratio}  color={
            macro.credit_regime_flag === 1 ? '#f87171' : '#4ade80'
          } description="High yield vs inv-grade" />
          <MacroStat label="HYG Z"      value={macro.hyg_lqd_zscore} unit="σ" color={
            macro.hyg_lqd_zscore < -1.5 ? '#f87171' : '#94a3b8'
          } />
          <MacroStat label="HY Spread"  value={macro.hy_credit_spread} unit="bps" color={
            macro.hy_credit_spread > 500 ? '#f87171' : '#94a3b8'
          } description="HY−IG spread" />
        </div>
        <CreditAlert ratio={macro.hyg_lqd_zscore} threshold={-1.5} />
      </div>

      {/* Rates section */}
      <div className="macro-section">
        <div className="macro-section-title">Fixed Income & FX</div>
        <div className="macro-stats-row">
          <MacroStat label="10Y−2Y"    value={macro.t10y2y_spread}   unit="%" color={yieldColor}
            description={macro.t10y2y_spread < 0 ? '⚠ Inverted yield curve' : 'Normal curve'} />
          <MacroStat label="Fed Funds" value={macro.fed_funds_rate}  unit="%" color="#94a3b8" />
          <MacroStat label="DXY 20D"   value={(macro.dxy_20d_momentum * 100)} unit="%" color={dxyColor}
            description="USD 20-day momentum" />
          <MacroStat label="Consumer" value={macro.consumer_sentiment} color="#94a3b8"
            description="UMich sentiment" />
        </div>
      </div>
    </div>
  );
}
