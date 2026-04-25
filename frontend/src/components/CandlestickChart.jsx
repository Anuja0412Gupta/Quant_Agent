import React from 'react';
import {
  ComposedChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';

/** Custom tooltip */
const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const isUp = d.close >= d.open;
  return (
    <div style={{
      background: '#1a2035', border: '1px solid rgba(148,163,184,0.2)',
      borderRadius: 10, padding: '12px 16px', fontSize: 13,
      fontFamily: "'JetBrains Mono', monospace", boxShadow: '0 8px 32px rgba(0,0,0,0.5)'
    }}>
      <p style={{ color: '#64748b', marginBottom: 8, fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 600 }}>
        {d.dateStr}
      </p>
      {[['O', d.open], ['H', d.high], ['L', d.low], ['C', d.close]].map(([k, v]) => (
        <p key={k} style={{
          color: k === 'C' ? (isUp ? '#00e5a0' : '#ff4f72') : '#cbd5e1',
          margin: '3px 0', display: 'flex', justifyContent: 'space-between', gap: 24
        }}>
          <span style={{ color: '#64748b' }}>{k}</span>
          <span>${Number(v).toFixed(2)}</span>
        </p>
      ))}
      <p style={{ color: '#64748b', marginTop: 8, fontSize: 11, paddingTop: 8, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        Vol: {Number(d.volume).toLocaleString()}
      </p>
    </div>
  );
};

export default function CandlestickChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
                    height: '100%', color: '#475569', fontSize: 14, gap: 8 }}>
        📊 No chart data — run Analyze to load price history
      </div>
    );
  }

  // Parse ohlcv bars — backend uses 'timestamp' as string like "2025-12-01 05:00:00"
  const chartData = data.map(d => {
    const open  = Number(d.open);
    const high  = Number(d.high);
    const low   = Number(d.low);
    const close = Number(d.close);
    const bodyLo = Math.min(open, close);
    const bodyHi = Math.max(open, close);
    const dateStr = (d.timestamp || d.time || '').toString().split(' ')[0];
    // Format date for display
    const parts = dateStr.split('-');
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const displayDate = parts.length >= 2 ? `${months[parseInt(parts[1]) - 1]} ${parts[2]}` : dateStr;
    return {
      dateStr: displayDate,
      rawDate: dateStr,
      open, high, low, close,
      volume: Number(d.volume) || 0,
      base:   bodyLo - low,
      body:   Math.max(bodyHi - bodyLo, 0.01),
      wickHi: high - bodyHi,
      isUp:   close >= open,
    };
  });

  // Sort by date — show last 50 bars so each candle is large and readable
  const display = chartData
    .sort((a, b) => a.rawDate.localeCompare(b.rawDate))
    .slice(-50);

  const allLows  = display.map(d => d.low).filter(Number.isFinite);
  const allHighs = display.map(d => d.high).filter(Number.isFinite);
  if (!allLows.length) return null;
  const domainLo = Math.min(...allLows)  * 0.998;
  const domainHi = Math.max(...allHighs) * 1.002;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={display} margin={{ top: 8, right: 16, left: 0, bottom: 0 }} barCategoryGap="20%" barGap={0}>
        <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
        <XAxis
          dataKey="dateStr"
          tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: 'Inter, sans-serif' }}
          axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
          tickLine={false}
          interval={Math.max(1, Math.floor(display.length / 8))}
        />
        <YAxis
          domain={[domainLo, domainHi]}
          tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: 'Inter, sans-serif' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={v => `$${v.toFixed(0)}`}
          width={65}
          orientation="right"
        />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }} />

        {/* Transparent base spacer */}
        <Bar dataKey="base" stackId="c" fill="transparent" stroke="none" />

        {/* Candle body */}
        <Bar dataKey="body" stackId="c" radius={[2, 2, 0, 0]}
          shape={(props) => {
            const d = props.payload;
            const color = d.isUp ? '#00e5a0' : '#ff4f72';
            return (
              <rect
                x={props.x + 1} y={props.y}
                width={Math.max(props.width - 2, 1)}
                height={Math.max(props.height, 1)}
                fill={color}
                fillOpacity={0.9}
              />
            );
          }}
        />

        {/* Upper wick */}
        <Bar dataKey="wickHi" stackId="c" fill="transparent" stroke="none"
          shape={(props) => {
            const d = props.payload;
            const color = d.isUp ? '#00e5a0' : '#ff4f72';
            const cx = props.x + props.width / 2;
            return (
              <line x1={cx} y1={props.y} x2={cx} y2={props.y + props.height}
                    stroke={color} strokeWidth={1.5} opacity={0.8} />
            );
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
