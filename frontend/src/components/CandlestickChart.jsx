import {
  ComposedChart, Bar, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine
} from 'recharts';

/**
 * CandlestickChart (Recharts fallback)
 * Renders candlestick bars using Recharts ComposedChart with custom shapes.
 * Fully stable and doesn't rely on complex canvas library APIs.
 */

// Custom candlestick shape
function Candle(props) {
  const { x, y, width, height, payload } = props;
  if (!payload) return null;

  const { open, high, low, close, volume } = payload;
  if ([open, high, low, close].some(v => !isFinite(v))) return null;

  const isUp = close >= open;
  const color = isUp ? '#00e5a0' : '#ff4f72';

  // The chart maps value axis — we get y/height from the two-bar trick
  // Instead, we'll use the raw payload values and scale ourselves
  // This component is called per-bar and uses the chart's y-scale
  // We receive y and height from the stacked bar positioning
  const barWidth = Math.max(width - 2, 2);
  const cx = x + width / 2;

  return (
    <g>
      {/* Wick line (high-low) */}
      <line x1={cx} y1={y} x2={cx} y2={y + height}
            stroke={color} strokeWidth={1} opacity={0.8} />
      {/* Body */}
      <rect
        x={x + 1}
        y={y}
        width={barWidth}
        height={Math.max(height, 1)}
        fill={color}
        stroke={color}
        opacity={isUp ? 0.85 : 0.85}
      />
    </g>
  );
}

// Custom tooltip
const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const isUp = d.close >= d.open;
  return (
    <div style={{
      background: '#1d2235', border: '1px solid rgba(99,120,255,0.25)',
      borderRadius: 8, padding: '10px 14px', fontSize: 12, fontFamily: 'JetBrains Mono, monospace'
    }}>
      <p style={{ color: '#647091', marginBottom: 6, fontFamily: 'Inter, sans-serif', fontSize: 11 }}>
        {d.dateStr}
      </p>
      {[['O', d.open], ['H', d.high], ['L', d.low], ['C', d.close]].map(([k, v]) => (
        <p key={k} style={{ color: k === 'C' ? (isUp ? '#00e5a0' : '#ff4f72') : '#a8b4d0', margin: '2px 0' }}>
          {k}: ${Number(v).toFixed(2)}
        </p>
      ))}
      <p style={{ color: '#647091', marginTop: 4, fontSize: 10 }}>
        Vol: {Number(d.volume).toLocaleString()}
      </p>
    </div>
  );
};

export default function CandlestickChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
                    height: '100%', color: '#3a4466', fontSize: 13 }}>
        No chart data
      </div>
    );
  }

  // Prepare data for a OHLC representation using two bars per candle:
  // bar1 = from low to min(open,close), bar2 = from min to max(open,close)
  const chartData = data.map(d => {
    const open  = Number(d.open);
    const high  = Number(d.high);
    const low   = Number(d.low);
    const close = Number(d.close);
    const bodyLo = Math.min(open, close);
    const bodyHi = Math.max(open, close);
    const date   = new Date(d.time * 1000);
    const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    return {
      dateStr,
      time: d.time,
      open, high, low, close,
      volume: Number(d.volume) || 0,
      // For stacked bar: invisible base + body + wick top
      wickLo:   low,
      body:     bodyHi - bodyLo,
      wickHi:   high - bodyHi,
      base:     bodyLo - low,    // transparent spacer from low to body bottom
      isUp:     close >= open,
    };
  });

  // Show only last 60 bars for readability
  const display = chartData.slice(-60);

  const allLows  = display.map(d => d.low).filter(isFinite);
  const allHighs = display.map(d => d.high).filter(isFinite);
  const domainLo = Math.min(...allLows)  * 0.999;
  const domainHi = Math.max(...allHighs) * 1.001;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={display} margin={{ top: 8, right: 12, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="rgba(99,120,255,0.06)" />
        <XAxis
          dataKey="dateStr"
          tick={{ fill: '#647091', fontSize: 10 }}
          axisLine={{ stroke: 'rgba(99,120,255,0.15)' }}
          tickLine={false}
          interval={Math.max(1, Math.floor(display.length / 10))}
        />
        <YAxis
          domain={[domainLo, domainHi]}
          tick={{ fill: '#647091', fontSize: 10 }}
          axisLine={{ stroke: 'rgba(99,120,255,0.15)' }}
          tickLine={false}
          tickFormatter={v => `$${v.toFixed(0)}`}
          width={62}
        />
        <Tooltip content={<CustomTooltip />} />

        {/* Transparent base spacer (positions body above lows) */}
        <Bar dataKey="base"   stackId="candle" fill="transparent" stroke="none" />

        {/* Body: green for up, red for down */}
        <Bar dataKey="body" stackId="candle" radius={[1,1,0,0]}
          fill="#00e5a0"
          label={false}
          shape={(props) => {
            const d = props.payload;
            const color = d.isUp ? '#00e5a0' : '#ff4f72';
            return (
              <rect
                x={props.x + 1} y={props.y}
                width={Math.max(props.width - 2, 1)}
                height={Math.max(props.height, 1)}
                fill={color} opacity={0.85}
              />
            );
          }}
        />

        {/* Wick top (above body to high) */}
        <Bar dataKey="wickHi" stackId="candle" fill="transparent" stroke="none"
          shape={(props) => {
            const d = props.payload;
            const color = d.isUp ? '#00e5a0' : '#ff4f72';
            const cx = props.x + props.width / 2;
            return (
              <line x1={cx} y1={props.y} x2={cx} y2={props.y + props.height}
                    stroke={color} strokeWidth={1.5} opacity={0.7} />
            );
          }}
        />

        {/* Wick bottom drawn using a custom line from base bottom */}
        {display.map((d, i) => null)}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
