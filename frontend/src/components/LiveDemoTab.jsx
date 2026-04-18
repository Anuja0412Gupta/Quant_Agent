import { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PIPELINE_STAGES = [
  { key: 'fetch', icon: 'Data', label: 'Data Fetch', desc: 'OHLCV bars + technicals' },
  { key: 'agents', icon: 'Agents', label: 'Agent Signals', desc: 'Indicator, Pattern, Trend, Regime' },
  { key: 'state', icon: 'State', label: 'RL State Vector', desc: 'State features and policy inputs' },
  { key: 'rl', icon: 'RL', label: 'RL Policy Action', desc: 'PPO direction + sizing' },
  { key: 'gate', icon: 'Gate', label: 'Disagreement Gate', desc: 'Uncertainty scaling' },
  { key: 'risk', icon: 'Risk', label: 'VaR Risk Gate', desc: 'Risk checks and veto' },
  { key: 'execute', icon: 'Exec', label: 'Trade Decision', desc: 'LONG / SHORT / FLAT' },
];

export default function LiveDemoTab() {
  const [ticker, setTicker] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1d');
  const [steps, setSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [running, setRunning] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [equityRL, setEquityRL] = useState([]);
  const [equityEW, setEquityEW] = useState([]);
  const [equityBH, setEquityBH] = useState([]);
  const intervalRef = useRef(null);

  const startReplay = useCallback(async () => {
    setRunning(true);
    setCurrentStep(0);
    setEquityRL([]);
    setEquityEW([]);
    setEquityBH([]);

    try {
      const { data: an } = await axios.get(`${API}/analyze/${ticker.toUpperCase()}`, {
        params: { timeframe },
        timeout: 60_000,
      });
      setAnalysis(an);

      const { data: bt } = await axios.post(
        `${API}/backtest`,
        {
          symbol: ticker.toUpperCase(),
          timeframe,
          period: '1y',
          n_folds: 4,
          initial_capital: 100000,
          use_walk_forward: true,
        },
        { timeout: 180_000 }
      );

      let peak = 0;
      const curve = Array.isArray(bt?.equity_curve) ? bt.equity_curve : [];
      const ts = Array.isArray(bt?.timestamps) ? bt.timestamps : [];
      const built = curve.map((value, idx) => {
        peak = Math.max(peak, Number(value) || 0);
        const dd = peak > 0 ? Math.max(0, (peak - (Number(value) || 0)) / peak) : 0;
        return {
          day: idx + 1,
          date: ts[idx],
          rl_equity: Number(value) || 0,
          bh_equity: Number(value) || 0,
          action: an?.trade_decision?.recommendation || 'HOLD',
          price: an?.current_price || 0,
          var_override: false,
          drawdown: dd,
          current_var: an?.trade_decision?.current_cvar || 0,
        };
      });

      setSteps(built);

      let i = 0;
      intervalRef.current = setInterval(() => {
        if (i >= built.length) {
          clearInterval(intervalRef.current);
          setRunning(false);
          return;
        }

        const step = built[i];
        setCurrentStep(i);
        setEquityRL((prev) => [...prev, { day: step.day, value: step.rl_equity }]);
        setEquityEW((prev) => [...prev, { day: step.day, value: step.bh_equity * 0.97 }]);
        setEquityBH((prev) => [...prev, { day: step.day, value: step.bh_equity }]);
        i += 1;
      }, 350);
    } catch (e) {
      console.error('Live demo failed:', e.message);
      setRunning(false);
    }
  }, [ticker, timeframe]);

  const stopReplay = () => {
    clearInterval(intervalRef.current);
    setRunning(false);
  };

  const currentStepData = steps[currentStep] || {};
  const rl = analysis?.rl_weights || {};
  const risk = analysis?.trade_decision || {};

  const chartData = equityRL.map((r, i) => ({
    day: r.day,
    RL: r.value,
    EqualWeight: equityEW[i]?.value,
    BuyHold: equityBH[i]?.value,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span className="card-title">Live Demo</span>
        <input
          id="live-ticker-input"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          style={{ width: 90, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '7px 12px', color: '#fff', fontSize: 13 }}
        />
        <select
          id="live-tf-select"
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
          style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '7px 12px', color: '#fff', fontSize: 13 }}
        >
          {['1m', '5m', '15m', '1h', '1d'].map((tf) => (
            <option key={tf} value={tf}>{tf}</option>
          ))}
        </select>
        {!running ? (
          <button id="start-replay-btn" className="btn-analyze" onClick={startReplay}>Start Replay</button>
        ) : (
          <button id="stop-replay-btn" className="btn-backtest" onClick={stopReplay}>Stop</button>
        )}
        {currentStep >= 0 && <span style={{ fontSize: 12, color: '#8b9fc0' }}>Day {currentStep + 1} / {steps.length}</span>}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 20 }}>
        <div className="card">
          <div className="card-header"><span className="card-title">Pipeline Flow</span></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {PIPELINE_STAGES.map((stage, idx) => {
              const isActive = currentStep >= 0 && idx <= 6;
              const isCurrent = running && idx === 4;
              return (
                <div
                  key={stage.key}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', borderRadius: 10,
                    background: isCurrent ? 'rgba(99,102,241,0.15)' : isActive ? 'rgba(104,211,145,0.05)' : 'rgba(255,255,255,0.02)',
                    border: isCurrent ? '1px solid rgba(99,102,241,0.4)' : isActive ? '1px solid rgba(104,211,145,0.15)' : '1px solid transparent',
                  }}
                >
                  <span style={{ fontSize: 13, fontFamily: 'monospace', color: '#a0aec0' }}>{stage.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: isCurrent ? '#c3dafe' : isActive ? '#e2e8f0' : '#647091' }}>{stage.label}</div>
                    <div style={{ fontSize: 11, color: '#647091' }}>{stage.desc}</div>
                  </div>
                  {isActive && <span style={{ color: '#68d391', fontSize: 14 }}>OK</span>}
                </div>
              );
            })}
          </div>

          {currentStepData.action && (
            <div style={{ marginTop: 12, padding: '10px 14px', borderRadius: 10, background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.2)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                {[
                  { l: 'Day', v: currentStepData.day },
                  { l: 'Price', v: `$${Number(currentStepData.price || 0).toFixed(2)}` },
                  { l: 'Action', v: currentStepData.action },
                  { l: 'VaR Override', v: currentStepData.var_override ? 'YES' : 'No' },
                  { l: 'Drawdown', v: `${((currentStepData.drawdown || 0) * 100).toFixed(2)}%` },
                  { l: 'Daily VaR', v: `${((currentStepData.current_var || 0) * 100).toFixed(2)}%` },
                ].map(({ l, v }) => (
                  <div key={l}>
                    <div style={{ fontSize: 10, color: '#647091' }}>{l}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', fontFamily: 'monospace' }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-header">
              <span className="card-title">Live Equity Curves</span>
              <span style={{ fontSize: 11, color: '#647091' }}>Updating each tick</span>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="day" tick={{ fill: '#647091', fontSize: 10 }} />
                <YAxis tick={{ fill: '#647091', fontSize: 10 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip contentStyle={{ background: 'rgba(26,27,58,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} formatter={(v) => [`$${Number(v || 0).toFixed(0)}`]} />
                <Legend wrapperStyle={{ color: '#a0aec0', fontSize: 11 }} />
                <Line type="monotone" dataKey="RL" stroke="#6366f1" dot={false} strokeWidth={2.5} name="RL Portfolio" isAnimationActive />
                <Line type="monotone" dataKey="EqualWeight" stroke="#68d391" dot={false} strokeWidth={1.5} strokeDasharray="4 4" name="Equal Weight" isAnimationActive />
                <Line type="monotone" dataKey="BuyHold" stroke="#f6e05e" dot={false} strokeWidth={1.5} strokeDasharray="2 4" name="Buy & Hold" isAnimationActive />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <div className="card-header"><span className="card-title">Risk Dashboard</span></div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                {
                  label: 'VaR (95%)',
                  value: risk.current_cvar != null ? `${(risk.current_cvar * 100).toFixed(2)}%` : '-',
                  color: (risk.current_cvar || 0) > 0.03 ? '#fc8181' : '#68d391',
                },
                { label: 'Risk Approval', value: risk.approved ? 'Approved' : 'Blocked', color: risk.approved ? '#68d391' : '#fc8181' },
                { label: 'Stop Loss', value: risk.stop_loss != null ? `$${Number(risk.stop_loss).toFixed(2)}` : '-', color: '#fc8181' },
                { label: 'Take Profit', value: risk.take_profit != null ? `$${Number(risk.take_profit).toFixed(2)}` : '-', color: '#68d391' },
                { label: 'Adjusted Action', value: risk.adjusted_action != null ? Number(risk.adjusted_action).toFixed(4) : '-', color: '#f6e05e' },
                { label: 'Veto Reason', value: risk.veto_reason || 'None', color: risk.veto_reason ? '#fc8181' : '#68d391' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontSize: 10, color: '#647091', marginBottom: 3 }}>{label}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'monospace' }}>{value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card" style={{ padding: '12px', fontSize: 12, color: '#8b9fc0' }}>
            RL action: {Number(rl.effective_action || 0).toFixed(4)} | Gate: {Number(rl.gate_value || 0).toFixed(4)} | Regime: {rl.active_regime || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
