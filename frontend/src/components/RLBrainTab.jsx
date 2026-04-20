import { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * RLBrainTab — RL Policy Visualization
 * =====================================
 * - Action gauge (rl_action ∈ [-1, +1])
 * - Reward curve + entropy
 * - Reward ablation bar chart (KEY showstopper)
 * - Disagreement gate visualization
 * - Active regime + MoE policy
 */
export default function RLBrainTab({ analysis }) {
  const [brain, setBrain]       = useState(null);
  const [ablation, setAblation] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [ablLoading, setAblLoading] = useState(false);
  const [ticker, setTicker]     = useState('AAPL');
  const [progress, setProgress] = useState(0);

  // Upload model state
  const [uploadFile, setUploadFile]     = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null); // null | 'uploading' | 'success' | 'error'
  const [uploadMsg, setUploadMsg]       = useState('');
  const [modelInfo, setModelInfo]       = useState(null);

  // Fetch model info on mount
  useEffect(() => {
    axios.get(`${API}/rl/model-info`, { params: { symbol: ticker } })
      .then(({ data }) => setModelInfo(data))
      .catch(() => {});
  }, [ticker]);

  // Poll brain when training is active
  useEffect(() => {
    let intervalId;
    if (brain?.training_active) {
      intervalId = setInterval(() => {
        setProgress(p => Math.min(p + (Math.random() * 5 + 1), 95));
        axios.get(`${API}/rl/brain`, { params: { symbol: ticker } })
          .then(({ data }) => {
            setBrain(data);
            if (!data.training_active) setProgress(100);
          }).catch(err => console.error(err));
      }, 3000);
    } else {
      if (progress >= 95) setProgress(100);
    }
    return () => clearInterval(intervalId);
  }, [brain?.training_active, ticker]);

  const fetchBrain = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API}/rl/brain`, { timeout: 30_000 });
      setBrain(data);
    } catch (e) {
      console.error('RL Brain fetch failed:', e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAblation = useCallback(async () => {
    setAblLoading(true);
    try {
      const { data } = await axios.post(`${API}/rl/ablation`, { symbol: ticker, timeframe: '1d', period: '2y' }, { timeout: 120_000 });
      setAblation(data);
    } catch (e) {
      console.error('Ablation fetch failed:', e.message);
    } finally {
      setAblLoading(false);
    }
  }, [ticker]);

  const handleTrain = async () => {
    setLoading(true);
    try {
      await axios.post(`${API}/rl/train?symbol=${analysis?.symbol || ticker}&timeframe=1d`);
      setBrain(prev => ({ ...(prev || {}), training_active: true }));
      setProgress(0);
    } catch (e) {
      console.error('Training failed:', e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadModel = async () => {
    if (!uploadFile) return;
    setUploadStatus('uploading');
    setUploadMsg('');
    try {
      const form = new FormData();
      form.append('file', uploadFile);
      const { data } = await axios.post(
        `${API}/rl/upload-model?symbol=${ticker}&timeframe=1d`,
        form,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 60_000 }
      );
      // model_ok = live verification passed; 'saved' = on disk, needs restart
      setUploadStatus(data.model_ok ? 'success' : 'saved');
      setUploadMsg(data.message || 'Model uploaded');
      setModelInfo({ exists: true, symbol: ticker, size_kb: data.size_kb, modified_at: new Date().toISOString() });
      setUploadFile(null);
    } catch (e) {
      setUploadStatus('error');
      setUploadMsg(e.response?.data?.detail || e.message || 'Upload failed');
    }
  };

  const rl      = analysis?.rl_weights || {};
  const action  = rl.effective_action ?? 0;
  const posSize = rl.effective_position ?? 0;
  const disagr  = rl.disagreement_score ?? 0;
  const gate    = rl.gate_value ?? 1;
  const regime  = rl.active_regime ?? '—';
  const actionThreshold = 0.02;

  // Gauge needle position: action ∈ [-1, +1] → rotate from -90° to +90°
  const needleDeg = action * 90;

  // Ablation chart data
  const ablationData = ablation ? Object.entries(ablation.variants || {}).map(([name, v]) => ({
    name: name.replace('_', ' ').replace(/^\w/, c => c.toUpperCase()),
    sharpe: v.sharpe,
    total: v.total_reward,
  })) : [];

  // Reward curve data
  const rewardData  = (brain?.reward_history  || []).map((v, i) => ({ step: i, reward: v }));
  const entropyData = (brain?.entropy_history || []).map((v, i) => ({ step: i, entropy: v }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Training Progress Banner */}
      {brain?.training_active && (
        <div style={{ background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.4)', borderRadius: 8, padding: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ color: '#c3dafe', fontSize: 13, fontWeight: 600 }}>⚡ Reinforcement Learning in Progress</span>
            <span style={{ color: '#818cf8', fontSize: 13, fontFamily: 'monospace' }}>{progress.toFixed(0)}%</span>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.05)', height: 6, borderRadius: 3, overflow: 'hidden' }}>
            <div style={{ background: '#818cf8', height: '100%', width: `${progress}%`, transition: 'width 0.5s ease-out' }} />
          </div>
          <div style={{ fontSize: 11, color: '#a0aec0', marginTop: 8 }}>
            Executing multi-phase curriculum (Trending → Mean-Reverting → High-Vol). Lagrangian constraints actively tightening.
          </div>
        </div>
      )}


      {/* Action Gauge + Regime */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        {/* Gauge */}
        <div className="card">
          <div className="card-header"><span className="card-title">⚡ RL Action</span></div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '12px 0' }}>
            <div style={{ position: 'relative', width: 140, height: 80 }}>
              <svg viewBox="0 0 140 80" width="140" height="80">
                {/* Arc background */}
                <path d="M 10 70 A 60 60 0 0 1 130 70" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="12" strokeLinecap="round" />
                {/* Negative zone */}
                <path d="M 10 70 A 60 60 0 0 1 70 10" fill="none" stroke="rgba(252,129,129,0.3)" strokeWidth="12" strokeLinecap="round" />
                {/* Positive zone */}
                <path d="M 70 10 A 60 60 0 0 1 130 70" fill="none" stroke="rgba(104,211,145,0.3)" strokeWidth="12" strokeLinecap="round" />
                {/* Needle */}
                <line
                  x1="70" y1="70"
                  x2={70 + 50 * Math.cos((needleDeg - 90) * Math.PI / 180)}
                  y2={70 + 50 * Math.sin((needleDeg - 90) * Math.PI / 180)}
                  stroke={action > actionThreshold ? '#68d391' : action < -actionThreshold ? '#fc8181' : '#f6e05e'}
                  strokeWidth="3" strokeLinecap="round"
                  style={{ transition: 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)' }}
                />
                <circle cx="70" cy="70" r="5" fill="#fff" />
              </svg>
            </div>
            <div style={{ fontSize: 28, fontWeight: 800, fontFamily: 'monospace',
                          color: action > actionThreshold ? '#68d391' : action < -actionThreshold ? '#fc8181' : '#f6e05e' }}>
              {action.toFixed(4)}
            </div>
            <div style={{ fontSize: 12, color: '#8b9fc0', marginTop: 4 }}>
              {rl.direction || 'FLAT'} · {(posSize * 100).toFixed(2)}% position
            </div>
          </div>
        </div>

        {/* Disagreement Gate */}
        <div className="card">
          <div className="card-header"><span className="card-title">🔒 Disagreement Gate</span></div>
          <div style={{ padding: '12px 0', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontFamily: 'monospace', fontSize: 12, color: '#c3dafe', textAlign: 'center', padding: '6px 8px',
                          background: 'rgba(99,102,241,0.1)', borderRadius: 6 }}>
              eff_action = {rl.rl_action?.toFixed(3) ?? '?'} × {gate.toFixed(3)}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { label: 'Disagreement Score', value: disagr, max: 1, color: '#fc8181' },
                { label: 'Gate Value (1-α·d)',  value: gate,   max: 1, color: '#68d391' },
                { label: 'Position Size',        value: posSize, max: 1, color: '#63b3ed' },
              ].map(({ label, value, max, color }) => (
                <div key={label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#8b9fc0', marginBottom: 3 }}>
                    <span>{label}</span>
                    <span style={{ color, fontFamily: 'monospace' }}>{(value * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.05)' }}>
                    <div style={{ width: `${Math.abs(value / max) * 100}%`, height: '100%',
                                  borderRadius: 3, background: color, transition: 'width 0.5s ease' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Active Regime Policy */}
        <div className="card">
          <div className="card-header"><span className="card-title">🌐 Active Regime</span></div>
          <div style={{ padding: '12px 0', display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
            {['trending', 'mean_reverting', 'high_volatility'].map(r => {
              const active = regime === r;
              const colors = { trending: '#68d391', mean_reverting: '#63b3ed', high_volatility: '#fc8181' };
              return (
                <div key={r} style={{
                  width: '90%', padding: '8px 14px', borderRadius: 8,
                  background: active ? `${colors[r]}22` : 'rgba(255,255,255,0.02)',
                  border: active ? `1.5px solid ${colors[r]}55` : '1px solid transparent',
                  transition: 'all 0.3s ease',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 12, color: active ? colors[r] : '#647091',
                                   fontWeight: active ? 700 : 400, textTransform: 'capitalize' }}>
                      {active && '▶ '}{r.replace('_', ' ')}
                    </span>
                    {active && (
                      <span style={{ fontSize: 11, color: colors[r],
                                     background: `${colors[r]}22`, padding: '1px 8px', borderRadius: 10 }}>
                        {(rl.regime_confidence * 100 || 0).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Reward Curve */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">📈 Reward Curve</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn-backtest" style={{ fontSize: 11, padding: '4px 12px' }}
                    onClick={handleTrain} disabled={loading}>
              {loading ? 'Starting…' : '⚡ Train RL Brain'}
            </button>
            <button className="btn-analyze" style={{ fontSize: 11, padding: '4px 12px' }}
                    onClick={fetchBrain} disabled={loading}>
              {loading ? 'Loading…' : '↻ Refresh'}
            </button>
          </div>
        </div>
        {brain ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={rewardData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" tick={{ fill: '#647091', fontSize: 10 }} />
              <YAxis tick={{ fill: '#647091', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: 'rgba(26,27,58,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
              <Line type="monotone" dataKey="reward" stroke="#68d391" dot={false}
                    strokeWidth={2} name="Reward" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ textAlign: 'center', padding: '32px 0', color: '#647091', fontSize: 13 }}>
            Click Refresh to load reward history
          </div>
        )}
      </div>

      {/* Reward Ablation — KEY SHOWSTOPPER */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">🧪 Reward Ablation Study</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
                   style={{ width: 70, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: 6, padding: '3px 8px', color: '#fff', fontSize: 12 }} />
            <button className="btn-backtest" style={{ fontSize: 11, padding: '4px 14px' }}
                    onClick={fetchAblation} disabled={ablLoading}>
              {ablLoading ? 'Running…' : '▶ Run Ablation'}
            </button>
          </div>
        </div>
        <div style={{ fontSize: 11, color: '#647091', marginBottom: 12 }}>
          Remove each penalty term from the reward function and compare resulting Sharpe ratios.
          A drop in Sharpe confirms each component is essential.
        </div>
        {ablationData.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ablationData} margin={{ top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: '#a0aec0', fontSize: 11 }} />
              <YAxis tick={{ fill: '#647091', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: 'rgba(26,27,58,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                formatter={(v) => [v.toFixed(4), 'Sharpe']}
              />
              <Bar dataKey="sharpe" name="Sharpe"
                   fill="#6366f1"
                   radius={[4,4,0,0]}
                   label={{ position: 'top', fill: '#a0aec0', fontSize: 10,
                            formatter: v => v?.toFixed(3) }} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ textAlign: 'center', padding: '32px 0', color: '#647091', fontSize: 13 }}>
            Run ablation to see how each penalty term affects strategy performance
          </div>
        )}
      </div>

      {/* ═══ Train on Google Colab ═══ */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">🚀 Train on Google Colab (GPU)</span>
        </div>
        <div style={{ fontSize: 12, color: '#a0aec0', marginBottom: 16, lineHeight: 1.6 }}>
          Train the RL model on a free GPU in ~30 min instead of 3-5 hours locally.
          <strong style={{ color: '#f6e05e' }}> 3 easy steps:</strong>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 20 }}>
          {[
            { step: '1', title: 'Open Colab', desc: 'Click the button below to open the training notebook', icon: '📓' },
            { step: '2', title: 'Run All Cells', desc: 'Select Runtime → Run All. Wait ~30 min on T4 GPU', icon: '⚡' },
            { step: '3', title: 'Upload Model', desc: 'Download the .zip file and upload it here', icon: '📤' },
          ].map(s => (
            <div key={s.step} style={{
              background: 'rgba(99,102,241,0.06)', borderRadius: 10, padding: 14,
              border: '1px solid rgba(99,102,241,0.15)', textAlign: 'center'
            }}>
              <div style={{ fontSize: 24, marginBottom: 6 }}>{s.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#c3dafe', marginBottom: 4 }}>Step {s.step}: {s.title}</div>
              <div style={{ fontSize: 11, color: '#8b9fc0', lineHeight: 1.4 }}>{s.desc}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Open Colab button */}
          <a
            href="https://colab.research.google.com/github/Anuja0412Gupta/Quant_Agent/blob/main/colab_train.ipynb"
            target="_blank" rel="noopener noreferrer"
            style={{
              background: 'linear-gradient(135deg, #f9ab00, #ea8600)', color: '#000',
              padding: '8px 20px', borderRadius: 8, fontSize: 13, fontWeight: 700,
              textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 6,
              transition: 'transform 0.2s, box-shadow 0.2s',
            }}
            onMouseEnter={e => { e.target.style.transform = 'scale(1.03)'; e.target.style.boxShadow = '0 4px 20px rgba(249,171,0,0.3)'; }}
            onMouseLeave={e => { e.target.style.transform = 'scale(1)'; e.target.style.boxShadow = 'none'; }}
          >
            <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="18" height="18" alt="" style={{ borderRadius: 3 }} />
            Open Colab Notebook
          </a>

          {/* Divider */}
          <div style={{ width: 1, height: 32, background: 'rgba(255,255,255,0.1)' }} />

          {/* File picker */}
          <label style={{
            cursor: 'pointer', background: 'rgba(255,255,255,0.05)', border: '1px dashed rgba(255,255,255,0.2)',
            borderRadius: 8, padding: '8px 16px', fontSize: 12, color: '#a0aec0',
            display: 'inline-flex', alignItems: 'center', gap: 6,
            transition: 'border-color 0.2s',
          }}>
            📁 {uploadFile ? uploadFile.name : 'Choose .zip model file'}
            <input type="file" accept=".zip" style={{ display: 'none' }}
              onChange={e => { setUploadFile(e.target.files[0] || null); setUploadStatus(null); }} />
          </label>

          {/* Upload button */}
          <button
            onClick={handleUploadModel}
            disabled={!uploadFile || uploadStatus === 'uploading'}
            style={{
              background: uploadFile ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : 'rgba(255,255,255,0.05)',
              color: uploadFile ? '#fff' : '#647091', border: 'none', borderRadius: 8,
              padding: '8px 18px', fontSize: 12, fontWeight: 600, cursor: uploadFile ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s',
            }}
          >
            {uploadStatus === 'uploading' ? '⏳ Uploading…' : '📤 Upload Model'}
          </button>
        </div>

        {/* Upload feedback — 4 states: uploading / success / saved / error */}
        {uploadStatus && (
          <div style={{
            marginTop: 12, padding: '10px 14px', borderRadius: 8, fontSize: 12,
            background:
              uploadStatus === 'success'  ? 'rgba(104,211,145,0.1)' :
              uploadStatus === 'saved'    ? 'rgba(246,224,94,0.08)' :
              uploadStatus === 'error'    ? 'rgba(252,129,129,0.1)' :
                                           'rgba(99,102,241,0.1)',
            border: `1px solid ${
              uploadStatus === 'success'  ? 'rgba(104,211,145,0.3)' :
              uploadStatus === 'saved'    ? 'rgba(246,224,94,0.3)'  :
              uploadStatus === 'error'    ? 'rgba(252,129,129,0.3)' :
                                           'rgba(99,102,241,0.3)'}`,
            color:
              uploadStatus === 'success'  ? '#68d391' :
              uploadStatus === 'saved'    ? '#f6e05e' :
              uploadStatus === 'error'    ? '#fc8181' :
                                           '#818cf8',
          }}>
            <div style={{ fontWeight: 600, marginBottom: uploadStatus === 'saved' ? 4 : 0 }}>
              {uploadStatus === 'success'  ? '✅' :
               uploadStatus === 'saved'    ? '⚠️' :
               uploadStatus === 'error'    ? '❌' : '⏳'} {uploadMsg}
            </div>
            {uploadStatus === 'saved' && (
              <div style={{ fontSize: 11, opacity: 0.8, marginTop: 4 }}>
                📌 Model is saved to disk. <strong>Restart your backend</strong> (Ctrl+C → python main.py) to activate it for backtest & analysis.
              </div>
            )}
          </div>
        )}

        {/* Current model info */}
        {modelInfo?.exists && (
          <div style={{ marginTop: 12, padding: '8px 14px', borderRadius: 8, background: 'rgba(255,255,255,0.03)',
                        fontSize: 11, color: '#8b9fc0', display: 'flex', gap: 16 }}>
            <span>📦 Current Model: <strong style={{ color: '#c3dafe' }}>{modelInfo.symbol}</strong></span>
            <span>Size: <strong>{modelInfo.size_kb} KB</strong></span>
            <span>Updated: <strong>{new Date(modelInfo.modified_at).toLocaleString()}</strong></span>
          </div>
        )}
      </div>
    </div>
  );
}

