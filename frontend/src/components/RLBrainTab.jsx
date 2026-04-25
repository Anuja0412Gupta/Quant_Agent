import { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const REGIME_CFG = {
  trending:        { color: '#00e5a0', icon: '📈', label: 'Trending'        },
  mean_reverting:  { color: '#60a5fa', icon: '↔️', label: 'Mean-Reverting'  },
  high_volatility: { color: '#ff4f72', icon: '⚡', label: 'High Volatility' },
};

export default function RLBrainTab({ analysis, onRefreshAnalysis }) {
  const [brain, setBrain]           = useState(null);
  const [ablation, setAblation]     = useState(null);
  const [loading, setLoading]       = useState(false);
  const [ablLoading, setAblLoading] = useState(false);
  const [ticker, setTicker]         = useState(analysis?.symbol || 'AAPL');
  const [progress, setProgress]     = useState(0);
  const [uploadFile, setUploadFile]     = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadMsg, setUploadMsg]       = useState('');
  const [modelInfo, setModelInfo]       = useState(null);

  useEffect(() => {
    if (analysis?.symbol) setTicker(analysis.symbol);
  }, [analysis?.symbol]);

  useEffect(() => {
    axios.get(`${API}/rl/model-info`, { params: { symbol: ticker } })
      .then(({ data }) => setModelInfo(data)).catch(() => {});
  }, [ticker]);

  useEffect(() => {
    let id;
    if (brain?.training_active) {
      id = setInterval(() => {
        setProgress(p => Math.min(p + (Math.random() * 5 + 1), 95));
        axios.get(`${API}/rl/brain`, { params: { symbol: ticker } })
          .then(({ data }) => { setBrain(data); if (!data.training_active) setProgress(100); })
          .catch(() => {});
      }, 3000);
    } else if (progress >= 95) setProgress(100);
    return () => clearInterval(id);
  }, [brain?.training_active, ticker]);

  const fetchBrain = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API}/rl/brain`, { timeout: 30_000 });
      setBrain(data);
    } catch (e) { console.error('RL Brain fetch failed:', e.message); }
    finally { setLoading(false); }
  }, []);

  const fetchAblation = useCallback(async () => {
    setAblLoading(true);
    try {
      const { data } = await axios.post(`${API}/rl/ablation`, { symbol: ticker, timeframe: '1d', period: '2y' }, { timeout: 120_000 });
      setAblation(data);
    } catch (e) { console.error('Ablation failed:', e.message); }
    finally { setAblLoading(false); }
  }, [ticker]);

  const handleTrain = async () => {
    setLoading(true);
    try {
      await axios.post(`${API}/rl/train?symbol=${analysis?.symbol || ticker}&timeframe=1d`);
      setBrain(prev => ({ ...(prev || {}), training_active: true }));
      setProgress(0);
    } catch (e) { console.error('Training failed:', e.message); }
    finally { setLoading(false); }
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploadStatus('uploading'); setUploadMsg('');
    try {
      const form = new FormData();
      form.append('file', uploadFile);
      const { data } = await axios.post(`${API}/rl/upload-model?symbol=${ticker}&timeframe=1d`, form,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 180_000 });
      setUploadStatus(data.model_ok ? 'success' : 'saved');
      setUploadMsg(data.message || 'Model uploaded');
      setModelInfo({ exists: true, symbol: ticker, size_kb: data.size_kb, modified_at: new Date().toISOString() });
      setUploadFile(null);
      if (onRefreshAnalysis) onRefreshAnalysis();
    } catch (e) {
      setUploadStatus('error');
      setUploadMsg(e.response?.data?.detail || e.message || 'Upload failed');
    }
  };

  const rl      = analysis?.rl_weights || {};
  const action  = rl.rl_action         ?? 0;
  const posSize = rl.effective_position ?? 0;
  const disagr  = rl.disagreement_score ?? 0;
  const gate    = rl.gate_value         ?? 1;
  const regime  = rl.active_regime      ?? 'trending';
  const regCfg  = REGIME_CFG[regime]    || { color: '#94a3b8', icon: '?', label: regime };

  const dirColor = rl.direction === 'BUY'  ? '#00e5a0'
                 : rl.direction === 'SELL' ? '#ff4f72'
                 : '#ffb830';

  const needleDeg  = action * 90;
  const rewardData = (brain?.reward_history  || []).map((v, i) => ({ step: i, reward: v }));
  const ablData    = ablation
    ? Object.entries(ablation.variants || {}).map(([name, v]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        sharpe: v.sharpe,
      }))
    : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Training progress */}
      {brain?.training_active && (
        <div style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 14, padding: '20px 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ color: '#a5b4fc', fontSize: 14, fontWeight: 700 }}>⚡ Reinforcement Learning Training in Progress</span>
            <span style={{ color: '#818cf8', fontSize: 14, fontFamily: 'var(--mono)', fontWeight: 800 }}>{progress.toFixed(0)}%</span>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.05)', height: 8, borderRadius: 4, overflow: 'hidden', marginBottom: 10 }}>
            <div style={{ background: 'linear-gradient(90deg, #6366f1, #a78bfa)', height: '100%', width: `${progress}%`, transition: 'width 0.5s ease', borderRadius: 4 }} />
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            Executing multi-phase curriculum (Trending → Mean-Reverting → High-Vol). Lagrangian constraints actively tightening.
          </div>
        </div>
      )}

      {/* ── Row 1: Action Gauge | Gate | Regime ─────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20 }}>

        {/* Action Gauge */}
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>RL ACTION OUTPUT</div>
          <div style={{ position: 'relative', width: 150, height: 90, margin: '0 auto 12px' }}>
            <svg viewBox="0 0 150 90" width="150" height="90">
              <path d="M 15 75 A 60 60 0 0 1 135 75" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="14" strokeLinecap="round" />
              <path d="M 15 75 A 60 60 0 0 1 75 15" fill="none" stroke="rgba(255,79,114,0.25)" strokeWidth="14" strokeLinecap="round" />
              <path d="M 75 15 A 60 60 0 0 1 135 75" fill="none" stroke="rgba(0,229,160,0.25)" strokeWidth="14" strokeLinecap="round" />
              <line
                x1="75" y1="75"
                x2={75 + 52 * Math.cos((needleDeg - 90) * Math.PI / 180)}
                y2={75 + 52 * Math.sin((needleDeg - 90) * Math.PI / 180)}
                stroke={dirColor}
                strokeWidth="3.5" strokeLinecap="round"
                style={{ transition: 'all 0.6s cubic-bezier(0.34,1.56,0.64,1)', filter: `drop-shadow(0 0 4px ${dirColor})` }}
              />
              <circle cx="75" cy="75" r="6" fill={dirColor} style={{ filter: `drop-shadow(0 0 6px ${dirColor})` }} />
            </svg>
          </div>
          <div style={{ fontSize: 32, fontWeight: 900, fontFamily: 'var(--mono)', color: dirColor, letterSpacing: -1, filter: `drop-shadow(0 0 8px ${dirColor})` }}>
            {action.toFixed(4)}
          </div>
          <div style={{ fontSize: 13, color: '#64748b', marginTop: 6 }}>Raw RL Action</div>
          <div style={{ marginTop: 12, padding: '8px 16px', background: `${dirColor}18`, border: `1px solid ${dirColor}40`, borderRadius: 20, display: 'inline-block' }}>
            <span style={{ color: dirColor, fontWeight: 800, fontSize: 14 }}>{rl.direction || 'FLAT'}</span>
            <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>{(posSize * 100).toFixed(2)}% position</span>
          </div>
        </div>

        {/* Disagreement Gate */}
        <div className="card">
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>DISAGREEMENT GATE</div>
          <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '12px 14px', marginBottom: 16, textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontFamily: 'var(--mono)' }}>FORMULA</div>
            <div style={{ fontSize: 14, color: '#a5b4fc', fontFamily: 'var(--mono)', fontWeight: 700 }}>
              eff = {rl.rl_action?.toFixed(3) ?? '?'} × {gate.toFixed(3)}
            </div>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>rl_action × gate_value</div>
          </div>
          {[
            { label: 'Disagreement Score', value: disagr,  color: '#ff4f72', desc: 'Higher = agents disagree more' },
            { label: 'Gate Value',         value: gate,    color: '#00e5a0', desc: 'Multiplier applied to action' },
            { label: 'Final Position %',   value: posSize, color: '#60a5fa', desc: 'Of total portfolio'           },
          ].map(({ label, value, color, desc }) => (
            <div key={label} style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                <div>
                  <span style={{ color: '#94a3b8', fontWeight: 600 }}>{label}</span>
                  <div style={{ fontSize: 10, color: '#475569' }}>{desc}</div>
                </div>
                <span style={{ color, fontFamily: 'var(--mono)', fontWeight: 800, fontSize: 15 }}>
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(100, Math.abs(value) * 100)}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.6s ease' }} />
              </div>
            </div>
          ))}
        </div>

        {/* Active Regime */}
        <div className="card">
          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 16 }}>ACTIVE REGIME POLICY</div>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>CURRENT POLICY</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '14px 16px', background: `${regCfg.color}14`, border: `1px solid ${regCfg.color}40`, borderRadius: 12 }}>
              <span style={{ fontSize: 24 }}>{regCfg.icon}</span>
              <div>
                <div style={{ fontSize: 18, fontWeight: 800, color: regCfg.color }}>{regCfg.label.toUpperCase()}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>RL sub-policy active</div>
              </div>
              <span style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontWeight: 800, fontSize: 18, color: regCfg.color }}>
                {((rl.regime_confidence || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(REGIME_CFG).map(([key, cfg]) => {
              const active = regime === key;
              return (
                <div key={key} style={{
                  padding: '10px 14px', borderRadius: 10,
                  background: active ? `${cfg.color}10` : 'rgba(255,255,255,0.02)',
                  border: active ? `1px solid ${cfg.color}40` : '1px solid rgba(255,255,255,0.04)',
                  display: 'flex', alignItems: 'center', gap: 10,
                }}>
                  <span style={{ fontSize: 16 }}>{cfg.icon}</span>
                  <span style={{ fontSize: 13, color: active ? cfg.color : '#475569', fontWeight: active ? 700 : 400 }}>
                    {cfg.label}
                  </span>
                  {active && (
                    <span style={{ marginLeft: 'auto', fontSize: 11, background: `${cfg.color}22`, color: cfg.color, padding: '2px 8px', borderRadius: 20, fontWeight: 700 }}>
                      ACTIVE
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Reward Curve ────────────────────────────────────────── */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>📈 TRAINING REWARD CURVE</div>
            <div style={{ fontSize: 13, color: '#94a3b8' }}>Cumulative reward per training step — should trend upward over time</div>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button onClick={fetchBrain} disabled={loading} style={{
              padding: '8px 18px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(255,255,255,0.05)', color: '#94a3b8', fontSize: 13, cursor: 'pointer',
              fontWeight: 600,
            }}>
              {loading ? '⏳ Loading…' : '↻ Refresh'}
            </button>
            <button onClick={handleTrain} disabled={loading} style={{
              padding: '8px 18px', borderRadius: 8, border: 'none',
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff', fontSize: 13,
              cursor: 'pointer', fontWeight: 700,
            }}>
              {loading ? '⏳ Starting…' : '⚡ Train RL Brain'}
            </button>
          </div>
        </div>
        {rewardData.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rewardData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 13 }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="reward" stroke="#00e5a0" dot={false} strokeWidth={2.5} name="Reward" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ textAlign: 'center', padding: '48px 0', color: '#475569' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>📊</div>
            <div style={{ fontSize: 14 }}>Click Refresh to load reward history</div>
            <div style={{ fontSize: 12, marginTop: 6, color: '#374151' }}>No training data loaded yet</div>
          </div>
        )}
      </div>

      {/* ── Feature Ablation ─────────────────────────────────────── */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>🧪 REWARD ABLATION STUDY</div>
            <div style={{ fontSize: 13, color: '#94a3b8', maxWidth: 480 }}>
              Each bar removes one penalty term from the reward function. A drop in Sharpe confirms that component is essential to profitability.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0 }}>
            <input
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              style={{
                width: 80, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8, padding: '8px 12px', color: '#fff', fontSize: 13,
              }}
            />
            <button onClick={fetchAblation} disabled={ablLoading} style={{
              padding: '8px 18px', borderRadius: 8, border: 'none',
              background: ablLoading ? 'rgba(255,255,255,0.05)' : 'linear-gradient(135deg, #f59e0b, #d97706)',
              color: ablLoading ? '#64748b' : '#000', fontSize: 13, cursor: 'pointer', fontWeight: 700,
            }}>
              {ablLoading ? '⏳ Running…' : '▶ Run Ablation'}
            </button>
          </div>
        </div>
        {ablData.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={ablData} margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10 }}
                formatter={v => [v?.toFixed(4), 'Sharpe Ratio']}
              />
              <Bar dataKey="sharpe" name="Sharpe" fill="#6366f1" radius={[6, 6, 0, 0]}
                label={{ position: 'top', fill: '#94a3b8', fontSize: 11, formatter: v => v?.toFixed(3) }} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ textAlign: 'center', padding: '48px 0', color: '#475569' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>🧪</div>
            <div style={{ fontSize: 14 }}>Run ablation to analyze reward component importance</div>
            <div style={{ fontSize: 12, marginTop: 6, color: '#374151' }}>Estimated runtime: ~2–5 minutes</div>
          </div>
        )}
      </div>

      {/* ── Colab Training ───────────────────────────────────────── */}
      <div className="card">
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 1.5, marginBottom: 20 }}>🚀 TRAIN ON GOOGLE COLAB (FREE GPU)</div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 24 }}>
          {[
            { step: '1', title: 'Open Notebook', desc: 'Click the Colab button below to open the GPU training notebook.', icon: '📓', color: '#f59e0b' },
            { step: '2', title: 'Run All Cells',  desc: 'Select Runtime → Run All. Completes in ~30 min on a T4 GPU.',      icon: '⚡', color: '#6366f1' },
            { step: '3', title: 'Upload Model',  desc: 'Download the .zip output and upload it in the panel below.',         icon: '📤', color: '#00e5a0' },
          ].map(s => (
            <div key={s.step} style={{ background: `${s.color}08`, borderRadius: 12, padding: '18px 16px', border: `1px solid ${s.color}25`, textAlign: 'center' }}>
              <div style={{ fontSize: 28, marginBottom: 10 }}>{s.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: s.color, marginBottom: 6 }}>Step {s.step}: {s.title}</div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>{s.desc}</div>
            </div>
          ))}
        </div>

        {/* Actions row */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 16 }}>
          <a
            href="https://colab.research.google.com/github/Anuja0412Gupta/Quant_Agent/blob/main/colab_train.ipynb"
            target="_blank" rel="noopener noreferrer"
            style={{
              background: 'linear-gradient(135deg, #f9ab00, #ea8600)', color: '#000',
              padding: '10px 22px', borderRadius: 10, fontSize: 13, fontWeight: 800,
              textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 8,
            }}
          >
            <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="18" height="18" alt="" style={{ borderRadius: 3 }} />
            Open Colab Notebook
          </a>

          <div style={{ width: 1, height: 32, background: 'rgba(255,255,255,0.08)' }} />

          <label style={{
            cursor: 'pointer', background: 'rgba(255,255,255,0.04)', border: '1px dashed rgba(255,255,255,0.15)',
            borderRadius: 10, padding: '10px 18px', fontSize: 13, color: uploadFile ? '#f1f5f9' : '#64748b',
            display: 'inline-flex', alignItems: 'center', gap: 8,
          }}>
            📁 {uploadFile ? uploadFile.name : 'Choose .zip model file'}
            <input type="file" accept=".zip" style={{ display: 'none' }}
              onChange={e => { setUploadFile(e.target.files[0] || null); setUploadStatus(null); }} />
          </label>

          <button
            onClick={handleUpload}
            disabled={!uploadFile || uploadStatus === 'uploading'}
            style={{
              background: uploadFile ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : 'rgba(255,255,255,0.04)',
              color: uploadFile ? '#fff' : '#475569', border: 'none', borderRadius: 10,
              padding: '10px 20px', fontSize: 13, fontWeight: 700,
              cursor: uploadFile ? 'pointer' : 'not-allowed',
            }}
          >
            {uploadStatus === 'uploading' ? '⏳ Uploading…' : '📤 Upload Model'}
          </button>
        </div>

        {/* Upload status */}
        {uploadStatus && (
          <div style={{
            padding: '14px 18px', borderRadius: 10, fontSize: 13,
            background:
              uploadStatus === 'success' ? 'rgba(0,229,160,0.08)' :
              uploadStatus === 'saved'   ? 'rgba(255,184,48,0.08)' :
              uploadStatus === 'error'   ? 'rgba(255,79,114,0.08)' :
                                          'rgba(99,102,241,0.08)',
            border: `1px solid ${
              uploadStatus === 'success' ? 'rgba(0,229,160,0.3)' :
              uploadStatus === 'saved'   ? 'rgba(255,184,48,0.3)'  :
              uploadStatus === 'error'   ? 'rgba(255,79,114,0.3)'  :
                                          'rgba(99,102,241,0.3)'}`,
            color:
              uploadStatus === 'success' ? '#00e5a0' :
              uploadStatus === 'saved'   ? '#ffb830' :
              uploadStatus === 'error'   ? '#ff4f72' :
                                          '#818cf8',
          }}>
            <div style={{ fontWeight: 700, marginBottom: uploadStatus === 'saved' ? 6 : 0 }}>
              {uploadStatus === 'success' ? '✅' : uploadStatus === 'saved' ? '⚠️' : uploadStatus === 'error' ? '❌' : '⏳'} {uploadMsg}
            </div>
            {uploadStatus === 'saved' && (
              <div style={{ fontSize: 12, opacity: 0.8 }}>
                📌 Model saved to disk. <strong>Restart backend</strong> (Ctrl+C → python main.py) to activate it.
              </div>
            )}
          </div>
        )}

        {/* Model info */}
        {modelInfo?.exists && (
          <div style={{ marginTop: 14, padding: '12px 16px', background: 'rgba(255,255,255,0.02)', borderRadius: 10, fontSize: 12, color: '#64748b', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            <span>📦 <strong style={{ color: '#94a3b8' }}>Loaded Model:</strong> {modelInfo.symbol}</span>
            <span>📏 <strong>Size:</strong> {modelInfo.size_kb} KB</span>
            <span>🕒 <strong>Updated:</strong> {new Date(modelInfo.modified_at).toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}
