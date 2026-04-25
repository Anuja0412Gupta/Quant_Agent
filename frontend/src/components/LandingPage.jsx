import React, { useState } from 'react';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function LandingPage({ onAuthSuccess }) {
  const [loginInput, setLoginInput] = useState('');
  const [passwordInput, setPasswordInput] = useState('');
  const [authMode, setAuthMode] = useState('login');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const handleAuth = async (e) => {
    e.preventDefault();
    if (!loginInput.trim() || !passwordInput.trim()) return;
    setAuthLoading(true);
    setAuthError('');
    try {
      if (authMode === 'signup') {
        await axios.post(`${API}/auth/signup`, { username: loginInput.trim(), password: passwordInput });
        onAuthSuccess(loginInput.trim());
      } else {
        await axios.post(`${API}/auth/login`, { username: loginInput.trim(), password: passwordInput });
        onAuthSuccess(loginInput.trim());
      }
    } catch (err) {
      setAuthError(err.response?.data?.detail || err.message || 'Authentication failed');
    } finally {
      setAuthLoading(false);
    }
  };

  return (
    <div className="landing-container">
      {/* Animated Background Elements */}
      <div className="landing-bg-elements">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <div className="landing-content">
        <div className="landing-hero">
          <div className="hero-logo-box">
             <span className="hero-logo-icon">⚡</span>
             <h1 className="hero-title">Quant<span className="hero-gradient">Agent</span></h1>
          </div>
          <p className="hero-subtitle">
            Institutional-Grade Reinforcement Learning Framework for Autonomous Trading
          </p>
          <div className="hero-features">
            <div className="feature-pill">🧠 45-dim Feature Vector Space</div>
            <div className="feature-pill">📈 Hidden Markov Regime Detection</div>
            <div className="feature-pill">📉 TCN-LSTM PPO Sub-Agents</div>
          </div>
        </div>

        <div className="auth-card glass-panel">
          <div className="auth-tabs">
            <button 
              type="button" 
              className={`auth-tab ${authMode === 'login' ? 'active' : ''}`}
              onClick={() => { setAuthMode('login'); setAuthError(''); }}
            >
              Sign In
            </button>
            <button 
              type="button" 
              className={`auth-tab ${authMode === 'signup' ? 'active' : ''}`}
              onClick={() => { setAuthMode('signup'); setAuthError(''); }}
            >
              Create Account
            </button>
          </div>

          <form onSubmit={handleAuth} className="auth-form">
            <div className="input-group">
              <label>Username</label>
              <input 
                type="text"
                className="landing-input"
                value={loginInput} 
                onChange={e => setLoginInput(e.target.value)} 
                placeholder="Enter your username" 
                required 
              />
            </div>
            
            <div className="input-group">
              <label>Password</label>
              <input 
                type="password"
                className="landing-input"
                value={passwordInput} 
                onChange={e => setPasswordInput(e.target.value)} 
                placeholder="Enter secure password" 
                required minLength={4}
              />
            </div>

            {authError && <div className="auth-error">{authError}</div>}
            
            <button type="submit" className="landing-btn-submit" disabled={authLoading}>
              {authLoading ? <div className="spinner-small" /> : (authMode === 'login' ? 'Secure Login  →' : 'Initialize Agent  →')}
            </button>
          </form>
          
          <div className="auth-footer">
            End-to-End Encrypted. Portfolio Isolated.
          </div>
        </div>
      </div>
    </div>
  );
}
