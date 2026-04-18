import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * TickerSearch — Smart autocomplete input
 * ========================================
 * - Searches by ticker OR company name (e.g. "Apple" → AAPL)
 * - Keyboard navigation (↑↓ Enter Escape)
 * - Validates non-empty before triggering analysis
 * - Debounced API search
 */
export default function TickerSearch({ value, onChange, onSubmit, loading, placeholder = 'Search stock…' }) {
  const [query,       setQuery]       = useState(value || '');
  const [suggestions, setSuggestions] = useState([]);
  const [open,        setOpen]        = useState(false);
  const [cursor,      setCursor]      = useState(-1);
  const [error,       setError]       = useState('');
  const debounceRef = useRef(null);
  const wrapperRef  = useRef(null);

  // Sync external value changes
  useEffect(() => { setQuery(value || ''); }, [value]);

  // Close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const search = useCallback(async (q) => {
    if (!q || q.length < 1) { setSuggestions([]); setOpen(false); return; }
    try {
      const { data } = await axios.get(`${API}/search`, { params: { q }, timeout: 5000 });
      setSuggestions(data.results || []);
      setOpen((data.results || []).length > 0);
      setCursor(-1);
    } catch { setSuggestions([]); setOpen(false); }
  }, []);

  const handleChange = (e) => {
    const q = e.target.value;
    setQuery(q);
    setError('');
    onChange?.(q);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => search(q), 220);
  };

  const select = (ticker, name) => {
    setQuery(ticker);
    onChange?.(ticker);
    setSuggestions([]);
    setOpen(false);
    setCursor(-1);
  };

  const handleKey = (e) => {
    if (!open) {
      if (e.key === 'Enter') handleSubmit();
      return;
    }
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, suggestions.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor(c => Math.max(c - 1, -1)); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      if (cursor >= 0 && suggestions[cursor]) {
        select(suggestions[cursor].ticker, suggestions[cursor].name);
      } else {
        setOpen(false);
        handleSubmit();
      }
    }
    else if (e.key === 'Escape') { setOpen(false); setCursor(-1); }
  };

  const handleSubmit = () => {
    const trimmed = query.trim();
    if (!trimmed) { setError('Please enter a stock ticker or company name'); return; }
    setError('');
    setOpen(false);
    onSubmit?.(trimmed.toUpperCase());
  };

  return (
    <div ref={wrapperRef} style={{ position: 'relative', display: 'flex', gap: 8, alignItems: 'flex-start', flexDirection: 'column' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {/* Input */}
        <div style={{ position: 'relative' }}>
          <input
            id="ticker-input"
            value={query}
            onChange={handleChange}
            onKeyDown={handleKey}
            onFocus={() => { if (suggestions.length > 0) setOpen(true); }}
            placeholder={placeholder}
            autoComplete="off"
            spellCheck={false}
            style={{
              width: 220,
              background: 'rgba(255,255,255,0.06)',
              border: `1px solid ${error ? '#fc8181' : open ? 'rgba(99,102,241,0.6)' : 'rgba(99,120,255,0.15)'}`,
              borderRadius: 10,
              color: '#f0f4ff',
              fontFamily: "'Inter', sans-serif",
              fontSize: 14,
              fontWeight: 500,
              padding: '8px 14px 8px 36px',
              outline: 'none',
              transition: 'all 0.2s',
              boxShadow: open ? '0 0 0 3px rgba(99,102,241,0.15)' : 'none',
            }}
          />
          {/* Search icon */}
          <span style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                          fontSize: 14, color: '#647091', pointerEvents: 'none' }}>🔍</span>
          {/* Clear button */}
          {query && (
            <button onClick={() => { setQuery(''); onChange?.(''); setSuggestions([]); setOpen(false); setError(''); }}
                    style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                             background: 'none', border: 'none', color: '#647091', cursor: 'pointer',
                             fontSize: 14, padding: 2, lineHeight: 1 }}>×</button>
          )}
        </div>

        {/* Timeframe slot is handled outside */}

        {/* Submit button */}
        <button
          id="analyze-btn"
          onClick={handleSubmit}
          disabled={loading}
          style={{
            background: 'linear-gradient(135deg, #4f7dff, #9b59ff)',
            border: 'none', borderRadius: 10, color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontFamily: "'Inter', sans-serif", fontSize: 13, fontWeight: 600,
            padding: '8px 20px', opacity: loading ? 0.5 : 1,
            transition: 'all 0.2s', whiteSpace: 'nowrap',
          }}
        >
          {loading ? '⏳ Analyzing…' : '▶ Analyze'}
        </button>
      </div>

      {/* Validation error */}
      {error && (
        <div style={{ fontSize: 11, color: '#fc8181', marginTop: -4, paddingLeft: 4,
                      display: 'flex', alignItems: 'center', gap: 4 }}>
          ⚠️ {error}
        </div>
      )}

      {/* Dropdown */}
      {open && suggestions.length > 0 && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, zIndex: 999, marginTop: 4,
          width: 300, background: '#141928',
          border: '1px solid rgba(99,102,241,0.3)', borderRadius: 12,
          boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
          overflow: 'hidden',
        }}>
          {suggestions.map((s, i) => (
            <div key={s.ticker}
                 onMouseDown={() => select(s.ticker, s.name)}
                 onMouseEnter={() => setCursor(i)}
                 style={{
                   display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                   padding: '10px 14px', cursor: 'pointer',
                   background: cursor === i ? 'rgba(99,102,241,0.15)' : 'transparent',
                   borderBottom: i < suggestions.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                   transition: 'background 0.15s',
                 }}>
              <span style={{ fontWeight: 700, fontFamily: 'monospace', fontSize: 13, color: '#c3dafe' }}>
                {s.ticker}
              </span>
              <span style={{ fontSize: 12, color: '#8b9fc0', textAlign: 'right', maxWidth: 180,
                             overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.name}
              </span>
            </div>
          ))}
          <div style={{ padding: '6px 14px', fontSize: 10, color: '#3a4466',
                        borderTop: '1px solid rgba(255,255,255,0.04)' }}>
            ↑↓ navigate · Enter select · Esc close
          </div>
        </div>
      )}
    </div>
  );
}
