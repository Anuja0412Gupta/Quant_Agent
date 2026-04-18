/**
 * Zustand Global State Store v3.0
 * ==================================
 * Extended with sentiment, regime probs, macro context,
 * SEC flags, lagrangian multipliers, and DCC-GARCH allocation.
 */
import { create } from 'zustand';

const useStore = create((set) => ({
  // ── Shared inputs ──────────────────────────────────────────────────────────
  ticker:    'AAPL',
  timeframe: '1d',
  setTicker:    (ticker)    => set({ ticker: ticker.toUpperCase() }),
  setTimeframe: (timeframe) => set({ timeframe }),

  // ── Loading states ─────────────────────────────────────────────────────────
  loadingAnalysis:  false,
  loadingBacktest:  false,
  loadingRLBrain:   false,
  loadingCompare:   false,
  loadingStress:    false,
  setLoading: (key, val) => set({ [key]: val }),

  // ── Error ──────────────────────────────────────────────────────────────────
  error: null,
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),

  // ── Analyze Tab ──────────────────────────────────────────────────────────
  analysis:  null,
  shap:      null,
  setAnalysis: (analysis) => set({ analysis }),
  setShap:     (shap)     => set({ shap }),

  // ── v3.0: Sentiment ──────────────────────────────────────────────────────
  sentiment:   null,
  secFlags:    null,
  macroCtx:    null,
  setSentiment:  (s) => set({ sentiment: s }),
  setSecFlags:   (f) => set({ secFlags: f }),
  setMacroCtx:   (m) => set({ macroCtx: m }),

  // ── v3.0: Regime ─────────────────────────────────────────────────────────
  regimeResult: null,
  setRegimeResult: (r) => set({ regimeResult: r }),

  // ── v3.0: Disagreement ───────────────────────────────────────────────────
  disagreement: null,
  setDisagreement: (d) => set({ disagreement: d }),

  // ── Backtest Tab ─────────────────────────────────────────────────────────
  backtest:         null,
  stressTest:       null,
  walkForwardFolds: [],
  setBacktest:         (b) => set({ backtest: b }),
  setStressTest:       (s) => set({ stressTest: s }),
  setWalkForwardFolds: (f) => set({ walkForwardFolds: f }),

  // ── RL Brain Tab ─────────────────────────────────────────────────────────
  rlBrain:    null,
  ablation:   null,
  lagrangian: null,
  setRLBrain:    (r) => set({ rlBrain: r }),
  setAblation:   (a) => set({ ablation: a }),
  setLagrangian: (l) => set({ lagrangian: l }),

  // ── Compare Tab ──────────────────────────────────────────────────────────
  compareStocks:     'AAPL,MSFT,NVDA',
  compareData:       null,
  portfolio:         null,
  allocationWeights: null,
  setCompareStocks:     (s) => set({ compareStocks: s }),
  setCompareData:       (d) => set({ compareData: d }),
  setPortfolio:         (p) => set({ portfolio: p }),
  setAllocationWeights: (w) => set({ allocationWeights: w }),

  // ── Live Demo Tab ────────────────────────────────────────────────────────
  liveStep:    0,
  liveRunning: false,
  liveSteps:   [],
  setLiveStep:    (n) => set({ liveStep: n }),
  setLiveRunning: (b) => set({ liveRunning: b }),
  setLiveSteps:   (s) => set({ liveSteps: s }),
}));

export default useStore;
