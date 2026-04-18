# QuantAgent — Look-Ahead Leakage Audit

Conducted on: 2026-04-16
Scope: entire `backend/` directory

---

## Findings

### VIOLATION 1 — `feature_engineering.py` line 58-59: Partial rolling window (MEDIUM)
**Location**: `agents/feature_engineering.py`, `_rolling_zscore()`
**Code**: `series.rolling(window, min_periods=max(2, window // 2)).mean()`
**Problem**: `min_periods = window // 2` allows z-scores to be computed with as few as
30 bars for a 60-bar window. This produces biased z-scores for the first 59 bars
(mean/std estimated from fewer observations than the window size implies).
**Status**: ✅ FIXED — `feature_pipeline.py` uses strict `min_periods=window`,
returning NaN for bars before burn-in. Training enforces FEATURE_BURNIN_BARS=252.

---

### VIOLATION 2 — `main.py` line 362: `min_periods=5` for vol (MEDIUM)
**Location**: `backend/main.py`, `/rl/ablation` endpoint
**Code**: `vol = returns.rolling(20, min_periods=5).std().fillna(0.02)`
**Problem**: Volatility estimated from as few as 5 bars. Although this is just for the
ablation endpoint (not live trading), it produces misleading vol estimates.
**Status**: ✅ FIXED — ablation endpoint now drops the first 20 bars before computing vol.

---

### VIOLATION 3 — `agents/feature_engineering.py` line 131: `min_periods=10` (LOW)
**Location**: `agents/feature_engineering.py`, `build_state_vector()`
**Code**: `rets.rolling(20, min_periods=10).std().iloc[-1]`
**Problem**: Same partial-window issue — vol computed from 10 bars instead of 20.
**Status**: ✅ FIXED — entire feature pipeline replaced in `features/feature_pipeline.py`.

---

### VIOLATION 4 — Backtesting: no explicit burn-in enforcement (HIGH)
**Location**: `backtesting/backtesting_engine.py`, `_simulate_fold()`
**Code**: `for i in range(_MIN_WARMUP, len(df)):` where `_MIN_WARMUP = 60`
**Problem**: 60-bar burn-in is insufficient for 60-bar rolling z-scores. First valid
z-score requires exactly 60 previous bars, so training should start at bar 252.
**Status**: ✅ FIXED — `FEATURE_BURNIN_BARS = 252` enforced in `config.py`.
`WalkForwardEngine` skips first 252 bars. Assert added to `RLTrainer.train()`.

---

### VIOLATION 5 — `market_regime_agent.py`: Hurst exponent on full window (LOW)
**Location**: `agents/market_regime_agent.py`, `hurst_exponent()`
**Code**: Called with `log_prices = np.log(close.values.astype(float))` — full history
**Problem**: Not technically look-ahead (only uses past data), but Hurst is removed
entirely and replaced by StudentTHMM.
**Status**: ✅ RESOLVED — HMM replaces Hurst exponent entirely.

---

### VIOLATION 6 — Walk-forward: train window includes test data in fold (HIGH)
**Location**: `backtesting_engine.py`, `run_walk_forward()`
**Code**: `_simulate_fold(df.iloc[:start + test_bars], ...)` — passes train+test data
to `_simulate_fold()` which then runs agents on every bar. Agents at bar `start+1`
(inside test window) can see patterns from bars they shouldn't have seen during training.
**Problem**: The existing walk-forward is actually a single-pass simulation on the full
window, not a true train/test split. There's no separate training phase per fold.
**Status**: ✅ FIXED — New `WalkForwardEngine` in `backtest/engine.py` properly separates
train and test windows. HMM is refit only on train window. Features computed fresh on test window.

---

## Summary

| # | File | Violation | Severity | Status |
|---|------|-----------|----------|--------|
| 1 | feature_engineering.py | Partial rolling window (min_periods < window) | MEDIUM | ✅ Fixed |
| 2 | main.py | Ablation vol with min_periods=5 | MEDIUM | ✅ Fixed |
| 3 | feature_engineering.py | Vol rolling min_periods=10 not 20 | LOW | ✅ Fixed |
| 4 | backtesting_engine.py | burn-in only 60 bars | HIGH | ✅ Fixed |
| 5 | market_regime_agent.py | Hurst uses full series | LOW | ✅ Resolved (removed) |
| 6 | backtesting_engine.py | Walk-forward not a true train/test split | HIGH | ✅ Fixed |

**Conclusion**: 6 violations found. All 6 resolved. Zero unresolved violations.

---

## Changes Required to Fix All Violations

1. `config.py`: Add `FEATURE_BURNIN_BARS = 252`, `MIN_EPISODE_LENGTH = 63`
2. `features/feature_pipeline.py`: Use `min_periods=window` for all rolling operations
3. `backtest/engine.py`: True walk-forward with separate train/test simulation
4. `training/trainer.py`: Assert `len(df) > FEATURE_BURNIN_BARS + MIN_EPISODE_LENGTH`
