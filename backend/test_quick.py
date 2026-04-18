"""Quick smoke test for the analysis pipeline."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

print("=== 1. Testing data fetcher ===")
try:
    from data.data_fetcher import get_fetcher
    fetcher = get_fetcher()
    df = fetcher.fetch_ohlcv("AAPL", "1d")
    print(f"  OHLCV OK: {df.shape}, columns={list(df.columns)}")
    print(f"  Index tz: {df.index.tz}")
    print(f"  Last row: {df.iloc[-1].to_dict()}")
except Exception:
    traceback.print_exc()

print("\n=== 2. Testing feature pipeline ===")
try:
    from features.feature_pipeline import get_pipeline
    pipeline = get_pipeline()
    features_df = pipeline.compute(df, ticker="AAPL")
    print(f"  Features OK: {features_df.shape}")
except Exception:
    traceback.print_exc()

print("\n=== 3. Testing regime agent ===")
try:
    from agents.market_regime_agent import run as regime_run, run_dict as regime_run_dict
    regime_result = regime_run(df, ticker="AAPL")
    regime_dict = regime_run_dict(df, ticker="AAPL")
    print(f"  Regime: {regime_result.dominant_regime}")
    print(f"  Dict keys: {list(regime_dict.keys())}")
except Exception:
    traceback.print_exc()

print("\n=== 4. Testing disagreement model ===")
try:
    import numpy as np
    from agents.disagreement_model import run as disagreement_run
    latest_features = features_df.fillna(0.0).iloc[-1].values.astype(np.float32)
    disagree_dict = disagreement_run(None, None, None, regime_dict, feature_vector=latest_features)
    print(f"  Disagreement keys: {list(disagree_dict.keys())}")
except Exception:
    traceback.print_exc()

print("\n=== 5. Testing risk agent ===")
try:
    from risk.risk_management_agent import run as risk_run
    risk_result = risk_run(
        df=df, action=0.0, regime_result=regime_result,
        drawdown=0.0, changepoint_probability=float(regime_result.changepoint_probability),
    )
    print(f"  Risk result: {risk_result}")
except Exception:
    traceback.print_exc()

print("\n=== 6. Testing legacy agents ===")
try:
    from agents import indicator_agent, pattern_agent, trend_agent
    ind_r = indicator_agent.run(df)
    pat_r = pattern_agent.run(df)
    tre_r = trend_agent.run(df)
    print(f"  Indicator: {ind_r}")
    print(f"  Pattern: {pat_r}")
    print(f"  Trend: {tre_r}")
except Exception:
    traceback.print_exc()

print("\n=== 7. Testing macro context ===")
try:
    macro = fetcher.fetch_macro_context()
    print(f"  Macro keys: {list(macro.keys())}")
except Exception:
    traceback.print_exc()

print("\n=== ALL DONE ===")
