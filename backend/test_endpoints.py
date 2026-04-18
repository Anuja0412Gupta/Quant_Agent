import sys, os, traceback, json
import torch # Pre-load to prevent Windows DLL lazy loading issues
sys.path.insert(0, os.path.dirname(__file__))

print("=== 1. Testing rl/brain ===")
try:
    from main import rl_brain
    import asyncio
    res = asyncio.run(rl_brain("AAPL", "1d"))
    print(f"RL brain OK: keys={list(res.keys())}")
except Exception:
    traceback.print_exc()

print("\n=== 2. Testing backtest ===")
try:
    from main import run_backtest_endpoint
    from schemas import BacktestRequest
    import asyncio
    req = BacktestRequest(symbol="AAPL", timeframe="1d", period="5y", initial_capital=100000.0, n_folds=5)
    res = asyncio.run(run_backtest_endpoint(req))
    print(f"Backtest OK: keys={list(res.keys())}")
except Exception:
    traceback.print_exc()
