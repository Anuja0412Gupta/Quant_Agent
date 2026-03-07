"""
QuantAgent — FastAPI Backend
============================
REST API exposing all agent pipelines, backtesting, and configuration.

Endpoints:
  GET  /health                  — Health check
  GET  /analyze/{symbol}        — Full analysis (all agents)
  GET  /backtest/{symbol}       — Historical backtest
  GET  /regime/{symbol}         — Market regime only
  GET  /agents/weights          — Current agent weights
  POST /agents/weights          — Override agent weights
  POST /critique                — Run self-critique after trade
  POST /rl/train                — Trigger RL training
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path fix so imports work from project root ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import API_HOST, API_PORT, CORS_ORIGINS, DEFAULT_TIMEFRAME
from data.data_fetcher import fetch_ohlcv
from agents import (
    indicator_agent,
    pattern_agent,
    trend_agent,
    market_regime_agent,
    decision_agent,
    disagreement_model,
    risk_management_agent,
    self_critique_agent,
    rl_meta_controller,
)
from backtesting.backtesting_engine import run as run_backtest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QuantAgent API",
    description="Multi-agent AI trading system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class WeightsPayload(BaseModel):
    indicator: float = 0.30
    pattern:   float = 0.25
    trend:     float = 0.25
    regime:    float = 0.20


class CritiquePayload(BaseModel):
    symbol:       str
    timeframe:    str = "1d"
    trade_result: str   # "profit" | "loss" | "breakeven"
    pnl_pct:      float = 0.0


class RLTrainPayload(BaseModel):
    timesteps: int = 10_000


# ── In-memory weight store ────────────────────────────────────────────────────
from config import DEFAULT_AGENT_WEIGHTS
_current_weights: Dict[str, float] = dict(DEFAULT_AGENT_WEIGHTS)


# ── Core pipeline helper ──────────────────────────────────────────────────────

def _run_full_pipeline(
    symbol: str,
    timeframe: str,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fetch data and run all agents. Returns combined result dict."""
    try:
        df = fetch_ohlcv(symbol, timeframe)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    wts = weights or _current_weights

    # ── Agent results ──────────────────────────────────────────────────────────
    ind_r  = indicator_agent.run(df)
    pat_r  = pattern_agent.run(df)
    tre_r  = trend_agent.run(df)
    reg_r  = market_regime_agent.run(df)
    dis_r  = disagreement_model.run(ind_r, pat_r, tre_r, reg_r)

    # ── RL weight adjustment ───────────────────────────────────────────────────
    rl_r   = rl_meta_controller.run(
        ind_r, pat_r, tre_r, reg_r,
        dis_r.get("disagreement_index", 0.0),
    )
    rl_weights = rl_r.get("weights", wts)

    dec_r  = decision_agent.run(df, ind_r, pat_r, tre_r, reg_r, rl_weights)

    # Override action if disagreement too high
    if dis_r.get("recommendation") == "NO_TRADE":
        dec_r["action"] = "NO_TRADE"
        dec_r["reasoning"] = "Overridden by high disagreement: " + dec_r["reasoning"]

    risk_r = risk_management_agent.run(df, dec_r, dis_r)

    # ── OHLCV for chart ────────────────────────────────────────────────────────
    recent = df.tail(120)
    ohlcv_data = [
        {
            "time":   int(ts.timestamp()),
            "open":   round(row["Open"],  4),
            "high":   round(row["High"],  4),
            "low":    round(row["Low"],   4),
            "close":  round(row["Close"], 4),
            "volume": int(row["Volume"]),
        }
        for ts, row in recent.iterrows()
    ]

    return {
        "symbol":      symbol,
        "timeframe":   timeframe,
        "indicator":   ind_r,
        "pattern":     pat_r,
        "trend":       tre_r,
        "regime":      reg_r,
        "disagreement": dis_r,
        "decision":    dec_r,
        "risk":        risk_r,
        "rl_weights":  rl_r,
        "ohlcv":       ohlcv_data,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/analyze/{symbol}")
async def analyze(
    symbol: str,
    timeframe: str = Query(DEFAULT_TIMEFRAME, description="1m|5m|15m|1h|1d"),
):
    """Run the full QuantAgent pipeline and return all agent results."""
    return _run_full_pipeline(symbol.upper(), timeframe)


@app.get("/backtest/{symbol}")
async def backtest(
    symbol:    str,
    timeframe: str   = Query("1d"),
    period:    str   = Query("5y"),
):
    """Run historical backtesting and return performance metrics."""
    try:
        df = fetch_ohlcv(symbol.upper(), timeframe, period=period)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = run_backtest(df, symbol=symbol.upper(), timeframe=timeframe, period=period)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/regime/{symbol}")
async def regime(
    symbol:    str,
    timeframe: str = Query(DEFAULT_TIMEFRAME),
):
    """Return market regime analysis only."""
    try:
        df = fetch_ohlcv(symbol.upper(), timeframe)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return market_regime_agent.run(df)


@app.get("/agents/weights")
async def get_weights():
    """Return current agent weights."""
    return {"weights": _current_weights}


@app.post("/agents/weights")
async def set_weights(payload: WeightsPayload):
    """Override agent weights."""
    global _current_weights
    raw = {"indicator": payload.indicator, "pattern": payload.pattern,
           "trend": payload.trend, "regime": payload.regime}
    total = sum(raw.values())
    _current_weights = {k: round(v / total, 4) for k, v in raw.items()}
    return {"weights": _current_weights, "message": "Weights updated."}


@app.post("/critique")
async def critique(payload: CritiquePayload):
    """Run self-critique after a trade and update agent weights."""
    try:
        df = fetch_ohlcv(payload.symbol.upper(), payload.timeframe)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ind_r = indicator_agent.run(df)
    pat_r = pattern_agent.run(df)
    tre_r = trend_agent.run(df)
    reg_r = market_regime_agent.run(df)

    result = self_critique_agent.run(
        ind_r, pat_r, tre_r, reg_r,
        trade_result=payload.trade_result,
        pnl_pct=payload.pnl_pct,
    )
    # Sync in-memory weights
    global _current_weights
    _current_weights = result["updated_weights"]
    return result


@app.post("/rl/train")
async def rl_train(payload: RLTrainPayload):
    """Trigger offline RL training run (runs synchronously — may take time)."""
    ctrl = rl_meta_controller.get_controller()
    ctrl.train(timesteps=payload.timesteps)
    return {"message": f"RL training complete ({payload.timesteps} steps). Model saved."}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
