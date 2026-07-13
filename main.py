"""
APEX-7 FastAPI Application
══════════════════════════
REST + WebSocket API powering the dashboard UI.

Changes vs v1:
  - /api/status now includes config validation warnings so problems
    (e.g. live mode with no API key) surface in the dashboard, not
    just server logs.
  - /api/trades exposes `notes` (used to flag a failed OCO exit — a
    position with no automatic TP/SL needs a human's attention).
  - clean shutdown of the shared aiohttp session in market_data.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select, desc

from config import settings
from database import init_db, AsyncSessionLocal, Trade, AgentSignal, PerformanceSnapshot
from market_data import fetch_price, fetch_all_prices, fetch_candles, shutdown as market_data_shutdown
from trading_engine import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex7.api")


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("✅ Database initialized")
    problems = settings.validate()
    if problems:
        for p in problems:
            logger.warning("⚠ Config issue: %s", p)
    broadcaster_task = asyncio.create_task(_ws_broadcaster())
    yield
    engine.stop()
    broadcaster_task.cancel()
    await market_data_shutdown()


app = FastAPI(title="APEX-7 Trading Bot", version="2.0.0", lifespan=lifespan)


async def _ws_broadcaster():
    while True:
        try:
            status = engine.get_status()
            prices = {}
            try:
                prices = await fetch_all_prices(settings.TRADING_PAIRS)
            except Exception:
                pass
            await ws_manager.broadcast({
                "type": "tick",
                "status": status,
                "prices": prices,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.debug(f"WS broadcast error: {e}")
        await asyncio.sleep(3)


# ─────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    status = engine.get_status()
    status["config_warnings"] = settings.validate()
    return status


@app.post("/api/engine/start")
async def api_start():
    engine.start()
    return {"ok": True, "message": "Engine started"}


@app.post("/api/engine/stop")
async def api_stop():
    engine.stop()
    return {"ok": True, "message": "Engine stopped"}


@app.post("/api/engine/resume")
async def api_resume():
    engine.risk.resume_trading()
    return {"ok": True, "message": "Trading resumed"}


@app.get("/api/trades")
async def api_trades(limit: int = 50, status: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        q = select(Trade).order_by(desc(Trade.opened_at)).limit(limit)
        if status:
            q = q.where(Trade.status == status.upper())
        result = await session.execute(q)
        trades = result.scalars().all()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "usdt_value": t.usdt_value,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "pnl_usdt": t.pnl_usdt,
            "pnl_pct": t.pnl_pct,
            "status": t.status,
            "consensus_score": t.consensus_score,
            "agents_agree": t.agents_agree,
            "regime": t.regime,
            "is_testnet": t.is_testnet,
            "notes": t.notes,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in trades
    ]


@app.get("/api/performance")
async def api_performance():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Trade).where(Trade.status == "CLOSED")
        )
        trades = result.scalars().all()

    if not trades:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate_pct": 0, "total_pnl_usdt": 0,
            "avg_win_usdt": 0, "avg_loss_usdt": 0,
            "profit_factor": 0, "best_trade_pct": 0, "worst_trade_pct": 0,
            "avg_hold_minutes": 0,
        }

    wins = [t for t in trades if (t.pnl_usdt or 0) > 0]
    losses = [t for t in trades if (t.pnl_usdt or 0) <= 0]
    total_pnl = sum(t.pnl_usdt or 0 for t in trades)
    gross_profit = sum(t.pnl_usdt for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_usdt for t in losses)) if losses else 1

    hold_minutes = []
    for t in trades:
        if t.opened_at and t.closed_at:
            diff = (t.closed_at - t.opened_at).total_seconds() / 60
            hold_minutes.append(diff)

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "total_pnl_usdt": round(total_pnl, 4),
        "avg_win_usdt": round(gross_profit / len(wins), 4) if wins else 0,
        "avg_loss_usdt": round(-gross_loss / len(losses), 4) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 3),
        "best_trade_pct": round(max((t.pnl_pct or 0) for t in trades), 3),
        "worst_trade_pct": round(min((t.pnl_pct or 0) for t in trades), 3),
        "avg_hold_minutes": round(sum(hold_minutes) / len(hold_minutes), 1) if hold_minutes else 0,
    }


@app.get("/api/signals")
async def api_signals(limit: int = 100, symbol: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        q = select(AgentSignal).order_by(desc(AgentSignal.created_at)).limit(limit)
        if symbol:
            q = q.where(AgentSignal.symbol == symbol)
        result = await session.execute(q)
        signals = result.scalars().all()
    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "timeframe": s.timeframe,
            "agent_name": s.agent_name,
            "signal": s.signal,
            "confidence": s.confidence,
            "reason": s.reason,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in signals
    ]


@app.get("/api/market/{symbol}")
async def api_market(symbol: str, interval: str = "5m"):
    symbol = symbol.upper()
    try:
        df = await fetch_candles(symbol, interval, 60)
        price = float(df["close"].iloc[-1])
        last = df.iloc[-1]

        def safe(val):
            return None if val != val else round(float(val), 6)  # NaN != NaN

        return {
            "symbol": symbol,
            "price": price,
            "rsi": safe(last["rsi"]),
            "macd_hist": safe(last["macd_hist"]),
            "bb_pct": safe(last["bb_pct"]),
            "vol_zscore": safe(last["vol_zscore"]),
            "adx": safe(last["adx"]),
            "atr": safe(last["atr"]),
            "candles": [
                {
                    "t": int(row["open_time"].timestamp() * 1000),
                    "o": float(row["open"]),
                    "h": float(row["high"]),
                    "l": float(row["low"]),
                    "c": float(row["close"]),
                    "v": float(row["volume"]),
                }
                for _, row in df.tail(60).iterrows()
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SettingsUpdate(BaseModel):
    trading_mode: Optional[str] = None
    trading_pairs: Optional[list[str]] = None
    min_consensus_score: Optional[float] = None
    min_agents_agree: Optional[int] = None
    max_portfolio_risk_pct: Optional[float] = None
    trade_usdt_cap: Optional[float] = None
    require_temporal_confluence: Optional[bool] = None
    regime_hard_block_volatile: Optional[bool] = None


@app.post("/api/settings")
async def api_update_settings(body: SettingsUpdate):
    if body.trading_mode in ("live", "testnet"):
        settings.TRADING_MODE = body.trading_mode
        engine.executor.reconnect()
    if body.trading_pairs:
        settings.TRADING_PAIRS = [p.strip().upper() for p in body.trading_pairs if p.strip()]
    if body.min_consensus_score is not None:
        settings.MIN_CONSENSUS_SCORE = body.min_consensus_score
    if body.min_agents_agree is not None:
        settings.MIN_AGENTS_AGREE = body.min_agents_agree
    if body.max_portfolio_risk_pct is not None:
        settings.MAX_PORTFOLIO_RISK_PCT = body.max_portfolio_risk_pct
    if body.trade_usdt_cap is not None:
        settings.TRADE_USDT_CAP = body.trade_usdt_cap
    if body.require_temporal_confluence is not None:
        settings.REQUIRE_TEMPORAL_CONFLUENCE = body.require_temporal_confluence
    if body.regime_hard_block_volatile is not None:
        settings.REGIME_HARD_BLOCK_VOLATILE = body.regime_hard_block_volatile

    warnings = settings.validate()
    return {"ok": True, "mode": settings.TRADING_MODE, "config_warnings": warnings}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())