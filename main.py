"""
APEX-8 FastAPI Application
══════════════════════════
REST + WebSocket API powering the dashboard UI.

Changes vs v1:
  - /api/status now includes config validation warnings so problems
    (e.g. live mode with no API key) surface in the dashboard, not
    just server logs.
  - /api/trades exposes `notes` (used to flag a failed OCO exit — a
    position with no automatic TP/SL needs a human's attention).
  - clean shutdown of the shared aiohttp session in market_data.

Changes vs v2:
  - /api/trades now exposes `risk_usdt`, and for OPEN trades also
    returns live `unrealized_pnl_usdt` / `unrealized_pnl_pct` computed
    from current market price — previously an open trade showed no
    PnL at all until it closed.
  - /api/performance now returns `unrealized_pnl_usdt` (sum across all
    open positions) and `combined_pnl_usdt` (realized + unrealized),
    instead of only ever reporting closed-trade PnL. The dashboard
    stat card was silently ignoring anything still open.

Changes vs v3 (Futures Mode):
  - Full parallel route set under /api/futures/* — status, engine
    start (accepts leverage + margin_type)/stop/resume, trades,
    performance, signals, market data, and settings. Same shapes as
    the spot routes plus leverage/margin/liquidation fields.
  - The WebSocket broadcaster now pushes BOTH engines' status/prices
    in one message (`status`/`prices` = spot, `futures_status`/
    `futures_prices` = futures) so the dashboard can run a single
    connection and just switch which half of the payload it reads
    when the person toggles Spot/Futures.
  - lifespan() now also shuts down futures_market_data's HTTP session
    and stops the futures engine on app shutdown.

Changes vs v4 (Signal Lab):
  - New /api/signal-lab/* routes exposing the "what if we'd taken this
    trade?" shadow-tracking data: raw outcomes, per-agent performance,
    per-combination leaderboard, per-agents-agree-count performance,
    and a weight-recommendation + apply-to-live-engine action. All
    routes take an optional `market_type` (SPOT/FUTURES) query param
    since Signal Lab data is stored in one shared table rather than
    split per market.

Changes vs v5 (Signal Lab fee adjustment):
  - lifespan() now runs signal_lab.backfill_fees() once at startup,
    after init_db(), so any resolved shadow outcomes from before fee
    adjustment shipped get a retroactive NET (fee-adjusted) figure
    instead of silently keeping the old, overstated GROSS-only numbers
    forever. Safe to run on every startup — it only touches rows still
    missing net_pnl_usdt.
  - /api/signal-lab/outcomes now also returns `fee_usdt` and
    `net_pnl_usdt` per row (GROSS pnl_usdt is unchanged/still present).

Changes vs v6 (real-trade fee adjustment):
  - _unrealized_pnl() / _futures_unrealized_pnl() now also return a
    live fee ESTIMATE and NET pnl for still-OPEN positions, and
    /api/trades, /api/futures/trades now expose fee_usdt/net_pnl_usdt/
    net_pnl_pct for CLOSED trades (persisted) and
    unrealized_fee_usdt/unrealized_net_pnl_usdt/unrealized_net_pnl_pct
    for OPEN ones (live estimate).
  - /api/performance and /api/futures/performance now report
    total_fees_usdt / total_net_pnl_usdt / net_win_rate_pct for closed
    trades, and unrealized_net_pnl_usdt / combined_net_pnl_usdt
    alongside the existing GROSS figures.
  - lifespan() now also runs a one-time backfill for historical CLOSED
    real trades that predate this feature (mirrors Signal Lab's own
    backfill_fees(), but for Trade/FuturesTrade instead of SignalOutcome).

Changes vs v7 (resilient startup backfill):
  - database.py's init_db() now self-heals ANY missing column on ANY
    known table on every boot (see that file's v7 changelog) — the
    `no such column: trades.risk_usdt` class of error that used to hit
    a persisted DB (e.g. a Railway Volume created before a schema
    change) should no longer happen. _backfill_trade_fees() is
    defense-in-depth on top of that: the spot-Trade block and the
    FuturesTrade block are now each wrapped independently, so a problem
    on ONE table (schema drift, a locked DB file, anything) can't
    silently prevent the OTHER table's historical rows from being
    backfilled, and both failures are logged individually instead of
    one exception aborting the whole function.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select, desc

from config import settings
from database import (
    init_db, AsyncSessionLocal, Trade, AgentSignal, PerformanceSnapshot,
    FuturesTrade, FuturesAgentSignal, FuturesPerformanceSnapshot,
    SignalOutcome,
)
from market_data import fetch_price, fetch_all_prices, fetch_candles, shutdown as market_data_shutdown
import futures_market_data as fmd
from trading_engine import engine
from futures_trading_engine import futures_engine
import signal_lab
from database import (
    init_db, AsyncSessionLocal, Trade, AgentSignal, PerformanceSnapshot,
    FuturesTrade, FuturesAgentSignal, FuturesPerformanceSnapshot,
    SignalOutcome, AgentWeightHistory,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("apex8.api")


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


async def _backfill_spot_trade_fees() -> int:
    """Backfill fee/net-PnL for historical CLOSED spot trades. Isolated
    from the futures backfill below so a failure here (e.g. schema
    drift, a locked DB file) can't prevent futures trades from being
    backfilled too — each table's problem is now visible on its own
    instead of one exception hiding the other."""
    updated = 0
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Trade).where(Trade.status == "CLOSED", Trade.net_pnl_usdt.is_(None))
        )
        rows = result.scalars().all()
        for t in rows:
            fee = round((t.usdt_value or 0.0) * (settings.SPOT_TAKER_FEE_PCT * 2 / 100), 4)
            net_usdt = round((t.pnl_usdt or 0.0) - fee, 4)
            net_pct = round(net_usdt / t.usdt_value * 100, 4) if t.usdt_value else t.pnl_pct
            t.fee_usdt, t.net_pnl_usdt, t.net_pnl_pct = fee, net_usdt, net_pct
            updated += 1
        if rows:
            await session.commit()
    return updated


async def _backfill_futures_trade_fees() -> int:
    """Backfill fee/net-PnL for historical CLOSED/LIQUIDATED futures
    trades. See _backfill_spot_trade_fees() docstring for why this is
    split out rather than sharing one try/except with the spot table."""
    updated = 0
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(FuturesTrade).where(
                FuturesTrade.status.in_(["CLOSED", "LIQUIDATED"]),
                FuturesTrade.net_pnl_usdt.is_(None),
            )
        )
        rows = result.scalars().all()
        for t in rows:
            # Futures fees are charged on NOTIONAL (usdt_value), but PnL%
            # is expressed as ROI-on-MARGIN, matching how the live close
            # path computes it (see futures_trading_engine.py).
            fee = round((t.usdt_value or 0.0) * (settings.FUTURES_TAKER_FEE_PCT * 2 / 100), 4)
            net_usdt = round((t.pnl_usdt or 0.0) - fee, 4)
            net_pct = round(net_usdt / t.margin_usdt * 100, 4) if t.margin_usdt else t.pnl_pct
            t.fee_usdt, t.net_pnl_usdt, t.net_pnl_pct = fee, net_usdt, net_pct
            updated += 1
        if rows:
            await session.commit()
    return updated


async def _backfill_trade_fees() -> int:
    """One-time-per-row (safe on every startup — only touches rows still
    missing net_pnl_usdt) retroactive fee/net-PnL computation for REAL
    trades that closed BEFORE this feature existed. Mirrors
    signal_lab.backfill_fees(), but for Trade/FuturesTrade instead of
    the shadow SignalOutcome table.

    v7: spot and futures are now backfilled independently (see the two
    helpers above) so a problem in one table doesn't take down the
    other, and each failure is logged with which table it was."""
    total_updated = 0

    try:
        spot_updated = await _backfill_spot_trade_fees()
        total_updated += spot_updated
    except Exception as e:
        logger.warning(
            f"⚠ Spot trade fee backfill failed: {type(e).__name__}: {e} — "
            f"if this mentions 'no such column', a schema migration didn't "
            f"complete; check the '✅ Migrated' / '❌ Schema sync failed' "
            f"log lines from database.init_db() just above this."
        )

    try:
        futures_updated = await _backfill_futures_trade_fees()
        total_updated += futures_updated
    except Exception as e:
        logger.warning(
            f"⚠ Futures trade fee backfill failed: {type(e).__name__}: {e} — "
            f"see database.init_db()'s migration log lines above for details."
        )

    return total_updated


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("✅ Database initialized (spot + futures + signal lab tables)")

    try:
        backfilled = await signal_lab.backfill_fees()
        if backfilled:
            logger.info(f"✅ Signal Lab: fee-adjusted {backfilled} historical resolved rows")
    except Exception as e:
        logger.warning(f"⚠ Signal Lab fee backfill failed: {e}")

    trades_backfilled = await _backfill_trade_fees()
    if trades_backfilled:
        logger.info(f"✅ Backfilled fee adjustment on {trades_backfilled} historical closed real trades")

    problems = settings.validate()
    if problems:
        for p in problems:
            logger.warning("⚠ Config issue: %s", p)
    broadcaster_task = asyncio.create_task(_ws_broadcaster())
    yield
    engine.stop()
    futures_engine.stop()
    broadcaster_task.cancel()
    await market_data_shutdown()
    await fmd.shutdown()


app = FastAPI(title="APEX-8 Trading Bot", version="7.0.0", lifespan=lifespan)


async def _ws_broadcaster():
    while True:
        try:
            spot_status = engine.get_status()
            futures_status = futures_engine.get_status()

            spot_prices = {}
            futures_prices = {}
            try:
                spot_prices = await fetch_all_prices(settings.TRADING_PAIRS)
            except Exception:
                pass
            try:
                futures_prices = await fmd.fetch_all_prices(settings.FUTURES_TRADING_PAIRS)
            except Exception:
                pass

            await ws_manager.broadcast({
                "type": "tick",
                "status": spot_status,
                "prices": spot_prices,
                "futures_status": futures_status,
                "futures_prices": futures_prices,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.debug(f"WS broadcast error: {e}")
        await asyncio.sleep(3)


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def _unrealized_pnl(trade: Trade, price: Optional[float]):
    """Return (gross_usdt, gross_pct, fee_usdt, net_usdt, net_pct) for an
    OPEN spot trade given a live price — fee/net here are an ESTIMATE
    (what it would cost/net to close right now), since the exit hasn't
    actually happened yet. Returns all-None if price isn't available."""
    if not price or price <= 0 or not trade.entry_price:
        return None, None, None, None, None
    pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
    if trade.side == "SELL":
        pnl_pct = -pnl_pct
    pnl_usdt = (trade.usdt_value or 0.0) * pnl_pct / 100
    fee_usdt = (trade.usdt_value or 0.0) * (settings.SPOT_TAKER_FEE_PCT * 2 / 100)
    net_usdt = pnl_usdt - fee_usdt
    net_pct = (net_usdt / trade.usdt_value * 100) if trade.usdt_value else pnl_pct
    return pnl_usdt, pnl_pct, fee_usdt, net_usdt, net_pct


def _futures_unrealized_pnl(trade: FuturesTrade, price: Optional[float]):
    """Return (gross_usdt, gross_pct, fee_usdt, net_usdt, net_pct) for an
    OPEN futures trade. gross_pct/net_pct are ROI-on-MARGIN (leverage-
    adjusted), matching how Binance itself displays futures PnL% — NOT
    the raw price move %, which is what the spot version returns. Fee is
    charged on NOTIONAL (trade.usdt_value), not margin."""
    if not price or price <= 0 or not trade.entry_price:
        return None, None, None, None, None
    pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
    if trade.side == "SELL":
        pnl_pct = -pnl_pct
    leveraged_pct = pnl_pct * (trade.leverage or 1)
    pnl_usdt = (trade.margin_usdt or 0.0) * leveraged_pct / 100
    fee_usdt = (trade.usdt_value or 0.0) * (settings.FUTURES_TAKER_FEE_PCT * 2 / 100)
    net_usdt = pnl_usdt - fee_usdt
    net_pct = (net_usdt / trade.margin_usdt * 100) if trade.margin_usdt else leveraged_pct
    return pnl_usdt, leveraged_pct, fee_usdt, net_usdt, net_pct


def _market_type_param(market_type: Optional[str]) -> Optional[str]:
    """Normalize/validate the market_type query param used across every
    Signal Lab route. Returns None (= no filter) if missing/invalid."""
    if not market_type:
        return None
    mt = market_type.upper()
    return mt if mt in ("SPOT", "FUTURES") else None


# ═══════════════════════════════════════════════════
#  SPOT API Routes
# ═══════════════════════════════════════════════════
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

    open_symbols = list({t.symbol for t in trades if t.status == "OPEN"})
    live_prices: dict[str, float] = {}
    if open_symbols:
        try:
            live_prices = await fetch_all_prices(open_symbols)
        except Exception:
            live_prices = {}

    out = []
    for t in trades:
        u_gross, u_gross_pct, u_fee, u_net, u_net_pct = (None,) * 5
        if t.status == "OPEN":
            u_gross, u_gross_pct, u_fee, u_net, u_net_pct = _unrealized_pnl(t, live_prices.get(t.symbol))

        out.append({
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "usdt_value": t.usdt_value,
            "risk_usdt": t.risk_usdt,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "pnl_usdt": t.pnl_usdt,                # GROSS — closed trades only
            "pnl_pct": t.pnl_pct,                  # GROSS
            "fee_usdt": t.fee_usdt,                # persisted, closed trades only
            "net_pnl_usdt": t.net_pnl_usdt,        # persisted, closed trades only
            "net_pnl_pct": t.net_pnl_pct,          # persisted, closed trades only
            "unrealized_pnl_usdt": round(u_gross, 4) if u_gross is not None else None,
            "unrealized_pnl_pct": round(u_gross_pct, 4) if u_gross_pct is not None else None,
            "unrealized_fee_usdt": round(u_fee, 4) if u_fee is not None else None,
            "unrealized_net_pnl_usdt": round(u_net, 4) if u_net is not None else None,
            "unrealized_net_pnl_pct": round(u_net_pct, 4) if u_net_pct is not None else None,
            "status": t.status,
            "consensus_score": t.consensus_score,
            "agents_agree": t.agents_agree,
            "regime": t.regime,
            "is_testnet": t.is_testnet,
            "notes": t.notes,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        })
    return out


@app.get("/api/performance")
async def api_performance():
    async with AsyncSessionLocal() as session:
        closed_result = await session.execute(
            select(Trade).where(Trade.status == "CLOSED")
        )
        closed = closed_result.scalars().all()

        open_result = await session.execute(
            select(Trade).where(Trade.status == "OPEN")
        )
        open_trades = open_result.scalars().all()

    unrealized_pnl_usdt = 0.0
    unrealized_net_pnl_usdt = 0.0
    if open_trades:
        try:
            live_prices = await fetch_all_prices(list({t.symbol for t in open_trades}))
        except Exception:
            live_prices = {}
        for t in open_trades:
            g, _, _, n, _ = _unrealized_pnl(t, live_prices.get(t.symbol))
            if g is not None:
                unrealized_pnl_usdt += g
            if n is not None:
                unrealized_net_pnl_usdt += n

    if not closed:
        base = {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate_pct": 0, "total_pnl_usdt": 0,
            "avg_win_usdt": 0, "avg_loss_usdt": 0,
            "profit_factor": 0, "best_trade_pct": 0, "worst_trade_pct": 0,
            "avg_hold_minutes": 0,
            "total_fees_usdt": 0, "total_net_pnl_usdt": 0, "net_win_rate_pct": 0,
        }
    else:
        wins = [t for t in closed if (t.pnl_usdt or 0) > 0]
        losses = [t for t in closed if (t.pnl_usdt or 0) <= 0]
        total_pnl = sum(t.pnl_usdt or 0 for t in closed)
        gross_profit = sum(t.pnl_usdt for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usdt for t in losses)) if losses else 1

        total_fees = sum(t.fee_usdt or 0.0 for t in closed)
        net_pnls = [(t.net_pnl_usdt if t.net_pnl_usdt is not None else (t.pnl_usdt or 0.0)) for t in closed]
        total_net_pnl = sum(net_pnls)
        net_wins = sum(1 for v in net_pnls if v > 0)

        hold_minutes = []
        for t in closed:
            if t.opened_at and t.closed_at:
                diff = (t.closed_at - t.opened_at).total_seconds() / 60
                hold_minutes.append(diff)

        base = {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 2),
            "total_pnl_usdt": round(total_pnl, 4),
            "avg_win_usdt": round(gross_profit / len(wins), 4) if wins else 0,
            "avg_loss_usdt": round(-gross_loss / len(losses), 4) if losses else 0,
            "profit_factor": round(gross_profit / gross_loss, 3),
            "best_trade_pct": round(max((t.pnl_pct or 0) for t in closed), 3),
            "worst_trade_pct": round(min((t.pnl_pct or 0) for t in closed), 3),
            "avg_hold_minutes": round(sum(hold_minutes) / len(hold_minutes), 1) if hold_minutes else 0,
            "total_fees_usdt": round(total_fees, 4),
            "total_net_pnl_usdt": round(total_net_pnl, 4),
            "net_win_rate_pct": round(net_wins / len(closed) * 100, 2),
        }

    base["open_trades_count"] = len(open_trades)
    base["unrealized_pnl_usdt"] = round(unrealized_pnl_usdt, 4)
    base["unrealized_net_pnl_usdt"] = round(unrealized_net_pnl_usdt, 4)
    base["combined_pnl_usdt"] = round(base["total_pnl_usdt"] + unrealized_pnl_usdt, 4)
    base["combined_net_pnl_usdt"] = round(base["total_net_pnl_usdt"] + unrealized_net_pnl_usdt, 4)
    return base


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
            return None if val != val else round(float(val), 6)

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
        futures_engine.executor.reconnect()  # shared TRADING_MODE — resync futures client too
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


# ═══════════════════════════════════════════════════
#  FUTURES API Routes
# ═══════════════════════════════════════════════════
@app.get("/api/futures/status")
async def api_futures_status():
    status = futures_engine.get_status()
    status["config_warnings"] = settings.validate()
    return status


class FuturesStartBody(BaseModel):
    leverage: Optional[int] = None
    margin_type: Optional[str] = None


@app.post("/api/futures/engine/start")
async def api_futures_start(body: FuturesStartBody):
    leverage = body.leverage if body.leverage is not None else settings.FUTURES_DEFAULT_LEVERAGE
    leverage = max(1, min(int(leverage), settings.FUTURES_MAX_LEVERAGE_ALLOWED))
    futures_engine.start(leverage=leverage, margin_type=body.margin_type)
    return {"ok": True, "message": f"Futures engine started @ {leverage}x", "leverage": leverage}


@app.post("/api/futures/engine/stop")
async def api_futures_stop():
    futures_engine.stop()
    return {"ok": True, "message": "Futures engine stopped"}


@app.post("/api/futures/engine/resume")
async def api_futures_resume():
    futures_engine.risk.resume_trading()
    return {"ok": True, "message": "Futures trading resumed"}


@app.get("/api/futures/trades")
async def api_futures_trades(limit: int = 50, status: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        q = select(FuturesTrade).order_by(desc(FuturesTrade.opened_at)).limit(limit)
        if status:
            q = q.where(FuturesTrade.status == status.upper())
        result = await session.execute(q)
        trades = result.scalars().all()

    open_symbols = list({t.symbol for t in trades if t.status == "OPEN"})
    live_prices: dict[str, float] = {}
    if open_symbols:
        try:
            live_prices = await fmd.fetch_all_prices(open_symbols)
        except Exception:
            live_prices = {}

    out = []
    for t in trades:
        u_gross, u_gross_pct, u_fee, u_net, u_net_pct = (None,) * 5
        if t.status == "OPEN":
            u_gross, u_gross_pct, u_fee, u_net, u_net_pct = _futures_unrealized_pnl(t, live_prices.get(t.symbol))

        out.append({
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "leverage": t.leverage,
            "margin_type": t.margin_type,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "usdt_value": t.usdt_value,           # notional
            "margin_usdt": t.margin_usdt,
            "risk_usdt": t.risk_usdt,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "liquidation_price": t.liquidation_price,
            "pnl_usdt": t.pnl_usdt,                # GROSS — closed trades only
            "pnl_pct": t.pnl_pct,                  # GROSS ROI on margin at close
            "fee_usdt": t.fee_usdt,                # persisted, closed trades only
            "net_pnl_usdt": t.net_pnl_usdt,        # persisted, closed trades only
            "net_pnl_pct": t.net_pnl_pct,          # persisted, closed trades only
            "unrealized_pnl_usdt": round(u_gross, 4) if u_gross is not None else None,
            "unrealized_pnl_pct": round(u_gross_pct, 4) if u_gross_pct is not None else None,
            "unrealized_fee_usdt": round(u_fee, 4) if u_fee is not None else None,
            "unrealized_net_pnl_usdt": round(u_net, 4) if u_net is not None else None,
            "unrealized_net_pnl_pct": round(u_net_pct, 4) if u_net_pct is not None else None,
            "status": t.status,
            "consensus_score": t.consensus_score,
            "agents_agree": t.agents_agree,
            "regime": t.regime,
            "is_testnet": t.is_testnet,
            "notes": t.notes,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        })
    return out


@app.get("/api/futures/performance")
async def api_futures_performance():
    async with AsyncSessionLocal() as session:
        closed_result = await session.execute(
            select(FuturesTrade).where(FuturesTrade.status.in_(["CLOSED", "LIQUIDATED"]))
        )
        closed = closed_result.scalars().all()

        open_result = await session.execute(
            select(FuturesTrade).where(FuturesTrade.status == "OPEN")
        )
        open_trades = open_result.scalars().all()

    unrealized_pnl_usdt = 0.0
    unrealized_net_pnl_usdt = 0.0
    if open_trades:
        try:
            live_prices = await fmd.fetch_all_prices(list({t.symbol for t in open_trades}))
        except Exception:
            live_prices = {}
        for t in open_trades:
            g, _, _, n, _ = _futures_unrealized_pnl(t, live_prices.get(t.symbol))
            if g is not None:
                unrealized_pnl_usdt += g
            if n is not None:
                unrealized_net_pnl_usdt += n

    if not closed:
        base = {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "liquidations": 0,
            "win_rate_pct": 0, "total_pnl_usdt": 0,
            "avg_win_usdt": 0, "avg_loss_usdt": 0,
            "profit_factor": 0, "best_trade_pct": 0, "worst_trade_pct": 0,
            "avg_hold_minutes": 0,
            "total_fees_usdt": 0, "total_net_pnl_usdt": 0, "net_win_rate_pct": 0,
        }
    else:
        wins = [t for t in closed if (t.pnl_usdt or 0) > 0]
        losses = [t for t in closed if (t.pnl_usdt or 0) <= 0]
        liquidations = [t for t in closed if t.status == "LIQUIDATED"]
        total_pnl = sum(t.pnl_usdt or 0 for t in closed)
        gross_profit = sum(t.pnl_usdt for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usdt for t in losses)) if losses else 1

        total_fees = sum(t.fee_usdt or 0.0 for t in closed)
        net_pnls = [(t.net_pnl_usdt if t.net_pnl_usdt is not None else (t.pnl_usdt or 0.0)) for t in closed]
        total_net_pnl = sum(net_pnls)
        net_wins = sum(1 for v in net_pnls if v > 0)

        hold_minutes = []
        for t in closed:
            if t.opened_at and t.closed_at:
                diff = (t.closed_at - t.opened_at).total_seconds() / 60
                hold_minutes.append(diff)

        base = {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "liquidations": len(liquidations),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 2),
            "total_pnl_usdt": round(total_pnl, 4),
            "avg_win_usdt": round(gross_profit / len(wins), 4) if wins else 0,
            "avg_loss_usdt": round(-gross_loss / len(losses), 4) if losses else 0,
            "profit_factor": round(gross_profit / gross_loss, 3),
            "best_trade_pct": round(max((t.pnl_pct or 0) for t in closed), 3),
            "worst_trade_pct": round(min((t.pnl_pct or 0) for t in closed), 3),
            "avg_hold_minutes": round(sum(hold_minutes) / len(hold_minutes), 1) if hold_minutes else 0,
            "total_fees_usdt": round(total_fees, 4),
            "total_net_pnl_usdt": round(total_net_pnl, 4),
            "net_win_rate_pct": round(net_wins / len(closed) * 100, 2),
        }

    base["open_trades_count"] = len(open_trades)
    base["unrealized_pnl_usdt"] = round(unrealized_pnl_usdt, 4)
    base["unrealized_net_pnl_usdt"] = round(unrealized_net_pnl_usdt, 4)
    base["combined_pnl_usdt"] = round(base["total_pnl_usdt"] + unrealized_pnl_usdt, 4)
    base["combined_net_pnl_usdt"] = round(base["total_net_pnl_usdt"] + unrealized_net_pnl_usdt, 4)
    return base


@app.get("/api/futures/signals")
async def api_futures_signals(limit: int = 100, symbol: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        q = select(FuturesAgentSignal).order_by(desc(FuturesAgentSignal.created_at)).limit(limit)
        if symbol:
            q = q.where(FuturesAgentSignal.symbol == symbol)
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


@app.get("/api/futures/market/{symbol}")
async def api_futures_market(symbol: str, interval: str = "3m"):
    symbol = symbol.upper()
    try:
        df = await fmd.fetch_candles(symbol, interval, 60)
        price = float(df["close"].iloc[-1])
        last = df.iloc[-1]

        def safe(val):
            return None if val != val else round(float(val), 6)

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


class FuturesSettingsUpdate(BaseModel):
    trading_mode: Optional[str] = None
    trading_pairs: Optional[list[str]] = None
    leverage: Optional[int] = None
    margin_type: Optional[str] = None
    min_consensus_score: Optional[float] = None
    min_agents_agree: Optional[int] = None
    max_portfolio_risk_pct: Optional[float] = None
    trade_margin_cap_usdt: Optional[float] = None


@app.post("/api/futures/settings")
async def api_update_futures_settings(body: FuturesSettingsUpdate):
    if body.trading_mode in ("live", "testnet"):
        settings.TRADING_MODE = body.trading_mode
        futures_engine.executor.reconnect()
        engine.executor.reconnect()  # shared TRADING_MODE — resync spot client too
    if body.trading_pairs:
        settings.FUTURES_TRADING_PAIRS = [p.strip().upper() for p in body.trading_pairs if p.strip()]
    if body.leverage is not None:
        futures_engine.update_leverage(body.leverage)
        settings.FUTURES_DEFAULT_LEVERAGE = futures_engine.leverage
    if body.margin_type is not None:
        futures_engine.update_margin_type(body.margin_type)
        settings.FUTURES_MARGIN_TYPE = futures_engine.margin_type
    if body.min_consensus_score is not None:
        settings.FUTURES_MIN_CONSENSUS_SCORE = body.min_consensus_score
    if body.min_agents_agree is not None:
        settings.FUTURES_MIN_AGENTS_AGREE = body.min_agents_agree
    if body.max_portfolio_risk_pct is not None:
        settings.FUTURES_MAX_PORTFOLIO_RISK_PCT = body.max_portfolio_risk_pct
    if body.trade_margin_cap_usdt is not None:
        settings.FUTURES_TRADE_MARGIN_CAP_USDT = body.trade_margin_cap_usdt

    warnings = settings.validate()
    return {
        "ok": True, "mode": settings.TRADING_MODE,
        "leverage": futures_engine.leverage, "margin_type": futures_engine.margin_type,
        "config_warnings": warnings,
    }


# ═══════════════════════════════════════════════════
#  SIGNAL LAB API Routes — "what if we'd taken this?" analytics
# ═══════════════════════════════════════════════════
@app.get("/api/signal-lab/status")
async def api_signal_lab_status(market_type: Optional[str] = None):
    return await signal_lab.status_summary(_market_type_param(market_type))


@app.post("/api/signal-lab/backfill-fees")
async def api_signal_lab_backfill_fees():
    """Manual trigger for the fee backfill (also runs automatically at
    startup) — useful right after changing SPOT_TAKER_FEE_PCT /
    FUTURES_TAKER_FEE_PCT so historical rows immediately reflect the new
    rate without a restart. NOTE: this only fills rows where
    net_pnl_usdt is NULL — it will NOT recompute rows that already have
    a net figure from a previous rate."""
    updated = await signal_lab.backfill_fees()
    return {"ok": True, "rows_updated": updated}


@app.get("/api/signal-lab/outcomes")
async def api_signal_lab_outcomes(
    market_type: Optional[str] = None,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
):
    mt = _market_type_param(market_type)
    async with AsyncSessionLocal() as session:
        q = select(SignalOutcome).order_by(desc(SignalOutcome.created_at)).limit(limit)
        if mt:
            q = q.where(SignalOutcome.market_type == mt)
        if symbol:
            q = q.where(SignalOutcome.symbol == symbol.upper())
        if status:
            q = q.where(SignalOutcome.status == status.upper())
        result = await session.execute(q)
        rows = result.scalars().all()

    out = []
    for r in rows:
        try:
            detail = json.loads(r.agent_detail_json or "[]")
        except Exception:
            detail = []
        out.append({
            "id": r.id,
            "market_type": r.market_type,
            "symbol": r.symbol,
            "direction": r.direction,
            "entry_price": r.entry_price,
            "consensus_score": r.consensus_score,
            "agents_agree": r.agents_agree,
            "total_agents": r.total_agents,
            "regime": r.regime,
            "stop_loss_price": r.stop_loss_price,
            "take_profit_price": r.take_profit_price,
            "was_taken": r.was_taken,
            "linked_trade_id": r.linked_trade_id,
            "agent_detail": detail,
            "status": r.status,
            "exit_price": r.exit_price,
            "pnl_pct": r.pnl_pct,
            "pnl_usdt": r.pnl_usdt,              # GROSS — before fees
            "fee_usdt": r.fee_usdt,
            "net_pnl_usdt": r.net_pnl_usdt,       # NET — after fees (the realistic figure)
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
        })
    return out


@app.get("/api/signal-lab/agent-performance")
async def api_signal_lab_agent_performance(
    market_type: Optional[str] = None, symbol: Optional[str] = None, min_samples: Optional[int] = None,
):
    return await signal_lab.agent_performance(
        _market_type_param(market_type), symbol.upper() if symbol else None, min_samples,
    )


@app.get("/api/signal-lab/combo-performance")
async def api_signal_lab_combo_performance(
    market_type: Optional[str] = None, symbol: Optional[str] = None,
    min_samples: Optional[int] = None, max_combos: int = 30,
):
    return await signal_lab.combo_performance(
        _market_type_param(market_type), symbol.upper() if symbol else None, min_samples, max_combos,
    )


@app.get("/api/signal-lab/agents-agree-performance")
async def api_signal_lab_agree_performance(market_type: Optional[str] = None, symbol: Optional[str] = None):
    return await signal_lab.agents_agree_bucket_performance(
        _market_type_param(market_type), symbol.upper() if symbol else None,
    )


@app.get("/api/signal-lab/weight-recommendations")
async def api_signal_lab_weight_recommendations(market_type: Optional[str] = None, min_samples: Optional[int] = None):
    mt = _market_type_param(market_type) or "SPOT"
    eng = futures_engine if mt == "FUTURES" else engine
    base_weights = {a.name: a.weight for a in eng.consensus.agents}
    return await signal_lab.recommend_weights(mt, min_samples, base_weights)


class ApplyWeightsBody(BaseModel):
    market_type: str
    weights: dict[str, float]


@app.get("/api/signal-lab/weight-recommendations")
async def api_signal_lab_weight_recommendations(market_type: Optional[str] = None, min_samples: Optional[int] = None):
    mt = _market_type_param(market_type) or "SPOT"
    eng = futures_engine if mt == "FUTURES" else engine
    # baseline = pristine class-default weight (never mutated), current =
    # whatever's live right now. Passing BOTH lets the UI show the diff
    # honestly without the recommendation itself depending on current.
    baseline_weights = {a.name: type(a).weight for a in eng.consensus.agents}
    current_weights = {a.name: a.weight for a in eng.consensus.agents}
    return await signal_lab.recommend_weights(mt, min_samples, baseline_weights, current_weights)


class ApplyWeightsBody(BaseModel):
    market_type: str
    agents: Optional[list[str]] = None   # None/empty = apply ALL current recommendations
    min_samples: Optional[int] = None


@app.post("/api/signal-lab/apply-weights")
async def api_signal_lab_apply_weights(body: ApplyWeightsBody):
    mt = _market_type_param(body.market_type) or "SPOT"
    eng = futures_engine if mt == "FUTURES" else engine

    # Recompute recommendations FRESH here rather than trusting whatever
    # weight values the client posts. This closes two problems at once:
    #   1. staleness — the client's cached recommendation could be from
    #      a page load several minutes/market-switches ago
    #   2. tampering/bugs — a client could otherwise submit an arbitrary
    #      {agent: weight} dict with no server-side sanity check at all
    # The client only ever tells us WHICH agent names it wants applied;
    # the actual numbers always come from a fresh server-side calculation
    # anchored to each agent's pristine baseline weight.
    baseline_weights = {a.name: type(a).weight for a in eng.consensus.agents}
    recs = await signal_lab.recommend_weights(mt, body.min_samples, baseline_weights)

    if not recs:
        raise HTTPException(
            status_code=400,
            detail="No weight recommendations available yet — not enough resolved Signal Lab data.",
        )

    selected = set(body.agents) if body.agents else None
    weights_to_apply, samples_map, win_rate_map = {}, {}, {}
    for r in recs:
        if selected is not None and r["agent"] not in selected:
            continue
        weights_to_apply[r["agent"]] = r["suggested_weight"]
        samples_map[r["agent"]] = r["samples"]
        win_rate_map[r["agent"]] = r["win_rate_when_agreed_pct"]

    if not weights_to_apply:
        raise HTTPException(status_code=400, detail="No matching agents to apply weights to.")

    applied = await eng.apply_agent_weights(weights_to_apply, samples_map, win_rate_map)
    return {"ok": True, "market_type": mt, "applied_weights": applied}


@app.get("/api/signal-lab/weight-history")
async def api_signal_lab_weight_history(market_type: Optional[str] = None, limit: int = 100):
    mt = _market_type_param(market_type)
    async with AsyncSessionLocal() as session:
        q = select(AgentWeightHistory).order_by(desc(AgentWeightHistory.applied_at)).limit(limit)
        if mt:
            q = q.where(AgentWeightHistory.market_type == mt)
        result = await session.execute(q)
        rows = result.scalars().all()
    return [
        {
            "id": r.id, "market_type": r.market_type, "agent_name": r.agent_name,
            "baseline_weight": r.baseline_weight, "old_weight": r.old_weight,
            "new_weight": r.new_weight, "samples": r.samples,
            "win_rate_when_agreed_pct": r.win_rate_when_agreed_pct,
            "applied_at": r.applied_at.isoformat() if r.applied_at else None,
        }
        for r in rows
    ]

@app.post("/api/trades/{trade_id}/close")
async def api_close_trade(trade_id: int):
    result = await engine.close_trade_manual(trade_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


@app.post("/api/futures/trades/{trade_id}/close")
async def api_close_futures_trade(trade_id: int):
    result = await futures_engine.close_trade_manual(trade_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


# ═══════════════════════════════════════════════════
#  WebSocket + Dashboard
# ═══════════════════════════════════════════════════
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