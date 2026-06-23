"""
APEX-7 Trading Engine
═════════════════════
The central orchestrator. Every SCAN_INTERVAL_SECONDS it:
  1. Fetches portfolio value
  2. Checks existing positions for SL/TP hits
  3. Runs the Polyphonic Consensus Engine on each symbol
  4. If consensus ≥ threshold → sizes position via Risk Manager
  5. Executes via Order Executor
  6. Persists everything to DB

This is the "brain stem" connecting all modules.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update

from config import settings
from consensus_engine import PolyphonicConsensusEngine
from risk_manager import RiskManager
from order_executor import OrderExecutor
from market_data import fetch_all_prices, fetch_price
from database import AsyncSessionLocal, Trade, AgentSignal as DBSignal, PerformanceSnapshot
from agents import (
    MomentumAgent, MeanReversionAgent, BreakoutAgent,
    VolumeAgent, SentimentAgent, OrderBookAgent,
    ScalpingAgent, RegimeAgent,
)

logger = logging.getLogger("apex7.engine")


class TradingEngine:
    def __init__(self):
        agents = [
            MomentumAgent(),
            MeanReversionAgent(),
            BreakoutAgent(),
            VolumeAgent(),
            SentimentAgent(),
            OrderBookAgent(),
            ScalpingAgent(),
            RegimeAgent(),
        ]
        self.consensus = PolyphonicConsensusEngine(agents)
        self.risk      = RiskManager()
        self.executor  = OrderExecutor()

        self._running  = False
        self._task: Optional[asyncio.Task] = None
        self._scan_count = 0
        self._last_scan: Optional[datetime] = None
        self._status_log: list[str] = []    # ring buffer for UI

    # ─────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"🚀 APEX-7 Engine STARTED — mode={settings.TRADING_MODE.upper()}")
        self._log(f"Engine started in {settings.TRADING_MODE.upper()} mode")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("🛑 APEX-7 Engine STOPPED")
        self._log("Engine stopped by user")

    @property
    def is_running(self) -> bool:
        return self._running

    # ─────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────
    async def _loop(self):
        while self._running:
            try:
                await self._scan_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                self._log(f"⚠ Loop error: {e}")
            await asyncio.sleep(settings.SCAN_INTERVAL_SECONDS)

    async def _scan_cycle(self):
        self._scan_count += 1
        self._last_scan = datetime.now(timezone.utc)
        logger.info(f"── Scan #{self._scan_count} ──────────────────")
        self._log(f"Scan #{self._scan_count} started")

        # 1. Portfolio value
        portfolio_usdt = await self.executor.get_usdt_balance()
        self.risk.update_portfolio_value(portfolio_usdt)

        # 2. Check existing open positions
        await self._check_open_positions(portfolio_usdt)

        # Halted? Skip new trades
        if self.risk.halted:
            self._log(f"🛑 Trading halted: {self.risk.halt_reason}")
            return

        # 3. Evaluate each symbol
        for symbol in settings.TRADING_PAIRS:
            try:
                await self._evaluate_symbol(symbol, portfolio_usdt)
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                self._log(f"⚠ {symbol} eval error: {e}")

        # 4. Save performance snapshot
        await self._save_snapshot(portfolio_usdt)

    async def _evaluate_symbol(self, symbol: str, portfolio_usdt: float):
        # Skip if already in a position for this symbol
        if symbol in self.risk.open_positions:
            return

        result = await self.consensus.evaluate(symbol)

        # Persist signals
        await self._save_signals(result.signals)

        if result.action == "HOLD":
            logger.debug(f"{symbol}: HOLD — {result.primary_reason}")
            return

        # Get current price
        try:
            price = await fetch_price(symbol)
        except Exception as e:
            logger.error(f"Price fetch failed {symbol}: {e}")
            return

        # Size position
        pos = self.risk.size_position(
            symbol=symbol,
            side=result.action,
            entry_price=price,
            stop_loss_pct=result.stop_loss_pct,
            take_profit_pct=result.take_profit_pct,
            portfolio_usdt=portfolio_usdt,
            consensus_score=result.score,
        )

        if not pos.approved:
            self._log(f"⚠ {symbol} rejected: {pos.rejection_reason}")
            return

        # Execute
        order = await self.executor.place_market_order(symbol, result.action, pos.quantity)
        if not order:
            self._log(f"❌ {symbol} order execution failed")
            return

        # Place exit OCO
        exit_side = "SELL" if result.action == "BUY" else "BUY"
        sl_buffer = 0.998 if result.action == "BUY" else 1.002
        await self.executor.place_oco_exit(
            symbol=symbol,
            side=exit_side,
            quantity=pos.quantity,
            take_profit_price=pos.take_profit_price,
            stop_price=pos.stop_loss_price,
            stop_limit_price=pos.stop_loss_price * sl_buffer,
        )

        # Track in risk manager
        self.risk.on_trade_open(symbol, pos.risk_usdt)

        # Persist to DB
        await self._save_trade(
            symbol=symbol,
            side=result.action,
            entry_price=price,
            quantity=pos.quantity,
            usdt_value=pos.usdt_value,
            stop_loss=pos.stop_loss_price,
            take_profit=pos.take_profit_price,
            consensus_score=result.score,
            agents_agree=result.agents_agree,
            regime=result.regime,
            order_id=str(order.get("orderId", "")),
        )

        msg = (f"✅ {result.action} {symbol} @ {price:.4f} "
               f"qty={pos.quantity:.6f} TP={pos.take_profit_price:.4f} "
               f"SL={pos.stop_loss_price:.4f} score={result.score:.3f}")
        logger.info(msg)
        self._log(msg)

    # ─────────────────────────────────────────
    #  Position monitoring (basic SL/TP poll)
    # ─────────────────────────────────────────
    async def _check_open_positions(self, portfolio_usdt: float):
        """Poll open DB trades and check if SL/TP has been hit."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == "OPEN")
            )
            trades = result.scalars().all()

        if not trades:
            return

        prices = await fetch_all_prices([t.symbol for t in trades])

        for trade in trades:
            price = prices.get(trade.symbol)
            if not price:
                continue

            hit_tp = hit_sl = False
            if trade.side == "BUY":
                hit_tp = price >= trade.take_profit
                hit_sl = price <= trade.stop_loss
            else:
                hit_tp = price <= trade.take_profit
                hit_sl = price >= trade.stop_loss

            if hit_tp or hit_sl:
                pnl_pct  = (price - trade.entry_price) / trade.entry_price * 100
                if trade.side == "SELL":
                    pnl_pct = -pnl_pct
                pnl_usdt = trade.usdt_value * pnl_pct / 100
                label    = "TP" if hit_tp else "SL"

                await self._close_trade(trade, price, pnl_usdt, pnl_pct)
                self.risk.on_trade_close(trade.symbol, pnl_usdt, hit_tp)
                self._log(
                    f"{'🎯' if hit_tp else '🔴'} {label} hit {trade.symbol} "
                    f"PnL={pnl_pct:+.2f}% (${pnl_usdt:+.2f})"
                )

    # ─────────────────────────────────────────
    #  DB helpers
    # ─────────────────────────────────────────
    async def _save_trade(self, **kwargs):
        async with AsyncSessionLocal() as session:
            trade = Trade(
                is_testnet=settings.is_testnet,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
                **kwargs,
            )
            session.add(trade)
            await session.commit()

    async def _close_trade(self, trade: Trade, exit_price: float,
                            pnl_usdt: float, pnl_pct: float):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Trade)
                .where(Trade.id == trade.id)
                .values(
                    status="CLOSED",
                    exit_price=exit_price,
                    pnl_usdt=pnl_usdt,
                    pnl_pct=pnl_pct,
                    closed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    async def _save_signals(self, signals):
        if not signals:
            return
        async with AsyncSessionLocal() as session:
            for s in signals:
                session.add(DBSignal(
                    symbol=s.symbol, timeframe=s.timeframe,
                    agent_name=s.agent_name, signal=s.signal,
                    confidence=s.confidence, reason=s.reason,
                ))
            await session.commit()

    async def _save_snapshot(self, portfolio_value: float):
        rm = self.risk.summary()
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    select(Trade).where(Trade.status == "CLOSED")
                )
                closed = result.scalars().all()
                total_pnl = sum(t.pnl_usdt or 0 for t in closed)
                wins = sum(1 for t in closed if (t.pnl_usdt or 0) > 0)
                session.add(PerformanceSnapshot(
                    total_trades=len(closed),
                    winning_trades=wins,
                    total_pnl_usdt=total_pnl,
                    win_rate=wins / len(closed) * 100 if closed else 0,
                    max_drawdown=rm["drawdown_pct"],
                    portfolio_value=portfolio_value,
                ))

    # ─────────────────────────────────────────
    #  Status for UI
    # ─────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self._status_log.append(entry)
        if len(self._status_log) > 200:
            self._status_log = self._status_log[-200:]

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "mode": settings.TRADING_MODE,
            "scan_count": self._scan_count,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "risk": self.risk.summary(),
            "pairs": settings.TRADING_PAIRS,
            "log": list(reversed(self._status_log[-50:])),
        }


# Singleton
engine = TradingEngine()
