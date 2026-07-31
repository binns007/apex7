"""
APEX-7 Trading Engine
═════════════════════
The central orchestrator. Every SCAN_INTERVAL_SECONDS it:
  1. Fetches portfolio value
  2. Checks existing positions for SL/TP hits
  3. Runs the Polyphonic Consensus Engine on each symbol
  4. If consensus is reached → sizes position via Risk Manager
  5. Executes via Order Executor
  6. Persists everything to DB

Changes vs v1:
  - price NaN/None guard before it reaches risk sizing
  - explicit handling when order placement succeeds but OCO exit fails
    (position is now flagged in the log/DB notes instead of silently
    left without protective exits)

Changes vs v2:
  - `pos.risk_usdt` (already computed by RiskManager.size_position) is
    now actually persisted on the Trade row via _save_trade(), so the
    dashboard can show $ at risk per trade instead of just usdt_value.

Changes vs v3 (Signal Lab):
  - Every directional candidate the consensus engine produces — whether
    or not it clears MIN_CONSENSUS_SCORE / MIN_AGENTS_AGREE and becomes
    a real order — is now recorded via signal_lab.record_candidate() and
    tracked forward via signal_lab.resolve_pending(), so "what if we'd
    taken this?" data accumulates for every candidate, not just the ones
    that fired. Real trades are still linked back to their shadow row
    (signal_lab.link_trade()) so the two can be cross-referenced.
"""
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update

from config import settings
from consensus_engine import PolyphonicConsensusEngine
from risk_manager import RiskManager
from order_executor import OrderExecutor
import market_data
from market_data import fetch_all_prices, fetch_price
from database import AsyncSessionLocal, Trade, AgentSignal as DBSignal, PerformanceSnapshot
import signal_lab
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
            ScalpingAgent() if settings.SCALPING_ENABLED else None,
            RegimeAgent(),
        ]
        agents = [a for a in agents if a is not None]
        if not settings.SENTIMENT_ENABLED:
            agents = [a for a in agents if a.name != "Sentiment"]

        self.consensus = PolyphonicConsensusEngine(agents)
        self.risk = RiskManager()
        self.executor = OrderExecutor()

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._scan_count = 0
        self._last_scan: Optional[datetime] = None
        self._status_log: list[str] = []

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

    def apply_agent_weights(self, weights: dict) -> dict:
        """Signal Lab hook: mutate LIVE agent instance `.weight` values
        based on measured hypothetical accuracy. Takes effect on the
        very next scan cycle — no restart needed. Returns the weights
        actually applied (unrecognized agent names are ignored)."""
        applied = {}
        for agent in self.consensus.agents:
            if agent.name in weights:
                try:
                    new_w = float(weights[agent.name])
                except (TypeError, ValueError):
                    continue
                agent.weight = new_w
                applied[agent.name] = new_w
        if applied:
            logger.info(f"⚙ Signal Lab applied agent weights: {applied}")
            self._log(f"⚙ Signal Lab applied agent weights: {applied}")
        return applied

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

        portfolio_usdt = await self.executor.get_usdt_balance()
        self.risk.update_portfolio_value(portfolio_usdt)

        await self._check_open_positions(portfolio_usdt)

        # Signal Lab: resolve shadow "what if" candidates against live
        # prices every cycle, independent of halt state — this is pure
        # observation and should keep running even while real trading
        # is paused, so the data doesn't have a gap right when it's
        # most interesting (post-drawdown, post-halt).
        try:
            await signal_lab.resolve_pending("SPOT", market_data, settings.SIGNAL_LAB_MAX_HOLD_MINUTES)
        except Exception as e:
            logger.debug(f"Signal Lab resolve failed: {e}")

        if self.risk.halted:
            self._log(f"🛑 Trading halted: {self.risk.halt_reason}")
            return

        for symbol in settings.TRADING_PAIRS:
            try:
                await self._evaluate_symbol(symbol, portfolio_usdt)
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                self._log(f"⚠ {symbol} eval error: {e}")

        await self._save_snapshot(portfolio_usdt)

    async def _evaluate_symbol(self, symbol: str, portfolio_usdt: float):
        if symbol in self.risk.open_positions:
            return

        result = await self.consensus.evaluate(symbol)
        await self._save_signals(result.signals)

        # ── Fetch price once, reused for both Signal Lab recording (any
        #    directional candidate) and real trade execution (only when
        #    action != HOLD). ──
        price = None
        if result.candidate_direction or result.action != "HOLD":
            try:
                price = await fetch_price(symbol)
            except Exception as e:
                logger.error(f"Price fetch failed {symbol}: {e}")
                price = None

        shadow_id = None
        if result.candidate_direction and price and not (isinstance(price, float) and math.isnan(price)) and price > 0:
            try:
                shadow_id = await signal_lab.record_candidate(
                    market_type="SPOT", symbol=symbol, price=price,
                    result=result, was_taken=(result.action != "HOLD"),
                )
            except Exception as e:
                logger.debug(f"Signal Lab record failed {symbol}: {e}")

        if result.action == "HOLD":
            logger.debug(f"{symbol}: HOLD — {result.primary_reason}")
            return

        if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
            logger.error(f"Invalid price for {symbol}: {price}")
            self._log(f"⚠ {symbol} invalid price, skipping")
            return

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

        order = await self.executor.place_market_order(symbol, result.action, pos.quantity)
        if not order:
            self._log(f"❌ {symbol} order execution failed")
            return

        exit_side = "SELL" if result.action == "BUY" else "BUY"
        sl_buffer = 0.998 if result.action == "BUY" else 1.002
        oco = await self.executor.place_oco_exit(
            symbol=symbol,
            side=exit_side,
            quantity=pos.quantity,
            take_profit_price=pos.take_profit_price,
            stop_price=pos.stop_loss_price,
            stop_limit_price=pos.stop_loss_price * sl_buffer,
        )
        exit_note = "" if oco else " (⚠ OCO exit failed — position has NO automatic TP/SL, monitor manually)"
        if not oco:
            self._log(f"⚠ {symbol} entered but OCO exit failed — manual monitoring required")

        self.risk.on_trade_open(symbol, pos.risk_usdt)

        trade_id = await self._save_trade(
            symbol=symbol,
            side=result.action,
            entry_price=price,
            quantity=pos.quantity,
            usdt_value=pos.usdt_value,
            risk_usdt=pos.risk_usdt,
            stop_loss=pos.stop_loss_price,
            take_profit=pos.take_profit_price,
            consensus_score=result.score,
            agents_agree=result.agents_agree,
            regime=result.regime,
            binance_order_id=str(order.get("orderId", "")),
            notes=(result.primary_reason + exit_note) if exit_note else result.primary_reason,
        )

        if shadow_id and trade_id:
            try:
                await signal_lab.link_trade(shadow_id, trade_id)
            except Exception as e:
                logger.debug(f"Signal Lab link failed {symbol}: {e}")

        msg = (f"✅ {result.action} {symbol} @ {price:.4f} "
               f"qty={pos.quantity:.6f} TP={pos.take_profit_price:.4f} "
               f"SL={pos.stop_loss_price:.4f} score={result.score:.3f}{exit_note}")
        logger.info(msg)
        self._log(msg)

    # ─────────────────────────────────────────
    #  Position monitoring (basic SL/TP poll)
    # ─────────────────────────────────────────
    async def _check_open_positions(self, portfolio_usdt: float):
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
                pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
                if trade.side == "SELL":
                    pnl_pct = -pnl_pct
                pnl_usdt = trade.usdt_value * pnl_pct / 100
                label = "TP" if hit_tp else "SL"

                await self._close_trade(trade, price, pnl_usdt, pnl_pct)
                self.risk.on_trade_close(trade.symbol, pnl_usdt, hit_tp)
                self._log(
                    f"{'🎯' if hit_tp else '🔴'} {label} hit {trade.symbol} "
                    f"PnL={pnl_pct:+.2f}% (${pnl_usdt:+.2f})"
                )

    # ─────────────────────────────────────────
    #  DB helpers
    # ─────────────────────────────────────────
    async def _save_trade(self, **kwargs) -> Optional[int]:
        async with AsyncSessionLocal() as session:
            trade = Trade(
                is_testnet=settings.is_testnet,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
                **kwargs,
            )
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id

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