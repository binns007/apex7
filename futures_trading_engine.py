"""
APEX-7 Futures Trading Engine
══════════════════════════════
Same orchestration pattern as the spot TradingEngine (fetch balance →
check open positions → run consensus per symbol → size → execute →
persist → snapshot), pointed at Binance USDT-M Futures instead of
spot, and tuned for the "quick, small, frequent" leveraged trade style:

  - Faster scan cadence (FUTURES_SCAN_INTERVAL_SECONDS, default 10s
    vs spot's 30s)
  - Micro timeframe stack (1m/3m/5m) via a dedicated
    PolyphonicConsensusEngine instance — same class as spot, different
    config (see consensus_engine.py's parameterization)
  - Leverage is chosen by the user at engine-start time (the dashboard's
    "Futures Mode" prompt) and applied per-symbol before each new entry
    via set_leverage()/set_margin_type() — changing it mid-run only
    affects NEW entries, not positions already open
  - Protective exits are two independent reduce-only orders
    (STOP_MARKET + TAKE_PROFIT_MARKET) since Futures has no single OCO
    primitive — whichever fills first, the position-monitor loop
    detects the resulting flat position and cancels the other via
    cancel_all_orders()

Hedge mode (independent long+short books per symbol) is NOT supported —
one open position per symbol, one-way mode, matching FUTURES_POSITION_MODE.
"""
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update

from config import settings
from consensus_engine import PolyphonicConsensusEngine
from futures_risk_manager import FuturesRiskManager
from futures_order_executor import FuturesOrderExecutor
import futures_market_data as fmd
from database import AsyncSessionLocal, FuturesTrade, FuturesAgentSignal, FuturesPerformanceSnapshot
from agents import (
    MomentumAgent, MeanReversionAgent, BreakoutAgent,
    VolumeAgent, SentimentAgent, OrderBookAgent,
    ScalpingAgent, RegimeAgent,
)

logger = logging.getLogger("apex7.futures_engine")


class FuturesTradingEngine:
    def __init__(self):
        # A DEDICATED RegimeAgent instance running on the futures
        # timeframe stack's slowest timeframe (5m), with its own
        # regime-state dict — completely independent from the spot
        # engine's RegimeAgent instance even when trading the same
        # symbol at the same time (see agents/regime_agent.py v3).
        regime = RegimeAgent(
            timeframe=settings.FUTURES_REGIME_TIMEFRAME,
            trend_adx=settings.FUTURES_REGIME_TREND_ADX,
            volatile_atr_pct=settings.FUTURES_REGIME_VOLATILE_ATR_PCT,
            volatile_bb_width=settings.FUTURES_REGIME_VOLATILE_BB_WIDTH,
        )
        agents = [
            MomentumAgent(),
            MeanReversionAgent(),
            BreakoutAgent(),
            VolumeAgent(),
            SentimentAgent(),
            OrderBookAgent(),
            ScalpingAgent() if settings.SCALPING_ENABLED else None,
            regime,
        ]
        agents = [a for a in agents if a is not None]
        if not settings.SENTIMENT_ENABLED:
            agents = [a for a in agents if a.name != "Sentiment"]

        # Same PolyphonicConsensusEngine class as spot — just pointed at
        # futures market data and configured with the FUTURES_* dials.
        self.consensus = PolyphonicConsensusEngine(
            agents,
            market_data_provider=fmd,
            timeframes=settings.FUTURES_TIMEFRAMES,
            primary_timeframe=settings.FUTURES_PRIMARY_TIMEFRAME,
            min_consensus_score=settings.FUTURES_MIN_CONSENSUS_SCORE,
            min_agents_agree=settings.FUTURES_MIN_AGENTS_AGREE,
            tie_break_action=settings.FUTURES_TIE_BREAK_ACTION,
            require_temporal_confluence=settings.FUTURES_REQUIRE_TEMPORAL_CONFLUENCE,
            confluence_score_bonus=settings.FUTURES_CONFLUENCE_SCORE_BONUS,
            no_confluence_score_penalty=settings.FUTURES_NO_CONFLUENCE_SCORE_PENALTY,
            regime_hard_block_volatile=settings.FUTURES_REGIME_HARD_BLOCK_VOLATILE,
            regime_volatile_score_multiplier=settings.FUTURES_REGIME_VOLATILE_SCORE_MULTIPLIER,
            candle_limit=settings.FUTURES_CANDLE_LIMIT,
            sl_atr_mult=settings.FUTURES_SL_ATR_MULT,
            sl_min_pct=settings.FUTURES_SL_MIN_PCT,
            sl_max_pct=settings.FUTURES_SL_MAX_PCT,
            rr_trend=settings.FUTURES_RR_TREND,
            rr_range=settings.FUTURES_RR_RANGE,
        )
        self.risk = FuturesRiskManager()
        self.executor = FuturesOrderExecutor()

        # Chosen at engine-start via the dashboard's leverage prompt;
        # falls back to config defaults if start() is called with none.
        self.leverage: int = settings.FUTURES_DEFAULT_LEVERAGE
        self.margin_type: str = settings.FUTURES_MARGIN_TYPE

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._scan_count = 0
        self._last_scan: Optional[datetime] = None
        self._status_log: list[str] = []

    # ─────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────
    def start(self, leverage: Optional[int] = None, margin_type: Optional[str] = None):
        if self._running:
            return
        if leverage is not None:
            self.leverage = max(1, min(int(leverage), settings.FUTURES_MAX_LEVERAGE_ALLOWED))
        if margin_type is not None and margin_type.upper() in ("ISOLATED", "CROSSED"):
            self.margin_type = margin_type.upper()

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"🚀 APEX-7 FUTURES Engine STARTED — mode={settings.TRADING_MODE.upper()} "
            f"leverage={self.leverage}x margin={self.margin_type}"
        )
        self._log(
            f"Futures engine started in {settings.TRADING_MODE.upper()} mode "
            f"@ {self.leverage}x {self.margin_type}"
        )

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("🛑 APEX-7 FUTURES Engine STOPPED")
        self._log("Futures engine stopped by user")

    @property
    def is_running(self) -> bool:
        return self._running

    def update_leverage(self, leverage: int):
        """Applies to NEW entries only — open positions keep their
        original leverage until closed."""
        self.leverage = max(1, min(int(leverage), settings.FUTURES_MAX_LEVERAGE_ALLOWED))
        self._log(f"Leverage updated to {self.leverage}x (applies to new entries)")

    def update_margin_type(self, margin_type: str):
        margin_type = margin_type.upper()
        if margin_type in ("ISOLATED", "CROSSED"):
            self.margin_type = margin_type
            self._log(f"Margin type updated to {self.margin_type} (applies to new entries)")

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
                logger.error(f"Futures loop error: {e}", exc_info=True)
                self._log(f"⚠ Loop error: {e}")
            await asyncio.sleep(settings.FUTURES_SCAN_INTERVAL_SECONDS)

    async def _scan_cycle(self):
        self._scan_count += 1
        self._last_scan = datetime.now(timezone.utc)
        logger.info(f"── Futures Scan #{self._scan_count} ──────────────────")
        self._log(f"Scan #{self._scan_count} started")

        portfolio_usdt = await self.executor.get_usdt_balance()
        self.risk.update_portfolio_value(portfolio_usdt)

        await self._check_open_positions()

        if self.risk.halted:
            self._log(f"🛑 Trading halted: {self.risk.halt_reason}")
            return

        for symbol in settings.FUTURES_TRADING_PAIRS:
            try:
                await self._evaluate_symbol(symbol, portfolio_usdt)
            except Exception as e:
                logger.error(f"Error evaluating futures {symbol}: {e}")
                self._log(f"⚠ {symbol} eval error: {e}")

        await self._save_snapshot(portfolio_usdt)

    async def _evaluate_symbol(self, symbol: str, portfolio_usdt: float):
        if symbol in self.risk.open_positions:
            return

        result = await self.consensus.evaluate(symbol)
        await self._save_signals(result.signals)

        if result.action == "HOLD":
            logger.debug(f"{symbol}: HOLD — {result.primary_reason}")
            return

        try:
            price = await fmd.fetch_price(symbol)
        except Exception as e:
            logger.error(f"Futures price fetch failed {symbol}: {e}")
            return

        if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
            logger.error(f"Invalid futures price for {symbol}: {price}")
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
            leverage=self.leverage,
        )

        if not pos.approved:
            self._log(f"⚠ {symbol} rejected: {pos.rejection_reason}")
            return

        # Sync leverage/margin type on the exchange BEFORE entering —
        # abort for safety if we can't confirm the leverage we sized for.
        await self.executor.set_margin_type(symbol, self.margin_type)
        lev_ok = await self.executor.set_leverage(symbol, pos.leverage)
        if not lev_ok:
            self._log(f"⚠ {symbol} leverage sync failed — skipping entry for safety")
            return

        order = await self.executor.place_market_order(symbol, result.action, pos.quantity)
        if not order:
            self._log(f"❌ {symbol} futures order execution failed")
            return

        tp_order, sl_order = await self.executor.place_protective_exits(
            symbol=symbol,
            side=result.action,
            quantity=pos.quantity,
            take_profit_price=pos.take_profit_price,
            stop_price=pos.stop_loss_price,
        )
        exit_note = ""
        if not tp_order or not sl_order:
            exit_note = " (⚠ one or both protective exits failed to place — monitor manually)"
            self._log(f"⚠ {symbol} entered but protective exit placement incomplete")

        self.risk.on_trade_open(symbol, pos.risk_usdt)

        await self._save_trade(
            symbol=symbol,
            side=result.action,
            entry_price=price,
            quantity=pos.quantity,
            usdt_value=pos.notional_usdt,
            margin_usdt=pos.margin_usdt,
            leverage=pos.leverage,
            margin_type=self.margin_type,
            risk_usdt=pos.risk_usdt,
            stop_loss=pos.stop_loss_price,
            take_profit=pos.take_profit_price,
            liquidation_price=pos.liquidation_price,
            consensus_score=result.score,
            agents_agree=result.agents_agree,
            regime=result.regime,
            binance_order_id=str(order.get("orderId", "")),
            notes=(result.primary_reason + exit_note) if exit_note else result.primary_reason,
        )

        msg = (f"✅ {result.action} {symbol} {pos.leverage}x @ {price:.4f} "
               f"qty={pos.quantity:.6f} margin=${pos.margin_usdt:.2f} "
               f"TP={pos.take_profit_price:.4f} SL={pos.stop_loss_price:.4f} "
               f"liq≈{pos.liquidation_price:.4f} score={result.score:.3f}{exit_note}")
        logger.info(msg)
        self._log(msg)

    # ─────────────────────────────────────────
    #  Position monitoring (SL/TP/liquidation poll)
    # ─────────────────────────────────────────
    async def _check_open_positions(self):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(FuturesTrade).where(FuturesTrade.status == "OPEN")
            )
            trades = result.scalars().all()

        if not trades:
            return

        prices = await fmd.fetch_all_prices([t.symbol for t in trades])

        for trade in trades:
            price = prices.get(trade.symbol)
            if not price:
                continue

            hit_tp = hit_sl = hit_liq = False
            if trade.side == "BUY":
                hit_tp = price >= trade.take_profit
                hit_sl = price <= trade.stop_loss
                hit_liq = bool(trade.liquidation_price) and price <= trade.liquidation_price
            else:
                hit_tp = price <= trade.take_profit
                hit_sl = price >= trade.stop_loss
                hit_liq = bool(trade.liquidation_price) and price >= trade.liquidation_price

            if hit_tp or hit_sl or hit_liq:
                pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
                if trade.side == "SELL":
                    pnl_pct = -pnl_pct
                # ROI on margin — the leverage-adjusted return, matching
                # how Binance itself displays futures PnL%.
                leveraged_pnl_pct = pnl_pct * (trade.leverage or 1)
                pnl_usdt = (trade.margin_usdt or 0.0) * leveraged_pnl_pct / 100
                label = "LIQUIDATION" if hit_liq else ("TP" if hit_tp else "SL")

                # Best-effort cleanup: whichever protective order didn't
                # fill is now stale — cancel it so it can't misfire later.
                await self.executor.cancel_all_orders(trade.symbol)

                await self._close_trade(trade, price, pnl_usdt, leveraged_pnl_pct, hit_liq)
                self.risk.on_trade_close(trade.symbol, pnl_usdt, hit_tp and not hit_liq)
                self._log(
                    f"{'💥' if hit_liq else ('🎯' if hit_tp else '🔴')} {label} {trade.symbol} "
                    f"PnL={leveraged_pnl_pct:+.2f}% (${pnl_usdt:+.2f})"
                )

    # ─────────────────────────────────────────
    #  DB helpers
    # ─────────────────────────────────────────
    async def _save_trade(self, **kwargs):
        async with AsyncSessionLocal() as session:
            trade = FuturesTrade(
                is_testnet=settings.is_testnet,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
                **kwargs,
            )
            session.add(trade)
            await session.commit()

    async def _close_trade(self, trade: FuturesTrade, exit_price: float,
                            pnl_usdt: float, pnl_pct: float, was_liquidation: bool):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(FuturesTrade)
                .where(FuturesTrade.id == trade.id)
                .values(
                    status="LIQUIDATED" if was_liquidation else "CLOSED",
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
                session.add(FuturesAgentSignal(
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
                    select(FuturesTrade).where(FuturesTrade.status.in_(["CLOSED", "LIQUIDATED"]))
                )
                closed = result.scalars().all()
                total_pnl = sum(t.pnl_usdt or 0 for t in closed)
                wins = sum(1 for t in closed if (t.pnl_usdt or 0) > 0)
                session.add(FuturesPerformanceSnapshot(
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
            "market_type": "FUTURES",
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "scan_count": self._scan_count,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "risk": self.risk.summary(),
            "pairs": settings.FUTURES_TRADING_PAIRS,
            "log": list(reversed(self._status_log[-50:])),
        }


# Singleton
futures_engine = FuturesTradingEngine()