"""
APEX-8 Futures Trading Engine
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

Changes vs v1 (Signal Lab):
  - Every directional candidate the (futures-tuned) consensus engine
    produces — whether or not it clears the futures thresholds and
    becomes a real leveraged order — is recorded via
    signal_lab.record_candidate() and tracked forward via
    signal_lab.resolve_pending(), using its own "FUTURES" market_type
    partition and its own (shorter) max-hold window, independent of the
    spot engine's shadow tracking.

Changes vs v2 (real-trade fee adjustment):
  - Closing a position now computes fee_usdt/net_pnl_usdt/net_pnl_pct
    alongside the existing GROSS pnl_usdt/pnl_pct and persists all of
    it. Futures fees are charged on NOTIONAL value traded (usdt_value),
    not margin — leverage changes how much margin backs a position, not
    how big a cut the exchange takes.
  - IMPORTANT BEHAVIOR CHANGE: RiskManager.on_trade_close() now receives
    the NET pnl and a NET-based win/loss flag instead of GROSS, same
    reasoning as the spot engine (see trading_engine.py's v4 note) —
    Kelly sizing and drawdown/heat tracking should reflect what the
    account actually keeps after fees.

Changes vs v3 (balance display fix):
  - get_status() previously fetched portfolio_usdt every scan cycle
    (via executor.get_usdt_balance()) but only ever used it to feed the
    risk manager / position sizing — it was never surfaced in the dict
    returned by get_status(), so the dashboard's "Futures Margin
    Balance" stat card had nothing to read and always showed "—". The
    scan loop now caches the last-fetched balance on
    `_last_balance_usdt`, and get_status() includes it as
    `balance_usdt` so the WebSocket status payload carries it to the
    frontend on every tick — mirrors the identical fix in
    trading_engine.py.

Changes vs v4 (manual-close / already-flat-position fix):
  - PROBLEM THIS FIXES: futures protective exits are TWO INDEPENDENT
    reduce-only orders (STOP_MARKET + TAKE_PROFIT_MARKET), not a real
    OCO. If one of them fills on the exchange (evaluated against MARK
    PRICE, in real time), the position goes flat on Binance immediately
    — but the local DB row still says status="OPEN" until the next
    `_check_open_positions()` poll notices the last-traded price has
    crossed the stored TP/SL level. In that window, a manual "Close"
    click would: cancel the now-stale resting order (succeeds, since it
    still exists as an order object even though the position is gone),
    then try to place a NEW reduceOnly MARKET order against a position
    that's already zero — which Binance correctly rejects with -2022
    ReduceOnly Order is rejected, because reducing a flat position isn't
    a valid reduce at all.
  - FIX: `close_trade_manual()` now calls `get_position_risk()` (already
    implemented on FuturesOrderExecutor, just previously unused here)
    to read the REAL position size on the exchange before attempting to
    close anything. If it's already flat, we reconcile the local Trade
    row against the current price instead of firing another order that
    would only ever be rejected. If a position genuinely remains open,
    we close using `min(trade.quantity, live position size)` so a
    partial-fill/rounding mismatch between our local qty and the
    exchange's can't trigger the same rejection.
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
import signal_lab
from agents import (
    MomentumAgent, MeanReversionAgent, BreakoutAgent,
    VolumeAgent, SentimentAgent, OrderBookAgent,
    ScalpingAgent, RegimeAgent,
)
from database import AsyncSessionLocal, FuturesTrade, FuturesAgentSignal, FuturesPerformanceSnapshot, AgentWeightHistory
logger = logging.getLogger("apex8.futures_engine")


class FuturesTradingEngine:
    MARKET_TYPE = "FUTURES"
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
        # Last futures margin balance fetched during a scan cycle —
        # cached here purely so get_status() has something to report
        # between scans (mirrors the identical field in TradingEngine).
        self._last_balance_usdt: float = 0.0

    async def close_trade_manual(self, trade_id: int) -> dict:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(FuturesTrade).where(FuturesTrade.id == trade_id))
            trade = result.scalar_one_or_none()

        if not trade:
            return {"ok": False, "error": "Trade not found"}
        if trade.status != "OPEN":
            return {"ok": False, "error": f"Trade is already {trade.status}"}

        await self.executor.cancel_all_orders(trade.symbol)

        # ── Reconcile against the REAL exchange position before trying
        #    to reduce it. Protective exits are two independent
        #    reduce-only orders (no true OCO on Futures) evaluated
        #    against mark price in real time by Binance — one of them
        #    can fill and flatten the position before our own
        #    _check_open_positions() poll notices via last price. In
        #    that window a naive close attempt would fire a NEW
        #    reduceOnly order against an already-flat position, which
        #    Binance rejects with -2022. ──
        live_amt = 0.0
        try:
            positions = await self.executor.get_position_risk(trade.symbol)
            if positions:
                live_amt = float(positions[0].get("positionAmt", 0.0))
        except Exception as e:
            logger.warning(f"⚠ Futures position risk check failed for {trade.symbol}: {type(e).__name__}: {e}")

        if abs(live_amt) < 1e-9:
            # Already flat on the exchange — TP or SL almost certainly
            # already filled. Reconcile the local record instead of
            # firing another order that would only ever be rejected.
            price = await fmd.fetch_price(trade.symbol)
            pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
            if trade.side == "SELL":
                pnl_pct = -pnl_pct
            leveraged_pct = pnl_pct * (trade.leverage or 1)
            pnl_usdt = (trade.margin_usdt or 0.0) * leveraged_pct / 100
            fee_usdt = round((trade.usdt_value or 0.0) * (settings.FUTURES_TAKER_FEE_PCT * 2 / 100), 4)
            net_pnl_usdt = round(pnl_usdt - fee_usdt, 4)
            net_pnl_pct = round(net_pnl_usdt / trade.margin_usdt * 100, 4) if trade.margin_usdt else leveraged_pct

            await self._close_trade(trade, price, pnl_usdt, leveraged_pct, False, fee_usdt, net_pnl_usdt, net_pnl_pct)
            self.risk.on_trade_close(trade.symbol, net_pnl_usdt, net_pnl_usdt > 0)

            msg = (f"ℹ️ {trade.symbol} was already flat on the exchange (TP/SL filled before manual "
                   f"close request arrived) — reconciled locally @ {price:.4f} "
                   f"PnL={leveraged_pct:+.2f}% (${pnl_usdt:+.2f} gross / ${net_pnl_usdt:+.2f} net)")
            logger.info(msg)
            self._log(msg)
            return {"ok": True, "trade_id": trade_id, "exit_price": price,
                    "pnl_usdt": round(pnl_usdt, 4), "net_pnl_usdt": net_pnl_usdt,
                    "note": "Position was already closed on the exchange (TP/SL filled) — reconciled, no new order placed."}

        # Genuinely still open — close using the SMALLER of our locally
        # recorded quantity and the live exchange size, so a rounding or
        # partial-fill mismatch can't itself trigger another -2022.
        close_qty = min(trade.quantity, abs(live_amt))

        order = await self.executor.close_position_market(trade.symbol, trade.side, close_qty)
        if not order:
            return {"ok": False, "error": "Market close order failed — check logs / exchange"}

        price = await fmd.fetch_price(trade.symbol)
        pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
        if trade.side == "SELL":
            pnl_pct = -pnl_pct
        leveraged_pct = pnl_pct * (trade.leverage or 1)
        pnl_usdt = (trade.margin_usdt or 0.0) * leveraged_pct / 100
        fee_usdt = round((trade.usdt_value or 0.0) * (settings.FUTURES_TAKER_FEE_PCT * 2 / 100), 4)
        net_pnl_usdt = round(pnl_usdt - fee_usdt, 4)
        net_pnl_pct = round(net_pnl_usdt / trade.margin_usdt * 100, 4) if trade.margin_usdt else leveraged_pct

        await self._close_trade(trade, price, pnl_usdt, leveraged_pct, False, fee_usdt, net_pnl_usdt, net_pnl_pct)
        self.risk.on_trade_close(trade.symbol, net_pnl_usdt, net_pnl_usdt > 0)

        msg = (f"✋ MANUAL close {trade.symbol} @ {price:.4f} "
            f"PnL={leveraged_pct:+.2f}% (${pnl_usdt:+.2f} gross / ${net_pnl_usdt:+.2f} net)")
        logger.info(msg)
        self._log(msg)
        return {"ok": True, "trade_id": trade_id, "exit_price": price,
                "pnl_usdt": round(pnl_usdt, 4), "net_pnl_usdt": net_pnl_usdt}

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
            f"🚀 APEX-8 FUTURES Engine STARTED — mode={settings.TRADING_MODE.upper()} "
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
        logger.info("🛑 APEX-8 FUTURES Engine STOPPED")
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

    async def apply_agent_weights(
            self, weights: dict,
            samples_map: Optional[dict] = None,
            win_rate_map: Optional[dict] = None,
        ) -> dict:
            """Signal Lab hook: mutate LIVE agent instance `.weight` values
            for the FUTURES engine's own agent instances (independent from
            the spot engine's instances), and persist an audit row per
            change to AgentWeightHistory. Takes effect on the very next scan
            cycle — no restart needed. `type(agent).weight` (the pristine
            class-level default) is captured as `baseline_weight` — this
            never changes no matter how many times weights are applied,
            since `agent.weight = new_w` sets an INSTANCE attribute that
            shadows the class attribute rather than mutating it. Pass a
            subset of `weights` to change only specific agents rather than
            the whole recommended batch at once."""
            applied = {}
            history_rows = []
            for agent in self.consensus.agents:
                if agent.name not in weights:
                    continue
                try:
                    new_w = float(weights[agent.name])
                except (TypeError, ValueError):
                    continue
                old_w = agent.weight
                baseline_w = type(agent).weight
                agent.weight = new_w
                applied[agent.name] = new_w
                history_rows.append(AgentWeightHistory(
                    market_type=self.MARKET_TYPE,
                    agent_name=agent.name,
                    baseline_weight=baseline_w,
                    old_weight=old_w,
                    new_weight=new_w,
                    samples=(samples_map or {}).get(agent.name),
                    win_rate_when_agreed_pct=(win_rate_map or {}).get(agent.name),
                ))
            if applied:
                logger.info(f"⚙ Signal Lab applied futures agent weights: {applied}")
                self._log(f"⚙ Signal Lab applied agent weights: {applied}")
                try:
                    async with AsyncSessionLocal() as session:
                        session.add_all(history_rows)
                        await session.commit()
                except Exception as e:
                    logger.warning(f"⚠ Failed to persist weight-change audit rows: {type(e).__name__}: {e}")
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
        self._last_balance_usdt = portfolio_usdt

        await self._check_open_positions()

        # Signal Lab: resolve shadow "what if" candidates against live
        # prices every cycle, independent of halt state (see spot
        # engine's identical comment for the reasoning).
        try:
            await signal_lab.resolve_pending(
                "FUTURES", fmd, settings.FUTURES_SIGNAL_LAB_MAX_HOLD_MINUTES
            )
        except Exception as e:
            # See trading_engine.py's identical comment — resolve_pending
            # already logs its own failures at WARNING; this is a
            # last-resort catch for anything that escapes it.
            logger.warning(f"⚠ Signal Lab resolve failed: {type(e).__name__}: {e}")

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

        # ── Fetch price once, reused for both Signal Lab recording (any
        #    directional candidate) and real trade execution (only when
        #    action != HOLD). ──
        price = None
        if result.candidate_direction or result.action != "HOLD":
            try:
                price = await fmd.fetch_price(symbol)
            except Exception as e:
                logger.error(f"Futures price fetch failed {symbol}: {e}")
                price = None

        shadow_id = None
        if result.candidate_direction and price and not (isinstance(price, float) and math.isnan(price)) and price > 0:
            try:
                shadow_id = await signal_lab.record_candidate(
                    market_type="FUTURES", symbol=symbol, price=price,
                    result=result, was_taken=(result.action != "HOLD"),
                )
            except Exception as e:
                logger.warning(f"⚠ Signal Lab record failed {symbol}: {type(e).__name__}: {e}")

        if result.action == "HOLD":
            logger.debug(f"{symbol}: HOLD — {result.primary_reason}")
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

        trade_id = await self._save_trade(
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

        if shadow_id and trade_id:
            try:
                await signal_lab.link_trade(shadow_id, trade_id)
            except Exception as e:
                logger.warning(f"⚠ Signal Lab link failed {symbol}: {type(e).__name__}: {e}")

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

                # Fee-adjusted NET figures. Futures fees are charged on
                # NOTIONAL value traded (trade.usdt_value), not margin —
                # leverage changes how much margin backs a position, not
                # how big a slice of it the exchange takes a cut of.
                fee_usdt = round((trade.usdt_value or 0.0) * (settings.FUTURES_TAKER_FEE_PCT * 2 / 100), 4)
                net_pnl_usdt = round(pnl_usdt - fee_usdt, 4)
                net_pnl_pct = (
                    round(net_pnl_usdt / trade.margin_usdt * 100, 4) if trade.margin_usdt else leveraged_pnl_pct
                )

                await self._close_trade(trade, price, pnl_usdt, leveraged_pnl_pct, hit_liq,
                                         fee_usdt, net_pnl_usdt, net_pnl_pct)
                # Risk tracking now based on NET pnl/win-flag — see this
                # file's v2 changelog note at the top for why.
                self.risk.on_trade_close(trade.symbol, net_pnl_usdt, (net_pnl_usdt > 0) and not hit_liq)
                self._log(
                    f"{'💥' if hit_liq else ('🎯' if hit_tp else '🔴')} {label} {trade.symbol} "
                    f"PnL={leveraged_pnl_pct:+.2f}% (${pnl_usdt:+.2f} gross / ${net_pnl_usdt:+.2f} net)"
                )

    # ─────────────────────────────────────────
    #  DB helpers
    # ─────────────────────────────────────────
    async def _save_trade(self, **kwargs) -> Optional[int]:
        async with AsyncSessionLocal() as session:
            trade = FuturesTrade(
                is_testnet=settings.is_testnet,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
                **kwargs,
            )
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id

    async def _close_trade(self, trade: FuturesTrade, exit_price: float,
                            pnl_usdt: float, pnl_pct: float, was_liquidation: bool,
                            fee_usdt: Optional[float] = None,
                            net_pnl_usdt: Optional[float] = None,
                            net_pnl_pct: Optional[float] = None):
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(FuturesTrade)
                .where(FuturesTrade.id == trade.id)
                .values(
                    status="LIQUIDATED" if was_liquidation else "CLOSED",
                    exit_price=exit_price,
                    pnl_usdt=pnl_usdt,
                    pnl_pct=pnl_pct,
                    fee_usdt=fee_usdt,
                    net_pnl_usdt=net_pnl_usdt,
                    net_pnl_pct=net_pnl_pct,
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
            "balance_usdt": self._last_balance_usdt,
            "risk": self.risk.summary(),
            "pairs": settings.FUTURES_TRADING_PAIRS,
            "log": list(reversed(self._status_log[-50:])),
        }


# Singleton
futures_engine = FuturesTradingEngine()