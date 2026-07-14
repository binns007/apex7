"""
APEX-7 Futures Order Executor
══════════════════════════════
Handles all Binance USDT-M Futures order placement, leverage/margin
sync, and position monitoring. Same defensive patterns as the spot
OrderExecutor (NaN/zero guards before anything reaches the Binance
client, safe step/tick rounding), adapted for futures specifics:

  - Leverage and margin type must be set PER SYMBOL before an order can
    use them — set_leverage()/set_margin_type() are synced lazily
    (once per symbol per process, or whenever the requested leverage
    changes) rather than on every scan cycle.
  - Binance Futures has no single OCO order type like spot. The
    equivalent here is two independent reduce-only conditional orders
    (STOP_MARKET + TAKE_PROFIT_MARKET, both closePosition=True) placed
    right after entry. Whichever the market hits first closes the
    position; the futures trading engine's position-monitor loop is
    responsible for cancelling the other once it detects the position
    is flat.
  - Uses MARK_PRICE as the trigger reference (workingType) for both
    exits, matching how Binance itself evaluates liquidation — this
    avoids stop-hunts on brief last-price wicks that never touched the
    mark price.
"""
import asyncio
import logging
import math
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import settings

logger = logging.getLogger("apex7.futures_executor")


def _make_client() -> Client:
    return Client(
        api_key=settings.active_futures_api_key,
        api_secret=settings.active_futures_api_secret,
        testnet=settings.is_testnet,
    )


class FuturesOrderExecutor:
    def __init__(self):
        self._client: Optional[Client] = None
        self._symbol_info: dict = {}
        self._leverage_synced: dict[str, int] = {}
        self._margin_type_synced: set[str] = set()

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = _make_client()
        return self._client

    def reconnect(self):
        """Force reconnect (e.g., after mode switch testnet <-> live)."""
        self._client = None
        self._symbol_info = {}
        self._leverage_synced = {}
        self._margin_type_synced = set()

    # ─────────────────────────────────────────
    #  Symbol metadata (step size / min qty / tick size)
    # ─────────────────────────────────────────
    def _get_symbol_info(self, symbol: str) -> dict:
        if symbol not in self._symbol_info:
            client = self._get_client()
            info = client.futures_exchange_info()
            match = next((s for s in info.get("symbols", []) if s["symbol"] == symbol), None)
            if match is None:
                raise ValueError(f"Unknown futures symbol on this Binance environment: {symbol}")
            self._symbol_info[symbol] = match
        return self._symbol_info[symbol]

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        if quantity is None or math.isnan(quantity) or quantity <= 0:
            return 0.0
        info = self._get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                if step <= 0:
                    return round(quantity, 6)
                precision = max(int(round(-math.log10(step))), 0)
                return round(math.floor(quantity / step) * step, precision)
        return round(quantity, 6)

    def _round_price(self, symbol: str, price: float) -> float:
        if price is None or math.isnan(price) or price <= 0:
            return 0.0
        info = self._get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
                if tick <= 0:
                    return round(price, 2)
                precision = max(int(round(-math.log10(tick))), 0)
                return round(round(price / tick) * tick, precision)
        return round(price, 2)

    # ─────────────────────────────────────────
    #  Leverage / margin type sync
    # ─────────────────────────────────────────
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._set_leverage_sync, symbol, leverage)

    def _set_leverage_sync(self, symbol: str, leverage: int) -> bool:
        leverage = max(1, min(int(leverage), settings.FUTURES_MAX_LEVERAGE_ALLOWED))
        if self._leverage_synced.get(symbol) == leverage:
            return True
        client = self._get_client()
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            self._leverage_synced[symbol] = leverage
            logger.info(f"Leverage set {symbol} → {leverage}x")
            return True
        except BinanceAPIException as e:
            logger.error(f"Leverage change failed {symbol}: {e}")
            return False

    async def set_margin_type(self, symbol: str, margin_type: Optional[str] = None) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._set_margin_type_sync, symbol, margin_type)

    def _set_margin_type_sync(self, symbol: str, margin_type: Optional[str]) -> bool:
        margin_type = (margin_type or settings.FUTURES_MARGIN_TYPE).upper()
        cache_key = f"{symbol}:{margin_type}"
        if cache_key in self._margin_type_synced:
            return True
        client = self._get_client()
        try:
            client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            self._margin_type_synced.add(cache_key)
            logger.info(f"Margin type set {symbol} → {margin_type}")
            return True
        except BinanceAPIException as e:
            # -4046 = "No need to change margin type" — already set to this value
            if getattr(e, "code", None) == -4046:
                self._margin_type_synced.add(cache_key)
                return True
            logger.error(f"Margin type change failed {symbol}: {e}")
            return False

    # ─────────────────────────────────────────
    #  Place entry market order
    # ─────────────────────────────────────────
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._place_market_sync, symbol, side, quantity)

    def _place_market_sync(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        if quantity is None or math.isnan(quantity) or quantity <= 0:
            logger.error(f"Refusing to place futures order with invalid quantity={quantity} for {symbol}")
            return None
        client = self._get_client()
        try:
            qty = self._round_quantity(symbol, quantity)
        except Exception as e:
            logger.error(f"Futures symbol info lookup failed for {symbol}: {e}")
            return None
        if qty <= 0:
            logger.error(f"Rounded futures quantity is zero for {symbol}")
            return None
        try:
            order = client.futures_create_order(
                symbol=symbol, side=side, type="MARKET", quantity=qty,
            )
            logger.info(f"✅ Futures Market {side} {qty} {symbol} → orderId={order['orderId']}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Futures market order failed {symbol}: {e}")
            return None

    # ─────────────────────────────────────────
    #  Protective exits — STOP_MARKET + TAKE_PROFIT_MARKET
    #  (Futures' answer to spot's single OCO order)
    # ─────────────────────────────────────────
    async def place_protective_exits(
        self, symbol: str, side: str, quantity: float,
        take_profit_price: float, stop_price: float,
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._place_protective_exits_sync, symbol, side, quantity,
            take_profit_price, stop_price,
        )

    def _place_protective_exits_sync(
        self, symbol, side, quantity, take_profit_price, stop_price
    ):
        client = self._get_client()
        exit_side = "SELL" if side == "BUY" else "BUY"
        try:
            tp = self._round_price(symbol, take_profit_price)
            sp = self._round_price(symbol, stop_price)
        except Exception as e:
            logger.error(f"Futures exit rounding failed for {symbol}: {e}")
            return None, None
        if tp <= 0 or sp <= 0:
            logger.error(f"Futures exit aborted — invalid rounded values for {symbol}: tp={tp} sp={sp}")
            return None, None

        tp_order = sl_order = None
        try:
            tp_order = client.futures_create_order(
                symbol=symbol, side=exit_side, type="TAKE_PROFIT_MARKET",
                stopPrice=str(tp), closePosition=True, workingType="MARK_PRICE",
                timeInForce="GTC",
            )
            logger.info(f"✅ Futures TP set {symbol} @ {tp}")
        except BinanceAPIException as e:
            logger.error(f"Futures TP order failed {symbol}: {e}")

        try:
            sl_order = client.futures_create_order(
                symbol=symbol, side=exit_side, type="STOP_MARKET",
                stopPrice=str(sp), closePosition=True, workingType="MARK_PRICE",
                timeInForce="GTC",
            )
            logger.info(f"✅ Futures SL set {symbol} @ {sp}")
        except BinanceAPIException as e:
            logger.error(f"Futures SL order failed {symbol}: {e}")

        return tp_order, sl_order

    # ─────────────────────────────────────────
    #  Cancel / close
    # ─────────────────────────────────────────
    async def cancel_all_orders(self, symbol: str) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cancel_all_sync, symbol)

    def _cancel_all_sync(self, symbol: str) -> bool:
        client = self._get_client()
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"Cancelled all open futures orders for {symbol}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Cancel futures orders failed {symbol}: {e}")
            return False

    async def close_position_market(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        """Flatten a position immediately at market — used for manual
        close / emergency halt. `side` is the ORIGINAL entry side."""
        loop = asyncio.get_event_loop()
        exit_side = "SELL" if side == "BUY" else "BUY"
        return await loop.run_in_executor(None, self._close_position_sync, symbol, exit_side, quantity)

    def _close_position_sync(self, symbol: str, exit_side: str, quantity: float) -> Optional[dict]:
        client = self._get_client()
        try:
            qty = self._round_quantity(symbol, quantity)
            if qty <= 0:
                return None
            order = client.futures_create_order(
                symbol=symbol, side=exit_side, type="MARKET",
                quantity=qty, reduceOnly=True,
            )
            logger.info(f"✅ Futures position closed {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Futures close position failed {symbol}: {e}")
            return None

    # ─────────────────────────────────────────
    #  Account / position info
    # ─────────────────────────────────────────
    async def get_usdt_balance(self) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_balance_sync)

    def _get_balance_sync(self) -> float:
        try:
            client = self._get_client()
            balances = client.futures_account_balance()
            for asset in balances:
                if asset["asset"] == "USDT":
                    return float(asset.get("availableBalance", asset.get("balance", 0.0)))
        except Exception as e:
            logger.error(f"Futures balance fetch failed: {e}")
        return 0.0

    async def get_open_orders(self, symbol: str) -> list[dict]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._get_client().futures_get_open_orders(symbol=symbol)
            )
        except Exception:
            return []

    async def get_position_risk(self, symbol: str) -> list[dict]:
        """Live position info straight from the exchange — entryPrice,
        markPrice, liquidationPrice, unRealizedProfit. Useful to
        reconcile against our locally-estimated liquidation price."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._get_client().futures_position_information(symbol=symbol)
            )
        except Exception as e:
            logger.error(f"Futures position risk fetch failed {symbol}: {e}")
            return []

    async def ping(self) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._get_client().futures_ping)
            return True
        except Exception:
            return False