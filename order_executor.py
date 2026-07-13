"""
APEX-7 Order Executor
═════════════════════
Handles all Binance order placement, modification, and monitoring.
Uses python-binance with testnet support.

Fixes vs v1:
  - _round_quantity/_round_price used math.log10(step) with no guard;
    a step/tick size of 0 (malformed filter data) would raise. Now
    falls back safely instead of crashing the scan cycle.
  - place_market_order now rejects NaN/zero/negative quantities before
    ever reaching the Binance client.
"""
import asyncio
import logging
import math
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import settings

logger = logging.getLogger("apex7.executor")


def _make_client() -> Client:
    if settings.is_testnet:
        return Client(
            api_key=settings.active_api_key,
            api_secret=settings.active_api_secret,
            testnet=True,
        )
    return Client(
        api_key=settings.active_api_key,
        api_secret=settings.active_api_secret,
    )


class OrderExecutor:
    def __init__(self):
        self._client: Optional[Client] = None
        self._symbol_info: dict = {}

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = _make_client()
        return self._client

    def reconnect(self):
        """Force reconnect (e.g., after mode switch)."""
        self._client = None
        self._symbol_info = {}

    # ─────────────────────────────────────────
    #  Symbol metadata (step size / min qty)
    # ─────────────────────────────────────────
    def _get_symbol_info(self, symbol: str) -> dict:
        if symbol not in self._symbol_info:
            client = self._get_client()
            info = client.get_symbol_info(symbol)
            if info is None:
                raise ValueError(f"Unknown symbol on this Binance environment: {symbol}")
            self._symbol_info[symbol] = info
        return self._symbol_info[symbol]

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to Binance's allowed step size."""
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
        """Round price to Binance's tick size."""
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
    #  Place entry market order
    # ─────────────────────────────────────────
    async def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Optional[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._place_market_sync, symbol, side, quantity)

    def _place_market_sync(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        if quantity is None or math.isnan(quantity) or quantity <= 0:
            logger.error(f"Refusing to place order with invalid quantity={quantity} for {symbol}")
            return None
        client = self._get_client()
        try:
            qty = self._round_quantity(symbol, quantity)
        except Exception as e:
            logger.error(f"Symbol info lookup failed for {symbol}: {e}")
            return None
        if qty <= 0:
            logger.error(f"Rounded quantity is zero for {symbol}")
            return None
        try:
            order = client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty,
            )
            logger.info(f"✅ Market {side} {qty} {symbol} → orderId={order['orderId']}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Market order failed {symbol}: {e}")
            return None

    # ─────────────────────────────────────────
    #  Place OCO exit order (TP + SL in one)
    # ─────────────────────────────────────────
    async def place_oco_exit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_price: float,
        stop_limit_price: float,
    ) -> Optional[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._place_oco_sync, symbol, side, quantity,
            take_profit_price, stop_price, stop_limit_price
        )

    def _place_oco_sync(
        self, symbol, side, quantity,
        take_profit_price, stop_price, stop_limit_price
    ) -> Optional[dict]:
        client = self._get_client()
        try:
            qty = self._round_quantity(symbol, quantity)
            tp = self._round_price(symbol, take_profit_price)
            sp = self._round_price(symbol, stop_price)
            sl = self._round_price(symbol, stop_limit_price)
        except Exception as e:
            logger.error(f"OCO rounding failed for {symbol}: {e}")
            return None
        if qty <= 0 or tp <= 0 or sp <= 0 or sl <= 0:
            logger.error(f"OCO aborted — invalid rounded values for {symbol}: "
                         f"qty={qty} tp={tp} sp={sp} sl={sl}")
            return None
        try:
            order = client.create_oco_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=str(tp),
                stopPrice=str(sp),
                stopLimitPrice=str(sl),
                stopLimitTimeInForce="GTC",
            )
            logger.info(f"✅ OCO exit {symbol}: TP={tp} SL={sp}")
            return order
        except BinanceAPIException as e:
            logger.error(f"OCO order failed {symbol}: {e}")
            try:
                order = client.create_order(
                    symbol=symbol, side=side, type="LIMIT",
                    quantity=qty, price=str(tp), timeInForce="GTC"
                )
                logger.info(f"Fallback limit TP placed for {symbol}")
                return order
            except Exception as e2:
                logger.error(f"Fallback limit also failed: {e2}")
                return None

    # ─────────────────────────────────────────
    #  Cancel all open orders for symbol
    # ─────────────────────────────────────────
    async def cancel_all_orders(self, symbol: str) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cancel_all_sync, symbol)

    def _cancel_all_sync(self, symbol: str) -> bool:
        client = self._get_client()
        try:
            client.cancel_open_orders(symbol=symbol)
            logger.info(f"Cancelled all open orders for {symbol}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Cancel orders failed {symbol}: {e}")
            return False

    # ─────────────────────────────────────────
    #  Account balance
    # ─────────────────────────────────────────
    async def get_usdt_balance(self) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_balance_sync)

    def _get_balance_sync(self) -> float:
        try:
            client = self._get_client()
            info = client.get_account()
            for asset in info["balances"]:
                if asset["asset"] == "USDT":
                    return float(asset["free"])
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
        return 0.0

    # ─────────────────────────────────────────
    #  Get open orders
    # ─────────────────────────────────────────
    async def get_open_orders(self, symbol: str) -> list[dict]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._get_client().get_open_orders(symbol=symbol)
            )
        except Exception:
            return []

    # ─────────────────────────────────────────
    #  Connection test
    # ─────────────────────────────────────────
    async def ping(self) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._get_client().ping)
            return True
        except Exception:
            return False