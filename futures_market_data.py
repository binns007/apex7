"""
APEX-7 — Futures Market Data
═══════════════════════════════
Mirrors market_data.py exactly in shape (same function names/signatures
so it's a drop-in `market_data_provider` for PolyphonicConsensusEngine)
but talks to Binance's USDT-M Futures REST API (fapi) instead of spot,
and adds a couple of futures-only reads (mark price / predicted funding)
used by the futures risk manager.

Indicator math (_compute_indicators) is imported straight from
market_data.py rather than duplicated — RSI/MACD/ADX/etc. mean the same
thing on a futures kline as a spot kline, only the data source differs.

Maintains its own HTTP session (independent from spot's) so futures and
spot can be started/stopped/shut down independently without one engine's
lifecycle affecting the other's in-flight requests.
"""
import time
import logging
from typing import Optional
import numpy as np
import pandas as pd
import aiohttp

from config import settings
from market_data import _compute_indicators, fetch_fear_greed  # reused as-is — macro, not market-specific

logger = logging.getLogger("apex7.futures_market_data")

FUTURES_LIVE_BASE = "https://fapi.binance.com"
FUTURES_TEST_BASE = "https://testnet.binancefuture.com"


def _base() -> str:
    return FUTURES_TEST_BASE if settings.is_testnet else FUTURES_LIVE_BASE


# ──────────────────────────────────────────────
#  Caches — same {value/df: ..., ts: float} pattern as market_data.py
# ──────────────────────────────────────────────
_candle_cache: dict[str, dict] = {}
_price_cache: dict[str, dict] = {}
_mark_price_cache: dict[str, dict] = {}
_orderbook_cache: dict[str, dict] = {}

CANDLE_CACHE_TTL = 15     # seconds — shorter than spot's 25s: futures scans faster
PRICE_CACHE_TTL = 3       # seconds — position sizing/liquidation checks need freshness
MARK_PRICE_CACHE_TTL = 3
ORDERBOOK_CACHE_TTL = 4

_http_session: Optional[aiohttp.ClientSession] = None


async def _get_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    return _http_session


async def _get(url: str, params: dict = None) -> dict | list:
    session = await _get_session()
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def shutdown():
    """Call on app shutdown to close the shared futures HTTP session."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()


# ──────────────────────────────────────────────
#  Candle Fetching
# ──────────────────────────────────────────────
async def fetch_candles(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Return OHLCV DataFrame with indicators pre-computed, from the
    Futures klines endpoint. Same DataFrame shape as spot's fetch_candles
    so every agent works unmodified against either source."""
    cache_key = f"{symbol}_{interval}"
    cached = _candle_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CANDLE_CACHE_TTL:
        return cached["df"]

    url = f"{_base()}/fapi/v1/klines"
    raw = await _get(url, {"symbol": symbol, "interval": interval, "limit": limit})

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    df = _compute_indicators(df)
    _candle_cache[cache_key] = {"df": df, "ts": time.time()}
    return df


# ──────────────────────────────────────────────
#  Live Price (last traded price)
# ──────────────────────────────────────────────
async def fetch_price(symbol: str) -> float:
    cached = _price_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < PRICE_CACHE_TTL:
        return cached["price"]
    url = f"{_base()}/fapi/v1/ticker/price"
    data = await _get(url, {"symbol": symbol})
    price = float(data["price"])
    _price_cache[symbol] = {"price": price, "ts": time.time()}
    return price


async def fetch_all_prices(symbols: list[str]) -> dict[str, float]:
    url = f"{_base()}/fapi/v1/ticker/price"
    data = await _get(url)
    now = time.time()
    prices = {}
    for d in data:
        if d["symbol"] in symbols:
            price = float(d["price"])
            prices[d["symbol"]] = price
            _price_cache[d["symbol"]] = {"price": price, "ts": now}
    return prices


# ──────────────────────────────────────────────
#  Mark Price + Predicted Funding (premiumIndex) — futures-only.
#  Mark price is what the exchange actually uses for PnL/liquidation,
#  distinct from the last-traded price above.
# ──────────────────────────────────────────────
async def fetch_mark_price(symbol: str) -> dict:
    cached = _mark_price_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < MARK_PRICE_CACHE_TTL:
        return cached["data"]
    url = f"{_base()}/fapi/v1/premiumIndex"
    data = await _get(url, {"symbol": symbol})
    result = {
        "mark_price": float(data["markPrice"]),
        "index_price": float(data.get("indexPrice", data["markPrice"])),
        "last_funding_rate": float(data.get("lastFundingRate", 0.0)),
        "next_funding_time": data.get("nextFundingTime"),
    }
    _mark_price_cache[symbol] = {"data": result, "ts": time.time()}
    return result


async def fetch_funding_rate(symbol: str) -> float:
    """Returns the current predicted funding rate. Positive = longs pay
    shorts (crowded long — mildly bearish signal for the Sentiment agent);
    negative = shorts pay longs."""
    try:
        info = await fetch_mark_price(symbol)
        return info["last_funding_rate"]
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
#  Order Book
# ──────────────────────────────────────────────
async def fetch_orderbook(symbol: str, limit: int = 20) -> dict:
    cached = _orderbook_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < ORDERBOOK_CACHE_TTL:
        return cached["data"]

    url = f"{_base()}/fapi/v1/depth"
    raw = await _get(url, {"symbol": symbol, "limit": limit})

    bids = [(float(p), float(q)) for p, q in raw["bids"]]
    asks = [(float(p), float(q)) for p, q in raw["asks"]]

    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    result = {
        "bids": bids,
        "asks": asks,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "imbalance": imbalance,
        "spread_pct": (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0,
    }
    _orderbook_cache[symbol] = {"data": result, "ts": time.time()}
    return result


# ──────────────────────────────────────────────
#  Open Interest — futures-only extra signal, not wired into an agent
#  by default but available for anyone extending the agent roster.
# ──────────────────────────────────────────────
async def fetch_open_interest(symbol: str) -> float:
    try:
        url = f"{_base()}/fapi/v1/openInterest"
        data = await _get(url, {"symbol": symbol})
        return float(data["openInterest"])
    except Exception:
        return 0.0