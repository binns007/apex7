"""
APEX-7 Market Data Engine
Fetches and caches OHLCV candles, order books, and live ticker prices.
All TA indicator calculations live here so agents stay clean.
"""
import asyncio
import time
import logging
from typing import Optional
import numpy as np
import pandas as pd
import aiohttp

from config import settings

logger = logging.getLogger("apex7.market_data")

# ──────────────────────────────────────────────
#  Binance REST endpoints
# ──────────────────────────────────────────────
LIVE_BASE = "https://api.binance.com"
TEST_BASE = "https://testnet.binance.vision"


def _base() -> str:
    return TEST_BASE if settings.is_testnet else LIVE_BASE


# ──────────────────────────────────────────────
#  Simple in-memory cache
# ──────────────────────────────────────────────
_candle_cache: dict[str, dict] = {}   # key: f"{symbol}_{tf}"
_price_cache: dict[str, float] = {}
_orderbook_cache: dict[str, dict] = {}
CACHE_TTL = 25  # seconds


async def _get(url: str, params: dict = None) -> dict | list:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()


# ──────────────────────────────────────────────
#  Candle Fetching
# ──────────────────────────────────────────────
async def fetch_candles(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Return OHLCV DataFrame with indicators pre-computed."""
    cache_key = f"{symbol}_{interval}"
    cached = _candle_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL:
        return cached["df"]

    url = f"{_base()}/api/v3/klines"
    raw = await _get(url, {"symbol": symbol, "interval": interval, "limit": limit})

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    df = _compute_indicators(df)
    _candle_cache[cache_key] = {"df": df, "ts": time.time()}
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all TA indicators to the DataFrame."""
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    vol   = df["volume"].values

    # ── RSI ───────────────────────────────────
    df["rsi"] = _rsi(close, 14)

    # ── MACD ─────────────────────────────────
    macd_line, signal_line, histogram = _macd(close)
    df["macd"]        = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"]   = histogram

    # ── Bollinger Bands ───────────────────────
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger(close, 20, 2.0)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── EMAs ─────────────────────────────────
    df["ema_9"]  = _ema(close, 9)
    df["ema_21"] = _ema(close, 21)
    df["ema_50"] = _ema(close, 50)
    df["ema_200"]= _ema(close, 200)

    # ── ATR ───────────────────────────────────
    df["atr"] = _atr(high, low, close, 14)

    # ── VWAP (rolling session proxy) ─────────
    df["vwap"] = _vwap(high, low, close, vol)

    # ── OBV ───────────────────────────────────
    df["obv"] = _obv(close, vol)

    # ── Stochastic ────────────────────────────
    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, 14, 3)

    # ── ROC (Rate of Change) ──────────────────
    df["roc_10"] = _roc(close, 10)

    # ── Volume z-score ────────────────────────
    vol_mean = pd.Series(vol).rolling(20).mean().values
    vol_std  = pd.Series(vol).rolling(20).std().values
    df["vol_zscore"] = (vol - vol_mean) / (vol_std + 1e-9)

    # ── Trend strength (ADX proxy) ────────────
    df["adx"] = _adx_proxy(high, low, close, 14)

    return df


# ──────────────────────────────────────────────
#  Indicator implementations (no deps)
# ──────────────────────────────────────────────
def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    k = 2 / (period + 1)
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(close: np.ndarray, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    valid = ~np.isnan(macd_line)
    sig = np.full(len(macd_line), np.nan)
    if valid.sum() >= signal:
        sig_vals = _ema(macd_line[valid], signal)
        sig[valid] = sig_vals
    histogram = macd_line - sig
    return macd_line, sig, histogram


def _bollinger(close: np.ndarray, period=20, num_std=2.0):
    mid = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close).rolling(period).std().values
    return mid + num_std * std, mid, mid - num_std * std


def _atr(high, low, close, period=14) -> np.ndarray:
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    atr = np.full(len(close), np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


def _vwap(high, low, close, volume) -> np.ndarray:
    tp = (high + low + close) / 3
    cum_tp_vol = np.cumsum(tp * volume)
    cum_vol = np.cumsum(volume)
    return cum_tp_vol / (cum_vol + 1e-9)


def _obv(close, volume) -> np.ndarray:
    direction = np.sign(np.diff(close))
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        obv[i] = obv[i - 1] + direction[i - 1] * volume[i]
    return obv


def _stochastic(high, low, close, k_period=14, d_period=3):
    k = np.full(len(close), np.nan)
    for i in range(k_period - 1, len(close)):
        lo = np.min(low[i - k_period + 1:i + 1])
        hi = np.max(high[i - k_period + 1:i + 1])
        k[i] = (close[i] - lo) / (hi - lo + 1e-9) * 100
    d = pd.Series(k).rolling(d_period).mean().values
    return k, d


def _roc(close: np.ndarray, period: int = 10) -> np.ndarray:
    roc = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        roc[i] = (close[i] - close[i - period]) / (close[i - period] + 1e-9) * 100
    return roc


def _adx_proxy(high, low, close, period=14) -> np.ndarray:
    """Simplified ADX-like trend strength 0–100."""
    atr_vals = _atr(high, low, close, period)
    price_change = np.abs(np.diff(close, prepend=close[0]))
    strength = pd.Series(price_change / (atr_vals + 1e-9)).rolling(period).mean().values * 100
    return np.clip(strength, 0, 100)


# ──────────────────────────────────────────────
#  Live Price
# ──────────────────────────────────────────────
async def fetch_price(symbol: str) -> float:
    cached = _price_cache.get(symbol)
    if cached:
        return cached
    url = f"{_base()}/api/v3/ticker/price"
    data = await _get(url, {"symbol": symbol})
    price = float(data["price"])
    _price_cache[symbol] = price
    # tiny TTL handled via periodic refresh in trading engine
    return price


async def fetch_all_prices(symbols: list[str]) -> dict[str, float]:
    url = f"{_base()}/api/v3/ticker/price"
    data = await _get(url)
    prices = {d["symbol"]: float(d["price"]) for d in data if d["symbol"] in symbols}
    _price_cache.update(prices)
    return prices


# ──────────────────────────────────────────────
#  Order Book
# ──────────────────────────────────────────────
async def fetch_orderbook(symbol: str, limit: int = 20) -> dict:
    cache_key = symbol
    cached = _orderbook_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < 5:
        return cached["data"]

    url = f"{_base()}/api/v3/depth"
    raw = await _get(url, {"symbol": symbol, "limit": limit})

    bids = [(float(p), float(q)) for p, q in raw["bids"]]
    asks = [(float(p), float(q)) for p, q in raw["asks"]]

    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)  # -1 to +1

    result = {
        "bids": bids,
        "asks": asks,
        "bid_volume": bid_vol,
        "ask_volume": ask_vol,
        "imbalance": imbalance,          # positive = buy pressure
        "spread_pct": (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0,
    }
    _orderbook_cache[cache_key] = {"data": result, "ts": time.time()}
    return result


# ──────────────────────────────────────────────
#  Fear & Greed (alternative.me public API)
# ──────────────────────────────────────────────
_fear_greed_cache = {"value": 50, "ts": 0}


async def fetch_fear_greed() -> int:
    """Returns 0-100 (0=extreme fear, 100=extreme greed)."""
    if time.time() - _fear_greed_cache["ts"] < 300:  # cache 5 min
        return _fear_greed_cache["value"]
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        data = await _get(url)
        val = int(data["data"][0]["value"])
        _fear_greed_cache.update({"value": val, "ts": time.time()})
        return val
    except Exception:
        return _fear_greed_cache["value"]


# ──────────────────────────────────────────────
#  Funding Rate (Futures) — useful sentiment
# ──────────────────────────────────────────────
async def fetch_funding_rate(symbol: str) -> float:
    """Returns funding rate as a float. Positive = longs pay shorts."""
    try:
        base = "https://fapi.binance.com" if not settings.is_testnet else "https://testnet.binancefuture.com"
        url = f"{base}/fapi/v1/fundingRate"
        data = await _get(url, {"symbol": symbol, "limit": 1})
        if data:
            return float(data[0]["fundingRate"])
    except Exception:
        pass
    return 0.0
