"""
APEX-7 Market Data Engine
Fetches and caches OHLCV candles, order books, and live ticker prices.
All TA indicator calculations live here so agents stay clean.

Fixes vs v1:
  - fetch_price() had NO TTL — it cached a price once and returned it
    forever unless something else happened to refresh the shared dict.
    Every cache in this module now uses the same {value, ts} + TTL pattern.
  - _adx_proxy() was a rough approximation, not real ADX, yet agents used
    it against textbook ADX thresholds (25/30). Replaced with a proper
    Wilder's ADX (_adx) so those thresholds mean what the code assumes.
  - VWAP was a rolling average that drifted with the cache window instead
    of a true session-anchored VWAP. Replaced with a UTC-day-anchored VWAP.
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
#  Simple in-memory caches — all {key: {"value"/"df"/"data": ..., "ts": float}}
# ──────────────────────────────────────────────
_candle_cache: dict[str, dict] = {}      # key: f"{symbol}_{tf}"
_price_cache: dict[str, dict] = {}       # key: symbol -> {"price": float, "ts": float}
_orderbook_cache: dict[str, dict] = {}   # key: symbol

CANDLE_CACHE_TTL = 25    # seconds
PRICE_CACHE_TTL = 4      # seconds — must be short, price sizing depends on it
ORDERBOOK_CACHE_TTL = 5  # seconds

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
    """Call on app shutdown to close the shared HTTP session cleanly."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()


# ──────────────────────────────────────────────
#  Candle Fetching
# ──────────────────────────────────────────────
async def fetch_candles(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Return OHLCV DataFrame with indicators pre-computed."""
    cache_key = f"{symbol}_{interval}"
    cached = _candle_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CANDLE_CACHE_TTL:
        return cached["df"]

    url = f"{_base()}/api/v3/klines"
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


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all TA indicators to the DataFrame."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values

    # ── RSI ───────────────────────────────────
    df["rsi"] = _rsi(close, 14)

    # ── MACD ─────────────────────────────────
    macd_line, signal_line, histogram = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # ── Bollinger Bands ───────────────────────
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger(close, 20, 2.0)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── EMAs ─────────────────────────────────
    df["ema_9"] = _ema(close, 9)
    df["ema_21"] = _ema(close, 21)
    df["ema_50"] = _ema(close, 50)
    df["ema_200"] = _ema(close, 200)

    # ── ATR ───────────────────────────────────
    df["atr"] = _atr(high, low, close, 14)

    # ── VWAP (true UTC-day anchored, resets at midnight UTC) ──
    df["vwap"] = _vwap_anchored(df)

    # ── OBV ───────────────────────────────────
    df["obv"] = _obv(close, vol)

    # ── Stochastic ────────────────────────────
    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, 14, 3)

    # ── ROC (Rate of Change) ──────────────────
    df["roc_10"] = _roc(close, 10)

    # ── Volume z-score ────────────────────────
    vol_mean = pd.Series(vol).rolling(20).mean().values
    vol_std = pd.Series(vol).rolling(20).std().values
    df["vol_zscore"] = (vol - vol_mean) / (vol_std + 1e-9)

    # ── True Wilder ADX (was a rough proxy in v1) ─
    df["adx"] = _adx(high, low, close, 14)

    return df


# ──────────────────────────────────────────────
#  Indicator implementations (no deps)
# ──────────────────────────────────────────────
def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    if len(arr) < period:
        return result
    k = 2 / (period + 1)
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    if n <= period:
        return np.full(n, np.nan)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
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
    return mid + num_std * std, pd.Series(mid), mid - num_std * std


def _atr(high, low, close, period=14) -> np.ndarray:
    n = len(close)
    if n <= period:
        return np.full(n, np.nan)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    atr = np.full(n, np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


def _vwap_anchored(df: pd.DataFrame) -> np.ndarray:
    """VWAP anchored to UTC calendar day (resets at 00:00 UTC), not a
    rolling window that silently drifts as the candle cache slides."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df["open_time"].dt.floor("D")
    tp_vol = tp * df["volume"]
    cum_tp_vol = tp_vol.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return (cum_tp_vol / (cum_vol + 1e-9)).values


def _obv(close, volume) -> np.ndarray:
    direction = np.sign(np.diff(close))
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        obv[i] = obv[i - 1] + direction[i - 1] * volume[i]
    return obv


def _stochastic(high, low, close, k_period=14, d_period=3):
    n = len(close)
    k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        lo = np.min(low[i - k_period + 1:i + 1])
        hi = np.max(high[i - k_period + 1:i + 1])
        k[i] = (close[i] - lo) / (hi - lo + 1e-9) * 100
    d = pd.Series(k).rolling(d_period).mean().values
    return k, d


def _roc(close: np.ndarray, period: int = 10) -> np.ndarray:
    n = len(close)
    roc = np.full(n, np.nan)
    for i in range(period, n):
        roc[i] = (close[i] - close[i - period]) / (close[i - period] + 1e-9) * 100
    return roc


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    if n <= period:
        return result
    result[period] = np.nansum(arr[1:period + 1])
    for i in range(period + 1, n):
        result[i] = result[i - 1] - (result[i - 1] / period) + arr[i]
    return result


def _adx(high, low, close, period=14) -> np.ndarray:
    """Real Wilder's ADX (0-100 trend-strength). v1 shipped a rough
    proxy (price-change / ATR) that isn't comparable to standard ADX
    even though thresholds like 25/30 assume the textbook definition."""
    n = len(close)
    if n <= period + 1:
        return np.full(n, np.nan)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / (smoothed_tr + 1e-9)
    minus_di = 100 * smoothed_minus_dm / (smoothed_tr + 1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)

    adx = np.full(n, np.nan)
    valid_start = period * 2
    if valid_start < n:
        adx[valid_start] = np.nanmean(dx[period:valid_start + 1])
        for i in range(valid_start + 1, n):
            if np.isnan(dx[i]) or np.isnan(adx[i - 1]):
                adx[i] = adx[i - 1]
            else:
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return np.clip(adx, 0, 100)


# ──────────────────────────────────────────────
#  Live Price — FIXED: previously had no TTL and could serve a
#  stale price indefinitely once cached (v1 bug).
# ──────────────────────────────────────────────
async def fetch_price(symbol: str) -> float:
    cached = _price_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < PRICE_CACHE_TTL:
        return cached["price"]
    url = f"{_base()}/api/v3/ticker/price"
    data = await _get(url, {"symbol": symbol})
    price = float(data["price"])
    _price_cache[symbol] = {"price": price, "ts": time.time()}
    return price


async def fetch_all_prices(symbols: list[str]) -> dict[str, float]:
    url = f"{_base()}/api/v3/ticker/price"
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
#  Order Book
# ──────────────────────────────────────────────
async def fetch_orderbook(symbol: str, limit: int = 20) -> dict:
    cached = _orderbook_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < ORDERBOOK_CACHE_TTL:
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
        "imbalance": imbalance,
        "spread_pct": (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0,
    }
    _orderbook_cache[symbol] = {"data": result, "ts": time.time()}
    return result


# ──────────────────────────────────────────────
#  Fear & Greed (alternative.me public API)
# ──────────────────────────────────────────────
_fear_greed_cache = {"value": 50, "ts": 0.0}


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
        base = "https://testnet.binancefuture.com" if settings.is_testnet else "https://fapi.binance.com"
        url = f"{base}/fapi/v1/fundingRate"
        data = await _get(url, {"symbol": symbol, "limit": 1})
        if data:
            return float(data[0]["fundingRate"])
    except Exception:
        pass
    return 0.0