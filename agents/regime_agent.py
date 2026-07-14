"""
Agent 8 — Regime Agent
Classifies the current market regime (trending up, trending down,
ranging, or volatile/choppy) and biases the signal accordingly.

v1 behavior: this agent's HOLD-on-volatile vote was redundant with a
SEPARATE hard block inside consensus_engine.py that killed ALL trading
outright whenever regime == VOLATILE. Crypto routinely exceeds naive
volatility thresholds intraday, so that combination could blackout
trading for long stretches even in tradeable conditions.

v2 design: this agent still classifies regime and still votes HOLD
when volatile (that's a legitimate vote), but the hard block has moved
out of here entirely. consensus_engine now applies a *soft* multiplier
by default, with an opt-in switch to restore the old behavior.
Thresholds are config-driven instead of hardcoded magic numbers.

v3 (Futures Mode support): the timeframe this agent runs on, and its
trend/volatility thresholds, are now constructor parameters instead of
hardcoded/class-level. Spot uses 15m with the original thresholds;
Futures Mode constructs a SEPARATE RegimeAgent instance on 5m with
tighter thresholds — each engine's regime state now lives on its own
instance (previously a class-level dict shared by ALL instances, which
would have let a spot BTCUSDT regime read collide with a futures
BTCUSDT regime read had both engines run at once).
"""
import math
import numpy as np
import pandas as pd

from config import settings
from agents.base_agent import BaseAgent, AgentSignal

REGIME_TREND_UP = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGING = "RANGING"
REGIME_VOLATILE = "VOLATILE"
REGIME_UNKNOWN = "UNKNOWN"


def classify_regime(
    df: pd.DataFrame,
    trend_adx: float,
    volatile_atr_pct: float,
    volatile_bb_width: float,
) -> tuple[str, float]:
    """Return (regime, confidence) based on the last N candles."""
    if len(df) < 50:
        return REGIME_RANGING, 0.5

    last = df.iloc[-1]
    fields = [last["adx"], last["atr"], last["bb_width"], last["ema_50"], last["ema_200"]]
    if any(math.isnan(v) for v in fields):
        return REGIME_UNKNOWN, 0.0

    close = last["close"]
    atr = last["atr"]
    adx = last["adx"]
    bb_w = last["bb_width"]
    ema50 = last["ema_50"]
    ema200 = last["ema_200"]

    if adx > trend_adx and close > ema50 > ema200:
        return REGIME_TREND_UP, min(adx / 50, 1.0)
    if adx > trend_adx and close < ema50 < ema200:
        return REGIME_TREND_DOWN, min(adx / 50, 1.0)

    atr_pct = atr / close * 100 if close else 0.0
    if atr_pct > volatile_atr_pct or bb_w > volatile_bb_width:
        return REGIME_VOLATILE, min(atr_pct / (volatile_atr_pct * 2), 1.0)

    return REGIME_RANGING, 0.6


class RegimeAgent(BaseAgent):
    name = "Regime"
    weight = 1.2

    def __init__(
        self,
        timeframe: str = "15m",
        trend_adx: float = None,
        volatile_atr_pct: float = None,
        volatile_bb_width: float = None,
    ):
        # Which timeframe this instance classifies regime on. Spot uses
        # 15m (the slowest of its 3 timeframes); Futures Mode passes 5m
        # (the slowest of its faster 1m/3m/5m stack) so "trend" still
        # means the highest timeframe available in that engine.
        self.timeframe = timeframe
        self.trend_adx = settings.REGIME_TREND_ADX if trend_adx is None else trend_adx
        self.volatile_atr_pct = settings.REGIME_VOLATILE_ATR_PCT if volatile_atr_pct is None else volatile_atr_pct
        self.volatile_bb_width = settings.REGIME_VOLATILE_BB_WIDTH if volatile_bb_width is None else volatile_bb_width

        # Per-symbol last-known regime, read by this engine's consensus
        # soft gate. Instance-level (not class-level) so a spot engine
        # and a futures engine each keep independent regime state even
        # when trading the same symbol at the same time.
        self._last_regime: dict[str, str] = {}
        self._last_regime_confidence: dict[str, float] = {}

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if timeframe != self.timeframe:
            return self.hold(symbol, timeframe, f"Regime runs on {self.timeframe} only")

        regime, confidence = classify_regime(
            df, self.trend_adx, self.volatile_atr_pct, self.volatile_bb_width
        )
        self._last_regime[symbol] = regime
        self._last_regime_confidence[symbol] = confidence

        if regime == REGIME_UNKNOWN:
            return self.hold(symbol, timeframe, "Regime indicators still warming up")

        last = df.iloc[-1]
        close = last["close"]
        ema50 = last["ema_50"]
        adx = last["adx"]

        if regime == REGIME_TREND_UP:
            if close < ema50 * 1.005:
                return self._signal(symbol, timeframe, "BUY", confidence,
                    f"Regime=TREND_UP pullback to EMA50, ADX={adx:.1f}")
            return self._signal(symbol, timeframe, "BUY", confidence * 0.7,
                f"Regime=TREND_UP continuation, ADX={adx:.1f}")

        if regime == REGIME_TREND_DOWN:
            if close > ema50 * 0.995:
                return self._signal(symbol, timeframe, "SELL", confidence,
                    f"Regime=TREND_DOWN rally to EMA50, ADX={adx:.1f}")
            return self._signal(symbol, timeframe, "SELL", confidence * 0.7,
                f"Regime=TREND_DOWN continuation, ADX={adx:.1f}")

        if regime == REGIME_VOLATILE:
            return self.hold(symbol, timeframe,
                f"Regime=VOLATILE — reduced conviction (ADX={adx:.1f})")

        return self.hold(symbol, timeframe, f"Regime=RANGING ADX={adx:.1f}")

    def get_regime(self, symbol: str) -> str:
        return self._last_regime.get(symbol, REGIME_UNKNOWN)

    def get_regime_confidence(self, symbol: str) -> float:
        return self._last_regime_confidence.get(symbol, 0.0)