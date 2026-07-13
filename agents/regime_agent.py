"""
Agent 8 — Regime Agent
Classifies the current market regime (trending up, trending down,
ranging, or volatile/choppy) and biases the signal accordingly.

v1 behavior: this agent's HOLD-on-volatile vote was redundant with a
SEPARATE hard block inside consensus_engine.py that killed ALL trading
outright whenever regime == VOLATILE. Crypto routinely exceeds naive
volatility thresholds intraday, so that combination could blackout
trading for long stretches even in tradeable conditions.

New design: this agent still classifies regime and still votes HOLD
when volatile (that's a legitimate vote), but the hard block has moved
out of here entirely. consensus_engine now applies a *soft* multiplier
(config: REGIME_VOLATILE_SCORE_MULTIPLIER) by default, with an opt-in
switch (REGIME_HARD_BLOCK_VOLATILE) to restore the old behavior.
Thresholds are now config-driven instead of hardcoded magic numbers.
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


def classify_regime(df: pd.DataFrame) -> tuple[str, float]:
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

    trend_adx = settings.REGIME_TREND_ADX

    if adx > trend_adx and close > ema50 > ema200:
        return REGIME_TREND_UP, min(adx / 50, 1.0)
    if adx > trend_adx and close < ema50 < ema200:
        return REGIME_TREND_DOWN, min(adx / 50, 1.0)

    atr_pct = atr / close * 100 if close else 0.0
    if atr_pct > settings.REGIME_VOLATILE_ATR_PCT or bb_w > settings.REGIME_VOLATILE_BB_WIDTH:
        return REGIME_VOLATILE, min(atr_pct / (settings.REGIME_VOLATILE_ATR_PCT * 2), 1.0)

    return REGIME_RANGING, 0.6


class RegimeAgent(BaseAgent):
    name = "Regime"
    weight = 1.2

    # Per-symbol last-known regime, read by the consensus engine's soft gate.
    _last_regime: dict[str, str] = {}
    _last_regime_confidence: dict[str, float] = {}

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if timeframe != "15m":
            return self.hold(symbol, timeframe, "Regime runs on 15m only")

        regime, confidence = classify_regime(df)
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

    @classmethod
    def get_regime(cls, symbol: str) -> str:
        return cls._last_regime.get(symbol, REGIME_UNKNOWN)

    @classmethod
    def get_regime_confidence(cls, symbol: str) -> float:
        return cls._last_regime_confidence.get(symbol, 0.0)