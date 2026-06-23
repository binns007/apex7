"""
Agent 8 — Regime Agent
Classifies the current market regime (trending up, trending down,
ranging, or volatile/choppy) and biases the signal accordingly.
This agent is unique: it ADJUSTS its signal based on what's working NOW.
"""
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


REGIME_TREND_UP   = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGING    = "RANGING"
REGIME_VOLATILE   = "VOLATILE"


def classify_regime(df: pd.DataFrame) -> tuple[str, float]:
    """Return (regime, confidence) based on last N candles."""
    if len(df) < 50:
        return REGIME_RANGING, 0.5

    close = df["close"].values[-50:]
    atr   = df["atr"].values[-1]
    adx   = df["adx"].values[-1]
    bb_w  = df["bb_width"].values[-1]
    ema50 = df["ema_50"].values[-1]
    ema200= df["ema_200"].values[-1]

    # Trend: ADX > 25 and EMA alignment
    if adx > 30 and close[-1] > ema50 > ema200:
        return REGIME_TREND_UP, min(adx / 50, 1.0)
    if adx > 30 and close[-1] < ema50 < ema200:
        return REGIME_TREND_DOWN, min(adx / 50, 1.0)

    # Volatile: high ATR relative to price, wide BBands
    price = close[-1]
    atr_pct = atr / price * 100
    if atr_pct > 2.5 or bb_w > 0.08:
        return REGIME_VOLATILE, min(atr_pct / 5, 1.0)

    # Default: ranging
    return REGIME_RANGING, 0.6


class RegimeAgent(BaseAgent):
    name = "Regime"
    weight = 1.2

    _last_regime: dict[str, str] = {}

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if timeframe != "15m":
            # Regime is a higher timeframe concept
            return self.hold(symbol, timeframe, "Regime runs on 15m only")

        regime, confidence = classify_regime(df)
        self._last_regime[symbol] = regime

        last   = df.iloc[-1]
        close  = last["close"]
        ema50  = last["ema_50"]
        ema200 = last["ema_200"]
        adx    = last["adx"]

        if regime == REGIME_TREND_UP:
            # In uptrend, favour buys on pullbacks
            if close < ema50 * 1.005:           # pulled back to EMA50
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
            # Reduce conviction in volatile markets
            return self.hold(symbol, timeframe,
                f"Regime=VOLATILE – skipping (ADX={adx:.1f})")

        # RANGING: let other agents handle it
        return self.hold(symbol, timeframe,
            f"Regime=RANGING ADX={adx:.1f}")

    @classmethod
    def get_regime(cls, symbol: str) -> str:
        return cls._last_regime.get(symbol, REGIME_RANGING)
