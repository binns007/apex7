"""
Agent 2 — Mean Reversion Agent
Identifies over-extended moves using Bollinger Bands, RSI extremes,
and Z-score to fade the move back to the mean.
"""
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class MeanReversionAgent(BaseAgent):
    name = "MeanReversion"
    weight = 1.1

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if len(df) < 30:
            return self.hold(symbol, timeframe, "Insufficient candles")

        last   = df.iloc[-1]
        close  = last["close"]
        rsi    = last["rsi"]
        bb_pct = last["bb_pct"]    # 0 = at lower band, 1 = at upper band
        bb_w   = last["bb_width"]  # band width relative to midline
        vwap   = last["vwap"]
        atr    = last["atr"]

        # Z-score of close vs 20-period mean
        window = df["close"].tail(20)
        z_score = (close - window.mean()) / (window.std() + 1e-9)

        buy_score = sell_score = 0.0

        # ── Oversold conditions ─────────────────────
        if rsi < 30:
            buy_score += 0.30
        elif rsi < 40:
            buy_score += 0.15

        if bb_pct < 0.05:                    # price touching/below lower BB
            buy_score += 0.30
        elif bb_pct < 0.15:
            buy_score += 0.15

        if z_score < -2.0:
            buy_score += 0.25
        elif z_score < -1.5:
            buy_score += 0.15

        if close < vwap * 0.995:             # price below VWAP
            buy_score += 0.10

        # ── Overbought conditions ───────────────────
        if rsi > 70:
            sell_score += 0.30
        elif rsi > 60:
            sell_score += 0.15

        if bb_pct > 0.95:
            sell_score += 0.30
        elif bb_pct > 0.85:
            sell_score += 0.15

        if z_score > 2.0:
            sell_score += 0.25
        elif z_score > 1.5:
            sell_score += 0.15

        if close > vwap * 1.005:
            sell_score += 0.10

        # Only trade mean reversion in low-trend environments (tight BB)
        bb_filter = bb_w < 0.06  # tight bands = range environment

        threshold = 0.50 if bb_filter else 0.70

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"MR_BUY: RSI={rsi:.1f} BB%={bb_pct:.2f} Z={z_score:.2f}")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"MR_SELL: RSI={rsi:.1f} BB%={bb_pct:.2f} Z={z_score:.2f}")

        return self.hold(symbol, timeframe, f"No extreme: RSI={rsi:.1f} Z={z_score:.2f}")
