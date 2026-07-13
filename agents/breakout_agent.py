"""
Agent 3 — Breakout Agent
Detects consolidation followed by expansion. Identifies key S/R levels
from pivot highs/lows and fires on confirmed breakouts with volume.
"""
import math
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class BreakoutAgent(BaseAgent):
    name = "Breakout"
    weight = 1.2

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if len(df) < 50:
            return self.hold(symbol, timeframe, "Insufficient candles")

        last = df.iloc[-1]
        fields = [last["vol_zscore"], last["bb_width"], last["atr"], last["adx"]]
        if any(math.isnan(v) for v in fields):
            return self.hold(symbol, timeframe, "Indicators still warming up")

        close = last["close"]
        vol_z = last["vol_zscore"]
        bb_w = last["bb_width"]
        atr = last["atr"]
        adx = last["adx"]

        pivot_window = 20
        recent = df.tail(pivot_window + 1).iloc[:-1]
        resistance = recent["high"].max()
        support = recent["low"].min()

        broke_up = close > resistance * 1.002
        broke_down = close < support * 0.998

        prev_bb_w = df["bb_width"].iloc[-5:-1].mean()
        bb_expanding = not math.isnan(prev_bb_w) and bb_w > prev_bb_w * 1.1

        vol_confirmed = vol_z > 0.8
        trending = adx > 25

        buy_score = sell_score = 0.0

        if broke_up:
            buy_score += 0.40
            if vol_confirmed:
                buy_score += 0.25
            if bb_expanding:
                buy_score += 0.15
            if trending:
                buy_score += 0.15
            body = close - last["open"]
            if body > 0 and body > atr * 0.4:
                buy_score += 0.10

        if broke_down:
            sell_score += 0.40
            if vol_confirmed:
                sell_score += 0.25
            if bb_expanding:
                sell_score += 0.15
            if trending:
                sell_score += 0.15
            body = last["open"] - close
            if body > 0 and body > atr * 0.4:
                sell_score += 0.10

        threshold = 0.60

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"Breakout above {resistance:.4f} | vol_z={vol_z:.2f} ADX={adx:.1f}")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"Breakdown below {support:.4f} | vol_z={vol_z:.2f} ADX={adx:.1f}")

        return self.hold(symbol, timeframe,
            f"No breakout: close={close:.4f} R={resistance:.4f} S={support:.4f}")