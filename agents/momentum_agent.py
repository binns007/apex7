"""
Agent 1 — Momentum Agent
Uses RSI, MACD, EMA alignment, and Rate-of-Change to detect strong directional momentum.
"""
import math
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class MomentumAgent(BaseAgent):
    name = "Momentum"
    weight = 1.3

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if len(df) < 50:
            return self.hold(symbol, timeframe, "Insufficient candles")

        last = df.iloc[-1]
        prev = df.iloc[-2]

        fields = [last["rsi"], last["macd"], last["macd_signal"], last["macd_hist"],
                  last["roc_10"], last["ema_9"], last["ema_21"], last["ema_50"]]
        if any(math.isnan(v) for v in fields):
            return self.hold(symbol, timeframe, "Indicators still warming up")

        rsi = last["rsi"]
        macd = last["macd"]
        sig = last["macd_signal"]
        hist = last["macd_hist"]
        roc = last["roc_10"]
        ema9 = last["ema_9"]
        ema21 = last["ema_21"]
        ema50 = last["ema_50"]
        close = last["close"]

        buy_score = 0.0
        sell_score = 0.0

        # RSI momentum
        if 45 < rsi < 65:
            buy_score += 0.25
        elif rsi > 65 and rsi < 80:
            buy_score += 0.15
        elif 35 < rsi < 55:
            sell_score += 0.25
        elif rsi < 35 and rsi > 20:
            sell_score += 0.15

        # MACD crossover
        if macd > sig and prev["macd"] <= prev["macd_signal"]:
            buy_score += 0.35  # fresh cross up
        elif macd > sig and hist > prev["macd_hist"]:
            buy_score += 0.20  # accelerating upward
        if macd < sig and prev["macd"] >= prev["macd_signal"]:
            sell_score += 0.35
        elif macd < sig and hist < prev["macd_hist"]:
            sell_score += 0.20

        # EMA stack alignment (bull: 9>21>50, bear: 9<21<50)
        if ema9 > ema21 > ema50 and close > ema9:
            buy_score += 0.25
        elif ema9 < ema21 < ema50 and close < ema9:
            sell_score += 0.25

        # Rate of change
        if roc > 1.5:
            buy_score += 0.15
        elif roc < -1.5:
            sell_score += 0.15

        if buy_score >= 0.55 and buy_score > sell_score * 1.3:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"RSI={rsi:.1f} MACD_hist={hist:.4f} EMA_aligned ROC={roc:.2f}%")
        if sell_score >= 0.55 and sell_score > buy_score * 1.3:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"RSI={rsi:.1f} MACD_hist={hist:.4f} EMA_bearish ROC={roc:.2f}%")

        return self.hold(symbol, timeframe, f"RSI={rsi:.1f}, mixed signals")