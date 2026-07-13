"""
Agent 7 — Scalping Agent
Operates on 1-minute micro-structure: rapid EMA crossovers,
Stochastic signals, and tick-level momentum.
"""
import math
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class ScalpingAgent(BaseAgent):
    name = "Scalping"
    weight = 0.85

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if timeframe not in ("1m", "3m"):
            return self.hold(symbol, timeframe, "Scalping only on 1m/3m")
        if len(df) < 20:
            return self.hold(symbol, timeframe, "Insufficient candles")

        last = df.iloc[-1]
        prev = df.iloc[-2]

        fields = [last["stoch_k"], last["stoch_d"], last["ema_9"], last["ema_21"]]
        if any(math.isnan(v) for v in fields):
            return self.hold(symbol, timeframe, "Indicators still warming up")

        close = last["close"]
        stoch_k = last["stoch_k"]
        stoch_d = last["stoch_d"]
        ema9 = last["ema_9"]
        ema21 = last["ema_21"]
        vol_z = last["vol_zscore"]

        buy_score = sell_score = 0.0

        if (prev["stoch_k"] < prev["stoch_d"] and stoch_k > stoch_d and stoch_k < 40):
            buy_score += 0.40
        elif stoch_k < 20:
            buy_score += 0.20

        if (prev["stoch_k"] > prev["stoch_d"] and stoch_k < stoch_d and stoch_k > 60):
            sell_score += 0.40
        elif stoch_k > 80:
            sell_score += 0.20

        if prev["ema_9"] < prev["ema_21"] and ema9 > ema21:
            buy_score += 0.30
        elif ema9 > ema21:
            buy_score += 0.10

        if prev["ema_9"] > prev["ema_21"] and ema9 < ema21:
            sell_score += 0.30
        elif ema9 < ema21:
            sell_score += 0.10

        closes = df["close"].tail(4).values
        if len(closes) == 4:
            if all(closes[i] > closes[i - 1] for i in range(1, 4)):
                buy_score += 0.15
            elif all(closes[i] < closes[i - 1] for i in range(1, 4)):
                sell_score += 0.15

        if not math.isnan(vol_z) and vol_z > 1.0:
            if close > last["open"]:
                buy_score += 0.10
            else:
                sell_score += 0.10

        threshold = 0.50

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"Scalp BUY: K={stoch_k:.1f} ema9>21={ema9 > ema21}")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"Scalp SELL: K={stoch_k:.1f} ema9<21={ema9 < ema21}")

        return self.hold(symbol, timeframe,
            f"Scalp neutral: K={stoch_k:.1f} D={stoch_d:.1f}")