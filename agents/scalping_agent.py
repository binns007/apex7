"""
Agent 7 — Scalping Agent
Operates on 1-minute micro-structure: rapid EMA crossovers,
Stochastic signals, and tick-level momentum.
"""
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class ScalpingAgent(BaseAgent):
    name = "Scalping"
    weight = 0.85   # slightly lower weight as it's noisier

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        # This agent is most useful on 1m timeframe
        if timeframe not in ("1m", "3m"):
            return self.hold(symbol, timeframe, "Scalping only on 1m/3m")
        if len(df) < 20:
            return self.hold(symbol, timeframe, "Insufficient candles")

        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        close   = last["close"]
        stoch_k = last["stoch_k"]
        stoch_d = last["stoch_d"]
        ema9    = last["ema_9"]
        ema21   = last["ema_21"]
        rsi     = last["rsi"]
        vol_z   = last["vol_zscore"]

        buy_score = sell_score = 0.0

        # ── Stochastic crossover ────────────────────
        # K crosses above D from oversold
        if (prev["stoch_k"] < prev["stoch_d"] and
                stoch_k > stoch_d and stoch_k < 40):
            buy_score += 0.40
        elif stoch_k < 20:
            buy_score += 0.20

        if (prev["stoch_k"] > prev["stoch_d"] and
                stoch_k < stoch_d and stoch_k > 60):
            sell_score += 0.40
        elif stoch_k > 80:
            sell_score += 0.20

        # ── Fast EMA micro cross ────────────────────
        if prev["ema_9"] < prev["ema_21"] and ema9 > ema21:
            buy_score += 0.30
        elif ema9 > ema21:
            buy_score += 0.10

        if prev["ema_9"] > prev["ema_21"] and ema9 < ema21:
            sell_score += 0.30
        elif ema9 < ema21:
            sell_score += 0.10

        # ── Momentum burst (3 consecutive same-direction candles) ──
        closes = df["close"].tail(4).values
        if all(closes[i] > closes[i-1] for i in range(1, 4)):
            buy_score += 0.15
        elif all(closes[i] < closes[i-1] for i in range(1, 4)):
            sell_score += 0.15

        # ── Volume boost ────────────────────────────
        if vol_z > 1.0:
            if close > last["open"]:
                buy_score += 0.10
            else:
                sell_score += 0.10

        threshold = 0.50

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"Scalp BUY: K={stoch_k:.1f} ema9>21={ema9>ema21}")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"Scalp SELL: K={stoch_k:.1f} ema9<21={ema9<ema21}")

        return self.hold(symbol, timeframe,
            f"Scalp neutral: K={stoch_k:.1f} D={stoch_d:.1f}")
