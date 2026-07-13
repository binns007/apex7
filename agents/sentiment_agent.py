"""
Agent 5 — Sentiment Agent
Reads macro crypto sentiment via Fear & Greed index and perpetual
futures funding rates. Contrarian on extremes, trend-following in the middle.
"""
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class SentimentAgent(BaseAgent):
    name = "Sentiment"
    weight = 0.9

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        fear_greed = extras.get("fear_greed", 50)
        funding = extras.get("funding_rate", {}).get(symbol, 0.0)

        buy_score = sell_score = 0.0

        if fear_greed <= 20:
            buy_score += 0.35
        elif fear_greed <= 35:
            buy_score += 0.20
        elif fear_greed >= 80:
            sell_score += 0.35
        elif fear_greed >= 65:
            sell_score += 0.20
        else:
            if fear_greed > 50:
                buy_score += 0.10
            else:
                sell_score += 0.10

        if funding > 0.0008:
            sell_score += 0.25
        elif funding > 0.0003:
            sell_score += 0.12
        elif funding < -0.0008:
            buy_score += 0.25
        elif funding < -0.0003:
            buy_score += 0.12

        threshold = 0.35

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"F&G={fear_greed} funding={funding:.5f} → contrarian BUY")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"F&G={fear_greed} funding={funding:.5f} → contrarian SELL")

        return self.hold(symbol, timeframe,
            f"Sentiment neutral: F&G={fear_greed} funding={funding:.5f}")