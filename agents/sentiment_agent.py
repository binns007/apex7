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
        funding    = extras.get("funding_rate", {}).get(symbol, 0.0)

        buy_score = sell_score = 0.0

        # ── Fear & Greed Index ──────────────────────
        # Contrarian extremes
        if fear_greed <= 20:               # Extreme Fear → buy dip
            buy_score += 0.35
        elif fear_greed <= 35:             # Fear
            buy_score += 0.20
        elif fear_greed >= 80:             # Extreme Greed → sell / caution
            sell_score += 0.35
        elif fear_greed >= 65:             # Greed
            sell_score += 0.20
        else:
            # Neutral 35–65: mild trend-following
            if fear_greed > 50:
                buy_score += 0.10
            else:
                sell_score += 0.10

        # ── Funding Rate ────────────────────────────
        # High positive funding = longs overcrowded → contrarian sell
        # High negative funding = shorts overcrowded → contrarian buy
        if funding > 0.0008:              # > ~0.08% per 8hr
            sell_score += 0.25
        elif funding > 0.0003:
            sell_score += 0.12
        elif funding < -0.0008:
            buy_score += 0.25
        elif funding < -0.0003:
            buy_score += 0.12
        # else near-zero: neutral, no adjustment

        threshold = 0.35

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"F&G={fear_greed} funding={funding:.5f} → contrarian BUY")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"F&G={fear_greed} funding={funding:.5f} → contrarian SELL")

        return self.hold(symbol, timeframe,
            f"Sentiment neutral: F&G={fear_greed} funding={funding:.5f}")
