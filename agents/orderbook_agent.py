"""
Agent 6 — Order Book Agent
Reads real-time order book microstructure: bid/ask imbalance,
large walls, and spread to infer near-term price direction.
"""
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class OrderBookAgent(BaseAgent):
    name = "OrderBook"
    weight = 1.0

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        ob = extras.get("orderbook", {}).get(symbol, {})
        if not ob:
            return self.hold(symbol, timeframe, "No order book data")

        imbalance  = ob.get("imbalance", 0.0)   # -1 to +1
        spread_pct = ob.get("spread_pct", 0.5)
        bids       = ob.get("bids", [])
        asks       = ob.get("asks", [])

        buy_score = sell_score = 0.0

        # ── Imbalance signal ────────────────────────
        if imbalance > 0.30:
            buy_score += 0.35
        elif imbalance > 0.15:
            buy_score += 0.20
        elif imbalance < -0.30:
            sell_score += 0.35
        elif imbalance < -0.15:
            sell_score += 0.20

        # ── Large wall detection ────────────────────
        if bids and asks:
            bid_vols = [q for _, q in bids]
            ask_vols = [q for _, q in asks]
            top_bid_wall = max(bid_vols[:5]) / (sum(bid_vols[:5]) + 1e-9)
            top_ask_wall = max(ask_vols[:5]) / (sum(ask_vols[:5]) + 1e-9)

            # Large bid wall = support → buy signal
            if top_bid_wall > 0.5:
                buy_score += 0.20
            # Large ask wall = resistance → sell signal
            if top_ask_wall > 0.5:
                sell_score += 0.20

        # ── Spread filter: tight spread = liquid → trade confidently
        if spread_pct < 0.02:
            confidence_boost = 1.15
        elif spread_pct > 0.10:
            confidence_boost = 0.80  # wide spread = risky
        else:
            confidence_boost = 1.0

        buy_score  *= confidence_boost
        sell_score *= confidence_boost

        threshold = 0.40

        if buy_score >= threshold and buy_score > sell_score:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"OB_imbalance={imbalance:.2f} spread={spread_pct:.3f}%")
        if sell_score >= threshold and sell_score > buy_score:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"OB_imbalance={imbalance:.2f} spread={spread_pct:.3f}%")

        return self.hold(symbol, timeframe,
            f"OB balanced: imbalance={imbalance:.2f}")
