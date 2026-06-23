"""
Agent 4 — Volume Agent
Analyses on-balance volume (OBV) trend, VWAP deviations, and
volume z-score to confirm or deny price moves.
"""
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentSignal


class VolumeAgent(BaseAgent):
    name = "Volume"
    weight = 1.1

    async def analyze(self, symbol, timeframe, df, extras) -> AgentSignal:
        if len(df) < 30:
            return self.hold(symbol, timeframe, "Insufficient candles")

        last  = df.iloc[-1]
        close = last["close"]
        vwap  = last["vwap"]
        vol_z = last["vol_zscore"]
        obv   = last["obv"]

        # OBV trend (slope of last 10 bars)
        obv_series = df["obv"].tail(10).values
        obv_slope = np.polyfit(range(len(obv_series)), obv_series, 1)[0]
        obv_trend_up = obv_slope > 0

        # VWAP deviation
        vwap_dev = (close - vwap) / vwap * 100  # % deviation from VWAP

        # Taker buy ratio (if available in candle data)
        taker_buy = df["taker_buy_base"].astype(float).tail(5).mean()
        total_vol  = df["volume"].astype(float).tail(5).mean()
        buy_ratio  = taker_buy / (total_vol + 1e-9)  # > 0.5 = buyers dominating

        buy_score = sell_score = 0.0

        # OBV direction
        if obv_trend_up:
            buy_score += 0.25
        else:
            sell_score += 0.25

        # VWAP distance: buying near VWAP is safest
        if -0.3 < vwap_dev < 0.3:
            pass  # near VWAP, neutral
        elif vwap_dev > 0.5:
            sell_score += 0.20  # overextended above VWAP
        elif vwap_dev < -0.5:
            buy_score += 0.20   # oversold below VWAP

        # Volume spike confirmation
        if vol_z > 1.5:
            # High-volume candle: direction matters
            if close > df["open"].iloc[-1]:
                buy_score += 0.30
            else:
                sell_score += 0.30
        elif vol_z > 0.8:
            if close > df["open"].iloc[-1]:
                buy_score += 0.15
            else:
                sell_score += 0.15

        # Taker buy pressure
        if buy_ratio > 0.58:
            buy_score += 0.20
        elif buy_ratio < 0.42:
            sell_score += 0.20

        threshold = 0.55

        if buy_score >= threshold and buy_score > sell_score * 1.2:
            return self._signal(symbol, timeframe, "BUY", buy_score,
                f"OBV_up={obv_trend_up} VWAP_dev={vwap_dev:.2f}% buy_ratio={buy_ratio:.2f}")
        if sell_score >= threshold and sell_score > buy_score * 1.2:
            return self._signal(symbol, timeframe, "SELL", sell_score,
                f"OBV_up={obv_trend_up} VWAP_dev={vwap_dev:.2f}% buy_ratio={buy_ratio:.2f}")

        return self.hold(symbol, timeframe,
            f"Volume neutral: vol_z={vol_z:.2f} buy_ratio={buy_ratio:.2f}")
