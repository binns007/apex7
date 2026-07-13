"""
APEX-7 Base Agent
All trading agents inherit from this class and implement analyze().
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class AgentSignal:
    agent_name: str
    symbol: str
    timeframe: str
    signal: str          # "BUY" | "SELL" | "HOLD"
    confidence: float    # 0.0 – 1.0
    reason: str

    def __post_init__(self):
        if self.signal not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"Invalid signal '{self.signal}' from {self.agent_name}")
        self.confidence = min(max(float(self.confidence), 0.0), 1.0)


class BaseAgent(ABC):
    name: str = "BaseAgent"
    weight: float = 1.0   # relative voting weight in consensus

    @abstractmethod
    async def analyze(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        extras: dict,
    ) -> AgentSignal:
        """
        Analyze market data and return a signal.
        extras may contain: orderbook, fear_greed, funding_rate, etc.
        """
        ...

    def _signal(self, symbol, tf, direction, confidence, reason) -> AgentSignal:
        return AgentSignal(
            agent_name=self.name,
            symbol=symbol,
            timeframe=tf,
            signal=direction,
            confidence=min(max(confidence, 0.0), 1.0),
            reason=reason,
        )

    def hold(self, symbol, tf, reason="No clear edge") -> AgentSignal:
        return self._signal(symbol, tf, "HOLD", 0.0, reason)

    def __repr__(self) -> str:
        return f"<{self.name} weight={self.weight}>"