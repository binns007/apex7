"""
APEX-7 — Polyphonic Consensus Engine (PCE)
══════════════════════════════════════════
The PCE is the defining innovation of APEX-7.

Instead of one strategy, eight specialized agents each independently
analyse the market from a different perspective. Each vote is WEIGHTED
by the agent's recent accuracy and by the signal's confidence.

A trade only fires when:
  1. Weighted consensus score ≥ MIN_CONSENSUS_SCORE
  2. At least MIN_AGENTS_AGREE agents point in the same direction
  3. Temporal confluence: the primary signal aligns on 2+ timeframes
  4. The regime agent confirms (or is neutral) — never fight the regime

This means APEX-7 only enters trades where multiple independent
evidence streams converge at the same moment — massively reducing
false positives while catching high-quality setups.
"""
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import settings
from agents.base_agent import BaseAgent, AgentSignal
from agents.regime_agent import RegimeAgent, REGIME_VOLATILE
from market_data import fetch_candles, fetch_orderbook, fetch_fear_greed, fetch_funding_rate

logger = logging.getLogger("apex7.consensus")


@dataclass
class ConsensusResult:
    symbol: str
    action: str             # "BUY" | "SELL" | "HOLD"
    score: float            # 0.0 – 1.0 weighted agreement
    agents_agree: int
    total_agents: int
    regime: str
    signals: list[AgentSignal] = field(default_factory=list)
    primary_reason: str = ""
    stop_loss_pct: float = 0.012   # default 1.2%
    take_profit_pct: float = 0.024  # default 2.4% (2:1 R/R)


class PolyphonicConsensusEngine:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        # Dynamic weights: adjusted based on agent rolling accuracy
        self._accuracy: dict[str, list[bool]] = defaultdict(list)

    # ─────────────────────────────────────────
    #  Main Entry Point
    # ─────────────────────────────────────────
    async def evaluate(self, symbol: str) -> ConsensusResult:
        """Run all agents across all timeframes and return consensus."""

        # ── 1. Fetch data for all timeframes ──
        candles: dict[str, pd.DataFrame] = {}
        for tf in settings.TIMEFRAMES:
            try:
                candles[tf] = await fetch_candles(symbol, tf, settings.CANDLE_LIMIT)
            except Exception as e:
                logger.warning(f"Candle fetch failed {symbol}/{tf}: {e}")

        if not candles:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                   agents_agree=0, total_agents=0, regime="UNKNOWN")

        # ── 2. Fetch extras ───────────────────
        extras = await self._fetch_extras(symbol)

        # ── 3. Collect signals from all agents ─
        all_signals: list[AgentSignal] = []
        tasks = []
        for agent in self.agents:
            for tf, df in candles.items():
                tasks.append(self._safe_analyze(agent, symbol, tf, df, extras))

        results = await asyncio.gather(*tasks)
        all_signals = [r for r in results if r is not None]

        # ── 4. Regime check ───────────────────
        regime = RegimeAgent.get_regime(symbol)
        if regime == REGIME_VOLATILE:
            logger.info(f"{symbol}: Regime=VOLATILE — skipping consensus")
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                   agents_agree=0, total_agents=len(all_signals),
                                   regime=regime, signals=all_signals,
                                   primary_reason="Regime: VOLATILE — trading paused")

        # ── 5. Temporal confluence filter ──────
        # Primary timeframe = 5m. Signal must also appear on 1m OR 15m
        primary_tf = "5m"
        primary_signals = [s for s in all_signals if s.timeframe == primary_tf and s.signal != "HOLD"]
        if not primary_signals:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                   agents_agree=0, total_agents=len(all_signals),
                                   regime=regime, signals=all_signals,
                                   primary_reason="No primary (5m) signal found")

        # Determine primary direction
        buy_count  = sum(1 for s in primary_signals if s.signal == "BUY")
        sell_count = sum(1 for s in primary_signals if s.signal == "SELL")
        direction  = "BUY" if buy_count > sell_count else "SELL"

        # Confirm on at least one other timeframe
        other_tfs = [tf for tf in settings.TIMEFRAMES if tf != primary_tf]
        confirmed = any(
            any(s.signal == direction and s.timeframe == tf for s in all_signals)
            for tf in other_tfs
        )
        if not confirmed:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                   agents_agree=0, total_agents=len(all_signals),
                                   regime=regime, signals=all_signals,
                                   primary_reason=f"No temporal confluence for {direction}")

        # ── 6. Weighted consensus score ────────
        score, agree_count = self._compute_weighted_score(all_signals, direction)

        # ── 7. Threshold check ────────────────
        if (score >= settings.MIN_CONSENSUS_SCORE and
                agree_count >= settings.MIN_AGENTS_AGREE):

            # Compute dynamic R/R from ATR
            sl_pct, tp_pct = self._dynamic_rr(candles.get("5m"), regime)

            top_reason = sorted(
                [s for s in all_signals if s.signal == direction],
                key=lambda s: s.confidence, reverse=True
            )[0].reason if all_signals else ""

            logger.info(f"✅ {symbol} CONSENSUS {direction} score={score:.3f} agents={agree_count}")
            return ConsensusResult(
                symbol=symbol,
                action=direction,
                score=score,
                agents_agree=agree_count,
                total_agents=len(set(s.agent_name for s in all_signals)),
                regime=regime,
                signals=all_signals,
                primary_reason=top_reason,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
            )

        return ConsensusResult(
            symbol=symbol, action="HOLD", score=score,
            agents_agree=agree_count,
            total_agents=len(set(s.agent_name for s in all_signals)),
            regime=regime, signals=all_signals,
            primary_reason=f"Threshold not met: score={score:.3f} agree={agree_count}"
        )

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────
    async def _safe_analyze(self, agent, symbol, tf, df, extras) -> Optional[AgentSignal]:
        try:
            return await agent.analyze(symbol, tf, df, extras)
        except Exception as e:
            logger.error(f"{agent.name}/{symbol}/{tf} error: {e}")
            return None

    async def _fetch_extras(self, symbol: str) -> dict:
        extras = {}
        try:
            extras["fear_greed"] = await fetch_fear_greed()
        except Exception:
            extras["fear_greed"] = 50
        try:
            extras["orderbook"] = {symbol: await fetch_orderbook(symbol)}
        except Exception:
            extras["orderbook"] = {}
        try:
            extras["funding_rate"] = {symbol: await fetch_funding_rate(symbol)}
        except Exception:
            extras["funding_rate"] = {symbol: 0.0}
        return extras

    def _compute_weighted_score(
        self, signals: list[AgentSignal], direction: str
    ) -> tuple[float, int]:
        """
        Weight each agent's signal by:
          agent.weight × signal.confidence × accuracy_multiplier
        """
        agent_map: dict[str, list[AgentSignal]] = defaultdict(list)
        for s in signals:
            agent_map[s.agent_name].append(s)

        weighted_agree = 0.0
        weighted_total = 0.0
        agree_count = 0

        agent_lookup = {a.name: a for a in self.agents}

        for agent_name, sigs in agent_map.items():
            agent = agent_lookup.get(agent_name)
            base_weight = agent.weight if agent else 1.0
            acc_mult = self._accuracy_multiplier(agent_name)

            # Use the highest confidence signal from this agent (any TF)
            best = max(sigs, key=lambda s: s.confidence if s.signal == direction else -1)

            if best.signal == direction and best.confidence > 0:
                weighted_agree += base_weight * best.confidence * acc_mult
                agree_count += 1

            # Count all signals (including HOLD) as total weight
            max_conf = max(s.confidence for s in sigs)
            weighted_total += base_weight * max(max_conf, 0.3) * acc_mult

        score = weighted_agree / (weighted_total + 1e-9)
        return min(score, 1.0), agree_count

    def _accuracy_multiplier(self, agent_name: str) -> float:
        """Agents with better recent accuracy get higher weights."""
        history = self._accuracy.get(agent_name, [])
        if len(history) < 5:
            return 1.0
        recent = history[-20:]
        win_rate = sum(recent) / len(recent)
        # Scale: 0.7× for bad agents, 1.3× for great agents
        return 0.7 + win_rate * 0.6

    def update_agent_accuracy(self, agent_name: str, was_correct: bool):
        self._accuracy[agent_name].append(was_correct)

    def _dynamic_rr(
        self, df: Optional[pd.DataFrame], regime: str
    ) -> tuple[float, float]:
        """Calculate ATR-based stop loss and take profit %."""
        if df is None or df.empty:
            return 0.012, 0.024

        atr     = df["atr"].iloc[-1]
        close   = df["close"].iloc[-1]
        atr_pct = atr / close

        # Stop: 1.5× ATR below entry
        sl = min(max(atr_pct * 1.5, 0.008), 0.025)

        # TP: 2:1 risk/reward in ranging, 3:1 in trending
        rr = 3.0 if "TREND" in regime else 2.0
        tp = sl * rr

        return round(sl, 4), round(tp, 4)
