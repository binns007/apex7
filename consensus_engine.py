"""
APEX-7 — Polyphonic Consensus Engine (PCE)
══════════════════════════════════════════
Eight specialized agents each independently analyse the market from a
different perspective. Each vote is WEIGHTED by the agent's recent
accuracy and by the signal's confidence.

Changes vs v1 (see inline comments for each):
  1. FIXED BUG: exact BUY/SELL ties on the primary timeframe used to
     silently resolve to SELL (`"BUY" if buy>sell else "SELL"` had no
     equality branch). Now explicit and configurable (default HOLD).
  2. The VOLATILE regime used to HARD BLOCK all trading before the
     score was even computed. It's now a configurable soft multiplier
     applied to the score (matches how a human desk would actually
     reduce, not zero out, conviction in choppy conditions).
  3. Temporal confluence used to be an unconditional hard gate. It's
     now a soft score bonus/penalty by default, with a config flag to
     restore the hard-gate behavior.
  4. NaN-safe: ATR-based R/R sizing no longer silently propagates NaN
     into stop-loss/take-profit percentages.
"""
import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import settings
from agents.base_agent import BaseAgent, AgentSignal
from agents.regime_agent import RegimeAgent, REGIME_VOLATILE, REGIME_UNKNOWN
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
    stop_loss_pct: float = 0.012
    take_profit_pct: float = 0.024


class PolyphonicConsensusEngine:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self._accuracy: dict[str, list[bool]] = defaultdict(list)

    # ─────────────────────────────────────────
    #  Main Entry Point
    # ─────────────────────────────────────────
    async def evaluate(self, symbol: str) -> ConsensusResult:
        """Run all agents across all timeframes and return consensus."""

        candles: dict[str, pd.DataFrame] = {}
        for tf in settings.TIMEFRAMES:
            try:
                candles[tf] = await fetch_candles(symbol, tf, settings.CANDLE_LIMIT)
            except Exception as e:
                logger.warning(f"Candle fetch failed {symbol}/{tf}: {e}")

        if not candles:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                    agents_agree=0, total_agents=0, regime=REGIME_UNKNOWN)

        extras = await self._fetch_extras(symbol)

        all_signals: list[AgentSignal] = []
        tasks = [
            self._safe_analyze(agent, symbol, tf, df, extras)
            for agent in self.agents
            for tf, df in candles.items()
        ]
        results = await asyncio.gather(*tasks)
        all_signals = [r for r in results if r is not None]

        # ── Regime: soft by default, hard-block only if explicitly enabled ──
        regime = RegimeAgent.get_regime(symbol)
        regime_multiplier = 1.0
        if regime == REGIME_VOLATILE:
            if settings.REGIME_HARD_BLOCK_VOLATILE:
                logger.info(f"{symbol}: Regime=VOLATILE — hard block enabled, skipping")
                return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                        agents_agree=0, total_agents=len(all_signals),
                                        regime=regime, signals=all_signals,
                                        primary_reason="Regime: VOLATILE — trading paused (hard block)")
            regime_multiplier = settings.REGIME_VOLATILE_SCORE_MULTIPLIER
            logger.info(f"{symbol}: Regime=VOLATILE — applying {regime_multiplier}x score penalty")

        primary_tf = settings.PRIMARY_TIMEFRAME
        primary_signals = [s for s in all_signals if s.timeframe == primary_tf and s.signal != "HOLD"]
        if not primary_signals:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                    agents_agree=0, total_agents=len(all_signals),
                                    regime=regime, signals=all_signals,
                                    primary_reason=f"No primary ({primary_tf}) signal found")

        # ── Determine primary direction — FIXED: exact ties no longer
        #    silently resolve to SELL. ──
        buy_count = sum(1 for s in primary_signals if s.signal == "BUY")
        sell_count = sum(1 for s in primary_signals if s.signal == "SELL")
        if buy_count == sell_count:
            direction = settings.TIE_BREAK_ACTION
            if direction == "HOLD":
                return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                        agents_agree=0, total_agents=len(all_signals),
                                        regime=regime, signals=all_signals,
                                        primary_reason=f"Primary timeframe tied {buy_count}-{sell_count} → HOLD")
        else:
            direction = "BUY" if buy_count > sell_count else "SELL"

        # ── Temporal confluence: soft bonus/penalty by default ──
        other_tfs = [tf for tf in settings.TIMEFRAMES if tf != primary_tf]
        confirmed = any(
            any(s.signal == direction and s.timeframe == tf for s in all_signals)
            for tf in other_tfs
        )
        if not confirmed and settings.REQUIRE_TEMPORAL_CONFLUENCE:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                    agents_agree=0, total_agents=len(all_signals),
                                    regime=regime, signals=all_signals,
                                    primary_reason=f"No temporal confluence for {direction}")

        # ── Weighted consensus score ────────
        score, agree_count = self._compute_weighted_score(all_signals, direction)
        score *= regime_multiplier
        if confirmed:
            score = min(score + settings.CONFLUENCE_SCORE_BONUS, 1.0)
        else:
            score = max(score - settings.NO_CONFLUENCE_SCORE_PENALTY, 0.0)

        if (score >= settings.MIN_CONSENSUS_SCORE and
                agree_count >= settings.MIN_AGENTS_AGREE):

            sl_pct, tp_pct = self._dynamic_rr(candles.get(primary_tf), regime)

            directional = [s for s in all_signals if s.signal == direction]
            top_reason = max(directional, key=lambda s: s.confidence).reason if directional else ""

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
            primary_reason=f"Threshold not met: score={score:.3f} agree={agree_count} "
                            f"(need score>={settings.MIN_CONSENSUS_SCORE}, agree>={settings.MIN_AGENTS_AGREE})"
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
        """Weight each agent's signal by agent.weight × confidence × accuracy."""
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

            best = max(sigs, key=lambda s: s.confidence if s.signal == direction else -1)

            if best.signal == direction and best.confidence > 0:
                weighted_agree += base_weight * best.confidence * acc_mult
                agree_count += 1

            max_conf = max(s.confidence for s in sigs)
            weighted_total += base_weight * max(max_conf, 0.3) * acc_mult

        if weighted_total <= 0:
            return 0.0, agree_count
        score = weighted_agree / weighted_total
        return min(score, 1.0), agree_count

    def _accuracy_multiplier(self, agent_name: str) -> float:
        history = self._accuracy.get(agent_name, [])
        if len(history) < 5:
            return 1.0
        recent = history[-20:]
        win_rate = sum(recent) / len(recent)
        return 0.7 + win_rate * 0.6

    def update_agent_accuracy(self, agent_name: str, was_correct: bool):
        self._accuracy[agent_name].append(was_correct)

    def _dynamic_rr(
        self, df: Optional[pd.DataFrame], regime: str
    ) -> tuple[float, float]:
        """Calculate ATR-based stop loss and take profit %.
        FIXED: v1 didn't guard against NaN ATR (possible before the ATR
        warmup period completes), which could silently propagate NaN
        into position sizing downstream."""
        if df is None or df.empty:
            return 0.012, 0.024

        atr = df["atr"].iloc[-1]
        close = df["close"].iloc[-1]

        if math.isnan(atr) or math.isnan(close) or close <= 0:
            return 0.012, 0.024

        atr_pct = atr / close
        if math.isnan(atr_pct):
            return 0.012, 0.024

        sl = min(max(atr_pct * 1.5, 0.008), 0.025)
        rr = 3.0 if "TREND" in regime else 2.0
        tp = sl * rr

        return round(sl, 4), round(tp, 4)