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

Changes vs v2 (Futures Mode support):
  5. PARAMETERIZED: every knob that used to be read straight off the
     module-level `settings` object (timeframes, primary timeframe,
     consensus threshold, agents-agree, tie-break, confluence bonus/
     penalty, regime multiplier, ATR sizing bounds) is now a
     constructor argument, defaulting to the original spot settings
     so `PolyphonicConsensusEngine(agents)` behaves EXACTLY as before.
     Futures Mode instantiates this same class with its own FUTURES_*
     config and a futures market-data provider — same mechanism,
     different dial settings, with no code fork.
  6. Regime lookups now go through the RegimeAgent INSTANCE passed in
     via `agents` (found by isinstance) instead of RegimeAgent
     classmethods, since regime_agent.py moved its state from
     class-level to instance-level (see agents/regime_agent.py).
  7. Market data access is injected (`market_data_provider`, duck-typed
     to expose fetch_candles/fetch_orderbook/fetch_fear_greed/
     fetch_funding_rate) instead of importing spot's market_data
     functions directly, so the same engine can point at Binance Spot
     or Binance Futures.
"""
import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import settings
import market_data as _spot_market_data
from agents.base_agent import BaseAgent, AgentSignal
from agents.regime_agent import RegimeAgent, REGIME_VOLATILE, REGIME_UNKNOWN

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
    def __init__(
        self,
        agents: list[BaseAgent],
        market_data_provider=None,
        timeframes: Optional[list[str]] = None,
        primary_timeframe: Optional[str] = None,
        min_consensus_score: Optional[float] = None,
        min_agents_agree: Optional[int] = None,
        tie_break_action: Optional[str] = None,
        require_temporal_confluence: Optional[bool] = None,
        confluence_score_bonus: Optional[float] = None,
        no_confluence_score_penalty: Optional[float] = None,
        regime_hard_block_volatile: Optional[bool] = None,
        regime_volatile_score_multiplier: Optional[float] = None,
        candle_limit: Optional[int] = None,
        sl_atr_mult: float = 1.5,
        sl_min_pct: float = 0.008,
        sl_max_pct: float = 0.025,
        rr_trend: float = 3.0,
        rr_range: float = 2.0,
    ):
        self.agents = agents
        # Defaults to spot market_data module — Futures Mode passes
        # futures_market_data instead (same function names, fapi URLs).
        self.md = market_data_provider or _spot_market_data

        self.timeframes = timeframes or settings.TIMEFRAMES
        self.primary_timeframe = primary_timeframe or settings.PRIMARY_TIMEFRAME
        self.min_consensus_score = (
            settings.MIN_CONSENSUS_SCORE if min_consensus_score is None else min_consensus_score
        )
        self.min_agents_agree = (
            settings.MIN_AGENTS_AGREE if min_agents_agree is None else min_agents_agree
        )
        self.tie_break_action = tie_break_action or settings.TIE_BREAK_ACTION
        self.require_temporal_confluence = (
            settings.REQUIRE_TEMPORAL_CONFLUENCE
            if require_temporal_confluence is None else require_temporal_confluence
        )
        self.confluence_score_bonus = (
            settings.CONFLUENCE_SCORE_BONUS if confluence_score_bonus is None else confluence_score_bonus
        )
        self.no_confluence_score_penalty = (
            settings.NO_CONFLUENCE_SCORE_PENALTY
            if no_confluence_score_penalty is None else no_confluence_score_penalty
        )
        self.regime_hard_block_volatile = (
            settings.REGIME_HARD_BLOCK_VOLATILE
            if regime_hard_block_volatile is None else regime_hard_block_volatile
        )
        self.regime_volatile_score_multiplier = (
            settings.REGIME_VOLATILE_SCORE_MULTIPLIER
            if regime_volatile_score_multiplier is None else regime_volatile_score_multiplier
        )
        self.candle_limit = candle_limit or settings.CANDLE_LIMIT

        # ATR-based exit sizing bounds — spot defaults match the original
        # hardcoded v1/v2 values exactly; Futures Mode passes tighter ones.
        self.sl_atr_mult = sl_atr_mult
        self.sl_min_pct = sl_min_pct
        self.sl_max_pct = sl_max_pct
        self.rr_trend = rr_trend
        self.rr_range = rr_range

        # Find the RegimeAgent instance in the provided agent list so we
        # read regime state from THIS engine's instance, not a shared
        # class-level cache (see agents/regime_agent.py v3 notes).
        self.regime_agent: Optional[RegimeAgent] = next(
            (a for a in agents if isinstance(a, RegimeAgent)), None
        )

        self._accuracy: dict[str, list[bool]] = defaultdict(list)

    # ─────────────────────────────────────────
    #  Main Entry Point
    # ─────────────────────────────────────────
    async def evaluate(self, symbol: str) -> ConsensusResult:
        """Run all agents across all timeframes and return consensus."""

        candles: dict[str, pd.DataFrame] = {}
        for tf in self.timeframes:
            try:
                candles[tf] = await self.md.fetch_candles(symbol, tf, self.candle_limit)
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
        regime = self.regime_agent.get_regime(symbol) if self.regime_agent else REGIME_UNKNOWN
        regime_multiplier = 1.0
        if regime == REGIME_VOLATILE:
            if self.regime_hard_block_volatile:
                logger.info(f"{symbol}: Regime=VOLATILE — hard block enabled, skipping")
                return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                        agents_agree=0, total_agents=len(all_signals),
                                        regime=regime, signals=all_signals,
                                        primary_reason="Regime: VOLATILE — trading paused (hard block)")
            regime_multiplier = self.regime_volatile_score_multiplier
            logger.info(f"{symbol}: Regime=VOLATILE — applying {regime_multiplier}x score penalty")

        primary_tf = self.primary_timeframe
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
            direction = self.tie_break_action
            if direction == "HOLD":
                return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                        agents_agree=0, total_agents=len(all_signals),
                                        regime=regime, signals=all_signals,
                                        primary_reason=f"Primary timeframe tied {buy_count}-{sell_count} → HOLD")
        else:
            direction = "BUY" if buy_count > sell_count else "SELL"

        # ── Temporal confluence: soft bonus/penalty by default ──
        other_tfs = [tf for tf in self.timeframes if tf != primary_tf]
        confirmed = any(
            any(s.signal == direction and s.timeframe == tf for s in all_signals)
            for tf in other_tfs
        )
        if not confirmed and self.require_temporal_confluence:
            return ConsensusResult(symbol=symbol, action="HOLD", score=0.0,
                                    agents_agree=0, total_agents=len(all_signals),
                                    regime=regime, signals=all_signals,
                                    primary_reason=f"No temporal confluence for {direction}")

        # ── Weighted consensus score ────────
        score, agree_count = self._compute_weighted_score(all_signals, direction)
        score *= regime_multiplier
        if confirmed:
            score = min(score + self.confluence_score_bonus, 1.0)
        else:
            score = max(score - self.no_confluence_score_penalty, 0.0)

        if (score >= self.min_consensus_score and
                agree_count >= self.min_agents_agree):

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
                            f"(need score>={self.min_consensus_score}, agree>={self.min_agents_agree})"
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
            extras["fear_greed"] = await self.md.fetch_fear_greed()
        except Exception:
            extras["fear_greed"] = 50
        try:
            extras["orderbook"] = {symbol: await self.md.fetch_orderbook(symbol)}
        except Exception:
            extras["orderbook"] = {}
        try:
            extras["funding_rate"] = {symbol: await self.md.fetch_funding_rate(symbol)}
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
        into position sizing downstream. Bounds/multiplier/RR ratios are
        now configurable per-engine (see __init__) instead of hardcoded,
        so Futures Mode can run tighter, faster exits."""
        fallback = (self.sl_min_pct, self.sl_min_pct * self.rr_range)

        if df is None or df.empty:
            return fallback

        atr = df["atr"].iloc[-1]
        close = df["close"].iloc[-1]

        if math.isnan(atr) or math.isnan(close) or close <= 0:
            return fallback

        atr_pct = atr / close
        if math.isnan(atr_pct):
            return fallback

        sl = min(max(atr_pct * self.sl_atr_mult, self.sl_min_pct), self.sl_max_pct)
        rr = self.rr_trend if "TREND" in regime else self.rr_range
        tp = sl * rr

        return round(sl, 4), round(tp, 4)