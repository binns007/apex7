"""
APEX-7 — Signal Lab
════════════════════
The "what if we'd taken this trade?" data-collection and analysis layer.

THE PROBLEM THIS SOLVES
────────────────────────
Every scan cycle, the Polyphonic Consensus Engine looks at 8 agents and
decides BUY / SELL / HOLD. When it decides HOLD because only 2-of-8
agents agreed (say Momentum at 60% and OrderBook at 55%), that candidate
signal is just... gone. There's no record of what would have happened
had we taken it. That means questions like "is 4-of-8 actually the right
threshold, or would 2-of-8 with the right two agents have been fine?" or
"is the Sentiment agent actually pulling its weight?" can never be
answered from real data — only from guesswork.

HOW IT WORKS
────────────
1. RECORD  — Every scan cycle, `record_candidate()` is called for every
   symbol where the consensus engine found a real directional majority
   on the primary timeframe (see consensus_engine.py's
   `candidate_direction`), REGARDLESS of whether the score/agents-agree
   thresholds were cleared and a real order was placed. Each row snapshots:
     - the candidate direction and entry price
     - the exact ATR-based SL/TP the real engine would have used
     - a full per-agent breakdown: who agreed, at what confidence, on
       which timeframe
     - whether it was ACTUALLY traded (was_taken) and, if so, a link to
       the real Trade/FuturesTrade row

2. RESOLVE — Every scan cycle, `resolve_pending()` polls all still-open
   ("PENDING") shadow candidates against live prices and marks them
   TP / SL / EXPIRED (forced resolution after a max hold time) exactly
   like a real position-monitor loop would, using the SAME SL/TP levels
   captured at record time. This produces both a GROSS hypothetical PnL
   (raw price movement on a fixed notional stake) and a FEE-ADJUSTED NET
   PnL (see "Fee adjustment" below) without ever touching the exchange.

3. ANALYZE — `agent_performance()`, `combo_performance()`, and
   `agents_agree_bucket_performance()` slice the resolved history by
   individual agent, by exact agreeing-agent combination, and by
   agents-agree count, so patterns like "Momentum+OrderBook alone has a
   71% hypothetical win rate" surface from real forward-tested data
   instead of a hunch. `recommend_weights()` turns per-agent accuracy
   into a bounded, transparent weight adjustment that can be applied to
   the LIVE engine via TradingEngine.apply_agent_weights() /
   FuturesTradingEngine.apply_agent_weights() — no redeploy required.

FEE ADJUSTMENT
──────────────
Raw price-move PnL (`pnl_usdt` / `pnl_pct`, referred to below and in the
API as GROSS) badly overstates real profitability at the scale this bot
trades at: many resolved outcomes move well under 0.5%, while a Binance
round-trip (entry taker + exit taker, the conservative assumption — see
config.py) already costs ~0.1-0.2% on spot. Every resolution now also
computes a NET figure (`net_pnl_usdt`) with that cost subtracted, and
every analytics function below is based on NET by default — "what if
we'd taken this?" should mean "in a real account," not "on a frictionless
simulator." GROSS is still reported alongside NET for comparison.
`backfill_fees()` retroactively computes NET for any row that resolved
before this feature existed, so historical data isn't stuck showing
inflated pre-fee numbers forever.

Two real costs are still NOT modeled anywhere in this file: slippage on
market-order fills, and (for futures) funding-rate carry cost on open
positions. Both are real money a live account pays. That means even
these NET numbers remain a best case, not a true worst case — keep that
in mind before treating a marginally-positive NET result as proven edge.

This module never places, modifies, or cancels a real order. It is pure
observation layered on top of what the consensus engine already
computes every cycle.
"""
import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update

from config import settings
from database import AsyncSessionLocal, SignalOutcome

logger = logging.getLogger("apex7.signal_lab")

RESOLVED_STATUSES = ("TP", "SL", "EXPIRED")


def _fee_pct_for(market_type: str) -> float:
    """Round-trip fee (%) — 2x the per-side taker rate, see config.py's
    fee-adjustment notes for why taker-on-both-legs is the assumption."""
    per_side = (
        settings.SIGNAL_LAB_SPOT_TAKER_FEE_PCT if market_type == "SPOT"
        else settings.SIGNAL_LAB_FUTURES_TAKER_FEE_PCT
    )
    return per_side * 2


def _net(row: SignalOutcome) -> float:
    """The realistic PnL for a resolved row: NET if already computed,
    otherwise falls back to GROSS (should only happen for rows that
    predate fee adjustment and haven't been through backfill_fees() yet)."""
    if row.net_pnl_usdt is not None:
        return row.net_pnl_usdt
    return row.pnl_usdt or 0.0


def _gross(row: SignalOutcome) -> float:
    return row.pnl_usdt or 0.0


# ─────────────────────────────────────────────────────
#  Recording
# ─────────────────────────────────────────────────────
async def record_candidate(
    market_type: str,
    symbol: str,
    price: float,
    result,                      # consensus_engine.ConsensusResult
    was_taken: bool,
    linked_trade_id: Optional[int] = None,
) -> Optional[int]:
    """Persist one shadow candidate for this scan cycle. Returns the new
    row id (so the caller can `link_trade()` it to a real Trade id once
    that trade is saved), or None if there was nothing to record."""
    if not settings.SIGNAL_LAB_ENABLED:
        return None
    if not getattr(result, "candidate_direction", None):
        return None
    if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
        return None

    direction = result.candidate_direction
    sl_pct = result.stop_loss_pct
    tp_pct = result.take_profit_pct

    if sl_pct is None or tp_pct is None or sl_pct <= 0:
        return None

    if direction == "BUY":
        sl_price = price * (1 - sl_pct)
        tp_price = price * (1 + tp_pct)
    else:
        sl_price = price * (1 + sl_pct)
        tp_price = price * (1 - tp_pct)

    row = SignalOutcome(
        market_type=market_type,
        symbol=symbol,
        direction=direction,
        entry_price=price,
        consensus_score=result.score,
        agents_agree=result.agents_agree,
        total_agents=result.total_agents,
        regime=result.regime,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        was_taken=was_taken,
        linked_trade_id=linked_trade_id,
        agent_detail_json=json.dumps(result.candidate_agent_detail),
        status="PENDING",
        created_at=datetime.now(timezone.utc),
    )
    try:
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.id
    except Exception as e:
        logger.debug(f"Signal Lab record_candidate failed for {symbol}: {e}")
        return None


async def link_trade(shadow_id: Optional[int], trade_id: Optional[int]):
    """Called once a shadow candidate turns into a real order, so the
    shadow row can be cross-referenced against the real Trade/
    FuturesTrade row later (e.g. to sanity-check real vs hypothetical
    PnL agree for taken trades)."""
    if not shadow_id or not trade_id:
        return
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(SignalOutcome).where(SignalOutcome.id == shadow_id)
                .values(linked_trade_id=trade_id, was_taken=True)
            )
            await session.commit()
    except Exception as e:
        logger.debug(f"Signal Lab link_trade failed: {e}")


# ─────────────────────────────────────────────────────
#  Resolution — forward-tracks PENDING candidates against live prices
# ─────────────────────────────────────────────────────
async def resolve_pending(market_type: str, market_data_module, max_hold_minutes: float):
    """Poll all PENDING shadow rows for this market and resolve them
    against live prices: TP hit, SL hit, or EXPIRED (forced resolution
    at the current price after max_hold_minutes with no touch, so slow
    candidates don't just accumulate forever). Computes both GROSS
    (raw price-move) and fee-adjusted NET PnL on resolution."""
    if not settings.SIGNAL_LAB_ENABLED:
        return

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SignalOutcome).where(
                SignalOutcome.status == "PENDING",
                SignalOutcome.market_type == market_type,
            )
        )
        rows = result.scalars().all()

    if not rows:
        return

    symbols = list({r.symbol for r in rows})
    try:
        prices = await market_data_module.fetch_all_prices(symbols)
    except Exception as e:
        logger.debug(f"Signal Lab price fetch failed ({market_type}): {e}")
        return

    now = datetime.now(timezone.utc)
    notional = settings.SIGNAL_LAB_NOTIONAL_USDT
    fee_pct_roundtrip = _fee_pct_for(market_type)
    fee_usdt = round(notional * fee_pct_roundtrip / 100, 4)

    for r in rows:
        price = prices.get(r.symbol)
        if not price:
            continue

        hit_tp = hit_sl = False
        if r.direction == "BUY":
            hit_tp = price >= r.take_profit_price
            hit_sl = price <= r.stop_loss_price
        else:
            hit_tp = price <= r.take_profit_price
            hit_sl = price >= r.stop_loss_price

        created = r.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_minutes = (now - created).total_seconds() / 60

        status = exit_price = None
        if hit_tp and hit_sl:
            # Both levels crossed between polls (gap/fast move) — resolve
            # conservatively as the stop, mirroring how the real position
            # monitor should be read when it can't tell which came first.
            status, exit_price = "SL", r.stop_loss_price
        elif hit_tp:
            status, exit_price = "TP", r.take_profit_price
        elif hit_sl:
            status, exit_price = "SL", r.stop_loss_price
        elif age_minutes >= max_hold_minutes:
            status, exit_price = "EXPIRED", price

        if status is None:
            continue

        pnl_pct = (exit_price - r.entry_price) / r.entry_price * 100
        if r.direction == "SELL":
            pnl_pct = -pnl_pct
        gross_pnl_usdt = round(notional * pnl_pct / 100, 4)
        net_pnl_usdt = round(gross_pnl_usdt - fee_usdt, 4)

        try:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(SignalOutcome).where(SignalOutcome.id == r.id).values(
                        status=status, exit_price=exit_price,
                        pnl_pct=round(pnl_pct, 4), pnl_usdt=gross_pnl_usdt,
                        fee_usdt=fee_usdt, net_pnl_usdt=net_pnl_usdt,
                        resolved_at=now,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.debug(f"Signal Lab resolve failed for row {r.id}: {e}")


async def backfill_fees(batch_size: int = 500) -> int:
    """One-time-per-row (safe to call on every startup — only touches
    rows still missing net_pnl_usdt), retroactive fee adjustment for any
    row that resolved BEFORE this feature existed. Without this, historical
    Signal Lab data would show only the old, fee-free GROSS numbers
    forever, silently overstating how well those candidates would
    actually have done. Returns the number of rows updated."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SignalOutcome).where(
                SignalOutcome.status.in_(RESOLVED_STATUSES),
                SignalOutcome.net_pnl_usdt.is_(None),
            )
        )
        rows = result.scalars().all()
        if not rows:
            return 0

        notional = settings.SIGNAL_LAB_NOTIONAL_USDT
        for i, r in enumerate(rows):
            fee_pct_roundtrip = _fee_pct_for(r.market_type)
            fee = round(notional * fee_pct_roundtrip / 100, 4)
            r.fee_usdt = fee
            r.net_pnl_usdt = round((r.pnl_usdt or 0.0) - fee, 4)
            if (i + 1) % batch_size == 0:
                await session.commit()
        await session.commit()

    logger.info(f"Signal Lab: backfilled fee adjustment on {len(rows)} historical resolved rows")
    return len(rows)


# ─────────────────────────────────────────────────────
#  Analysis — everything below is NET (fee-adjusted) by default,
#  with GROSS reported alongside for comparison.
# ─────────────────────────────────────────────────────
async def _fetch_resolved(
    market_type: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 10000,
) -> list:
    async with AsyncSessionLocal() as session:
        q = select(SignalOutcome).where(SignalOutcome.status.in_(RESOLVED_STATUSES))
        if market_type:
            q = q.where(SignalOutcome.market_type == market_type)
        if symbol:
            q = q.where(SignalOutcome.symbol == symbol)
        q = q.order_by(SignalOutcome.resolved_at.desc()).limit(limit)
        result = await session.execute(q)
        return result.scalars().all()


async def agent_performance(
    market_type: Optional[str] = None,
    symbol: Optional[str] = None,
    min_samples: Optional[int] = None,
) -> list[dict]:
    """Per-agent hypothetical performance, split by whether that agent
    AGREED with the eventual candidate direction that cycle or not.
    A well-calibrated agent should show a materially better NET win rate
    / avg PnL in the "agreed" bucket than the "disagreed" bucket."""
    min_samples = settings.SIGNAL_LAB_MIN_SAMPLES if min_samples is None else min_samples
    rows = await _fetch_resolved(market_type, symbol)

    stats = defaultdict(lambda: {
        "agreed_wins": 0, "agreed_total": 0, "agreed_net_pnl": 0.0, "agreed_gross_pnl": 0.0,
        "disagreed_wins": 0, "disagreed_total": 0, "disagreed_net_pnl": 0.0,
    })
    for r in rows:
        try:
            detail = json.loads(r.agent_detail_json or "[]")
        except Exception:
            continue
        net = _net(r)
        gross = _gross(r)
        is_win = net > 0
        for d in detail:
            name = d.get("agent")
            if not name:
                continue
            bucket = "agreed" if d.get("agreed") else "disagreed"
            stats[name][f"{bucket}_total"] += 1
            stats[name][f"{bucket}_net_pnl"] += net
            if bucket == "agreed":
                stats[name]["agreed_gross_pnl"] += gross
            if is_win:
                stats[name][f"{bucket}_wins"] += 1

    out = []
    for name, s in stats.items():
        agreed_total = s["agreed_total"]
        if agreed_total < min_samples:
            continue
        disagreed_total = s["disagreed_total"]
        out.append({
            "agent": name,
            "samples_when_agreed": agreed_total,
            "win_rate_when_agreed_pct": round(s["agreed_wins"] / agreed_total * 100, 2),
            "avg_pnl_usdt_when_agreed": round(s["agreed_net_pnl"] / agreed_total, 4),
            "avg_gross_pnl_usdt_when_agreed": round(s["agreed_gross_pnl"] / agreed_total, 4),
            "samples_when_disagreed": disagreed_total,
            "win_rate_when_disagreed_pct": (
                round(s["disagreed_wins"] / disagreed_total * 100, 2) if disagreed_total else None
            ),
            "avg_pnl_usdt_when_disagreed": (
                round(s["disagreed_net_pnl"] / disagreed_total, 4) if disagreed_total else None
            ),
        })
    out.sort(key=lambda x: x["avg_pnl_usdt_when_agreed"], reverse=True)
    return out


async def combo_performance(
    market_type: Optional[str] = None,
    symbol: Optional[str] = None,
    min_samples: Optional[int] = None,
    max_combos: int = 30,
) -> list[dict]:
    """Hypothetical performance grouped by the EXACT set of agreeing
    agents — this is the direct answer to "what if only Momentum and
    OrderBook had agreed, would that have been enough?": every cycle
    where exactly that pair (and no one else) agreed gets bucketed
    together and scored on real forward outcomes (NET of fees)."""
    min_samples = settings.SIGNAL_LAB_MIN_SAMPLES if min_samples is None else min_samples
    rows = await _fetch_resolved(market_type, symbol)

    stats = defaultdict(lambda: {"wins": 0, "total": 0, "net_pnl": 0.0, "gross_pnl": 0.0})
    for r in rows:
        try:
            detail = json.loads(r.agent_detail_json or "[]")
        except Exception:
            continue
        agreeing = tuple(sorted(d["agent"] for d in detail if d.get("agreed")))
        if not agreeing:
            continue
        net = _net(r)
        gross = _gross(r)
        stats[agreeing]["total"] += 1
        stats[agreeing]["net_pnl"] += net
        stats[agreeing]["gross_pnl"] += gross
        if net > 0:
            stats[agreeing]["wins"] += 1

    out = []
    for combo, s in stats.items():
        if s["total"] < min_samples:
            continue
        out.append({
            "agents": list(combo),
            "agent_count": len(combo),
            "samples": s["total"],
            "win_rate_pct": round(s["wins"] / s["total"] * 100, 2),
            "avg_pnl_usdt": round(s["net_pnl"] / s["total"], 4),
            "avg_gross_pnl_usdt": round(s["gross_pnl"] / s["total"], 4),
            "total_pnl_usdt": round(s["net_pnl"], 4),
        })
    out.sort(key=lambda x: x["avg_pnl_usdt"], reverse=True)
    return out[:max_combos]


async def agents_agree_bucket_performance(
    market_type: Optional[str] = None,
    symbol: Optional[str] = None,
) -> list[dict]:
    """Hypothetical performance grouped by RAW agents-agree count
    (2-of-8, 3-of-8, ... 8-of-8) — the most direct evidence for whether
    MIN_AGENTS_AGREE is set too high, too low, or about right. NET of fees."""
    rows = await _fetch_resolved(market_type, symbol)
    stats = defaultdict(lambda: {"wins": 0, "total": 0, "net_pnl": 0.0, "gross_pnl": 0.0})
    for r in rows:
        net = _net(r)
        gross = _gross(r)
        stats[r.agents_agree]["total"] += 1
        stats[r.agents_agree]["net_pnl"] += net
        stats[r.agents_agree]["gross_pnl"] += gross
        if net > 0:
            stats[r.agents_agree]["wins"] += 1

    out = []
    for n, s in sorted(stats.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if s["total"] == 0:
            continue
        out.append({
            "agents_agree": n,
            "samples": s["total"],
            "win_rate_pct": round(s["wins"] / s["total"] * 100, 2),
            "avg_pnl_usdt": round(s["net_pnl"] / s["total"], 4),
            "avg_gross_pnl_usdt": round(s["gross_pnl"] / s["total"], 4),
        })
    return out


async def recommend_weights(
    market_type: str,
    min_samples: Optional[int] = None,
    base_weights: Optional[dict] = None,
) -> list[dict]:
    """A simple, transparent re-weighting rule, NOT a black box: each
    agent's live weight is nudged toward its measured hypothetical
    accuracy (NET win rate when it agreed with the eventual candidate
    direction), bounded to ±50% per pass so one noisy batch of data
    can't swing the engine to an extreme. Intended to be re-run
    periodically (e.g. weekly) as more resolved history accumulates —
    each pass compounds gently rather than leaping straight to whatever
    the current sample suggests.

        multiplier = 1.0 + (win_rate - 0.50) * 2.5   clamped to [0.5, 1.5]

    50% win rate → 1.00x (no change). 60% → 1.25x. 70% → 1.50x (capped).
    40% → 0.75x. 30% → 0.50x (capped). This mirrors, at a much simpler
    level, the same idea as the engine's built-in rolling accuracy
    multiplier (_accuracy_multiplier in consensus_engine.py) but is
    explicit, inspectable, and only applied when YOU choose to apply it.

    Uses NET (fee-adjusted) win rate, since a weight recommendation
    based on GROSS numbers could recommend increasing an agent's
    influence based on "wins" that wouldn't survive real trading costs.
    """
    min_samples = settings.SIGNAL_LAB_MIN_SAMPLES if min_samples is None else min_samples
    perf = await agent_performance(market_type=market_type, min_samples=min_samples)
    base_weights = base_weights or {}

    recs = []
    for p in perf:
        name = p["agent"]
        base = base_weights.get(name, 1.0)
        win_rate = p["win_rate_when_agreed_pct"] / 100
        multiplier = 1.0 + (win_rate - 0.50) * 2.5
        multiplier = max(0.5, min(1.5, multiplier))
        new_weight = round(base * multiplier, 3)
        recs.append({
            "agent": name,
            "current_weight": round(base, 3),
            "samples": p["samples_when_agreed"],
            "win_rate_when_agreed_pct": p["win_rate_when_agreed_pct"],
            "avg_pnl_usdt_when_agreed": p["avg_pnl_usdt_when_agreed"],
            "suggested_weight": new_weight,
        })
    recs.sort(key=lambda x: x["suggested_weight"] - x["current_weight"], reverse=True)
    return recs


async def status_summary(market_type: Optional[str] = None) -> dict:
    async with AsyncSessionLocal() as session:
        q = select(SignalOutcome)
        if market_type:
            q = q.where(SignalOutcome.market_type == market_type)
        result = await session.execute(q)
        rows = result.scalars().all()

    total = len(rows)
    pending = sum(1 for r in rows if r.status == "PENDING")
    resolved = [r for r in rows if r.status in RESOLVED_STATUSES]
    taken = sum(1 for r in rows if r.was_taken)

    net_wins = sum(1 for r in resolved if _net(r) > 0)
    gross_wins = sum(1 for r in resolved if _gross(r) > 0)
    total_fees = sum(r.fee_usdt or 0.0 for r in resolved)
    total_gross_pnl = sum(_gross(r) for r in resolved)
    total_net_pnl = sum(_net(r) for r in resolved)

    return {
        "enabled": settings.SIGNAL_LAB_ENABLED,
        "total_candidates": total,
        "pending": pending,
        "resolved": len(resolved),
        "taken_as_real_trades": taken,
        "shadow_only": total - taken,
        "resolved_win_rate_pct": round(net_wins / len(resolved) * 100, 2) if resolved else 0,
        "resolved_win_rate_gross_pct": round(gross_wins / len(resolved) * 100, 2) if resolved else 0,
        "total_fees_usdt": round(total_fees, 2),
        "total_gross_pnl_usdt": round(total_gross_pnl, 2),
        "total_net_pnl_usdt": round(total_net_pnl, 2),
        "notional_stake_usdt": settings.SIGNAL_LAB_NOTIONAL_USDT,
    }