"""
APEX-8 Database Layer
SQLite via SQLAlchemy — stores trades, signals, and performance snapshots.

Changes vs v1: added indices on the columns actually queried
(status, symbol, created_at/opened_at) since the API filters/sorts on
these constantly as trade history grows.

Changes vs v2:
  - Added `risk_usdt` to Trade. The Risk Manager already computed this
    per-trade (PositionSize.risk_usdt) but it was never persisted, so
    the dashboard/API had no way to show per-trade risk. Existing rows
    will show NULL for this column (SQLite doesn't backfill history) —
    only trades opened after this change populate it.

Changes vs v3 (Futures Mode):
  - Added FuturesTrade / FuturesAgentSignal / FuturesPerformanceSnapshot
    as SEPARATE tables rather than bolting leverage/margin/liquidation
    columns onto the spot Trade table. Futures positions carry enough
    extra state (leverage, margin, liquidation price, ROI-on-margin %
    instead of raw price-%) that mixing them into one heavily-nullable
    table would make every spot query filter noise. Same column
    conventions and index strategy as the spot tables, so the two are
    easy to reason about side by side.

Changes vs v4 (Signal Lab):
  - Added SignalOutcome — ONE shared table (not split spot/futures,
    unlike Trade/FuturesTrade) that records every directional candidate
    the Polyphonic Consensus Engine produces each scan cycle, whether or
    not it actually became a real trade. A `market_type` column
    ("SPOT"/"FUTURES") distinguishes the two instead of a table split,
    because the whole point of this table is cross-cutting analysis
    (per-agent accuracy, per-combo win rate, per agents-agree-count
    bucket) that's easier to run over one table with a filter than to
    UNION two tables for. See signal_lab.py for the read/write logic.

Changes vs v5 (Signal Lab fee adjustment):
  - Added `fee_usdt` / `net_pnl_usdt` to SignalOutcome.

Changes vs v6 (real-trade fee adjustment):
  - Added `fee_usdt` / `net_pnl_usdt` / `net_pnl_pct` to Trade AND
    FuturesTrade too — the fee adjustment that used to only apply to
    Signal Lab's shadow candidates now applies to real closed trades as
    well, so "Realized PnL" on the dashboard reflects actual trading
    costs. `pnl_usdt` / `pnl_pct` are unchanged in meaning (GROSS, raw
    price move) — the new columns are the NET (post-fee) figures.

Changes vs v7 (self-healing schema migration):
  - PROBLEM THIS FIXES: on any deployment where data/apex7.db persists
    across code updates (e.g. a Railway Volume), Base.metadata.create_all()
    is a no-op for tables that already exist — it NEVER alters columns on
    an existing table. Every schema change above therefore needed someone
    to remember to also add an entry to a hand-maintained
    `_add_missing_columns(conn, "trades", {...})` call. That list drifted:
    `risk_usdt` (added in v2) was never added to it, only the v6 fee
    columns were. Result: a persisted DB created before v2 would run this
    app for months, then suddenly throw
    `sqlite3.OperationalError: no such column: trades.risk_usdt` the
    moment ANY query touched the Trade model — including plain
    `/api/trades` reads, not just the fee backfill — because SQLAlchemy's
    `select(Trade)` selects every mapped column by default, not just the
    ones a given query happens to need.
  - FIX: `_sync_table_columns()` replaces the curated dict entirely. It
    diffs each ORM model's declared columns against `PRAGMA table_info`
    on the real table and ALTERs in whatever is missing, for every table
    APEX-8 defines. This is generic — the next time a column gets added
    to Trade/FuturesTrade/SignalOutcome/etc., it self-heals on the next
    startup with zero additional migration code required. Safe to run on
    every boot; it only ever adds columns, never touches existing data.
"""
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Float, Integer, DateTime, Text, Boolean, Index, text


DATABASE_URL = "sqlite+aiosqlite:///./data/apex7.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

logger = logging.getLogger("apex8.db")


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(10))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    quantity: Mapped[float] = mapped_column(Float)
    usdt_value: Mapped[float] = mapped_column(Float)
    risk_usdt: Mapped[float] = mapped_column(Float, nullable=True)  # $ risked per trade
    stop_loss: Mapped[float] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float] = mapped_column(Float, nullable=True)
    pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)   # GROSS — raw price move, before fees
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)    # GROSS
    fee_usdt: Mapped[float] = mapped_column(Float, nullable=True)       # round-trip fee (v6)
    net_pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)   # pnl_usdt - fee_usdt (v6)
    net_pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)    # net_pnl_usdt as % of usdt_value (v6)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    consensus_score: Mapped[float] = mapped_column(Float, nullable=True)
    agents_agree: Mapped[int] = mapped_column(Integer, nullable=True)
    regime: Mapped[str] = mapped_column(String(20), nullable=True)
    is_testnet: Mapped[bool] = mapped_column(Boolean, default=True)
    binance_order_id: Mapped[str] = mapped_column(String(50), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_trades_status", "status"),
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_opened_at", "opened_at"),
    )


class AgentSignal(Base):
    __tablename__ = "agent_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(5))
    agent_name: Mapped[str] = mapped_column(String(50))
    signal: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_signals_symbol", "symbol"),
        Index("ix_signals_created_at", "created_at"),
        Index("ix_signals_agent_name", "agent_name"),
    )


class PerformanceSnapshot(Base):
    __tablename__ = "performance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl_usdt: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    portfolio_value: Mapped[float] = mapped_column(Float, default=0.0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ══════════════════════════════════════════════════
#  FUTURES MODE — separate tables, same conventions
# ══════════════════════════════════════════════════

class FuturesTrade(Base):
    __tablename__ = "futures_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(10))
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    margin_type: Mapped[str] = mapped_column(String(10), default="ISOLATED")
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    quantity: Mapped[float] = mapped_column(Float)
    usdt_value: Mapped[float] = mapped_column(Float)          # notional value
    margin_usdt: Mapped[float] = mapped_column(Float, nullable=True)   # actual capital committed
    risk_usdt: Mapped[float] = mapped_column(Float, nullable=True)     # $ risked at stop-loss
    stop_loss: Mapped[float] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float] = mapped_column(Float, nullable=True)
    liquidation_price: Mapped[float] = mapped_column(Float, nullable=True)  # estimate at entry
    pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)   # GROSS $, before fees
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)    # GROSS ROI on MARGIN (leverage-adjusted)
    fee_usdt: Mapped[float] = mapped_column(Float, nullable=True)       # round-trip fee, charged on NOTIONAL (v6)
    net_pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)   # pnl_usdt - fee_usdt (v6)
    net_pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)    # net_pnl_usdt as ROI on MARGIN (v6)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")   # OPEN | CLOSED | LIQUIDATED
    consensus_score: Mapped[float] = mapped_column(Float, nullable=True)
    agents_agree: Mapped[int] = mapped_column(Integer, nullable=True)
    regime: Mapped[str] = mapped_column(String(20), nullable=True)
    is_testnet: Mapped[bool] = mapped_column(Boolean, default=True)
    binance_order_id: Mapped[str] = mapped_column(String(50), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_futures_trades_status", "status"),
        Index("ix_futures_trades_symbol", "symbol"),
        Index("ix_futures_trades_opened_at", "opened_at"),
    )


class FuturesAgentSignal(Base):
    __tablename__ = "futures_agent_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(5))
    agent_name: Mapped[str] = mapped_column(String(50))
    signal: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_futures_signals_symbol", "symbol"),
        Index("ix_futures_signals_created_at", "created_at"),
        Index("ix_futures_signals_agent_name", "agent_name"),
    )



class AgentWeightHistory(Base):
    """Audit trail for every Signal Lab weight change actually applied to
    a live engine. Without this there was no record of when a weight
    changed or what it changed from/to — impossible to correlate a later
    performance shift with a specific weight adjustment."""
    __tablename__ = "agent_weight_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_type: Mapped[str] = mapped_column(String(10))     # SPOT | FUTURES
    agent_name: Mapped[str] = mapped_column(String(50))
    baseline_weight: Mapped[float] = mapped_column(Float)    # pristine class-default — the fixed anchor
    old_weight: Mapped[float] = mapped_column(Float)         # live weight immediately before this change
    new_weight: Mapped[float] = mapped_column(Float)         # weight actually applied
    samples: Mapped[int] = mapped_column(Integer, nullable=True)
    win_rate_when_agreed_pct: Mapped[float] = mapped_column(Float, nullable=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_weight_history_market_type", "market_type"),
        Index("ix_weight_history_agent_name", "agent_name"),
        Index("ix_weight_history_applied_at", "applied_at"),
    )

class FuturesPerformanceSnapshot(Base):
    __tablename__ = "futures_performance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl_usdt: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    portfolio_value: Mapped[float] = mapped_column(Float, default=0.0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ══════════════════════════════════════════════════
#  SIGNAL LAB — "what if we'd taken this?" shadow
#  tracking. One row per directional candidate per
#  scan cycle, whether or not it became a real trade.
# ══════════════════════════════════════════════════

class SignalOutcome(Base):
    __tablename__ = "signal_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_type: Mapped[str] = mapped_column(String(10))   # "SPOT" | "FUTURES"
    symbol: Mapped[str] = mapped_column(String(20))
    direction: Mapped[str] = mapped_column(String(10))      # "BUY" | "SELL"
    entry_price: Mapped[float] = mapped_column(Float)

    # Consensus state AT THE MOMENT this candidate was produced — includes
    # candidates that never cleared MIN_CONSENSUS_SCORE / MIN_AGENTS_AGREE.
    consensus_score: Mapped[float] = mapped_column(Float)
    agents_agree: Mapped[int] = mapped_column(Integer)
    total_agents: Mapped[int] = mapped_column(Integer)
    regime: Mapped[str] = mapped_column(String(20), nullable=True)

    # The SAME ATR-based exits the real engine would have used for this
    # candidate, so the hypothetical trade is judged by the same rules
    # as a real one — not some simplified fixed %.
    stop_loss_pct: Mapped[float] = mapped_column(Float)
    take_profit_pct: Mapped[float] = mapped_column(Float)
    stop_loss_price: Mapped[float] = mapped_column(Float)
    take_profit_price: Mapped[float] = mapped_column(Float)

    # Whether this candidate actually cleared the gate and became a real
    # order (allows separating "shadow-only" analysis from "confirms our
    # real trades" analysis).
    was_taken: Mapped[bool] = mapped_column(Boolean, default=False)
    linked_trade_id: Mapped[int] = mapped_column(Integer, nullable=True)

    # Per-agent snapshot for this cycle: JSON list of
    # {agent, weight, timeframe, signal, confidence, agreed}. This is
    # what lets analysis regroup by ANY combination after the fact
    # without having had to pre-simulate all 2^8 subsets.
    agent_detail_json: Mapped[str] = mapped_column(Text)

    status: Mapped[str] = mapped_column(String(20), default="PENDING")  # PENDING|TP|SL|EXPIRED
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)    # raw price-move % in trade direction
    pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)   # GROSS — on the fixed Signal Lab notional, before fees

    # ── Fee adjustment (v5) ──
    fee_usdt: Mapped[float] = mapped_column(Float, nullable=True)      # round-trip fee charged on the notional stake
    net_pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)  # pnl_usdt - fee_usdt — the realistic figure

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    resolved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_signal_outcomes_status", "status"),
        Index("ix_signal_outcomes_symbol", "symbol"),
        Index("ix_signal_outcomes_market_type", "market_type"),
        Index("ix_signal_outcomes_created_at", "created_at"),
    )


# All ORM models that init_db() should keep schema-synced. Add any new
# model to this tuple and it's automatically covered by the generic
# migration below — no more hand-maintained column dicts to forget.
_ALL_MODELS = (
    Trade, AgentSignal, PerformanceSnapshot,
    FuturesTrade, FuturesAgentSignal, FuturesPerformanceSnapshot,
    SignalOutcome,
)


async def _sync_table_columns(conn, model) -> list[str]:
    """Self-healing migration: for the given ORM model, ALTER TABLE ADD
    COLUMN any column that's declared on the model but missing from the
    actual SQLite table. Generic replacement for a hand-maintained
    per-column dict — every future model change (new column on Trade,
    FuturesTrade, SignalOutcome, etc.) now self-heals on the next startup
    instead of silently drifting until someone hits
    `sqlite3.OperationalError: no such column: ...` in production (which
    is exactly how `risk_usdt` got missed here — it predates the fee
    columns and was never added to the old curated migration list).

    Safe to run on every boot: only ADDs columns, never drops or alters
    existing ones, and is a no-op once a table is fully in sync.
    """
    table = model.__table__
    result = await conn.execute(text(f"PRAGMA table_info({table.name})"))
    existing_cols = {row[1] for row in result.fetchall()}  # row[1] = column name

    added = []
    for col in table.columns:
        if col.name in existing_cols:
            continue
        col_type = col.type.compile(dialect=conn.dialect)
        await conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {col_type}'))
        added.append(col.name)
    return added


async def init_db():
    """Create all tables (spot + futures + signal lab), then self-heal
    any column drift on tables that already existed from an older
    version of the schema (see _sync_table_columns docstring)."""
    import os
    os.makedirs("data", exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for model in _ALL_MODELS:
            try:
                added = await _sync_table_columns(conn, model)
                if added:
                    logger.info(f"✅ Migrated '{model.__tablename__}': added columns {added}")
            except Exception as e:
                # A migration failure on one table should be loud (schema
                # drift left half-fixed is worse than not fixed at all if
                # it's silent) but must not prevent the other tables from
                # being checked/migrated.
                logger.error(f"❌ Schema sync failed for '{model.__tablename__}': {type(e).__name__}: {e}")


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session