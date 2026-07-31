"""
APEX-7 Database Layer
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
  - Added `fee_usdt` / `net_pnl_usdt` to SignalOutcome. Raw price-move
    PnL (`pnl_usdt`, kept as-is, now referred to as the GROSS figure)
    overstates real profitability at the scale this bot trades at —
    round-trip fees are frequently larger than the edge itself. Every
    resolution now also computes a fee-adjusted NET figure; see
    signal_lab.py's resolve_pending() and backfill_fees(). Because these
    are new columns on a table that may already have real data in it
    (unlike a brand-new install), `init_db()` now runs a small idempotent
    SQLite migration (ALTER TABLE ADD COLUMN) after create_all(), since
    SQLAlchemy's create_all() only creates missing TABLES — it does not
    alter columns on tables that already exist.
"""
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Float, Integer, DateTime, Text, Boolean, Index, text


DATABASE_URL = "sqlite+aiosqlite:///./data/apex7.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


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
    pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)
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
    pnl_usdt: Mapped[float] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)  # ROI on MARGIN (leverage-adjusted)
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


async def _migrate_signal_outcomes_columns(conn):
    """Idempotent SQLite migration: add any SignalOutcome columns that
    don't exist yet on an ALREADY-CREATED table. Base.metadata.create_all()
    only creates tables that are missing entirely — it silently does
    nothing to a table that already exists with an older column set, so
    a person who was already running Signal Lab before fee adjustment
    shipped would otherwise get sqlite3.OperationalError: no such column
    the first time resolve_pending() tries to write fee_usdt/net_pnl_usdt."""
    result = await conn.execute(text("PRAGMA table_info(signal_outcomes)"))
    existing_cols = {row[1] for row in result.fetchall()}  # row[1] = column name
    if "fee_usdt" not in existing_cols:
        await conn.execute(text("ALTER TABLE signal_outcomes ADD COLUMN fee_usdt FLOAT"))
    if "net_pnl_usdt" not in existing_cols:
        await conn.execute(text("ALTER TABLE signal_outcomes ADD COLUMN net_pnl_usdt FLOAT"))


async def init_db():
    """Create all tables (spot + futures + signal lab), then run any
    small column migrations needed on tables that already existed."""
    import os
    os.makedirs("data", exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_signal_outcomes_columns(conn)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session