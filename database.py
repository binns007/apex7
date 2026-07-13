"""
APEX-7 Database Layer
SQLite via SQLAlchemy — stores trades, signals, and performance snapshots.

Changes vs v1: added indices on the columns actually queried
(status, symbol, created_at/opened_at) since the API filters/sorts on
these constantly as trade history grows.
"""
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Float, Integer, DateTime, Text, Boolean, Index


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


async def init_db():
    """Create all tables."""
    import os
    os.makedirs("data", exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session