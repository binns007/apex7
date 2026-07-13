"""
APEX-7 Configuration
═════════════════════
All runtime knobs live here. Every threshold that used to be a magic
number buried in an agent or the consensus engine is now surfaced here,
validated, and safe to tune from the dashboard without redeploying.

Key philosophy change from v1:
  - The engine gates used to stack three independent hard filters
    (regime block + temporal confluence + consensus threshold), which
    meant the bot could go days without a single trade even in a
    perfectly tradeable market. Regime is now a SOFT penalty by
    default (config-switchable back to a hard block if you want it).
"""
import os
import math
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # ── Binance credentials ──────────────────────────
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_API_SECRET: str = os.getenv("BINANCE_TESTNET_API_SECRET", "")

    # ── Mode ──────────────────────────────────────────
    TRADING_MODE: str = os.getenv("TRADING_MODE", "testnet")  # "live" | "testnet"

    # ── Trading pairs ─────────────────────────────────
    TRADING_PAIRS: list[str] = [
        p.strip().upper() for p in os.getenv(
            "TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT"
        ).split(",") if p.strip()
    ]

    # ── Risk controls ─────────────────────────────────
    MAX_PORTFOLIO_RISK_PCT: float = _env_float("MAX_PORTFOLIO_RISK_PCT", 1.5)
    MAX_PORTFOLIO_HEAT_PCT: float = _env_float("MAX_PORTFOLIO_HEAT_PCT", 8.0)
    MAX_DRAWDOWN_HALT_PCT: float = _env_float("MAX_DRAWDOWN_HALT_PCT", 15.0)
    TRADE_USDT_CAP: float = _env_float("TRADE_USDT_CAP", 500.0)
    MIN_TRADE_USDT: float = _env_float("MIN_TRADE_USDT", 10.0)

    # ── Strategy thresholds (loosened from v1 defaults) ─
    # v1 defaults (0.68 / 5-of-8) combined with a hard regime block and a
    # hard temporal-confluence requirement meant the bot rarely fired.
    # These defaults are still conservative but no longer triple-gated.
    MIN_CONSENSUS_SCORE: float = _env_float("MIN_CONSENSUS_SCORE", 0.60)
    MIN_AGENTS_AGREE: int = _env_int("MIN_AGENTS_AGREE", 4)
    SCALPING_ENABLED: bool = _env_bool("SCALPING_ENABLED", True)
    SENTIMENT_ENABLED: bool = _env_bool("SENTIMENT_ENABLED", True)

    # ── Temporal confluence ───────────────────────────
    # If True, the 5m primary signal must also appear on >=1 other TF.
    # If False, confluence still boosts score but no longer hard-blocks.
    REQUIRE_TEMPORAL_CONFLUENCE: bool = _env_bool("REQUIRE_TEMPORAL_CONFLUENCE", False)
    CONFLUENCE_SCORE_BONUS: float = _env_float("CONFLUENCE_SCORE_BONUS", 0.08)
    NO_CONFLUENCE_SCORE_PENALTY: float = _env_float("NO_CONFLUENCE_SCORE_PENALTY", 0.10)

    # ── Regime gate ────────────────────────────────────
    # v1 hard-blocked ALL trading in a VOLATILE regime. Crypto routinely
    # sits above naive volatility thresholds intraday, so a hard block
    # can blackout trading for long stretches. Default is now a soft
    # multiplier; flip REGIME_HARD_BLOCK_VOLATILE=true to restore v1 behavior.
    REGIME_HARD_BLOCK_VOLATILE: bool = _env_bool("REGIME_HARD_BLOCK_VOLATILE", False)
    REGIME_VOLATILE_SCORE_MULTIPLIER: float = _env_float("REGIME_VOLATILE_SCORE_MULTIPLIER", 0.55)
    REGIME_TREND_ADX: float = _env_float("REGIME_TREND_ADX", 28.0)
    REGIME_VOLATILE_ATR_PCT: float = _env_float("REGIME_VOLATILE_ATR_PCT", 4.0)
    REGIME_VOLATILE_BB_WIDTH: float = _env_float("REGIME_VOLATILE_BB_WIDTH", 0.12)

    # ── Consensus tie-break ───────────────────────────
    # v1 bug: an exact BUY/SELL tie on the primary timeframe silently
    # resolved to SELL (`x if cond else y` with no equality branch).
    # Now configurable; default is the only sane choice: HOLD.
    TIE_BREAK_ACTION: str = os.getenv("TIE_BREAK_ACTION", "HOLD")  # HOLD | BUY | SELL

    # ── Server ────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me")

    # ── Timeframes used for multi-frame confluence ────
    TIMEFRAMES: list[str] = ["1m", "5m", "15m"]
    PRIMARY_TIMEFRAME: str = "5m"

    # ── Agent scan cycle ──────────────────────────────
    SCAN_INTERVAL_SECONDS: int = _env_int("SCAN_INTERVAL_SECONDS", 30)

    # ── Candle history depth per timeframe ────────────
    CANDLE_LIMIT: int = _env_int("CANDLE_LIMIT", 200)

    @property
    def is_testnet(self) -> bool:
        return self.TRADING_MODE == "testnet"

    @property
    def active_api_key(self) -> str:
        return self.BINANCE_TESTNET_API_KEY if self.is_testnet else self.BINANCE_API_KEY

    @property
    def active_api_secret(self) -> str:
        return self.BINANCE_TESTNET_API_SECRET if self.is_testnet else self.BINANCE_API_SECRET

    def validate(self) -> list[str]:
        """Return a list of human-readable config problems (empty = OK)."""
        problems = []
        if self.TRADING_MODE not in ("live", "testnet"):
            problems.append(f"TRADING_MODE must be 'live' or 'testnet', got '{self.TRADING_MODE}'")
        if not self.TRADING_PAIRS:
            problems.append("TRADING_PAIRS is empty")
        if not (0.0 < self.MIN_CONSENSUS_SCORE <= 1.0):
            problems.append("MIN_CONSENSUS_SCORE must be in (0, 1]")
        if self.MIN_AGENTS_AGREE < 1:
            problems.append("MIN_AGENTS_AGREE must be >= 1")
        if self.MAX_PORTFOLIO_RISK_PCT <= 0 or self.MAX_PORTFOLIO_RISK_PCT > 25:
            problems.append("MAX_PORTFOLIO_RISK_PCT looks unsafe (expected 0–25)")
        if self.TIE_BREAK_ACTION not in ("HOLD", "BUY", "SELL"):
            problems.append("TIE_BREAK_ACTION must be HOLD, BUY, or SELL")
        if not self.is_testnet and not (self.BINANCE_API_KEY and self.BINANCE_API_SECRET):
            problems.append("TRADING_MODE=live but BINANCE_API_KEY/SECRET are not set")
        return problems


settings = Settings()

_problems = settings.validate()
if _problems:
    import logging
    _log = logging.getLogger("apex7.config")
    for p in _problems:
        _log.warning("⚠ Config issue: %s", p)