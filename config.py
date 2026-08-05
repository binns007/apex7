"""
APEX-8 Configuration
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

NEW — Futures Mode:
  - Binance USDT-M Futures needs its OWN API keys (Spot Testnet and
    Futures Testnet are entirely separate platforms/accounts on
    Binance — the spot testnet key will NOT work against
    testnet.binancefuture.com and vice versa). See
    BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_TESTNET_API_KEY below.
  - Futures Mode reuses the exact same Polyphonic Consensus Engine and
    Kelly-based Risk Manager mechanism as spot — it's the same classes,
    just constructed with a different (faster, tighter, leverage-aware)
    config block below, prefixed FUTURES_*.
  - TRADING_MODE (testnet/live) is a single global safety switch shared
    by both spot and futures — flipping to "live" makes BOTH markets
    trade with real funds, using their respective credential pairs.

NEW — Signal Lab ("what-if" shadow tracking):
  - Every scan cycle produces a directional candidate (Momentum+OrderBook
    leaning BUY, say) whether or not it clears MIN_CONSENSUS_SCORE /
    MIN_AGENTS_AGREE. Previously that candidate was discarded the moment
    the engine decided HOLD — there was no record of what WOULD have
    happened had it been taken. Signal Lab records every such candidate
    with its full per-agent breakdown and tracks it forward against the
    same ATR-based SL/TP the engine would have used, whether or not a
    real trade was ever opened. See signal_lab.py.

NEW — Fee adjustment (real trades AND Signal Lab shadow trades):
  - SPOT_TAKER_FEE_PCT / FUTURES_TAKER_FEE_PCT below are the single
    source of truth for round-trip trading cost, used in TWO places:
      1. signal_lab.py — every resolved shadow outcome gets a GROSS
         (raw price move) and NET (fee-adjusted) PnL.
      2. trading_engine.py / futures_trading_engine.py — every REAL
         closed trade now also gets fee_usdt/net_pnl_usdt/net_pnl_pct
         persisted, and net_pnl_usdt (not gross) is what feeds the Risk
         Manager's win/loss bookkeeping (Kelly win-rate estimate,
         drawdown tracking) — see the trading engines for why.
  - Both entry (always MARKET) and exit are conservatively charged at
    the TAKER rate on both legs; slippage and futures funding-rate
    carry cost are still NOT modeled anywhere, so even these NET
    numbers remain a best case, not a true worst case.
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
    # ── Binance credentials — SPOT ────────────────────
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_API_SECRET: str = os.getenv("BINANCE_TESTNET_API_SECRET", "")

    # ── Binance credentials — FUTURES (separate platform/account) ────
    # Live:    generate at https://www.binance.com/en/my/settings/api-management
    #          (needs "Enable Futures" permission)
    # Testnet: generate at https://testnet.binancefuture.com
    BINANCE_FUTURES_API_KEY: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
    BINANCE_FUTURES_API_SECRET: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")
    BINANCE_FUTURES_TESTNET_API_KEY: str = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "")
    BINANCE_FUTURES_TESTNET_API_SECRET: str = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "")

    # ── Mode (shared by spot + futures) ───────────────
    TRADING_MODE: str = os.getenv("TRADING_MODE", "testnet")  # "live" | "testnet"

    # ── Trading pairs — SPOT ───────────────────────────
    TRADING_PAIRS: list[str] = [
        p.strip().upper() for p in os.getenv(
            "TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT"
        ).split(",") if p.strip()
    ]

    # ── Risk controls — SPOT ───────────────────────────
    MAX_PORTFOLIO_RISK_PCT: float = _env_float("MAX_PORTFOLIO_RISK_PCT", 1.5)
    MAX_PORTFOLIO_HEAT_PCT: float = _env_float("MAX_PORTFOLIO_HEAT_PCT", 8.0)
    MAX_DRAWDOWN_HALT_PCT: float = _env_float("MAX_DRAWDOWN_HALT_PCT", 15.0)
    TRADE_USDT_CAP: float = _env_float("TRADE_USDT_CAP", 500.0)
    MIN_TRADE_USDT: float = _env_float("MIN_TRADE_USDT", 10.0)

    # ── Strategy thresholds — SPOT (loosened from v1 defaults) ────────
    # v1 defaults (0.68 / 5-of-8) combined with a hard regime block and a
    # hard temporal-confluence requirement meant the bot rarely fired.
    # These defaults are still conservative but no longer triple-gated.
    MIN_CONSENSUS_SCORE: float = _env_float("MIN_CONSENSUS_SCORE", 0.55)
    MIN_AGENTS_AGREE: int = _env_int("MIN_AGENTS_AGREE", 3)
    SCALPING_ENABLED: bool = _env_bool("SCALPING_ENABLED", True)
    SENTIMENT_ENABLED: bool = _env_bool("SENTIMENT_ENABLED", True)

    # ── Temporal confluence — SPOT ─────────────────────
    REQUIRE_TEMPORAL_CONFLUENCE: bool = _env_bool("REQUIRE_TEMPORAL_CONFLUENCE", False)
    CONFLUENCE_SCORE_BONUS: float = _env_float("CONFLUENCE_SCORE_BONUS", 0.08)
    NO_CONFLUENCE_SCORE_PENALTY: float = _env_float("NO_CONFLUENCE_SCORE_PENALTY", 0.10)

    # ── Regime gate — SPOT ──────────────────────────────
    REGIME_HARD_BLOCK_VOLATILE: bool = _env_bool("REGIME_HARD_BLOCK_VOLATILE", False)
    REGIME_VOLATILE_SCORE_MULTIPLIER: float = _env_float("REGIME_VOLATILE_SCORE_MULTIPLIER", 0.55)
    REGIME_TREND_ADX: float = _env_float("REGIME_TREND_ADX", 28.0)
    REGIME_VOLATILE_ATR_PCT: float = _env_float("REGIME_VOLATILE_ATR_PCT", 4.0)
    REGIME_VOLATILE_BB_WIDTH: float = _env_float("REGIME_VOLATILE_BB_WIDTH", 0.12)

    # ── Consensus tie-break — SPOT ─────────────────────
    TIE_BREAK_ACTION: str = os.getenv("TIE_BREAK_ACTION", "HOLD")  # HOLD | BUY | SELL

    # ── Server ────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me")

    # ── Timeframes — SPOT ──────────────────────────────
    TIMEFRAMES: list[str] = ["1m", "5m", "15m"]
    PRIMARY_TIMEFRAME: str = "5m"

    # ── Agent scan cycle — SPOT ────────────────────────
    SCAN_INTERVAL_SECONDS: int = _env_int("SCAN_INTERVAL_SECONDS", 15)

    # ── Candle history depth per timeframe ────────────
    CANDLE_LIMIT: int = _env_int("CANDLE_LIMIT", 200)

    # ══════════════════════════════════════════════════
    #  TRADING FEES — used for BOTH real trade PnL display
    #  (trading_engine.py / futures_trading_engine.py) AND
    #  Signal Lab's fee-adjusted shadow PnL (signal_lab.py)
    # ══════════════════════════════════════════════════
    # Per-SIDE taker rate. Both engines always enter with a MARKET order
    # (taker). Spot exits go through an OCO whose take-profit leg COULD
    # fill as maker if price reaches it passively; futures exits are
    # MARKET-triggered (STOP_MARKET/TAKE_PROFIT_MARKET, always taker).
    # We conservatively charge taker on BOTH legs of every trade rather
    # than assume the cheaper maker fill — overstating cost is safer
    # than overstating profit. Defaults match Binance's standard
    # (non-BNB-discounted) fee schedule; override via env if your
    # account has a different tier or a BNB fee discount enabled.
    SPOT_TAKER_FEE_PCT: float = _env_float("SPOT_TAKER_FEE_PCT", 0.10)
    FUTURES_TAKER_FEE_PCT: float = _env_float("FUTURES_TAKER_FEE_PCT", 0.05)

    # ══════════════════════════════════════════════════
    #  SIGNAL LAB — "what if we'd taken this?" shadow
    #  tracking + agent/weight analytics (spot + futures)
    # ══════════════════════════════════════════════════
    SIGNAL_LAB_ENABLED: bool = _env_bool("SIGNAL_LAB_ENABLED", True)
    # How long a shadow candidate is allowed to sit PENDING before it's
    # force-resolved at the current price (labeled EXPIRED rather than
    # left open forever). Spot trades hold longer than futures scalps.
    SIGNAL_LAB_MAX_HOLD_MINUTES: float = _env_float("SIGNAL_LAB_MAX_HOLD_MINUTES", 360.0)
    FUTURES_SIGNAL_LAB_MAX_HOLD_MINUTES: float = _env_float("FUTURES_SIGNAL_LAB_MAX_HOLD_MINUTES", 90.0)
    # Fixed hypothetical stake used to express every shadow outcome in
    # comparable dollar terms, independent of whatever the real Risk
    # Manager would have actually sized that trade at.
    SIGNAL_LAB_NOTIONAL_USDT: float = _env_float("SIGNAL_LAB_NOTIONAL_USDT", 100.0)
    # Minimum resolved samples before an agent/combo is surfaced in
    # analytics — keeps single-digit-sample noise out of the leaderboard.
    SIGNAL_LAB_MIN_SAMPLES: int = _env_int("SIGNAL_LAB_MIN_SAMPLES", 5)

    # ══════════════════════════════════════════════════
    #  FUTURES MODE — same mechanism, tuned for fast,
    #  small, leveraged trades. Every knob below is the
    #  futures analogue of a spot knob above.
    # ══════════════════════════════════════════════════

    FUTURES_TRADING_PAIRS: list[str] = [
        p.strip().upper() for p in os.getenv(
            "FUTURES_TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT"
        ).split(",") if p.strip()
    ]

    # ── Leverage / margin ──────────────────────────────
    FUTURES_DEFAULT_LEVERAGE: int = _env_int("FUTURES_DEFAULT_LEVERAGE", 5)
    FUTURES_MAX_LEVERAGE_ALLOWED: int = _env_int("FUTURES_MAX_LEVERAGE_ALLOWED", 20)
    FUTURES_MARGIN_TYPE: str = os.getenv("FUTURES_MARGIN_TYPE", "ISOLATED")  # ISOLATED | CROSSED
    # Hedge mode (separate long/short books per symbol) is NOT implemented
    # in this version — every futures position is one-way, reduce-only
    # exits, one open position per symbol at a time. Kept here so it's
    # validated/surfaced rather than silently ignored if someone sets it.
    FUTURES_POSITION_MODE: str = os.getenv("FUTURES_POSITION_MODE", "ONE_WAY")

    # ── Scan cadence — fast/small trades need to react quickly ───────
    FUTURES_SCAN_INTERVAL_SECONDS: int = _env_int("FUTURES_SCAN_INTERVAL_SECONDS", 10)

    # ── Timeframes — biased to micro-timeframes vs spot's 1m/5m/15m ──
    FUTURES_TIMEFRAMES: list[str] = ["1m", "3m", "5m"]
    FUTURES_PRIMARY_TIMEFRAME: str = "3m"
    FUTURES_REGIME_TIMEFRAME: str = "5m"   # spot's Regime agent runs on 15m
    FUTURES_CANDLE_LIMIT: int = _env_int("FUTURES_CANDLE_LIMIT", 200)

    # ── Strategy thresholds — FUTURES ─────────────────
    FUTURES_MIN_CONSENSUS_SCORE: float = _env_float("FUTURES_MIN_CONSENSUS_SCORE", 0.52)
    FUTURES_MIN_AGENTS_AGREE: int = _env_int("FUTURES_MIN_AGENTS_AGREE", 3)
    FUTURES_TIE_BREAK_ACTION: str = os.getenv("FUTURES_TIE_BREAK_ACTION", "HOLD")

    # ── Temporal confluence — FUTURES ─────────────────
    FUTURES_REQUIRE_TEMPORAL_CONFLUENCE: bool = _env_bool("FUTURES_REQUIRE_TEMPORAL_CONFLUENCE", False)
    FUTURES_CONFLUENCE_SCORE_BONUS: float = _env_float("FUTURES_CONFLUENCE_SCORE_BONUS", 0.06)
    FUTURES_NO_CONFLUENCE_SCORE_PENALTY: float = _env_float("FUTURES_NO_CONFLUENCE_SCORE_PENALTY", 0.08)

    # ── Regime gate — FUTURES ──────────────────────────
    FUTURES_REGIME_HARD_BLOCK_VOLATILE: bool = _env_bool("FUTURES_REGIME_HARD_BLOCK_VOLATILE", False)
    FUTURES_REGIME_VOLATILE_SCORE_MULTIPLIER: float = _env_float("FUTURES_REGIME_VOLATILE_SCORE_MULTIPLIER", 0.50)
    FUTURES_REGIME_TREND_ADX: float = _env_float("FUTURES_REGIME_TREND_ADX", 25.0)
    FUTURES_REGIME_VOLATILE_ATR_PCT: float = _env_float("FUTURES_REGIME_VOLATILE_ATR_PCT", 5.0)
    FUTURES_REGIME_VOLATILE_BB_WIDTH: float = _env_float("FUTURES_REGIME_VOLATILE_BB_WIDTH", 0.14)

    # ── Exit sizing — FUTURES (tighter/faster than spot's 1.5x ATR,
    #    0.8–2.5% band) since these are meant to be quick, small trades ──
    FUTURES_SL_ATR_MULT: float = _env_float("FUTURES_SL_ATR_MULT", 1.1)
    FUTURES_SL_MIN_PCT: float = _env_float("FUTURES_SL_MIN_PCT", 0.0035)
    FUTURES_SL_MAX_PCT: float = _env_float("FUTURES_SL_MAX_PCT", 0.012)
    FUTURES_RR_TREND: float = _env_float("FUTURES_RR_TREND", 1.8)
    FUTURES_RR_RANGE: float = _env_float("FUTURES_RR_RANGE", 1.3)

    # ── Risk controls — FUTURES ────────────────────────
    # Risk % is of account margin balance, applied BEFORE leverage —
    # leverage only changes how much margin a given $ risk requires
    # (see futures_risk_manager.py). Kept smaller than spot per-trade
    # risk since trade frequency is much higher here.
    FUTURES_MAX_PORTFOLIO_RISK_PCT: float = _env_float("FUTURES_MAX_PORTFOLIO_RISK_PCT", 0.8)
    FUTURES_MAX_PORTFOLIO_HEAT_PCT: float = _env_float("FUTURES_MAX_PORTFOLIO_HEAT_PCT", 6.0)
    FUTURES_MAX_DRAWDOWN_HALT_PCT: float = _env_float("FUTURES_MAX_DRAWDOWN_HALT_PCT", 12.0)
    FUTURES_TRADE_MARGIN_CAP_USDT: float = _env_float("FUTURES_TRADE_MARGIN_CAP_USDT", 100.0)
    FUTURES_MIN_TRADE_MARGIN_USDT: float = _env_float("FUTURES_MIN_TRADE_MARGIN_USDT", 5.0)

    # ── Liquidation safety ─────────────────────────────
    # Estimated liquidation distance ≈ (1/leverage) − maintenance margin
    # rate. The stop-loss must sit comfortably INSIDE that distance (this
    # fraction of it) so ordinary noise can't liquidate a position that
    # should have been stopped out first. This is a conservative estimate
    # for pre-trade safety, not Binance's exact tiered-bracket formula —
    # the executor can pull the real liquidation price from the exchange
    # via get_position_risk() for reconciliation.
    FUTURES_LIQUIDATION_SAFETY_FACTOR: float = _env_float("FUTURES_LIQUIDATION_SAFETY_FACTOR", 0.6)
    FUTURES_MAINTENANCE_MARGIN_RATE: float = _env_float("FUTURES_MAINTENANCE_MARGIN_RATE", 0.005)

    @property
    def is_testnet(self) -> bool:
        return self.TRADING_MODE == "testnet"

    @property
    def active_api_key(self) -> str:
        return self.BINANCE_TESTNET_API_KEY if self.is_testnet else self.BINANCE_API_KEY

    @property
    def active_api_secret(self) -> str:
        return self.BINANCE_TESTNET_API_SECRET if self.is_testnet else self.BINANCE_API_SECRET

    @property
    def active_futures_api_key(self) -> str:
        return self.BINANCE_FUTURES_TESTNET_API_KEY if self.is_testnet else self.BINANCE_FUTURES_API_KEY

    @property
    def active_futures_api_secret(self) -> str:
        return self.BINANCE_FUTURES_TESTNET_API_SECRET if self.is_testnet else self.BINANCE_FUTURES_API_SECRET

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

        # ── Futures validation ──
        if not self.FUTURES_TRADING_PAIRS:
            problems.append("FUTURES_TRADING_PAIRS is empty")
        if not (0.0 < self.FUTURES_MIN_CONSENSUS_SCORE <= 1.0):
            problems.append("FUTURES_MIN_CONSENSUS_SCORE must be in (0, 1]")
        if self.FUTURES_MIN_AGENTS_AGREE < 1:
            problems.append("FUTURES_MIN_AGENTS_AGREE must be >= 1")
        if not (1 <= self.FUTURES_DEFAULT_LEVERAGE <= self.FUTURES_MAX_LEVERAGE_ALLOWED):
            problems.append(
                f"FUTURES_DEFAULT_LEVERAGE ({self.FUTURES_DEFAULT_LEVERAGE}) must be between "
                f"1 and FUTURES_MAX_LEVERAGE_ALLOWED ({self.FUTURES_MAX_LEVERAGE_ALLOWED})"
            )
        if self.FUTURES_MAX_LEVERAGE_ALLOWED > 50:
            problems.append("FUTURES_MAX_LEVERAGE_ALLOWED > 50x is not recommended for automated trading")
        if self.FUTURES_MARGIN_TYPE not in ("ISOLATED", "CROSSED"):
            problems.append("FUTURES_MARGIN_TYPE must be ISOLATED or CROSSED")
        if self.FUTURES_POSITION_MODE != "ONE_WAY":
            problems.append("FUTURES_POSITION_MODE=HEDGE is not implemented in this version — using ONE_WAY")
        if self.FUTURES_MAX_PORTFOLIO_RISK_PCT <= 0 or self.FUTURES_MAX_PORTFOLIO_RISK_PCT > 10:
            problems.append("FUTURES_MAX_PORTFOLIO_RISK_PCT looks unsafe (expected 0–10)")
        if self.FUTURES_TIE_BREAK_ACTION not in ("HOLD", "BUY", "SELL"):
            problems.append("FUTURES_TIE_BREAK_ACTION must be HOLD, BUY, or SELL")
        if not self.is_testnet and not (self.BINANCE_FUTURES_API_KEY and self.BINANCE_FUTURES_API_SECRET):
            problems.append("TRADING_MODE=live but BINANCE_FUTURES_API_KEY/SECRET are not set (needed for Futures Mode)")
        if self.is_testnet and not (self.BINANCE_FUTURES_TESTNET_API_KEY and self.BINANCE_FUTURES_TESTNET_API_SECRET):
            problems.append(
                "Futures Testnet keys are not set (BINANCE_FUTURES_TESTNET_API_KEY/SECRET) — "
                "these are separate from the Spot Testnet keys, generate at testnet.binancefuture.com"
            )

        # ── Signal Lab / fee validation ──
        if self.SIGNAL_LAB_MAX_HOLD_MINUTES <= 0:
            problems.append("SIGNAL_LAB_MAX_HOLD_MINUTES must be > 0")
        if self.FUTURES_SIGNAL_LAB_MAX_HOLD_MINUTES <= 0:
            problems.append("FUTURES_SIGNAL_LAB_MAX_HOLD_MINUTES must be > 0")
        if self.SIGNAL_LAB_NOTIONAL_USDT <= 0:
            problems.append("SIGNAL_LAB_NOTIONAL_USDT must be > 0")
        if self.SPOT_TAKER_FEE_PCT < 0:
            problems.append("SPOT_TAKER_FEE_PCT must be >= 0")
        if self.FUTURES_TAKER_FEE_PCT < 0:
            problems.append("FUTURES_TAKER_FEE_PCT must be >= 0")
        return problems


settings = Settings()

_problems = settings.validate()
if _problems:
    import logging
    _log = logging.getLogger("apex8.config")
    for p in _problems:
        _log.warning("⚠ Config issue: %s", p)