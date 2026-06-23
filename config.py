"""
APEX-7 Configuration
Loads environment variables and provides a single settings object.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Binance credentials
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_API_SECRET: str = os.getenv("BINANCE_TESTNET_API_SECRET", "")

    # Mode
    TRADING_MODE: str = os.getenv("TRADING_MODE", "testnet")  # "live" | "testnet"

    # Trading pairs
    TRADING_PAIRS: list[str] = [
        p.strip() for p in os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT").split(",")
    ]

    # Risk controls
    MAX_PORTFOLIO_RISK_PCT: float = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "2.0"))
    MAX_PORTFOLIO_HEAT_PCT: float = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", "8.0"))
    MAX_DRAWDOWN_HALT_PCT: float = float(os.getenv("MAX_DRAWDOWN_HALT_PCT", "15.0"))
    TRADE_USDT_CAP: float = float(os.getenv("TRADE_USDT_CAP", "500.0"))

    # Strategy thresholds
    MIN_CONSENSUS_SCORE: float = float(os.getenv("MIN_CONSENSUS_SCORE", "0.68"))
    MIN_AGENTS_AGREE: int = int(os.getenv("MIN_AGENTS_AGREE", "5"))
    SCALPING_ENABLED: bool = os.getenv("SCALPING_ENABLED", "true").lower() == "true"
    SENTIMENT_ENABLED: bool = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me")

    @property
    def is_testnet(self) -> bool:
        return self.TRADING_MODE == "testnet"

    @property
    def active_api_key(self) -> str:
        return self.BINANCE_TESTNET_API_KEY if self.is_testnet else self.BINANCE_API_KEY

    @property
    def active_api_secret(self) -> str:
        return self.BINANCE_TESTNET_API_SECRET if self.is_testnet else self.BINANCE_API_SECRET

    # Timeframes used for multi-frame confluence
    TIMEFRAMES: list[str] = ["1m", "5m", "15m"]

    # Agent scan cycle
    SCAN_INTERVAL_SECONDS: int = 30

    # Candle history depth per timeframe
    CANDLE_LIMIT: int = 200


settings = Settings()
