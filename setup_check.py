#!/usr/bin/env python3
"""
APEX-7 Quick Start Script
Run this first to verify your setup before starting the bot.
"""
import asyncio
import os
import sys

print("""
╔══════════════════════════════════════════════════════╗
║           APEX-7 Setup Verification                  ║
╚══════════════════════════════════════════════════════╝
""")

# 1. Check .env file
if not os.path.exists('.env'):
    print("❌ .env file not found. Copy .env.example to .env and fill in your API keys.")
    sys.exit(1)
else:
    print("✅ .env file found")

# 2. Check dependencies
try:
    import fastapi, uvicorn, pandas, numpy, binance, sqlalchemy, aiohttp, dotenv
    print("✅ All Python dependencies installed")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# 3. Check data directory
os.makedirs("data", exist_ok=True)
print("✅ Data directory ready")

# 4. Load config
from config import settings
print(f"✅ Config loaded — Mode: {settings.TRADING_MODE.upper()}")
print(f"   Pairs: {', '.join(settings.TRADING_PAIRS)}")
print(f"   Consensus threshold: {settings.MIN_CONSENSUS_SCORE}")

# 5. Test Binance connection
async def test_binance():
    from market_data import fetch_price
    try:
        price = await fetch_price("BTCUSDT")
        print(f"✅ Binance API reachable — BTC price: ${price:,.2f}")
    except Exception as e:
        print(f"⚠  Binance API error: {e}")
        print("   Check your internet connection. API keys are only needed for live trading.")

asyncio.run(test_binance())

# 6. Test DB
async def test_db():
    from database import init_db
    await init_db()
    print("✅ Database initialized")

asyncio.run(test_db())

print("""
╔══════════════════════════════════════════════════════╗
║  Setup complete! Start APEX-7 with:                  ║
║                                                      ║
║  uvicorn main:app --host 0.0.0.0 --port 8000         ║
║                                                      ║
║  Then open: http://localhost:8000                    ║
╚══════════════════════════════════════════════════════╝
""")
