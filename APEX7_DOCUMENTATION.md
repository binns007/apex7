# APEX-7 — Polyphonic Consensus Intelligence
## Production-Ready Crypto Trading Bot

---

## Table of Contents

1. [What Is APEX-7?](#1-what-is-apex-7)
2. [Core Innovation — Polyphonic Consensus Engine](#2-core-innovation)
3. [The 8 Agents — What Each One Does](#3-the-8-agents)
4. [Risk Management System](#4-risk-management)
5. [How a Trade Happens Step by Step](#5-trade-lifecycle)
6. [Testnet vs Live Mode](#6-testnet-vs-live)
7. [Installation & Setup](#7-installation)
8. [Configuration (.env Guide)](#8-configuration)
9. [Running the Bot](#9-running)
10. [Dashboard UI Guide](#10-dashboard)
11. [Architecture Overview](#11-architecture)
12. [Performance Expectations](#12-performance)
13. [Known Limitations & Warnings](#13-warnings)

---

## 1. What Is APEX-7?

APEX-7 is a multi-agent AI trading bot for Binance spot markets. It is designed around one founding idea:

> **No single strategy is consistently profitable. But when 8 independent strategies all agree at once, the edge compounds dramatically.**

Instead of running one algorithm and hoping for the best, APEX-7 runs **8 specialist AI agents simultaneously** — each reading the market through a completely different lens. A trade only fires when enough agents reach **weighted consensus** above a configurable threshold.

This mimics how a trading desk with multiple specialists works: the momentum trader, the quant, the risk officer, the sentiment analyst, and the market microstructure expert all need to agree before capital is deployed.

### What Makes It Different from Retail Bots?

| Feature | Typical Retail Bot | APEX-7 |
|---|---|---|
| Strategy | Single (e.g. RSI crossover) | 8 independent agents |
| Timeframes | 1 | 3 (1m, 5m, 15m) simultaneous |
| Entry filter | Static threshold | Temporal confluence required |
| Position sizing | Fixed lot | Fractional Kelly Criterion |
| Market regime | None | Active regime detection |
| Risk management | Basic stop loss | Portfolio heat + drawdown halt |
| Adaptability | Static | Agent weights update on accuracy |
| Exit | Manual or trailing | ATR-based OCO (TP + SL in one order) |

---

## 2. Core Innovation — Polyphonic Consensus Engine (PCE)

The PCE is APEX-7's brain. Here's exactly what it does every 30 seconds:

### Step 1: Parallel Data Collection
All 8 agents receive the same market data simultaneously across 3 timeframes (1m, 5m, 15m) — that's up to 24 independent analysis tasks running in parallel via `asyncio`.

### Step 2: Independent Voting
Each agent returns a signal: `BUY`, `SELL`, or `HOLD`, plus a confidence score (0–1) and a reason string. Agents never talk to each other — they are fully isolated.

### Step 3: Regime Gate
The Regime Agent classifies the current market condition (TREND_UP, TREND_DOWN, RANGING, VOLATILE). If the regime is VOLATILE, all trading is paused — no signal passes through.

### Step 4: Temporal Confluence Filter
The primary signal must exist on the **5m timeframe**. It must **also appear on at least one other timeframe** (1m or 15m). This eliminates noise signals that only appear on one timeframe.

### Step 5: Weighted Consensus Score
Each agent's vote is weighted by:
- Agent's base weight (experience level)
- Signal confidence (0–1)
- Agent accuracy multiplier (rolling win rate adjusts weights dynamically)

```
consensus_score = Σ(weight × confidence × accuracy) / Σ(total_weights)
```

### Step 6: Threshold Check
Trade fires only if:
- `consensus_score ≥ MIN_CONSENSUS_SCORE` (default 0.68)
- `agents_agree ≥ MIN_AGENTS_AGREE` (default 5 out of 8)

Both conditions must be true. This is the core filter that prevents overtrading.

---

## 3. The 8 Agents

### Agent 1: Momentum Agent (Weight: 1.3)
**What it reads:** RSI, MACD, EMA stack alignment (9/21/50), Rate of Change.

**Specialty:** Detects strong directional momentum — situations where price is moving with conviction. High weight because momentum is one of the most reliable persistent edges in crypto.

**Fires when:** MACD crosses above signal line, EMAs align bullishly (9>21>50), RSI in 45–65 zone (momentum, not overbought), positive ROC.

---

### Agent 2: Mean Reversion Agent (Weight: 1.1)
**What it reads:** Bollinger Bands %B, RSI extremes, Z-score of price vs 20-period mean, VWAP.

**Specialty:** Finds over-extended moves and fades them back to the mean. Automatically reduces its threshold in ranging markets (tight Bollinger Bands).

**Fires when:** Price below lower Bollinger Band + RSI < 30 + Z-score < -2.0. The three conditions together indicate a statistically extreme oversold condition.

---

### Agent 3: Breakout Agent (Weight: 1.2)
**What it reads:** Pivot high/low support & resistance, Bollinger Band width expansion, volume Z-score, ADX.

**Specialty:** Catches momentum-driven breakouts from consolidation zones. Requires volume confirmation — breakouts on thin volume are frequently false.

**Fires when:** Price breaks above recent 20-bar high + volume Z-score > 0.8 + Bollinger Bands expanding + strong candle body.

---

### Agent 4: Volume Agent (Weight: 1.1)
**What it reads:** On-Balance Volume (OBV) trend slope, VWAP deviation, taker buy/sell ratio from Binance candle data.

**Specialty:** Volume doesn't lie. A price move with no volume support is suspect. This agent confirms (or denies) what price is doing. The taker buy ratio reveals who is aggressive — buyers or sellers.

**Key insight:** High taker buy ratio (>58%) means buyers are hitting asks — genuine aggression. This is significantly more bullish than passive bid-stacking.

---

### Agent 5: Sentiment Agent (Weight: 0.9)
**What it reads:** Fear & Greed Index (alternative.me API), Binance perpetual futures funding rate.

**Specialty:** Contrarian at extremes, trend-following in the middle. When everyone is in extreme fear, it's often the best buying opportunity. When funding rates are very positive, too many people are long — crowded trade.

**Fires:** Contrarian BUY at F&G ≤ 20 (extreme fear) + negative funding. Contrarian SELL at F&G ≥ 80 + high positive funding.

---

### Agent 6: Order Book Agent (Weight: 1.0)
**What it reads:** Real-time bid/ask order book depth, imbalance ratio, large wall detection, bid-ask spread.

**Specialty:** Microstructure analysis. Large bid walls indicate institutional support. Imbalance > 0.30 means buyers are stacking heavily. Wide spreads reduce confidence (thin market).

**Key insight:** This is the only agent reading real-time microstructure — what's happening *right now* at the order book level, not historical candles.

---

### Agent 7: Scalping Agent (Weight: 0.85)
**What it reads:** Stochastic %K/%D crossovers, fast EMA (9/21) crossover, 3-candle momentum bursts.

**Specialty:** Pure micro-timeframe momentum. Only fires on 1m and 3m timeframes. Slightly lower weight because 1m signals are noisier. Most effective when other agents are already bullish and this provides the final micro-timing trigger.

---

### Agent 8: Regime Agent (Weight: 1.2)
**What it reads:** ADX for trend strength, EMA 50/200 alignment, ATR-to-price ratio, Bollinger Band width.

**Special role:** This agent doesn't just vote — it classifies the entire market environment:
- `TREND_UP`: ADX > 30, price > EMA50 > EMA200
- `TREND_DOWN`: ADX > 30, price < EMA50 < EMA200  
- `RANGING`: Low ADX, tight bands
- `VOLATILE`: High ATR%, wide bands → **all trading paused**

In trending markets, it biases toward pullbacks for better entries. In volatile conditions, it's the circuit breaker.

---

## 4. Risk Management System

APEX-7's risk engine uses professional-grade position sizing:

### Kelly Criterion Position Sizing
```
Kelly fraction = (p × (b+1) - 1) / b
where:
  p = win rate (from live trade history, defaults to 52%)
  b = R/R ratio (take profit / stop loss)

APEX-7 uses 25% of Kelly (fractional Kelly) for safety.
```

Fractional Kelly is standard practice at hedge funds. Full Kelly maximizes long-run growth but produces 30%+ drawdowns. Quarter Kelly provides ~80% of the growth with dramatically smoother equity curve.

### Conviction Boost
High consensus scores allow slightly larger positions:
```
final_size = kelly_size × (0.8 + consensus_score × 0.4)
```
A 0.68 consensus score gives 1.07× boost. A 0.95 score gives 1.18× boost. This means your highest-conviction trades are sized larger.

### Portfolio Heat Limit
Maximum 8% of portfolio at risk across all open positions simultaneously. Even if 4 pairs signal at once, only enough positions to fill 8% heat are allowed.

### Drawdown Circuit Breaker
If portfolio drawdown exceeds 15% (configurable), all trading halts automatically. You must manually resume via the dashboard. This prevents catastrophic losses from rare market dislocations.

### ATR-Based Stop Loss
Stop loss is dynamically placed at 1.5× ATR below entry (not a fixed %).
- Low volatility environment → tighter stops → smaller losses if wrong
- High volatility environment → wider stops → not stopped out by normal noise

Take profit is set at 2:1 RR in ranging markets, 3:1 in trending markets.

### OCO Exit Orders
Immediately after entry, an OCO (One-Cancels-Other) order is placed:
- Limit order at take profit price
- Stop-limit order at stop loss price
- Whichever triggers first cancels the other

This means exits are managed by Binance's servers — no connectivity required.

---

## 5. Trade Lifecycle (Step by Step)

```
[Every 30 seconds]
        │
        ▼
1. Fetch USDT balance from Binance
        │
        ▼
2. Check all open DB trades for SL/TP hits (price poll)
        │
        ▼
3. For each trading pair (BTC, ETH, SOL, BNB):
   │
   ├─ Already in position? → SKIP
   │
   ├─ Fetch candles (1m/5m/15m) + order book + sentiment
   │
   ├─ Run all 8 agents in parallel (24 analysis tasks)
   │
   ├─ Regime check → VOLATILE? → SKIP
   │
   ├─ Temporal confluence check (5m + 1 other TF)
   │
   ├─ Compute weighted consensus score
   │
   ├─ Score < threshold? → HOLD (no trade)
   │
   ├─ Risk Manager: size position (Kelly + heat check)
   │
   ├─ Position rejected (heat full / drawdown)? → SKIP
   │
   ├─ Place market entry order via Binance API
   │
   ├─ Place OCO exit (TP + SL) immediately
   │
   └─ Save trade to SQLite DB, update risk tracker
        │
        ▼
4. Save performance snapshot to DB
        │
        ▼
5. Broadcast status via WebSocket to dashboard
        │
        ▼
[Sleep 30s, repeat]
```

---

## 6. Testnet vs Live Mode

### Testnet Mode (Default)
- Uses `https://testnet.binance.vision`
- Real market data, simulated paper money
- Orders are executed as if real but no actual funds move
- Set up at: https://testnet.binance.vision (click "Generate HMAC_SHA256 Key")
- **Always start here and observe for several days before going live**

### Live Mode
- Uses `https://api.binance.com`
- Real money, real orders
- Enable by setting `TRADING_MODE=live` in `.env`
- You can also switch via the dashboard Settings page

### Recommended Protocol
1. Run testnet for 1–2 weeks
2. Verify win rate > 50%, profit factor > 1.2
3. Start live with minimal capital (e.g. $100 USDT)
4. Scale up gradually as live results match testnet

---

## 7. Installation & Setup

### Requirements
- Python 3.11+
- Binance account (https://www.binance.com)
- Testnet account (https://testnet.binance.vision)

### Steps

```bash
# 1. Clone / copy the apex7 directory

# 2. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys (see Section 8)

# 5. Create data directory
mkdir -p data

# 6. Run the bot
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 7. Open dashboard
# Navigate to http://localhost:8000
```

---

## 8. Configuration (.env Guide)

```bash
# ── Binance Live API (only needed for live trading) ──
BINANCE_API_KEY=xxxx
BINANCE_API_SECRET=xxxx

# ── Binance Testnet API ──
# Get from: https://testnet.binance.vision → Generate HMAC Key
BINANCE_TESTNET_API_KEY=xxxx
BINANCE_TESTNET_API_SECRET=xxxx

# ── Mode ──
TRADING_MODE=testnet   # Change to "live" for real trading

# ── Pairs ──
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT

# ── Risk (start conservative, adjust after testing) ──
MAX_PORTFOLIO_RISK_PCT=2.0    # 2% risk per trade (very conservative)
MAX_PORTFOLIO_HEAT_PCT=8.0    # Max 8% at risk total
MAX_DRAWDOWN_HALT_PCT=15.0    # Halt if 15% drawdown
TRADE_USDT_CAP=500.0          # Never risk more than $500 per trade

# ── Strategy ──
MIN_CONSENSUS_SCORE=0.68      # 68% weighted agreement required
MIN_AGENTS_AGREE=5            # At least 5 of 8 agents must agree
```

**Conservative settings** (recommended for new users):
- `MAX_PORTFOLIO_RISK_PCT=1.0`
- `MIN_CONSENSUS_SCORE=0.75`
- `TRADE_USDT_CAP=100.0`

---

## 9. Running the Bot

### Development mode (auto-reload on code changes)
```bash
uvicorn main:app --reload --port 8000
```

### Production mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```
> Use `--workers 1` only. The trading engine is a singleton — multiple workers would create duplicate orders.

### As a background service (Linux systemd)
```ini
[Unit]
Description=APEX-7 Trading Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/path/to/apex7
ExecStart=/path/to/apex7/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 10. Dashboard UI Guide

### Header
- **Green pulse dot** = Engine running
- **TESTNET/LIVE badge** = Current mode
- **SCAN #N** = Number of complete scan cycles
- **START/STOP button** = Engine control

### Dashboard Page
- **Stat cards**: Portfolio value, total PnL, win rate, open positions
- **Price chart**: Live OHLCV candlestick chart with volume
- **Indicator row**: RSI, MACD histogram, BB%, Volume Z-score, ADX
- **Recent trades**: Last 8 trades with PnL

### Trades Page
Full trade history with filtering by status (Open/Closed/All).

### Performance Page
- Win rate, profit factor, average win/loss
- Best and worst trade
- Average holding time
- Equity curve chart

### Agent Signals Page
- Live card for each agent showing latest signal and confidence bar
- Full signal table with reasons

### Settings Page
- Switch between testnet and live
- Adjust consensus threshold (higher = fewer but higher quality trades)
- Adjust risk parameters
- Changes take effect on next scan cycle

### Right Sidebar
- **Consensus Dial**: Visual score gauge
- **Risk Monitor**: Heat, drawdown, win rate progress bars
- **Live Signal Stream**: Real-time agent votes as they come in
- **Engine Log**: Last 50 engine events

---

## 11. Architecture Overview

```
apex7/
├── main.py              → FastAPI app, all API routes, WebSocket server
├── config.py            → Settings loaded from .env
├── database.py          → SQLAlchemy models (Trade, AgentSignal, PerformanceSnapshot)
├── market_data.py       → Binance API client, all TA indicators, caching
├── consensus_engine.py  → Polyphonic Consensus Engine (PCE) — the brain
├── risk_manager.py      → Kelly sizing, heat, drawdown, position tracking
├── order_executor.py    → Binance order placement, OCO exits, quantity rounding
├── trading_engine.py    → Main orchestrator loop, all modules connected here
├── agents/
│   ├── base_agent.py    → Abstract base class for all agents
│   ├── momentum_agent.py
│   ├── mean_reversion_agent.py
│   ├── breakout_agent.py
│   ├── volume_agent.py
│   ├── sentiment_agent.py
│   ├── orderbook_agent.py
│   ├── scalping_agent.py
│   └── regime_agent.py
├── static/
│   └── index.html       → Full dashboard UI
├── data/
│   └── apex7.db         → SQLite database (auto-created)
├── .env                 → Your credentials (never commit this)
├── .env.example         → Template for .env
└── requirements.txt     → Python dependencies
```

### Data Flow

```
Binance API
    │
    ▼
market_data.py (candles, orderbook, prices)
    │
    ▼
consensus_engine.py → 8 agents × 3 timeframes = up to 24 analyses
    │
    ▼
PolyphonicConsensusEngine → weighted vote → ConsensusResult
    │
    ▼
risk_manager.py → Kelly sizing → PositionSize
    │
    ▼
order_executor.py → Binance market order → OCO exit
    │
    ▼
database.py → SQLite persistence
    │
    ▼
main.py → WebSocket broadcast → Dashboard UI
```

---

## 12. Performance Expectations

### Realistic Expectations
- **Trade frequency**: 2–10 trades per day across 4 pairs (high consensus threshold means selectivity)
- **Win rate target**: 55–65% (the Kelly sizing ensures wins > losses even at lower win rates)
- **Risk per trade**: 1–2% of portfolio
- **Drawdown**: Should stay below 10% in normal markets

### What Can Hurt Performance
- **Gap openings**: Crypto trades 24/7 but sudden news gaps can gap through stop losses
- **Low liquidity hours**: 3–6 AM UTC can produce false signals
- **Correlated pairs**: BTC, ETH, SOL, BNB are all correlated — in a BTC crash, all 4 pairs may trigger simultaneously

### How to Improve Performance
1. Raise `MIN_CONSENSUS_SCORE` to 0.75+ for higher quality but fewer trades
2. Limit pairs to 1–2 to reduce correlation risk
3. Disable scalping agent for more conservative signals
4. Monitor which agents have the highest accuracy in your live logs and adjust weights in code

---

## 13. Known Limitations & Warnings

### ⚠ THIS IS NOT FINANCIAL ADVICE
Trading cryptocurrencies involves substantial risk of loss. Past performance of any strategy does not guarantee future results. Only trade with capital you can afford to lose entirely.

### Technical Limitations
1. **Spot only**: This bot trades spot markets. Futures/margin require significant additional safety code.
2. **No slippage modeling**: Market orders in low liquidity can fill at worse prices than expected.
3. **API rate limits**: Running on many pairs simultaneously increases API call frequency. Binance allows 1200 requests/minute.
4. **Single process**: The bot must run in a single uvicorn worker. Multi-worker deployment is not supported.
5. **OCO limitation**: Some symbols on Binance testnet don't support OCO orders. The code falls back to a simple limit order.

### Before Going Live Checklist
- [ ] Ran on testnet for at least 7 days
- [ ] Win rate > 50% on testnet
- [ ] Profit factor > 1.0 on testnet
- [ ] Verified orders appear correctly in Binance testnet UI
- [ ] Set conservative USDT cap (start with $50–$100 maximum per trade)
- [ ] Enabled 2FA on your Binance account
- [ ] API key has only "Spot Trading" permission — no withdrawal permission

---

*APEX-7 v1.0 — Built with Python, FastAPI, SQLAlchemy, and Binance API*
