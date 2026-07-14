"""
APEX-7 Futures Risk Manager
════════════════════════════
Same fractional-Kelly + portfolio-heat + drawdown-halt mechanism as
the spot RiskManager (subclassed directly — halt tracking, heat
tracking, drawdown tracking, and win-rate bookkeeping are identical
and reused as-is), extended for leveraged futures:

  - Sizing targets a $ risk (max loss at stop-loss) exactly like spot,
    then derives the MARGIN required to hold that risk at the chosen
    leverage:

        notional  = risk_usdt / stop_loss_pct
        margin    = notional / leverage

    Higher leverage means the same $ risk needs less margin — that's
    the capital efficiency futures gives you for the "quick, small,
    frequent" trade style this mode is built for: more of your account
    stays free to size the NEXT signal instead of being tied up in the
    current one.

  - NEW: liquidation safety check. A leveraged position gets liquidated
    at roughly entry × (1 − 1/leverage + maintenance_margin_rate) for a
    long (mirrored for a short) — see config.py's
    FUTURES_MAINTENANCE_MARGIN_RATE comment for caveats on this being
    an estimate, not Binance's exact tiered-bracket formula. If the
    consensus engine's stop-loss distance isn't comfortably inside that
    estimated liquidation distance, the position is REJECTED rather
    than opened — ordinary price noise should never be able to
    liquidate a position that was supposed to get stopped out first.
    Caller should retry with lower leverage and/or let the consensus
    engine's ATR-based stop widen naturally in that symbol's regime.
"""
import logging
import math
from dataclasses import dataclass

from config import settings
from risk_manager import RiskManager, PositionSize

logger = logging.getLogger("apex7.futures_risk")


@dataclass
class FuturesPositionSize(PositionSize):
    leverage: int = 1
    margin_usdt: float = 0.0
    notional_usdt: float = 0.0
    liquidation_price: float = 0.0
    liquidation_distance_pct: float = 0.0


class FuturesRiskManager(RiskManager):
    """Inherits halt/heat/drawdown/win-rate state + on_trade_open/close/
    update_portfolio_value/resume_trading/summary from RiskManager
    unmodified. Only size_position (renamed reject helper) is leverage-
    aware and futures-specific."""

    def size_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        portfolio_usdt: float,
        consensus_score: float,
        leverage: int,
    ) -> FuturesPositionSize:
        # ── NaN / sanity guards ────────────────
        for name, val in (
            ("entry_price", entry_price),
            ("stop_loss_pct", stop_loss_pct),
            ("take_profit_pct", take_profit_pct),
            ("portfolio_usdt", portfolio_usdt),
            ("consensus_score", consensus_score),
        ):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return self._reject_futures(0.0, 0.01, 0.02, side, leverage, f"Invalid {name}: NaN/None")

        if entry_price <= 0:
            return self._reject_futures(0.0, stop_loss_pct, take_profit_pct, side, leverage,
                                         f"Invalid entry_price={entry_price}")
        if stop_loss_pct <= 0:
            return self._reject_futures(entry_price, 0.01, take_profit_pct, side, leverage,
                                         f"Invalid stop_loss_pct={stop_loss_pct}")

        leverage = max(1, min(int(leverage), settings.FUTURES_MAX_LEVERAGE_ALLOWED))

        # ── Liquidation safety check ───────────
        liq_distance_pct = max((1.0 / leverage) - settings.FUTURES_MAINTENANCE_MARGIN_RATE, 0.0)
        max_allowed_sl_pct = liq_distance_pct * settings.FUTURES_LIQUIDATION_SAFETY_FACTOR
        if stop_loss_pct >= max_allowed_sl_pct:
            return self._reject_futures(
                entry_price, stop_loss_pct, take_profit_pct, side, leverage,
                f"Stop-loss ({stop_loss_pct*100:.2f}%) too close to est. liquidation "
                f"({liq_distance_pct*100:.2f}%) at {leverage}x — lower leverage or widen stop"
            )

        # ── Guard: zero balance almost always means bad API key ──
        if portfolio_usdt < 1.0:
            logger.warning(
                "❌ Futures margin balance is $%.2f — check API key / testnet configuration. "
                "No trades will be placed until balance > $1.", portfolio_usdt
            )
            return self._reject_futures(
                entry_price, stop_loss_pct, take_profit_pct, side, leverage,
                f"Futures balance ${portfolio_usdt:.2f} — verify API key and testnet settings"
            )

        if self.halted:
            return self._reject_futures(
                entry_price, stop_loss_pct, take_profit_pct, side, leverage,
                f"Trading halted: {self.halt_reason}"
            )

        # ── Drawdown check ─────────────────────
        if self._current_value > 0 and self._peak_value > 0:
            dd = (self._peak_value - self._current_value) / self._peak_value * 100
            if dd >= settings.FUTURES_MAX_DRAWDOWN_HALT_PCT:
                self.halted = True
                self.halt_reason = f"Max drawdown {dd:.1f}% exceeded"
                return self._reject_futures(
                    entry_price, stop_loss_pct, take_profit_pct, side, leverage, self.halt_reason
                )

        # ── Portfolio heat check ───────────────
        total_risk = sum(self._open_risk_usdt.values())
        heat_pct = total_risk / (portfolio_usdt + 1e-9) * 100
        if heat_pct >= settings.FUTURES_MAX_PORTFOLIO_HEAT_PCT:
            return self._reject_futures(
                entry_price, stop_loss_pct, take_profit_pct, side, leverage,
                f"Portfolio heat {heat_pct:.1f}% at max ({settings.FUTURES_MAX_PORTFOLIO_HEAT_PCT}%)"
            )

        # ── Already exposed to this symbol? (one position per symbol) ──
        if symbol in self._open_risk_usdt and self._open_risk_usdt[symbol] > 0:
            return self._reject_futures(
                entry_price, stop_loss_pct, take_profit_pct, side, leverage,
                f"Already have an open futures position on {symbol}"
            )

        # ── Kelly position sizing (identical formula to spot) ──
        win_rate = self._win_rate()
        rr_ratio = take_profit_pct / (stop_loss_pct + 1e-9)

        kelly_f = max((win_rate * (rr_ratio + 1) - 1) / (rr_ratio + 1e-9), 0.0)
        fractional_kelly = kelly_f * 0.25
        max_risk_pct = settings.FUTURES_MAX_PORTFOLIO_RISK_PCT / 100
        risk_pct = min(fractional_kelly, max_risk_pct)

        conviction_boost = 0.8 + consensus_score * 0.4
        risk_pct = min(risk_pct * conviction_boost, max_risk_pct * 1.3)

        if self._total_trades < 10:
            risk_pct = max(risk_pct, max_risk_pct * 0.5)

        risk_usdt = portfolio_usdt * risk_pct
        notional_usdt = risk_usdt / (stop_loss_pct + 1e-9)

        # ── Leverage converts risk → margin requirement ──
        margin_usdt = notional_usdt / leverage
        margin_usdt = min(margin_usdt, settings.FUTURES_TRADE_MARGIN_CAP_USDT)
        # Recompute notional/risk after the margin cap so risk_usdt (used
        # for heat tracking) reflects what was ACTUALLY sized, not the
        # pre-cap target.
        notional_usdt = margin_usdt * leverage
        risk_usdt = notional_usdt * stop_loss_pct

        if margin_usdt < settings.FUTURES_MIN_TRADE_MARGIN_USDT:
            return self._reject_futures(
                entry_price, stop_loss_pct, take_profit_pct, side, leverage,
                f"Margin too small (${margin_usdt:.2f} < ${settings.FUTURES_MIN_TRADE_MARGIN_USDT} USDT)"
            )

        quantity = notional_usdt / entry_price

        if side == "BUY":
            sl_price = entry_price * (1 - stop_loss_pct)
            tp_price = entry_price * (1 + take_profit_pct)
            liq_price = entry_price * (1 - liq_distance_pct)
        else:
            sl_price = entry_price * (1 + stop_loss_pct)
            tp_price = entry_price * (1 - take_profit_pct)
            liq_price = entry_price * (1 + liq_distance_pct)

        logger.info(
            "✅ Futures risk approved %s %s %sx: qty=%.6f notional=$%.2f margin=$%.2f "
            "risk=$%.2f SL=%.4f TP=%.4f liq≈%.4f (kelly=%.3f conv_boost=%.2f)",
            symbol, side, leverage, quantity, notional_usdt, margin_usdt,
            risk_usdt, sl_price, tp_price, liq_price, fractional_kelly, conviction_boost
        )

        return FuturesPositionSize(
            quantity=quantity,
            usdt_value=notional_usdt,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            risk_usdt=risk_usdt,
            approved=True,
            leverage=leverage,
            margin_usdt=margin_usdt,
            notional_usdt=notional_usdt,
            liquidation_price=liq_price,
            liquidation_distance_pct=liq_distance_pct,
        )

    def _reject_futures(
        self, entry: float, sl_pct: float, tp_pct: float, side: str, leverage: int, reason: str,
    ) -> FuturesPositionSize:
        try:
            if side == "BUY":
                sl_price = entry * (1 - sl_pct)
                tp_price = entry * (1 + tp_pct)
            else:
                sl_price = entry * (1 + sl_pct)
                tp_price = entry * (1 - tp_pct)
        except Exception:
            sl_price = tp_price = 0.0
        logger.warning("❌ Futures position rejected: %s", reason)
        return FuturesPositionSize(
            quantity=0, usdt_value=0, stop_loss_price=sl_price, take_profit_price=tp_price,
            risk_usdt=0, approved=False, rejection_reason=reason,
            leverage=leverage, margin_usdt=0, notional_usdt=0,
            liquidation_price=0, liquidation_distance_pct=0,
        )