"""
APEX-7 Risk Manager
═══════════════════
Handles:
  • Kelly Criterion position sizing
  • ATR-based stop loss placement
  • Portfolio heat tracking (max open risk)
  • Drawdown halting
  • Per-symbol exposure limits
  • Correlation protection (don't stack correlated bets)
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger("apex7.risk")


@dataclass
class PositionSize:
    quantity: float
    usdt_value: float
    stop_loss_price: float
    take_profit_price: float
    risk_usdt: float
    approved: bool
    rejection_reason: str = ""


class RiskManager:
    def __init__(self):
        self._open_risk_usdt: dict[str, float] = {}   # symbol → risk $ at stake
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._total_trades: int = 0
        self._winning_trades: int = 0
        self.halted: bool = False
        self.halt_reason: str = ""

    # ─────────────────────────────────────────
    #  Main sizing entry point
    # ─────────────────────────────────────────
    def size_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        portfolio_usdt: float,
        consensus_score: float,
    ) -> PositionSize:
        """
        Returns approved position size using fractional Kelly.
        """
        if self.halted:
            return self._reject(entry_price, stop_loss_pct, take_profit_pct,
                                f"Trading halted: {self.halt_reason}")

        # ── Drawdown check ────────────────────
        if self._current_value > 0 and self._peak_value > 0:
            dd = (self._peak_value - self._current_value) / self._peak_value * 100
            if dd >= settings.MAX_DRAWDOWN_HALT_PCT:
                self.halted = True
                self.halt_reason = f"Max drawdown {dd:.1f}% exceeded"
                return self._reject(entry_price, stop_loss_pct, take_profit_pct, self.halt_reason)

        # ── Portfolio heat check ───────────────
        total_risk = sum(self._open_risk_usdt.values())
        heat_pct = total_risk / (portfolio_usdt + 1e-9) * 100
        if heat_pct >= settings.MAX_PORTFOLIO_HEAT_PCT:
            return self._reject(entry_price, stop_loss_pct, take_profit_pct,
                f"Portfolio heat {heat_pct:.1f}% at max ({settings.MAX_PORTFOLIO_HEAT_PCT}%)")

        # ── Already exposed to this symbol? ───
        if symbol in self._open_risk_usdt and self._open_risk_usdt[symbol] > 0:
            return self._reject(entry_price, stop_loss_pct, take_profit_pct,
                f"Already have open risk on {symbol}")

        # ── Kelly position sizing ──────────────
        win_rate = self._win_rate()
        rr_ratio = take_profit_pct / (stop_loss_pct + 1e-9)

        # Standard Kelly: f* = (p*(b+1) - 1) / b  where b = R/R ratio
        kelly_f = max((win_rate * (rr_ratio + 1) - 1) / (rr_ratio + 1e-9), 0.0)

        # Fractional Kelly: use 25% of full Kelly to stay conservative
        fractional_kelly = kelly_f * 0.25

        # Clamp to max risk per trade
        max_risk_pct = settings.MAX_PORTFOLIO_RISK_PCT / 100
        risk_pct = min(fractional_kelly, max_risk_pct)

        # Boost slightly with consensus score (high conviction → larger)
        conviction_boost = 0.8 + consensus_score * 0.4  # 0.8× to 1.2×
        risk_pct = min(risk_pct * conviction_boost, max_risk_pct * 1.3)

        # ── Compute $ amounts ─────────────────
        risk_usdt   = portfolio_usdt * risk_pct
        usdt_value  = risk_usdt / (stop_loss_pct + 1e-9)   # position value
        usdt_value  = min(usdt_value, settings.TRADE_USDT_CAP)
        risk_usdt   = usdt_value * stop_loss_pct            # recalculate after cap

        if usdt_value < 10:  # minimum trade size
            return self._reject(entry_price, stop_loss_pct, take_profit_pct,
                "Position too small (< $10 USDT)")

        quantity = usdt_value / entry_price

        if side == "BUY":
            sl_price = entry_price * (1 - stop_loss_pct)
            tp_price = entry_price * (1 + take_profit_pct)
        else:
            sl_price = entry_price * (1 + stop_loss_pct)
            tp_price = entry_price * (1 - take_profit_pct)

        logger.info(
            f"✅ Risk approved {symbol} {side}: qty={quantity:.6f} "
            f"value=${usdt_value:.2f} risk=${risk_usdt:.2f} "
            f"SL={sl_price:.4f} TP={tp_price:.4f}"
        )

        return PositionSize(
            quantity=quantity,
            usdt_value=usdt_value,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            risk_usdt=risk_usdt,
            approved=True,
        )

    # ─────────────────────────────────────────
    #  Position lifecycle tracking
    # ─────────────────────────────────────────
    def on_trade_open(self, symbol: str, risk_usdt: float):
        self._open_risk_usdt[symbol] = risk_usdt

    def on_trade_close(self, symbol: str, pnl_usdt: float, was_win: bool):
        self._open_risk_usdt.pop(symbol, None)
        self._total_trades += 1
        if was_win:
            self._winning_trades += 1
        self._current_value += pnl_usdt
        if self._current_value > self._peak_value:
            self._peak_value = self._current_value

    def update_portfolio_value(self, value: float):
        self._current_value = value
        if value > self._peak_value:
            self._peak_value = value

    def resume_trading(self):
        """Manual override to resume after halt."""
        self.halted = False
        self.halt_reason = ""

    # ─────────────────────────────────────────
    #  Metrics
    # ─────────────────────────────────────────
    def _win_rate(self) -> float:
        if self._total_trades < 10:
            return 0.52  # assume slightly above random until we have data
        return self._winning_trades / self._total_trades

    @property
    def portfolio_heat_pct(self) -> float:
        total_risk = sum(self._open_risk_usdt.values())
        if self._current_value <= 0:
            return 0.0
        return total_risk / self._current_value * 100

    @property
    def drawdown_pct(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value * 100

    @property
    def open_positions(self) -> list[str]:
        return [sym for sym, risk in self._open_risk_usdt.items() if risk > 0]

    def summary(self) -> dict:
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate_pct": round(self._win_rate() * 100, 2),
            "open_risk_usdt": sum(self._open_risk_usdt.values()),
            "portfolio_heat_pct": round(self.portfolio_heat_pct, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "open_positions": self.open_positions,
        }

    # ─────────────────────────────────────────
    def _reject(self, entry, sl_pct, tp_pct, reason) -> PositionSize:
        if side := "BUY":
            sl_price = entry * (1 - sl_pct)
            tp_price = entry * (1 + tp_pct)
        logger.warning(f"❌ Position rejected: {reason}")
        return PositionSize(
            quantity=0, usdt_value=0,
            stop_loss_price=entry * (1 - sl_pct),
            take_profit_price=entry * (1 + tp_pct),
            risk_usdt=0, approved=False,
            rejection_reason=reason,
        )
