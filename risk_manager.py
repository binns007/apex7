"""
APEX-7 Risk Manager
════════════════════
Fixes vs v1:
  - _reject() no longer has the walrus-operator side always defaulting
    incorrectly; it always takes an explicit `side` parameter (this was
    already patched in the file we inherited, kept here).
  - NEW: NaN/zero guards on entry_price, stop_loss_pct, take_profit_pct
    before they're used in division or price math. A NaN slipping in
    from an indicator warmup period could previously have produced NaN
    order parameters, propagated silently through `min()/max()` clamps.
  - NEW: guards against non-positive stop_loss_pct which would otherwise
    divide-by-near-zero and produce absurd position sizes.
"""
import logging
import math
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
        self._open_risk_usdt: dict[str, float] = {}
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
        # ── NaN / sanity guards (new) ─────────
        for name, val in (
            ("entry_price", entry_price),
            ("stop_loss_pct", stop_loss_pct),
            ("take_profit_pct", take_profit_pct),
            ("portfolio_usdt", portfolio_usdt),
            ("consensus_score", consensus_score),
        ):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return self._reject(0.0, 0.01, 0.02, side, f"Invalid {name}: NaN/None")

        if entry_price <= 0:
            return self._reject(0.0, stop_loss_pct, take_profit_pct, side,
                                 f"Invalid entry_price={entry_price}")
        if stop_loss_pct <= 0:
            return self._reject(entry_price, 0.01, take_profit_pct, side,
                                 f"Invalid stop_loss_pct={stop_loss_pct}")

        # ── Guard: zero balance almost always means bad API key ──
        if portfolio_usdt < 1.0:
            logger.warning(
                "❌ Portfolio balance is $%.2f — check API key / testnet configuration. "
                "No trades will be placed until balance > $1.", portfolio_usdt
            )
            return self._reject(
                entry_price, stop_loss_pct, take_profit_pct, side,
                f"Portfolio balance ${portfolio_usdt:.2f} — verify API key and testnet settings"
            )

        if self.halted:
            return self._reject(
                entry_price, stop_loss_pct, take_profit_pct, side,
                f"Trading halted: {self.halt_reason}"
            )

        # ── Drawdown check ────────────────────
        if self._current_value > 0 and self._peak_value > 0:
            dd = (self._peak_value - self._current_value) / self._peak_value * 100
            if dd >= settings.MAX_DRAWDOWN_HALT_PCT:
                self.halted = True
                self.halt_reason = f"Max drawdown {dd:.1f}% exceeded"
                return self._reject(
                    entry_price, stop_loss_pct, take_profit_pct, side, self.halt_reason
                )

        # ── Portfolio heat check ───────────────
        total_risk = sum(self._open_risk_usdt.values())
        heat_pct = total_risk / (portfolio_usdt + 1e-9) * 100
        if heat_pct >= settings.MAX_PORTFOLIO_HEAT_PCT:
            return self._reject(
                entry_price, stop_loss_pct, take_profit_pct, side,
                f"Portfolio heat {heat_pct:.1f}% at max ({settings.MAX_PORTFOLIO_HEAT_PCT}%)"
            )

        # ── Already exposed to this symbol? ───
        if symbol in self._open_risk_usdt and self._open_risk_usdt[symbol] > 0:
            return self._reject(
                entry_price, stop_loss_pct, take_profit_pct, side,
                f"Already have open risk on {symbol}"
            )

        # ── Kelly position sizing ──────────────
        win_rate = self._win_rate()
        rr_ratio = take_profit_pct / (stop_loss_pct + 1e-9)

        kelly_f = max((win_rate * (rr_ratio + 1) - 1) / (rr_ratio + 1e-9), 0.0)
        fractional_kelly = kelly_f * 0.25
        max_risk_pct = settings.MAX_PORTFOLIO_RISK_PCT / 100
        risk_pct = min(fractional_kelly, max_risk_pct)

        conviction_boost = 0.8 + consensus_score * 0.4
        risk_pct = min(risk_pct * conviction_boost, max_risk_pct * 1.3)

        # Floor: with limited trade history, use half the configured max
        # rather than letting an unproven Kelly estimate zero out sizing.
        if self._total_trades < 10:
            risk_pct = max(risk_pct, max_risk_pct * 0.5)

        risk_usdt = portfolio_usdt * risk_pct
        usdt_value = risk_usdt / (stop_loss_pct + 1e-9)
        usdt_value = min(usdt_value, settings.TRADE_USDT_CAP)
        risk_usdt = usdt_value * stop_loss_pct

        if usdt_value < settings.MIN_TRADE_USDT:
            return self._reject(
                entry_price, stop_loss_pct, take_profit_pct, side,
                f"Position too small (${usdt_value:.2f} < ${settings.MIN_TRADE_USDT} USDT)"
            )

        quantity = usdt_value / entry_price

        if side == "BUY":
            sl_price = entry_price * (1 - stop_loss_pct)
            tp_price = entry_price * (1 + take_profit_pct)
        else:
            sl_price = entry_price * (1 + stop_loss_pct)
            tp_price = entry_price * (1 - take_profit_pct)

        logger.info(
            "✅ Risk approved %s %s: qty=%.6f value=$%.2f risk=$%.2f "
            "SL=%.4f TP=%.4f (kelly=%.3f conv_boost=%.2f)",
            symbol, side, quantity, usdt_value, risk_usdt,
            sl_price, tp_price, fractional_kelly, conviction_boost
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
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self._current_value = value
        if value > self._peak_value:
            self._peak_value = value

    def resume_trading(self):
        self.halted = False
        self.halt_reason = ""

    # ─────────────────────────────────────────
    #  Metrics
    # ─────────────────────────────────────────
    def _win_rate(self) -> float:
        if self._total_trades < 10:
            return 0.52
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

    def _reject(
        self,
        entry: float,
        sl_pct: float,
        tp_pct: float,
        side: str,
        reason: str,
    ) -> PositionSize:
        try:
            if side == "BUY":
                sl_price = entry * (1 - sl_pct)
                tp_price = entry * (1 + tp_pct)
            else:
                sl_price = entry * (1 + sl_pct)
                tp_price = entry * (1 - tp_pct)
        except Exception:
            sl_price = tp_price = 0.0
        logger.warning("❌ Position rejected: %s", reason)
        return PositionSize(
            quantity=0,
            usdt_value=0,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            risk_usdt=0,
            approved=False,
            rejection_reason=reason,
        )