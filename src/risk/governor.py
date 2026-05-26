"""
Risk Governor (REQ-P1-08).

A small, deterministic gatekeeper that decides whether the bot is allowed to
open a new position right now. Failure modes ranked from "stop hard" to
"shrink size":

  1. Equity floor breached                → permanent stop
  2. Daily-loss cap hit                   → stop until next UTC day
  3. Max-drawdown cap hit                 → permanent stop (until manual reset)
  4. Consecutive-loss circuit breaker     → stop until reset() or a winner
  5. Order-rate limiter cooling           → wait
  6. Drawdown soft-scaling                → allow but scale risk_percent down

The governor is *advisory* — it returns a decision; the executor is responsible
for honoring it. That separation is intentional so the same governor can be
wired into the live loop and the backtester.

The governor is intentionally synchronous and stateless beyond its own
counters. It does not reach into MT5 or the database; the caller passes in an
``AccountState`` snapshot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from config import settings
from core.logger import get_logger

logger = get_logger("midas.risk")


@dataclass
class AccountState:
    """A point-in-time view of the live account.

    All values are in account currency (USD on most XAUUSD ECN brokers).
    """

    equity: float
    balance: float
    open_positions: int = 0
    # Optional: caller may pass these if it tracks them. The governor will fall
    # back to its own internal state when omitted.
    day_start_equity: Optional[float] = None
    peak_equity: Optional[float] = None


@dataclass
class RiskDecision:
    """The governor's verdict for a candidate order.

    ``allow``: whether to send the order at all.
    ``reason``: short, human-readable explanation (also used for logging).
    ``risk_scale``: multiplier in [0, 1] applied to ``risk_percent``. ``1.0``
    means full size; ``0.5`` means halve the lot size; ``0.0`` means decline.
    """

    allow: bool
    reason: str
    risk_scale: float = 1.0


@dataclass
class _GovernorMemory:
    """Internal counters the governor maintains across calls."""

    day: Optional[date] = None
    day_start_equity: Optional[float] = None
    peak_equity: Optional[float] = None
    consecutive_losses: int = 0
    last_order_ts: float = 0.0
    halted_until: Optional[datetime] = None
    halt_reason: str = ""


class RiskGovernor:
    """Gatekeeper for new orders. See module docstring."""

    def __init__(
        self,
        *,
        daily_loss_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        equity_floor: Optional[float] = None,
        max_consecutive_losses: Optional[int] = None,
        dd_scale_threshold: Optional[float] = None,
        dd_scale_floor: Optional[float] = None,
        min_seconds_between_orders: Optional[int] = None,
        clock=None,
    ) -> None:
        # Pull defaults from validated settings; callers may override per-instance.
        self.daily_loss_pct = (
            daily_loss_pct if daily_loss_pct is not None else settings.risk_daily_loss_pct
        )
        self.max_drawdown_pct = (
            max_drawdown_pct if max_drawdown_pct is not None else settings.risk_max_drawdown_pct
        )
        self.equity_floor = (
            equity_floor if equity_floor is not None else settings.risk_equity_floor
        )
        self.max_consecutive_losses = (
            max_consecutive_losses
            if max_consecutive_losses is not None
            else settings.risk_max_consecutive_losses
        )
        self.dd_scale_threshold = (
            dd_scale_threshold
            if dd_scale_threshold is not None
            else settings.risk_dd_scale_threshold
        )
        self.dd_scale_floor = (
            dd_scale_floor if dd_scale_floor is not None else settings.risk_dd_scale_floor
        )
        self.min_seconds_between_orders = (
            min_seconds_between_orders
            if min_seconds_between_orders is not None
            else settings.risk_min_seconds_between_orders
        )

        self._mem = _GovernorMemory()
        # Test seam — `clock()` returns float seconds since epoch (UTC).
        self._clock = clock or time.time

    # ------------------------------------------------------------------ state

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc)

    def _refresh_day_marker(self, account: AccountState) -> None:
        """Reset the daily-loss marker at each UTC day boundary."""
        today = self._now().date()
        if self._mem.day != today:
            self._mem.day = today
            # Prefer the account's own day-start if the caller tracks it
            self._mem.day_start_equity = (
                account.day_start_equity
                if account.day_start_equity is not None
                else account.equity
            )
            # Clear daily halts (DD halts persist intentionally)
            if self._mem.halt_reason.startswith("daily-loss"):
                self._mem.halted_until = None
                self._mem.halt_reason = ""

    def _refresh_peak(self, account: AccountState) -> None:
        peak = account.peak_equity or self._mem.peak_equity or account.equity
        if account.equity > peak:
            peak = account.equity
        self._mem.peak_equity = peak

    # --------------------------------------------------------------- mutators

    def record_trade_result(self, profit: float) -> None:
        """Update the consecutive-loss counter after a trade closes."""
        if profit < 0:
            self._mem.consecutive_losses += 1
            logger.info(
                f"Risk governor: loss recorded "
                f"(consecutive={self._mem.consecutive_losses}/{self.max_consecutive_losses})"
            )
        else:
            if self._mem.consecutive_losses:
                logger.info("Risk governor: loss streak broken")
            self._mem.consecutive_losses = 0

    def record_order_sent(self) -> None:
        """Mark that an order was just sent so the rate limiter can throttle."""
        self._mem.last_order_ts = self._clock()

    def reset(self) -> None:
        """Operator action: clear halts and the consecutive-loss counter."""
        logger.warning("Risk governor manually reset")
        self._mem.consecutive_losses = 0
        self._mem.halted_until = None
        self._mem.halt_reason = ""

    # --------------------------------------------------------------- queries

    @property
    def consecutive_losses(self) -> int:
        return self._mem.consecutive_losses

    @property
    def is_halted(self) -> bool:
        return bool(self._mem.halt_reason)

    def status(self) -> dict:
        return {
            "halted": self.is_halted,
            "halt_reason": self._mem.halt_reason,
            "consecutive_losses": self._mem.consecutive_losses,
            "peak_equity": self._mem.peak_equity,
            "day_start_equity": self._mem.day_start_equity,
            "day": self._mem.day.isoformat() if self._mem.day else None,
        }

    # ------------------------------------------------------------- decisions

    def can_open(self, account: AccountState) -> RiskDecision:
        """Decide whether a new order is allowed right now."""
        self._refresh_day_marker(account)
        self._refresh_peak(account)

        equity = account.equity

        # 1. Equity floor (permanent).
        if equity <= self.equity_floor:
            return self._block(
                f"equity-floor: equity ${equity:.2f} <= floor ${self.equity_floor:.2f}"
            )

        # 2. Daily-loss cap.
        day_start = self._mem.day_start_equity or equity
        if day_start > 0:
            day_pnl_pct = (equity - day_start) / day_start
            if day_pnl_pct <= -self.daily_loss_pct:
                self._mem.halt_reason = "daily-loss"
                return self._block(
                    f"daily-loss: {day_pnl_pct:+.2%} <= -{self.daily_loss_pct:.2%}"
                )

        # 3. Max drawdown.
        peak = self._mem.peak_equity or equity
        dd_pct = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
        if dd_pct >= self.max_drawdown_pct:
            self._mem.halt_reason = "max-drawdown"
            return self._block(
                f"max-drawdown: {dd_pct:.2%} >= {self.max_drawdown_pct:.2%}"
            )

        # 4. Consecutive-loss circuit breaker.
        if self._mem.consecutive_losses >= self.max_consecutive_losses:
            return RiskDecision(
                allow=False,
                reason=(
                    f"consecutive-losses: {self._mem.consecutive_losses}"
                    f" >= {self.max_consecutive_losses}"
                ),
                risk_scale=0.0,
            )

        # 5. Order rate limiter.
        seconds_since_last = self._clock() - self._mem.last_order_ts
        if seconds_since_last < self.min_seconds_between_orders:
            wait = self.min_seconds_between_orders - seconds_since_last
            return RiskDecision(
                allow=False,
                reason=f"rate-limit: wait {wait:.0f}s before next order",
                risk_scale=0.0,
            )

        # 6. Drawdown-aware soft scaling.
        scale = self._dd_scale(dd_pct)
        if scale < 1.0:
            return RiskDecision(
                allow=True,
                reason=f"dd-scaled: {dd_pct:.2%} dd, risk x{scale:.2f}",
                risk_scale=scale,
            )

        return RiskDecision(allow=True, reason="ok", risk_scale=1.0)

    # ---------------------------------------------------------- internals

    def _block(self, reason: str) -> RiskDecision:
        if not self._mem.halt_reason:
            self._mem.halt_reason = reason
        logger.error(f"Risk governor BLOCK: {reason}")
        return RiskDecision(allow=False, reason=reason, risk_scale=0.0)

    def _dd_scale(self, dd_pct: float) -> float:
        """Linear ramp between dd_scale_threshold and max_drawdown_pct."""
        if dd_pct <= self.dd_scale_threshold:
            return 1.0
        if dd_pct >= self.max_drawdown_pct:
            return self.dd_scale_floor
        # Linear interpolation
        span = self.max_drawdown_pct - self.dd_scale_threshold
        progressed = dd_pct - self.dd_scale_threshold
        ratio = progressed / span if span > 0 else 1.0
        return max(self.dd_scale_floor, 1.0 - ratio * (1.0 - self.dd_scale_floor))
