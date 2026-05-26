"""
Live Shadow Mode (REQ-P3-09).

Runs a challenger model/strategy alongside the champion in paper-trade mode.
The challenger generates signals and tracks hypothetical PnL without placing
real orders. After a configurable observation period (default 4 weeks), the
operator can compare performance and decide whether to promote.

This is the gate between "backtested well" and "deployed with real money."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from core.logger import get_logger

logger = get_logger("midas.shadow")


@dataclass
class ShadowTrade:
    """A hypothetical trade tracked by the shadow system."""
    signal_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit: float = 0.0
    status: str = "OPEN"  # OPEN, TP_HIT, SL_HIT, EXPIRED


@dataclass
class ShadowReport:
    """Performance comparison between champion and challenger."""
    challenger_name: str
    observation_days: int
    champion_trades: int = 0
    champion_pnl: float = 0.0
    challenger_trades: int = 0
    challenger_pnl: float = 0.0
    challenger_win_rate: float = 0.0
    ready_to_promote: bool = False
    reason: str = ""

    def summary(self) -> str:
        return (
            f"Shadow Report ({self.challenger_name}, {self.observation_days}d):\n"
            f"  Champion:   {self.champion_trades} trades, PnL=${self.champion_pnl:+.2f}\n"
            f"  Challenger: {self.challenger_trades} trades, PnL=${self.challenger_pnl:+.2f}, "
            f"WR={self.challenger_win_rate:.1%}\n"
            f"  Promote: {'YES' if self.ready_to_promote else 'NO'} — {self.reason}"
        )


class ShadowRunner:
    """Tracks a challenger strategy's hypothetical performance.

    Usage:
        shadow = ShadowRunner("lgbm_v2", min_days=28)
        # Each cycle:
        shadow.record_signal(direction, price, sl, tp, confidence)
        shadow.update_open_trades(current_price)
        # After observation period:
        report = shadow.evaluate(champion_pnl, champion_trades)
    """

    def __init__(
        self,
        challenger_name: str,
        min_days: int = 28,
        min_trades: int = 20,
    ) -> None:
        self.challenger_name = challenger_name
        self.min_days = min_days
        self.min_trades = min_trades
        self.start_time = datetime.now(timezone.utc)
        self.trades: List[ShadowTrade] = []

    @property
    def elapsed_days(self) -> int:
        return (datetime.now(timezone.utc) - self.start_time).days

    @property
    def is_observation_complete(self) -> bool:
        return self.elapsed_days >= self.min_days and len(self.trades) >= self.min_trades

    def record_signal(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
    ) -> None:
        """Record a challenger signal as a hypothetical trade."""
        if direction not in ("BUY", "SELL"):
            return
        trade = ShadowTrade(
            signal_time=datetime.now(timezone.utc),
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
        )
        self.trades.append(trade)
        logger.debug(f"Shadow trade: {direction} @ {entry_price:.2f}")

    def update_open_trades(self, current_price: float) -> None:
        """Check SL/TP for open shadow trades."""
        for trade in self.trades:
            if trade.status != "OPEN":
                continue

            if trade.direction == "BUY":
                if current_price <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.profit = (trade.stop_loss - trade.entry_price) * 100  # simplified
                    trade.status = "SL_HIT"
                elif current_price >= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.profit = (trade.take_profit - trade.entry_price) * 100
                    trade.status = "TP_HIT"
            else:  # SELL
                if current_price >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.profit = (trade.entry_price - trade.stop_loss) * 100
                    trade.status = "SL_HIT"
                elif current_price <= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.profit = (trade.entry_price - trade.take_profit) * 100
                    trade.status = "TP_HIT"

            if trade.status != "OPEN":
                trade.exit_time = datetime.now(timezone.utc)

    def evaluate(self, champion_pnl: float = 0.0, champion_trades: int = 0) -> ShadowReport:
        """Evaluate whether the challenger is ready for promotion."""
        closed = [t for t in self.trades if t.status != "OPEN"]
        wins = [t for t in closed if t.profit > 0]
        total_pnl = sum(t.profit for t in closed)
        win_rate = len(wins) / len(closed) if closed else 0.0

        ready = False
        reason = ""

        if not self.is_observation_complete:
            reason = f"observation incomplete ({self.elapsed_days}/{self.min_days} days, {len(closed)}/{self.min_trades} trades)"
        elif total_pnl <= 0:
            reason = "challenger PnL is negative"
        elif total_pnl <= champion_pnl and champion_pnl > 0:
            reason = f"challenger PnL ${total_pnl:.2f} <= champion ${champion_pnl:.2f}"
        elif win_rate < 0.45:
            reason = f"challenger win rate {win_rate:.1%} < 45%"
        else:
            ready = True
            reason = f"challenger outperforms: PnL=${total_pnl:.2f}, WR={win_rate:.1%}"

        return ShadowReport(
            challenger_name=self.challenger_name,
            observation_days=self.elapsed_days,
            champion_trades=champion_trades,
            champion_pnl=champion_pnl,
            challenger_trades=len(closed),
            challenger_pnl=total_pnl,
            challenger_win_rate=win_rate,
            ready_to_promote=ready,
            reason=reason,
        )
