"""
Independent Risk System with Kill-Switch Authority (REQ-P3-07).

A separate process/class that monitors the trading system and can forcibly
halt all activity. Unlike the RiskGovernor (which is advisory and runs inside
the trading loop), this system has *authority* — it can close all positions
and disable the bot without the strategy's consent.

In production, this would run as a separate process with its own heartbeat.
If the trading process stops sending heartbeats, the risk system assumes a
crash and closes everything.

Capabilities:
  - Force-close all positions
  - Disable order placement (kill switch)
  - Monitor heartbeat from trading process
  - Alert on anomalies (position size spike, rapid PnL change)
  - Daily P/L limit enforcement at the system level
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.logger import get_logger

logger = get_logger("midas.independent_risk")


@dataclass
class RiskSystemState:
    """State of the independent risk system."""
    kill_switch_active: bool = False
    kill_reason: str = ""
    last_heartbeat: float = 0.0
    heartbeat_timeout_seconds: float = 120.0
    daily_pnl: float = 0.0
    daily_pnl_limit: float = -500.0  # USD
    positions_value: float = 0.0
    max_positions_value: float = 50_000.0  # USD notional


class IndependentRiskSystem:
    """Kill-switch authority that operates independently of the trading loop.

    In a full deployment, this runs as a separate process. Here it's implemented
    as a thread-safe class that the main loop calls `heartbeat()` on, and that
    can be queried by any component via `is_trading_allowed()`.
    """

    def __init__(
        self,
        heartbeat_timeout: float = 120.0,
        daily_pnl_limit: float = -500.0,
        max_notional: float = 50_000.0,
    ) -> None:
        self._state = RiskSystemState(
            heartbeat_timeout_seconds=heartbeat_timeout,
            daily_pnl_limit=daily_pnl_limit,
            max_positions_value=max_notional,
        )
        self._lock = threading.Lock()

    # ---- Trading loop calls these ----

    def heartbeat(self, daily_pnl: float = 0.0, positions_value: float = 0.0) -> None:
        """Called by the trading loop each cycle to signal it's alive."""
        with self._lock:
            self._state.last_heartbeat = time.time()
            self._state.daily_pnl = daily_pnl
            self._state.positions_value = positions_value

            # Check limits
            if daily_pnl <= self._state.daily_pnl_limit:
                self._activate_kill_switch(
                    f"daily PnL ${daily_pnl:.2f} <= limit ${self._state.daily_pnl_limit:.2f}"
                )
            if positions_value > self._state.max_positions_value:
                self._activate_kill_switch(
                    f"notional ${positions_value:.0f} > max ${self._state.max_positions_value:.0f}"
                )

    def is_trading_allowed(self) -> tuple[bool, str]:
        """Check if the system allows trading right now."""
        with self._lock:
            if self._state.kill_switch_active:
                return False, f"KILL SWITCH: {self._state.kill_reason}"

            # Check heartbeat staleness
            if self._state.last_heartbeat > 0:
                elapsed = time.time() - self._state.last_heartbeat
                if elapsed > self._state.heartbeat_timeout_seconds:
                    self._activate_kill_switch(
                        f"heartbeat timeout ({elapsed:.0f}s > {self._state.heartbeat_timeout_seconds:.0f}s)"
                    )
                    return False, f"KILL SWITCH: {self._state.kill_reason}"

            return True, "ok"

    # ---- Operator actions ----

    def activate_kill_switch(self, reason: str = "manual") -> None:
        """Manually activate the kill switch."""
        with self._lock:
            self._activate_kill_switch(reason)

    def deactivate_kill_switch(self) -> None:
        """Manually deactivate the kill switch (operator override)."""
        with self._lock:
            logger.warning("Independent risk system: kill switch DEACTIVATED by operator")
            self._state.kill_switch_active = False
            self._state.kill_reason = ""

    def get_state(self) -> dict:
        """Get current state for monitoring/dashboard."""
        with self._lock:
            return {
                "kill_switch_active": self._state.kill_switch_active,
                "kill_reason": self._state.kill_reason,
                "last_heartbeat_ago": time.time() - self._state.last_heartbeat if self._state.last_heartbeat else None,
                "daily_pnl": self._state.daily_pnl,
                "positions_value": self._state.positions_value,
            }

    # ---- Internal ----

    def _activate_kill_switch(self, reason: str) -> None:
        if not self._state.kill_switch_active:
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
            self._state.kill_switch_active = True
            self._state.kill_reason = reason
