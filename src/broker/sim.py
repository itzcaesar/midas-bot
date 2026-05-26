"""
Simulated Broker (REQ-P1-02).

A lightweight in-memory broker that implements BrokerProtocol for testing and
backtesting without requiring MetaTrader5. Useful for:
  - Unit tests that need a broker without MT5 installed
  - Offline dry-run mode
  - Integration tests of the full trading loop
"""
from __future__ import annotations

import time
from itertools import count
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from core.logger import get_logger

logger = get_logger("midas.sim_broker")

# Start ticket counter from a time-based offset so restarts don't collide
# with tickets already in the DB from previous runs.
_TICKET_SEQ = count(start=int(time.time()) % 10_000_000 * 100)


class SimBroker:
    """In-memory simulated broker implementing BrokerProtocol."""

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        data: Optional[pd.DataFrame] = None,
        start_cursor: int = 500,
    ) -> None:
        self._balance = initial_balance
        self._equity = initial_balance
        self._positions: Dict[int, dict] = {}
        self._data = data  # optional pre-loaded OHLCV for get_ohlcv
        self._connected = False
        # Cursor advances each get_ohlcv call so the strategy sees new bars
        # each cycle, simulating real-time data flow through history.
        self._cursor = start_cursor  # start after enough warmup bars

    # ---- connection ----

    def connect(self) -> bool:
        self._connected = True
        logger.info("SimBroker connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("SimBroker disconnected")

    # ---- data ----

    def get_ohlcv(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        if self._data is None:
            return None
        # Advance cursor by 1 each call so each cycle sees the next bar
        self._cursor = min(self._cursor + 1, len(self._data))
        slice_end = self._cursor
        slice_start = max(0, slice_end - bars)
        result = self._data.iloc[slice_start:slice_end].copy()
        if len(result) < 50:
            return None
        # Update open position prices to the latest close
        current_price = float(result['close'].iloc[-1])
        for ticket, pos in self._positions.items():
            pos['price_current'] = current_price
            # Compute unrealized profit (simplified: 1 pip = $0.10 per 0.01 lot)
            direction = pos['type']
            lot = pos['volume']
            entry = pos['price_open']
            if entry > 0:
                if direction == 'BUY':
                    pos['profit'] = round((current_price - entry) * 10 * lot * 10, 2)
                else:
                    pos['profit'] = round((entry - current_price) * 10 * lot * 10, 2)
        return result

    # ---- account ----

    def get_account_balance(self) -> float:
        return self._balance

    def get_account_equity(self) -> float:
        return self._equity

    def set_equity(self, equity: float) -> None:
        """Test helper: set equity externally."""
        self._equity = equity

    # ---- sizing ----

    def calculate_lot_size(self, symbol: str, stop_loss_pips: float) -> float:
        risk_amount = self._balance * (settings.risk_percent / 100)
        if stop_loss_pips <= 0:
            return 0.01
        lot = risk_amount / (stop_loss_pips * 10)
        return max(0.01, min(round(lot, 2), 10.0))

    # ---- orders ----

    def place_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        sl_pips: float | None = None,
        tp_pips: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        comment: str = "SimBot",
    ) -> Optional[int]:
        ticket = next(_TICKET_SEQ)
        # Use the current bar's close as the entry price
        current_price = 0.0
        if self._data is not None and self._cursor > 0:
            current_price = float(self._data.iloc[min(self._cursor - 1, len(self._data) - 1)]['close'])
        self._positions[ticket] = {
            "ticket": ticket,
            "symbol": symbol,
            "type": order_type,
            "volume": lot,
            "price_open": current_price,
            "price_current": current_price,
            "sl": sl_price or 0.0,
            "tp": tp_price or 0.0,
            "profit": 0.0,
            "swap": 0.0,
            "time": time.time(),
            "magic": 123456,
        }
        logger.info(f"SimBroker: placed {order_type} {lot} lots {symbol} @ {current_price:.2f} ticket={ticket}")
        return ticket

    def get_bot_positions(self, symbol: str | None = None) -> List[dict]:
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        return positions

    def get_open_positions(self, symbol: str | None = None) -> List[dict]:
        return self.get_bot_positions(symbol)

    def close_position(self, ticket: int) -> bool:
        if ticket in self._positions:
            pos = self._positions.pop(ticket)
            self._balance += pos.get("profit", 0.0)
            logger.info(f"SimBroker: closed ticket={ticket}")
            return True
        return False

    def modify_position(self, ticket: int, sl: float | None = None, tp: float | None = None) -> bool:
        if ticket not in self._positions:
            return False
        if sl is not None:
            self._positions[ticket]["sl"] = sl
        if tp is not None:
            self._positions[ticket]["tp"] = tp
        return True

    def update_trailing_stop(self, ticket: int, trailing_pips: float | None = None) -> bool:
        # Simplified: no-op in sim
        return False
