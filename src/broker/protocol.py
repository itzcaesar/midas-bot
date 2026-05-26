"""
Broker Protocol (REQ-P1-02).

Defines the interface that both the live MT5 broker and the simulated broker
must implement. This allows the backtester, live executor, and tests to share
the same contract without coupling to MetaTrader5.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BrokerProtocol(Protocol):
    """Minimal broker interface consumed by the trading loop and backtester."""

    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_ohlcv(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]: ...
    def get_account_balance(self) -> float: ...
    def get_account_equity(self) -> float: ...
    def calculate_lot_size(self, symbol: str, stop_loss_pips: float) -> float: ...
    def place_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        sl_pips: float | None = None,
        tp_pips: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        comment: str = "MT5Bot",
    ) -> Optional[int]: ...
    def get_bot_positions(self, symbol: str | None = None) -> List[dict]: ...
    def close_position(self, ticket: int) -> bool: ...
    def modify_position(self, ticket: int, sl: float | None = None, tp: float | None = None) -> bool: ...
    def update_trailing_stop(self, ticket: int, trailing_pips: float | None = None) -> bool: ...
