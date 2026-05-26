"""
Signal Source Protocol (REQ-P1-01).

Defines the interface that all signal generators must implement. The backtester,
live executor, and Discord notifier all consume this protocol — ensuring the
system you test is the system that trades.
"""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd


class SignalLike(Protocol):
    """Minimal signal shape consumed by the executor and backtester."""

    direction: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    stop_loss: float
    take_profit: float
    reasons: list


class SignalSourceProtocol(Protocol):
    """Any strategy or ML generator that can produce a signal from market data."""

    def analyze(
        self,
        xau_df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame] = None,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> SignalLike:
        """Generate a trading signal from the provided market data."""
        ...
