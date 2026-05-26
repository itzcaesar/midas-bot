"""
Tick Data Ingestion + Microstructure Features (REQ-P3-01).

Provides infrastructure for consuming real-time tick data from MT5 and
computing order-flow / microstructure features that are unavailable from
bar-level OHLCV data:

  - Tick arrival rate (trades per second)
  - Bid-ask spread (live)
  - Trade flow imbalance (buy vs sell volume)
  - Volume-weighted average price (VWAP)
  - Price impact per unit volume

These features are most useful for scalping (5m) and intraday (15m) styles.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger("midas.tick_data")


@dataclass
class Tick:
    """Single market tick."""
    timestamp: float  # Unix epoch seconds
    bid: float
    ask: float
    last: float
    volume: float
    flags: int = 0  # MT5 tick flags (buy/sell indicator)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def is_buy(self) -> bool:
        """Heuristic: last >= ask means buyer-initiated."""
        return self.last >= self.ask

    @property
    def is_sell(self) -> bool:
        return self.last <= self.bid


class TickBuffer:
    """Rolling buffer of recent ticks with microstructure feature computation."""

    def __init__(self, max_ticks: int = 10_000, window_seconds: float = 300.0):
        self._buffer: Deque[Tick] = deque(maxlen=max_ticks)
        self._window = window_seconds

    def add(self, tick: Tick) -> None:
        self._buffer.append(tick)

    def add_raw(self, timestamp: float, bid: float, ask: float, last: float, volume: float, flags: int = 0) -> None:
        self._buffer.append(Tick(timestamp, bid, ask, last, volume, flags))

    @property
    def count(self) -> int:
        return len(self._buffer)

    def _recent(self, seconds: Optional[float] = None) -> List[Tick]:
        """Get ticks within the last N seconds."""
        cutoff = time.time() - (seconds or self._window)
        return [t for t in self._buffer if t.timestamp >= cutoff]

    def compute_features(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Compute microstructure features from recent ticks.

        Returns dict with keys:
          tick_rate: ticks per second
          avg_spread: average bid-ask spread
          spread_volatility: std of spread
          buy_volume_ratio: fraction of volume that's buyer-initiated
          vwap: volume-weighted average price
          price_impact: price change per unit volume (Kyle's lambda proxy)
          tick_imbalance: (buy_ticks - sell_ticks) / total_ticks
        """
        ticks = self._recent(window_seconds)
        if len(ticks) < 10:
            return self._empty_features()

        duration = ticks[-1].timestamp - ticks[0].timestamp
        if duration <= 0:
            duration = 1.0

        spreads = [t.spread for t in ticks]
        volumes = [t.volume for t in ticks]
        prices = [t.last for t in ticks]
        buy_vol = sum(t.volume for t in ticks if t.is_buy)
        sell_vol = sum(t.volume for t in ticks if t.is_sell)
        total_vol = sum(volumes) or 1.0

        # VWAP
        vwap = sum(t.last * t.volume for t in ticks) / total_vol if total_vol > 0 else prices[-1]

        # Price impact (Kyle's lambda approximation)
        if len(prices) > 1 and total_vol > 0:
            price_change = abs(prices[-1] - prices[0])
            price_impact = price_change / total_vol
        else:
            price_impact = 0.0

        # Tick imbalance
        buy_ticks = sum(1 for t in ticks if t.is_buy)
        sell_ticks = sum(1 for t in ticks if t.is_sell)
        total_ticks = buy_ticks + sell_ticks or 1

        return {
            "tick_rate": len(ticks) / duration,
            "avg_spread": float(np.mean(spreads)),
            "spread_volatility": float(np.std(spreads)),
            "buy_volume_ratio": buy_vol / total_vol,
            "vwap": vwap,
            "price_impact": price_impact,
            "tick_imbalance": (buy_ticks - sell_ticks) / total_ticks,
        }

    @staticmethod
    def _empty_features() -> Dict[str, float]:
        return {
            "tick_rate": 0.0,
            "avg_spread": 0.0,
            "spread_volatility": 0.0,
            "buy_volume_ratio": 0.5,
            "vwap": 0.0,
            "price_impact": 0.0,
            "tick_imbalance": 0.0,
        }


class TickCollector:
    """Collects ticks from MT5 in a background-friendly manner.

    In production, this would run in a separate thread/process and push ticks
    into the buffer. For now it provides a `poll()` method that the main loop
    can call each cycle.
    """

    def __init__(self, symbol: str = "XAUUSD", buffer_size: int = 50_000):
        self.symbol = symbol
        self.buffer = TickBuffer(max_ticks=buffer_size)
        self._last_poll_time: float = 0.0

    def poll(self, mt5_module=None) -> int:
        """Fetch new ticks from MT5 since last poll.

        Args:
            mt5_module: The MetaTrader5 module (or mock). If None, does nothing.

        Returns:
            Number of new ticks added.
        """
        if mt5_module is None:
            return 0

        try:
            from_time = self._last_poll_time or (time.time() - 60)
            ticks = mt5_module.copy_ticks_from(
                self.symbol,
                datetime.fromtimestamp(from_time, tz=timezone.utc),
                1000,
                mt5_module.COPY_TICKS_ALL,
            )
            if ticks is None or len(ticks) == 0:
                return 0

            count = 0
            for t in ticks:
                self.buffer.add_raw(
                    timestamp=float(t['time']),
                    bid=float(t['bid']),
                    ask=float(t['ask']),
                    last=float(t.get('last', (t['bid'] + t['ask']) / 2)),
                    volume=float(t.get('volume', 0)),
                    flags=int(t.get('flags', 0)),
                )
                count += 1

            if count > 0:
                self._last_poll_time = float(ticks[-1]['time'])

            return count
        except Exception as e:
            logger.debug(f"Tick poll failed: {e}")
            return 0
