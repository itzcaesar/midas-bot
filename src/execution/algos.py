"""
Execution Algorithms (REQ-P3-05).

Instead of firing a single market order, execution algos split the order into
smaller child orders spread over time to reduce market impact and slippage.

Implemented:
  - TWAP (Time-Weighted Average Price): splits order into N equal slices
    executed at regular intervals.

Future:
  - VWAP: weight slices by historical volume profile.
  - Iceberg: show only a fraction of the order at a time.
  - Adaptive: adjust pace based on real-time spread/volume.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from core.logger import get_logger

logger = get_logger("midas.execution")


class BrokerOrderFn(Protocol):
    """Callable that places a single child order and returns a ticket."""
    def __call__(self, symbol: str, order_type: str, lot: float, **kwargs) -> Optional[int]: ...


@dataclass
class ChildOrder:
    """A single slice of a parent algo order."""
    ticket: Optional[int] = None
    lot: float = 0.0
    filled: bool = False
    fill_time: float = 0.0
    error: str = ""


@dataclass
class AlgoResult:
    """Result of an execution algo run."""
    parent_lot: float
    filled_lot: float = 0.0
    children: List[ChildOrder] = field(default_factory=list)
    avg_fill_time: float = 0.0
    success: bool = False

    @property
    def fill_rate(self) -> float:
        return self.filled_lot / self.parent_lot if self.parent_lot > 0 else 0.0


class ExecutionAlgo(ABC):
    """Base class for execution algorithms."""

    @abstractmethod
    def execute(
        self,
        symbol: str,
        direction: str,
        total_lot: float,
        place_order_fn: BrokerOrderFn,
        **kwargs,
    ) -> AlgoResult:
        """Execute the algo. Returns AlgoResult."""
        ...


class TWAPExecutor(ExecutionAlgo):
    """Time-Weighted Average Price execution.

    Splits the order into `n_slices` equal parts, each executed `interval_seconds`
    apart. If a child order fails, it retries once before moving on.

    Args:
        n_slices: Number of child orders (default 3).
        interval_seconds: Seconds between slices (default 10).
        retry_on_fail: Whether to retry a failed child once.
    """

    def __init__(
        self,
        n_slices: int = 3,
        interval_seconds: float = 10.0,
        retry_on_fail: bool = True,
    ) -> None:
        self.n_slices = max(1, n_slices)
        self.interval_seconds = interval_seconds
        self.retry_on_fail = retry_on_fail

    def execute(
        self,
        symbol: str,
        direction: str,
        total_lot: float,
        place_order_fn: BrokerOrderFn,
        **kwargs,
    ) -> AlgoResult:
        """Execute TWAP."""
        slice_lot = round(total_lot / self.n_slices, 2)
        # Ensure minimum lot
        if slice_lot < 0.01:
            slice_lot = 0.01
            self.n_slices = max(1, int(total_lot / slice_lot))

        result = AlgoResult(parent_lot=total_lot)
        logger.info(
            f"TWAP: {direction} {total_lot} lots of {symbol} "
            f"in {self.n_slices} slices of {slice_lot} every {self.interval_seconds}s"
        )

        for i in range(self.n_slices):
            # Last slice gets the remainder
            lot = slice_lot if i < self.n_slices - 1 else round(total_lot - result.filled_lot, 2)
            if lot <= 0:
                break

            child = self._place_child(symbol, direction, lot, place_order_fn, **kwargs)
            result.children.append(child)

            if child.filled:
                result.filled_lot += lot

            # Wait between slices (except after the last one)
            if i < self.n_slices - 1:
                time.sleep(self.interval_seconds)

        result.success = result.fill_rate >= 0.9
        if result.children:
            fill_times = [c.fill_time for c in result.children if c.filled]
            result.avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0.0

        logger.info(
            f"TWAP complete: filled {result.filled_lot}/{total_lot} lots "
            f"({result.fill_rate:.0%})"
        )
        return result

    def _place_child(
        self,
        symbol: str,
        direction: str,
        lot: float,
        place_order_fn: BrokerOrderFn,
        **kwargs,
    ) -> ChildOrder:
        """Place a single child order with optional retry."""
        child = ChildOrder(lot=lot)
        start = time.time()

        try:
            ticket = place_order_fn(symbol=symbol, order_type=direction, lot=lot, **kwargs)
            if ticket:
                child.ticket = ticket
                child.filled = True
                child.fill_time = time.time() - start
                return child
            child.error = "returned None"
        except Exception as e:
            child.error = str(e)

        # Retry once
        if self.retry_on_fail:
            time.sleep(1.0)
            try:
                ticket = place_order_fn(symbol=symbol, order_type=direction, lot=lot, **kwargs)
                if ticket:
                    child.ticket = ticket
                    child.filled = True
                    child.fill_time = time.time() - start
                    child.error = ""
                    return child
            except Exception as e:
                child.error = f"retry failed: {e}"

        logger.warning(f"TWAP child failed: {child.error}")
        return child
