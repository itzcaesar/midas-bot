"""REQ-P1-09: DB ↔ MT5 reconciliation on startup."""
from __future__ import annotations

from datetime import datetime
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Fakes — kept tiny on purpose so the test reads top-down.
# ---------------------------------------------------------------------------


class FakeTrade:
    """Mirrors the SQLAlchemy `Trade` shape that `get_open_trades()` returns."""

    def __init__(self, ticket: int, status: str = "OPEN", profit: float = 0.0):
        self.ticket = ticket
        self.status = status
        self.profit = profit
        self.exit_price = None
        self.exit_time = None
        self.symbol = "XAUUSD"
        self.direction = "BUY"
        self.lot_size = 0.1
        self.entry_price = 1900.0
        self.stop_loss = 1890.0
        self.take_profit = 1920.0


class FakeDB:
    def __init__(self, open_trades: List[FakeTrade] | None = None):
        self.trades_by_ticket = {t.ticket: t for t in (open_trades or [])}
        self.exit_calls: List[dict] = []
        self.entry_calls: List[dict] = []

    def get_open_trades(self) -> List[FakeTrade]:
        return [t for t in self.trades_by_ticket.values() if t.status == "OPEN"]

    def record_trade_exit(self, ticket, exit_price, profit, **_):
        if ticket in self.trades_by_ticket:
            t = self.trades_by_ticket[ticket]
            t.exit_price = exit_price
            t.profit = profit
            t.status = "CLOSED"
        self.exit_calls.append(
            {"ticket": ticket, "exit_price": exit_price, "profit": profit}
        )
        return self.trades_by_ticket.get(ticket)

    def record_trade_entry(self, **kwargs):
        ticket = kwargs["ticket"]
        trade = FakeTrade(ticket)
        for k, v in kwargs.items():
            if hasattr(trade, k):
                setattr(trade, k, v)
        self.trades_by_ticket[ticket] = trade
        self.entry_calls.append(kwargs)
        return trade


class FakeBroker:
    def __init__(self, positions=None):
        self._positions = positions or []
        self._tick_price = 1925.0

    def get_bot_positions(self, symbol=None):
        return list(self._positions)

    def get_ohlcv(self, symbol, timeframe, bars=1):
        # Minimal: a one-row "DataFrame-like" wrapper so reconciler can read close
        import pandas as pd

        return pd.DataFrame(
            {"close": [self._tick_price]},
            index=pd.to_datetime([datetime.utcnow()]),
        )


def _pos(ticket, **overrides):
    base = {
        "ticket": ticket,
        "symbol": "XAUUSD",
        "type": "BUY",
        "volume": 0.1,
        "price_open": 1900.0,
        "price_current": 1925.0,
        "sl": 1890.0,
        "tp": 1920.0,
        "profit": 25.0,
        "swap": 0.0,
        "time": datetime.utcnow(),
        "magic": 123456,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_drift_clean_state() -> None:
    from reconciliation.sync import Reconciler

    db = FakeDB(open_trades=[FakeTrade(1001)])
    broker = FakeBroker(positions=[_pos(1001)])
    report = Reconciler(broker=broker, db=db).reconcile_on_startup(symbol="XAUUSD")
    assert report.in_sync_count == 1
    assert report.missing_in_db == []
    assert report.missing_in_broker == []
    assert report.dry_run_orphans == []
    assert db.exit_calls == []
    assert db.entry_calls == []


def test_orphan_in_broker_recreates_in_db() -> None:
    """Position open in broker, missing from DB → insert a synthetic DB row."""
    from reconciliation.sync import Reconciler

    db = FakeDB(open_trades=[])
    broker = FakeBroker(positions=[_pos(2001, profit=12.34)])
    report = Reconciler(broker=broker, db=db).reconcile_on_startup(symbol="XAUUSD")

    assert report.missing_in_db == [2001]
    assert len(db.entry_calls) == 1
    inserted = db.entry_calls[0]
    assert inserted["ticket"] == 2001
    assert inserted["direction"] == "BUY"
    assert "RECONCILED_RECREATED" in (inserted.get("factors") or [""])[0]


def test_missing_in_broker_closes_db_row() -> None:
    """DB has OPEN; broker shows nothing → close in DB and tag as reconciled."""
    from reconciliation.sync import Reconciler

    db = FakeDB(open_trades=[FakeTrade(3001)])
    broker = FakeBroker(positions=[])
    report = Reconciler(broker=broker, db=db).reconcile_on_startup(symbol="XAUUSD")

    assert report.missing_in_broker == [3001]
    assert len(db.exit_calls) == 1
    closed_call = db.exit_calls[0]
    assert closed_call["ticket"] == 3001
    assert closed_call["exit_price"] == pytest.approx(1925.0)
    # Trade got closed in DB
    assert db.trades_by_ticket[3001].status == "CLOSED"


def test_dry_run_orphans_are_ignored() -> None:
    """Negative tickets are dry-run sentinels and must not be touched."""
    from reconciliation.sync import Reconciler

    db = FakeDB(open_trades=[FakeTrade(-12345), FakeTrade(4001)])
    broker = FakeBroker(positions=[_pos(4001)])
    report = Reconciler(broker=broker, db=db).reconcile_on_startup(symbol="XAUUSD")

    assert -12345 in report.dry_run_orphans
    # Live ticket matches → in sync, no fixups
    assert 4001 not in report.missing_in_broker
    assert db.exit_calls == []  # we don't auto-close dry-run rows
    assert db.entry_calls == []


def test_dry_run_run_in_paper_does_not_recreate_negative_tickets() -> None:
    """Even when the broker reports nothing, a negative DB ticket isn't 'missing'."""
    from reconciliation.sync import Reconciler

    db = FakeDB(open_trades=[FakeTrade(-99)])
    broker = FakeBroker(positions=[])
    report = Reconciler(broker=broker, db=db).reconcile_on_startup(symbol="XAUUSD")

    assert report.missing_in_broker == []
    assert report.dry_run_orphans == [-99]
    assert db.exit_calls == []


def test_report_human_readable() -> None:
    from reconciliation.sync import Reconciler

    db = FakeDB(open_trades=[FakeTrade(5001), FakeTrade(5002)])
    broker = FakeBroker(positions=[_pos(5002), _pos(5003)])
    report = Reconciler(broker=broker, db=db).reconcile_on_startup(symbol="XAUUSD")

    assert report.missing_in_db == [5003]
    assert report.missing_in_broker == [5001]
    assert report.in_sync_count == 1
    text = str(report)
    assert "5003" in text and "5001" in text
