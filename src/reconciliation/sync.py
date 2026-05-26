"""
Startup DB ↔ MT5 reconciliation (REQ-P1-09).

If the bot crashes mid-trade, the persisted trade ledger and the broker's open
positions can drift apart. This module brings them back into sync.

Three drift cases:

  1. **In sync** — same ticket open in both. No action.
  2. **Missing in DB** — broker shows an open position with our magic number,
     but the DB has no OPEN row. We insert a synthetic entry row so that
     subsequent close events are tracked correctly. The ``factors`` field is
     tagged ``RECONCILED_RECREATED`` so analytics can spot it.
  3. **Missing in broker** — DB has an OPEN row, but the broker no longer
     shows the position. We close the DB row using the latest market close as
     the exit price and tag it ``RECONCILED_CLOSED``. The exit price is best-
     effort — operators should review.

Negative tickets (dry-run sentinels created by ``broker.mt5._make_dry_run_ticket``)
are *never* touched: they are not real broker tickets, so "missing in broker"
is the only state they can be in. They are reported as ``dry_run_orphans`` so
the operator sees the count, but no DB writes happen.

The reconciler is intentionally conservative: it never *closes* a real broker
position, never opens a new one, and never modifies SL/TP. Any real-money
adjustment must be an explicit operator action.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Protocol

logger = logging.getLogger("midas.reconciliation")


# ---------------------------------------------------------------------------
# Protocols — kept loose so the reconciler can be unit-tested without MT5.
# ---------------------------------------------------------------------------


class _BrokerProtocol(Protocol):
    def get_bot_positions(self, symbol: Optional[str] = ...) -> List[dict]: ...
    def get_ohlcv(self, symbol: str, timeframe: str, bars: int = ...) -> Any: ...


class _DBProtocol(Protocol):
    def get_open_trades(self) -> List[Any]: ...
    def record_trade_exit(self, ticket: int, exit_price: float, profit: float, **_) -> Any: ...
    def record_trade_entry(self, **kwargs) -> Any: ...


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationReport:
    """Structured outcome of a reconciliation run."""

    in_sync_count: int = 0
    missing_in_db: List[int] = field(default_factory=list)
    missing_in_broker: List[int] = field(default_factory=list)
    dry_run_orphans: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            "Reconciliation report: "
            f"{self.in_sync_count} in sync, "
            f"missing_in_db={self.missing_in_db}, "
            f"missing_in_broker={self.missing_in_broker}, "
            f"dry_run_orphans={self.dry_run_orphans}, "
            f"errors={self.errors}"
        )

    @property
    def has_drift(self) -> bool:
        return bool(self.missing_in_db or self.missing_in_broker)


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class Reconciler:
    """One-shot DB ↔ broker reconciler. See module docstring."""

    def __init__(self, broker: _BrokerProtocol, db: _DBProtocol) -> None:
        self.broker = broker
        self.db = db

    def reconcile_on_startup(
        self,
        symbol: str,
        timeframe: str = "M1",
        log_to: Optional[logging.Logger] = None,
    ) -> ReconciliationReport:
        """Compare DB open trades with broker open positions and patch the DB."""
        log = log_to or logger
        report = ReconciliationReport()

        # ----- pull both sides -----
        try:
            broker_positions = self.broker.get_bot_positions(symbol)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"broker.get_bot_positions failed: {exc}")
            log.error(f"Reconciliation: broker query failed: {exc}")
            return report

        try:
            db_open = self.db.get_open_trades()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"db.get_open_trades failed: {exc}")
            log.error(f"Reconciliation: DB query failed: {exc}")
            return report

        broker_tickets = {p["ticket"]: p for p in broker_positions}
        db_tickets = {t.ticket: t for t in db_open}

        # ----- partition -----
        for ticket, db_trade in db_tickets.items():
            if ticket < 0:
                # dry-run sentinel — never reconcile against broker
                report.dry_run_orphans.append(ticket)
                continue
            if ticket in broker_tickets:
                report.in_sync_count += 1
            else:
                report.missing_in_broker.append(ticket)
                self._close_orphaned_db_row(ticket, db_trade, symbol, timeframe, report, log)

        for ticket, position in broker_tickets.items():
            if ticket not in db_tickets:
                report.missing_in_db.append(ticket)
                self._recreate_db_row(ticket, position, report, log)

        log.info(str(report))
        return report

    # ------------------------------------------------------------- helpers

    def _close_orphaned_db_row(
        self,
        ticket: int,
        db_trade: Any,
        symbol: str,
        timeframe: str,
        report: ReconciliationReport,
        log: logging.Logger,
    ) -> None:
        """Mark a DB-only OPEN row as CLOSED at the latest market price."""
        try:
            ohlcv = self.broker.get_ohlcv(symbol, timeframe, bars=1)
            exit_price = float(ohlcv["close"].iloc[-1]) if ohlcv is not None else 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning(
                f"Reconciliation: couldn't fetch latest price for {symbol}, using 0.0: {exc}"
            )
            exit_price = 0.0

        # Best-effort PnL: lots * (exit - entry) * pip_value
        # We don't have a generic pip-value helper here, so only set price/zero PnL.
        # Operator can correct this from broker history.
        try:
            self.db.record_trade_exit(
                ticket=ticket,
                exit_price=exit_price,
                profit=getattr(db_trade, "profit", 0.0) or 0.0,
            )
            log.warning(
                f"Reconciliation: closed orphan DB ticket {ticket} at {exit_price:.2f} "
                "(broker had no matching position)"
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"close orphan {ticket} failed: {exc}")
            log.error(f"Reconciliation: failed to close orphan {ticket}: {exc}")

    def _recreate_db_row(
        self,
        ticket: int,
        position: dict,
        report: ReconciliationReport,
        log: logging.Logger,
    ) -> None:
        """Insert a synthetic OPEN row for a broker position that's missing in the DB."""
        try:
            self.db.record_trade_entry(
                ticket=ticket,
                symbol=position.get("symbol", "UNKNOWN"),
                direction=position.get("type", "BUY"),
                lot_size=position.get("volume", 0.0),
                entry_price=position.get("price_open", 0.0),
                stop_loss=position.get("sl", 0.0),
                take_profit=position.get("tp", 0.0),
                factors=[
                    f"RECONCILED_RECREATED at {datetime.utcnow().isoformat()}Z; "
                    "broker showed open position with no DB row"
                ],
                confidence=0.0,
            )
            log.warning(
                f"Reconciliation: recreated DB row for orphan broker ticket {ticket}"
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"recreate ticket {ticket} failed: {exc}")
            log.error(f"Reconciliation: failed to recreate {ticket}: {exc}")
