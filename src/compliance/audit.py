"""
Compliance & Audit Trail (REQ-P3-08).

Every trading decision must be reproducible from:
  (model_version, data_hash, features_snapshot, code_sha)

This module provides:
  - Decision logging: every signal, order, and rejection is recorded with full context
  - Immutable audit log (append-only JSONL file)
  - Reproducibility metadata attached to every trade
  - Query interface for compliance review
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import get_logger

logger = get_logger("midas.audit")


def _get_git_sha() -> str:
    """Get current git commit SHA (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _hash_array(data) -> str:
    """SHA256 hash of a numpy array or bytes."""
    import numpy as np
    if hasattr(data, 'tobytes'):
        raw = data.tobytes()
    elif isinstance(data, bytes):
        raw = data
    else:
        raw = str(data).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


class AuditEntry:
    """A single audit log entry."""

    def __init__(
        self,
        event_type: str,
        details: Dict[str, Any],
        model_version: str = "",
        data_hash: str = "",
        code_sha: str = "",
    ) -> None:
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.event_type = event_type
        self.details = details
        self.model_version = model_version
        self.data_hash = data_hash
        self.code_sha = code_sha or _get_git_sha()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "model_version": self.model_version,
            "data_hash": self.data_hash,
            "code_sha": self.code_sha,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditTrail:
    """Append-only audit log for compliance.

    Writes to a JSONL file (one JSON object per line). Each entry contains
    enough metadata to reproduce the decision.
    """

    def __init__(self, log_dir: str = "logs/audit") -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        self._code_sha = _get_git_sha()

    def log_signal(
        self,
        direction: str,
        confidence: float,
        factors: List[str],
        model_version: str = "",
        data_hash: str = "",
        extra: Optional[Dict] = None,
    ) -> None:
        """Log a signal generation event."""
        details = {
            "direction": direction,
            "confidence": confidence,
            "factors": factors,
            **(extra or {}),
        }
        self._write(AuditEntry("SIGNAL", details, model_version, data_hash, self._code_sha))

    def log_order(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        lot: float,
        entry_price: float,
        sl: float,
        tp: float,
        model_version: str = "",
        extra: Optional[Dict] = None,
    ) -> None:
        """Log an order placement."""
        details = {
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "lot": lot,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            **(extra or {}),
        }
        self._write(AuditEntry("ORDER", details, model_version, code_sha=self._code_sha))

    def log_rejection(
        self,
        reason: str,
        source: str = "",
        extra: Optional[Dict] = None,
    ) -> None:
        """Log an order rejection (by governor, filter, or risk system)."""
        details = {"reason": reason, "source": source, **(extra or {})}
        self._write(AuditEntry("REJECTION", details, code_sha=self._code_sha))

    def log_close(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        reason: str = "",
    ) -> None:
        """Log a position close."""
        details = {"ticket": ticket, "exit_price": exit_price, "profit": profit, "reason": reason}
        self._write(AuditEntry("CLOSE", details, code_sha=self._code_sha))

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a generic event."""
        self._write(AuditEntry(event_type, details, code_sha=self._code_sha))

    def _write(self, entry: AuditEntry) -> None:
        try:
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            logger.error(f"Audit write failed: {e}")

    def query(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Read recent audit entries (for dashboard/review)."""
        entries = []
        try:
            # Read all JSONL files in reverse chronological order
            files = sorted(self._dir.glob("audit_*.jsonl"), reverse=True)
            for f in files:
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        entry = json.loads(line.strip())
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        entries.append(entry)
                        if len(entries) >= limit:
                            return entries
        except Exception as e:
            logger.error(f"Audit query failed: {e}")
        return entries
