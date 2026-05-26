"""
Pre-trade filter helper (REQ-P0-10).

Centralizes session-window and economic-news guards so every signal source
(rule-based and ML) honors the same pre-trade rules. The legacy `XAUUSDStrategy`
already runs these checks; this helper extends the same protection to the ML
pipelines (`MLSignalGenerator`, `StyledSignalGenerator`, `SignalGenerator`)
which previously bypassed all filters.

Filters short-circuit on `settings.session_filter_enabled` and
`settings.news_filter_enabled` so backtests can disable them deterministically.
"""
from __future__ import annotations

from typing import Tuple

from config import settings
from core.logger import get_logger

logger = get_logger("mt5bot.pre_trade")


def pre_trade_filters_passed(
    session_filter=None,
    news_filter=None,
) -> Tuple[bool, str]:
    """Return ``(passed, reason)`` for the current pre-trade gate.

    Args:
        session_filter: Optional shared ``SessionFilter`` instance. A new one is
            constructed lazily if ``None`` is passed and session filtering is
            enabled.
        news_filter: Optional shared ``NewsFilter`` instance, constructed lazily
            when needed.

    Returns:
        ``(True, "filters passed")`` when a trade may proceed,
        ``(False, "<filter>: <reason>")`` otherwise.

    Notes:
        Failures inside the filter implementations are caught and treated as
        "filter unavailable, allow trade" rather than blocking, because the
        opposite is more dangerous (silent permanent halt). Failures are
        logged so they can be investigated.
    """
    # Session filter ----------------------------------------------------------
    if getattr(settings, "session_filter_enabled", False):
        try:
            if session_filter is None:
                from analysis.session_filter import SessionFilter

                session_filter = SessionFilter()
            ok, reason = session_filter.is_valid_session()
            if not ok:
                return False, f"Session filter: {reason}"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Session filter unavailable, allowing trade: {exc}")

    # News filter -------------------------------------------------------------
    if getattr(settings, "news_filter_enabled", False):
        try:
            if news_filter is None:
                from analysis.news_filter import NewsFilter

                news_filter = NewsFilter()
            ok, reason = news_filter.is_safe_to_trade()
            if not ok:
                return False, f"News filter: {reason}"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"News filter unavailable, allowing trade: {exc}")

    return True, "filters passed"
