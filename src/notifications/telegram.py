"""
Telegram Notifier (stub)

This module exists so `from notifications.telegram import TelegramNotifier` succeeds
on every install. It is intentionally minimal: when no bot token is configured,
all methods are safe no-ops; when a token + chat ID are present, messages are
sent via the simple Telegram Bot HTTP API (no third-party async dependency).

Failures never raise — they are logged and swallowed — because notification
failures must not stop the trading loop.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Optional

from core.logger import get_logger

logger = get_logger("mt5bot.telegram")

try:
    import requests  # used only when actually sending
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False


class TelegramNotifier:
    """Minimal Telegram bot wrapper.

    The notifier is enabled only when both a bot token and chat ID are present
    (via constructor args or the ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``
    environment variables) AND the ``requests`` package is importable.

    All public methods catch and log exceptions; they never raise.
    """

    API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if enabled is None:
            self._enabled = bool(self._token and self._chat_id and _REQUESTS_AVAILABLE)
        else:
            self._enabled = bool(enabled)

        if not _REQUESTS_AVAILABLE and (self._token or self._chat_id):
            logger.warning(
                "TelegramNotifier configured but `requests` is not installed; running disabled."
            )

    # ----- public API used by main.py ----------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def send_startup_message(self, balance: float) -> bool:
        return self._send(
            f"🚀 <b>MIDAS started</b>\nBalance: <b>${balance:,.2f}</b>\n{self._now()}"
        )

    def send_shutdown_message(self, reason: str = "Normal shutdown") -> bool:
        return self._send(f"🛑 <b>MIDAS stopped</b>\nReason: {reason}\n{self._now()}")

    def send_trade_opened(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        lot: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        emoji = "🟢" if direction == "BUY" else "🔴"
        text = (
            f"{emoji} <b>{direction} {symbol}</b>\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"Lot: {lot}\n"
            f"Entry: ${entry_price:.2f}\n"
            f"SL: ${stop_loss:.2f}\n"
            f"TP: ${take_profit:.2f}"
        )
        return self._send(text)

    def send_trade_closed(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        profit: float,
        reason: str = "",
    ) -> bool:
        emoji = "✅" if profit > 0 else "❌"
        text = (
            f"{emoji} <b>Closed {direction} {symbol}</b>\n"
            f"Ticket: <code>{ticket}</code>\n"
            f"Entry: ${entry_price:.2f}\n"
            f"Exit: ${exit_price:.2f}\n"
            f"P/L: <b>${profit:+.2f}</b>\n"
            f"Reason: {reason or 'manual'}"
        )
        return self._send(text)

    def send_error_alert(self, message: str, context: str = "") -> bool:
        text = f"⚠️ <b>Error</b>\n{context}\n<code>{_truncate(message, 500)}</code>"
        return self._send(text)

    def send_daily_summary(self, stats: dict) -> bool:
        lines: Iterable[str] = (
            "📊 <b>Daily summary</b>",
            f"Date: {self._now()}",
            *(f"{k}: <b>{v}</b>" for k, v in stats.items()),
        )
        return self._send("\n".join(lines))

    # ----- internals ---------------------------------------------------------

    def _send(self, text: str) -> bool:
        if not self._enabled:
            logger.debug("Telegram disabled, dropping message")
            return False
        try:
            response = requests.post(  # type: ignore[name-defined]
                self.API_BASE.format(token=self._token),
                data={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
                timeout=5,
            )
            if not response.ok:
                logger.warning(
                    f"Telegram send failed: {response.status_code} {response.text[:200]}"
                )
                return False
            return True
        except Exception as exc:  # noqa: BLE001 — never propagate notifier errors
            logger.warning(f"Telegram send error: {exc}")
            return False

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
