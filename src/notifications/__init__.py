"""
Notifications Module
Discord and Telegram notification services.
"""
from .discord import DiscordNotifier

# Conditionally import Telegram
try:
    from .telegram import TelegramNotifier
except ImportError:
    TelegramNotifier = None

__all__ = [
    "DiscordNotifier",
    "TelegramNotifier",
]
