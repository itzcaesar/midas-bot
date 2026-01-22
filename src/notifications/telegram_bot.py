"""
Telegram Bot Notifications
Send trade alerts, signals, and daily summaries via Telegram.
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, List



from config import settings
from core.logger import get_logger

logger = get_logger("mt5bot.telegram")

# Try to import telegram
try:
    from telegram import Bot
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram notifications disabled.")


class TelegramNotifier:
    """
    Send trading notifications via Telegram.
    
    Setup:
    1. Create a bot via @BotFather on Telegram
    2. Get your bot token
    3. Start a chat with your bot and get the chat ID
    4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
    """
    
    def __init__(self, token: str = None, chat_id: str = None):
        """
        Initialize Telegram notifier.
        
        Args:
            token: Telegram bot token (or uses settings.telegram_bot_token)
            chat_id: Telegram chat ID (or uses settings.telegram_chat_id)
        """
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.bot = None
        self._enabled = False
        
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram library not available")
            return
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return
        
        try:
            self.bot = Bot(token=self.token)
            self._enabled = True
            logger.info("Telegram notifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.bot is not None
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                return asyncio.ensure_future(coro)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(coro)
    
    async def _send_message(self, message: str, parse_mode: str = None) -> bool:
        """Send a message via Telegram."""
        if not self.is_enabled:
            logger.debug(f"Telegram disabled, would send: {message[:100]}...")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode or ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_message(self, message: str) -> bool:
        """Send a plain text message (sync wrapper)."""
        return self._run_async(self._send_message(message))
    
    def send_signal_alert(self, direction: str, symbol: str, confidence: float,
                           stop_loss: float, take_profit: float, 
                           factors: List[str], executed: bool = False) -> bool:
        """
        Send a trading signal alert.
        
        Args:
            direction: 'BUY' or 'SELL'
            symbol: Trading symbol
            confidence: Signal confidence (0-1)
            stop_loss: Stop loss price
            take_profit: Take profit price
            factors: List of signal factors
            executed: Whether the trade was executed
        """
        emoji = "🟢" if direction == "BUY" else "🔴"
        status = "✅ EXECUTED" if executed else "📊 SIGNAL"
        
        message = f"""
{emoji} *{status}: {direction} {symbol}*

📊 Confidence: {confidence:.0%}
🎯 Take Profit: ${take_profit:.2f}
🛑 Stop Loss: ${stop_loss:.2f}

*Factors:*
"""
        for factor in factors:
            message += f"• {factor}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self._run_async(self._send_message(message))
    
    def send_trade_opened(self, ticket: int, symbol: str, direction: str,
                          lot_size: float, entry_price: float,
                          stop_loss: float, take_profit: float) -> bool:
        """Send trade opened notification."""
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        message = f"""
{emoji} *TRADE OPENED*

🎫 Ticket: #{ticket}
💱 {symbol} - {direction}
📦 Lot Size: {lot_size}
💰 Entry: ${entry_price:.2f}
🛑 SL: ${stop_loss:.2f}
🎯 TP: ${take_profit:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self._run_async(self._send_message(message))
    
    def send_trade_closed(self, ticket: int, symbol: str, direction: str,
                          entry_price: float, exit_price: float,
                          profit: float, reason: str) -> bool:
        """Send trade closed notification."""
        emoji = "✅" if profit > 0 else "❌"
        pnl_emoji = "💰" if profit > 0 else "📉"
        
        message = f"""
{emoji} *TRADE CLOSED*

🎫 Ticket: #{ticket}
💱 {symbol} - {direction}
💰 Entry: ${entry_price:.2f}
🏁 Exit: ${exit_price:.2f}
{pnl_emoji} P/L: *${profit:+.2f}*
📝 Reason: {reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self._run_async(self._send_message(message))
    
    def send_daily_summary(self, stats: Dict) -> bool:
        """
        Send daily trading summary.
        
        Args:
            stats: Dictionary with daily statistics
        """
        pnl = stats.get('pnl', 0)
        emoji = "📈" if pnl >= 0 else "📉"
        
        message = f"""
{emoji} *DAILY TRADING SUMMARY*

💰 P/L: *${pnl:+.2f}*
📊 Trades: {stats.get('trades', 0)}
✅ Win Rate: {stats.get('win_rate', 0):.0%}
📉 Max Drawdown: {stats.get('max_dd', 0):.1%}
💵 Balance: ${stats.get('balance', 0):,.2f}
📈 Equity: ${stats.get('equity', 0):,.2f}

📅 {datetime.now().strftime('%Y-%m-%d')}
"""
        return self._run_async(self._send_message(message))
    
    def send_error_alert(self, error: str, context: str = None) -> bool:
        """Send error notification."""
        message = f"""
⚠️ *BOT ERROR*

❌ *Error:* {error}
"""
        if context:
            message += f"📝 *Context:* {context}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self._run_async(self._send_message(message))
    
    def send_startup_message(self, balance: float, settings_info: Dict = None) -> bool:
        """Send bot startup notification."""
        message = f"""
🚀 *MT5 BOT STARTED*

💵 Balance: ${balance:,.2f}
📊 Symbol: {settings.symbol}
⏱️ Timeframe: {settings.timeframe}
💰 Risk: {settings.risk_percent}%
🧪 Dry Run: {'Yes' if settings.dry_run else 'No ⚠️'}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self._run_async(self._send_message(message))
    
    def send_shutdown_message(self, reason: str = "Normal shutdown") -> bool:
        """Send bot shutdown notification."""
        message = f"""
🛑 *MT5 BOT STOPPED*

📝 Reason: {reason}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self._run_async(self._send_message(message))


# Convenience function for quick notifications
def send_notification(message: str) -> bool:
    """
    Quick function to send a Telegram notification.
    Uses default settings from .env file.
    """
    notifier = TelegramNotifier()
    return notifier.send_message(message)
