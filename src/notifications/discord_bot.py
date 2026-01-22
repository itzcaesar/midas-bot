"""
Discord Bot Integration
Send trading signals to Discord via webhook.
"""
import json
from datetime import datetime
from typing import Optional, Dict, List
import sys
sys.path.append('../..')

from core.logger import get_logger

logger = get_logger("mt5bot.discord")

# Import discord webhook
try:
    from discord_webhook import DiscordWebhook, DiscordEmbed
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("discord-webhook not installed")


class DiscordNotifier:
    """
    Send trading signals and alerts to Discord via webhook.
    
    Setup:
    1. Create a Discord webhook in your server (Server Settings > Integrations > Webhooks)
    2. Copy the webhook URL
    3. Set DISCORD_WEBHOOK_URL in your .env file
    """
    
    # Color scheme
    COLORS = {
        'buy': 0x00FF00,      # Green
        'sell': 0xFF0000,     # Red
        'hold': 0xFFFF00,     # Yellow
        'info': 0x3498DB,     # Blue
        'error': 0xE74C3C,    # Orange-red
        'success': 0x2ECC71,  # Light green
    }
    
    def __init__(self, webhook_url: str = None):
        """
        Initialize Discord notifier.
        
        Args:
            webhook_url: Discord webhook URL (or set DISCORD_WEBHOOK_URL env var)
        """
        import os
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self._enabled = DISCORD_AVAILABLE and bool(self.webhook_url)
        
        if not DISCORD_AVAILABLE:
            logger.warning("Discord webhook library not available")
        elif not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
        else:
            logger.info("Discord notifier initialized")
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def send_signal(
        self,
        direction: str,
        symbol: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        factors: List[str] = None,
        model_name: str = None,
        timeframe: str = None
    ) -> bool:
        """
        Send a trading signal to Discord.
        
        Args:
            direction: 'BUY', 'SELL', or 'HOLD'
            symbol: Trading symbol (e.g., 'XAUUSD')
            confidence: Signal confidence (0-1)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            factors: List of contributing factors
            model_name: ML model that generated the signal
            timeframe: Trading timeframe
            
        Returns:
            True if sent successfully
        """
        if not self.is_enabled:
            logger.debug(f"Discord disabled. Signal: {direction} {symbol}")
            return False
        
        try:
            webhook = DiscordWebhook(url=self.webhook_url, rate_limit_retry=True)
            
            # Determine color and emoji
            color = self.COLORS.get(direction.lower(), self.COLORS['info'])
            emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "🟡"
            
            # Create embed
            embed = DiscordEmbed(
                title=f"{emoji} {direction} Signal: {symbol}",
                description=f"**Confidence: {confidence:.0%}**",
                color=color
            )
            
            # Add fields
            embed.add_embed_field(name="📊 Entry Price", value=f"${entry_price:.2f}", inline=True)
            embed.add_embed_field(name="🛑 Stop Loss", value=f"${stop_loss:.2f}", inline=True)
            embed.add_embed_field(name="🎯 Take Profit", value=f"${take_profit:.2f}", inline=True)
            
            # Risk/Reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            embed.add_embed_field(name="📈 Risk/Reward", value=f"1:{rr_ratio:.1f}", inline=True)
            
            if timeframe:
                embed.add_embed_field(name="⏱️ Timeframe", value=timeframe, inline=True)
            
            if model_name:
                embed.add_embed_field(name="🤖 Model", value=model_name, inline=True)
            
            # Add factors
            if factors:
                factors_text = "\n".join([f"• {f}" for f in factors[:5]])  # Limit to 5
                embed.add_embed_field(name="🔍 Factors", value=factors_text, inline=False)
            
            # Timestamp
            embed.set_timestamp()
            embed.set_footer(text="MT5Bot ML Trading System")
            
            webhook.add_embed(embed)
            response = webhook.execute()
            
            logger.info(f"Discord signal sent: {direction} {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord signal: {e}")
            return False
    
    def send_daily_summary(
        self,
        stats: Dict,
        model_performance: Dict = None
    ) -> bool:
        """
        Send daily performance summary to Discord.
        
        Args:
            stats: Dictionary with daily statistics
            model_performance: Per-model performance metrics
            
        Returns:
            True if sent successfully
        """
        if not self.is_enabled:
            return False
        
        try:
            webhook = DiscordWebhook(url=self.webhook_url, rate_limit_retry=True)
            
            pnl = stats.get('pnl', 0)
            color = self.COLORS['success'] if pnl >= 0 else self.COLORS['error']
            emoji = "📈" if pnl >= 0 else "📉"
            
            embed = DiscordEmbed(
                title=f"{emoji} Daily Trading Summary",
                description=f"**Performance for {datetime.now().strftime('%Y-%m-%d')}**",
                color=color
            )
            
            embed.add_embed_field(name="💰 P/L", value=f"${pnl:+.2f}", inline=True)
            embed.add_embed_field(name="📊 Signals", value=str(stats.get('total_signals', 0)), inline=True)
            embed.add_embed_field(name="✅ Win Rate", value=f"{stats.get('win_rate', 0):.0%}", inline=True)
            embed.add_embed_field(name="📈 Best Trade", value=f"${stats.get('best_trade', 0):+.2f}", inline=True)
            embed.add_embed_field(name="📉 Worst Trade", value=f"${stats.get('worst_trade', 0):+.2f}", inline=True)
            embed.add_embed_field(name="💵 Balance", value=f"${stats.get('balance', 0):,.2f}", inline=True)
            
            if model_performance:
                perf_text = "\n".join([f"• {k}: {v:.0%}" for k, v in model_performance.items()])
                embed.add_embed_field(name="🤖 Model Accuracy", value=perf_text, inline=False)
            
            embed.set_timestamp()
            embed.set_footer(text="MT5Bot ML Trading System")
            
            webhook.add_embed(embed)
            webhook.execute()
            
            logger.info("Discord daily summary sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord summary: {e}")
            return False
    
    def send_alert(
        self,
        title: str,
        message: str,
        alert_type: str = 'info'
    ) -> bool:
        """
        Send a general alert to Discord.
        
        Args:
            title: Alert title
            message: Alert message
            alert_type: 'info', 'error', 'success'
            
        Returns:
            True if sent successfully
        """
        if not self.is_enabled:
            return False
        
        try:
            webhook = DiscordWebhook(url=self.webhook_url, rate_limit_retry=True)
            
            color = self.COLORS.get(alert_type, self.COLORS['info'])
            emoji_map = {'info': 'ℹ️', 'error': '⚠️', 'success': '✅'}
            emoji = emoji_map.get(alert_type, 'ℹ️')
            
            embed = DiscordEmbed(
                title=f"{emoji} {title}",
                description=message,
                color=color
            )
            
            embed.set_timestamp()
            embed.set_footer(text="MT5Bot ML Trading System")
            
            webhook.add_embed(embed)
            webhook.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def send_model_update(
        self,
        model_name: str,
        metrics: Dict,
        is_training: bool = False
    ) -> bool:
        """
        Send model training/update notification.
        
        Args:
            model_name: Name of the ML model
            metrics: Model performance metrics
            is_training: True if model is training, False if completed
            
        Returns:
            True if sent successfully
        """
        if not self.is_enabled:
            return False
        
        try:
            webhook = DiscordWebhook(url=self.webhook_url, rate_limit_retry=True)
            
            if is_training:
                title = f"🔄 Training: {model_name}"
                color = self.COLORS['info']
            else:
                title = f"✅ Model Updated: {model_name}"
                color = self.COLORS['success']
            
            embed = DiscordEmbed(title=title, color=color)
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    embed.add_embed_field(name=key.replace('_', ' ').title(), value=f"{value:.2%}", inline=True)
                else:
                    embed.add_embed_field(name=key.replace('_', ' ').title(), value=str(value), inline=True)
            
            embed.set_timestamp()
            embed.set_footer(text="MT5Bot ML Trading System")
            
            webhook.add_embed(embed)
            webhook.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord model update: {e}")
            return False


def send_discord_signal(
    direction: str,
    confidence: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    webhook_url: str = None
) -> bool:
    """
    Convenience function to send a quick signal.
    
    Args:
        direction: 'BUY' or 'SELL'
        confidence: Signal confidence
        entry_price: Entry price
        stop_loss: Stop loss
        take_profit: Take profit
        webhook_url: Discord webhook URL
        
    Returns:
        True if sent successfully
    """
    notifier = DiscordNotifier(webhook_url)
    return notifier.send_signal(
        direction=direction,
        symbol="XAUUSD",
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
