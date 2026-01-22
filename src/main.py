"""
XAUUSD Trading Bot - Main Entry Point
Enhanced with logging, notifications, database persistence, and advanced filters.
"""
import time
import sys
from datetime import datetime
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from config import settings
from core.logger import setup_logger, TradeLogger
from core.database import Database
from core.exceptions import ConnectionError, OrderError
from broker.mt5_manager import MT5Manager
from strategy.xauusd_strategy import XAUUSDStrategy
from analysis import dxy
from notifications.telegram_bot import TelegramNotifier

# Setup main logger
logger = setup_logger("mt5bot.main", log_dir=settings.log_dir)
trade_logger = TradeLogger(settings.log_dir)


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║               🏆 XAUUSD TRADING BOT v2.0 🏆                  ║
║    Strategy: Liquidity + MACD + Breakout + Structure + DXY   ║
║              + HTF Bias + Session Filter + News              ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol: {symbol:<10}  Timeframe: {tf:<8}  Risk: {risk}%         ║
║  Dry Run: {dry:<8}   Session Filter: {session:<8}              ║
║  HTF Enabled: {htf:<6}  News Filter: {news:<8}                 ║
╚══════════════════════════════════════════════════════════════╝
""".format(
        symbol=settings.symbol,
        tf=settings.timeframe,
        risk=settings.risk_percent,
        dry="YES" if settings.dry_run else "NO ⚠️",
        session="ON" if settings.session_filter_enabled else "OFF",
        htf="ON" if settings.htf_enabled else "OFF",
        news="ON" if settings.news_filter_enabled else "OFF"
    )
    print(banner)
    logger.info("=" * 60)
    logger.info("XAUUSD Trading Bot Starting")
    logger.info("=" * 60)


def main():
    """Main bot loop."""
    print_banner()
    
    # Initialize components
    mt5 = MT5Manager()
    db = Database()
    notifier = TelegramNotifier()
    
    # Connect to MT5
    try:
        if not mt5.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        return
    
    # Initialize strategy with MT5 reference for HTF data
    strategy = XAUUSDStrategy(mt5_manager=mt5)
    
    # Send startup notification
    if notifier.is_enabled:
        notifier.send_startup_message(mt5.get_account_balance())
    
    try:
        while True:
            loop_start = datetime.now()
            logger.info(f"Running analysis cycle at {loop_start}")
            
            try:
                # --- Fetch Data ---
                xau_df = mt5.get_ohlcv(settings.symbol, settings.timeframe, bars=500)
                if xau_df is None:
                    logger.warning("Failed to fetch XAUUSD data. Retrying...")
                    time.sleep(settings.loop_interval_seconds)
                    continue
                
                # Try to get DXY data
                dxy_df = mt5.get_ohlcv(settings.dxy_symbol, settings.timeframe, bars=100)
                if dxy_df is None:
                    logger.debug("DXY not available on MT5, trying yfinance fallback...")
                    dxy_df = dxy.get_dxy_from_yfinance()
                
                # Fetch HTF data if enabled
                htf_df = None
                if settings.htf_enabled:
                    htf = strategy.mtf_analyzer.get_htf_for_ltf(settings.timeframe)
                    if htf:
                        htf_df = mt5.get_ohlcv(settings.symbol, htf, bars=100)
                
                # --- Check Existing Positions ---
                positions = mt5.get_bot_positions(settings.symbol)
                
                # Update trailing stops for existing positions
                for pos in positions:
                    mt5.update_trailing_stop(pos['ticket'])
                
                if len(positions) >= settings.max_positions:
                    logger.info(f"Max positions ({settings.max_positions}) reached. Checking for exits...")
                    
                    for pos in positions:
                        should_close, reason = strategy.should_close_position(pos, xau_df)
                        if should_close:
                            logger.info(f"Closing position {pos['ticket']}: {reason}")
                            
                            # Get current profit before closing
                            profit = pos.get('profit', 0)
                            
                            if mt5.close_position(pos['ticket']):
                                # Record in database
                                db.record_trade_exit(
                                    ticket=pos['ticket'],
                                    exit_price=xau_df['close'].iloc[-1],
                                    profit=profit
                                )
                                
                                # Log and notify
                                trade_logger.log_exit(
                                    pos['ticket'], settings.symbol, pos['type'],
                                    xau_df['close'].iloc[-1], profit, reason
                                )
                                
                                if notifier.is_enabled:
                                    notifier.send_trade_closed(
                                        pos['ticket'], settings.symbol, pos['type'],
                                        pos['price_open'], xau_df['close'].iloc[-1],
                                        profit, reason
                                    )
                    
                    time.sleep(settings.loop_interval_seconds)
                    continue
                
                # --- Generate Signal ---
                signal = strategy.analyze(xau_df, dxy_df, htf_df)
                
                # Log signal
                logger.info(
                    f"Signal: {signal.direction} | Confidence: {signal.confidence:.0%} | "
                    f"HTF: {signal.htf_bias} | {signal.session_info}"
                )
                
                if signal.reasons:
                    for reason in signal.reasons:
                        logger.debug(f"  - {reason}")
                
                # Record signal in database
                db.record_signal(
                    symbol=settings.symbol,
                    timeframe=settings.timeframe,
                    direction=signal.direction,
                    confidence=signal.confidence,
                    factors=signal.reasons,
                    executed=False
                )
                
                # --- Execute Trade ---
                if signal.direction in ["BUY", "SELL"] and signal.confidence >= settings.min_confidence:
                    current_price = xau_df['close'].iloc[-1]
                    sl_pips = abs(current_price - signal.stop_loss) / 0.1  # Convert to pips
                    
                    lot_size = mt5.calculate_lot_size(settings.symbol, sl_pips)
                    tp_pips = abs(signal.take_profit - current_price) / 0.1
                    
                    logger.info(f"Executing {signal.direction} order:")
                    logger.info(f"  Lot Size: {lot_size}")
                    logger.info(f"  SL: {signal.stop_loss:.2f} ({sl_pips:.0f} pips)")
                    logger.info(f"  TP: {signal.take_profit:.2f} ({tp_pips:.0f} pips)")
                    
                    try:
                        ticket = mt5.place_order(
                            symbol=settings.symbol,
                            order_type=signal.direction,
                            lot=lot_size,
                            sl_price=signal.stop_loss,
                            tp_price=signal.take_profit,
                            comment=f"MT5Bot {signal.htf_bias}"
                        )
                        
                        if ticket:
                            logger.info(f"Order placed! Ticket: {ticket}")
                            
                            # Record in database
                            db.record_trade_entry(
                                ticket=ticket if ticket > 0 else 0,
                                symbol=settings.symbol,
                                direction=signal.direction,
                                lot_size=lot_size,
                                entry_price=current_price,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                factors=signal.reasons,
                                confidence=signal.confidence
                            )
                            
                            # Log trade
                            trade_logger.log_entry(
                                ticket, settings.symbol, signal.direction, lot_size,
                                current_price, signal.stop_loss, signal.take_profit,
                                signal.reasons
                            )
                            
                            # Notify
                            if notifier.is_enabled:
                                notifier.send_trade_opened(
                                    ticket, settings.symbol, signal.direction,
                                    lot_size, current_price, signal.stop_loss,
                                    signal.take_profit
                                )
                        else:
                            logger.warning("Order placement returned None")
                    
                    except OrderError as e:
                        logger.error(f"Order failed: {e}")
                        if notifier.is_enabled:
                            notifier.send_error_alert(str(e), "Order placement")
                
                # --- Wait for next cycle ---
                elapsed = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, settings.loop_interval_seconds - elapsed)
                logger.debug(f"Sleeping {sleep_time:.0f}s until next cycle...")
                time.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if notifier.is_enabled:
                    notifier.send_error_alert(str(e), "Main loop")
                time.sleep(settings.loop_interval_seconds)
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    
    finally:
        mt5.disconnect()
        
        if notifier.is_enabled:
            notifier.send_shutdown_message("Normal shutdown")
        
        logger.info("Bot stopped.")
        print("\n🛑 Bot stopped. Goodbye!")


if __name__ == "__main__":
    main()
