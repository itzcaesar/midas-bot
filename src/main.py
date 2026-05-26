"""
XAUUSD Trading Bot - Main Entry Point
Enhanced with logging, notifications, database persistence, and advanced filters.
"""
import sys
import time

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
from broker.mt5 import MT5Manager
from broker import SimBroker
from strategy.xauusd_strategy import XAUUSDStrategy
from analysis import dxy
from notifications.telegram import TelegramNotifier
from risk import AccountState, RiskGovernor
from reconciliation import Reconciler

# Setup main logger
logger = setup_logger("mt5bot.main", log_dir=settings.log_dir)
trade_logger = TradeLogger(settings.log_dir)


def print_banner():
    """Print startup banner."""
    banner = (
        "\n"
        "+--------------------------------------------------------------+\n"
        "|             XAUUSD TRADING BOT v2.0 (MIDAS)                  |\n"
        "|  Strategy: Liquidity + MACD + Breakout + Structure + DXY     |\n"
        "|            + HTF Bias + Session Filter + News                |\n"
        "+--------------------------------------------------------------+\n"
        f"|  Symbol: {settings.symbol:<10}  Timeframe: {settings.timeframe:<8}"
        f"  Risk: {settings.risk_percent}%\n"
        f"|  Dry Run: {'YES' if settings.dry_run else 'NO':<8}"
        f"   Session Filter: {'ON' if settings.session_filter_enabled else 'OFF':<8}\n"
        f"|  HTF Enabled: {'ON' if settings.htf_enabled else 'OFF':<6}"
        f"  News Filter: {'ON' if settings.news_filter_enabled else 'OFF':<8}\n"
        "+--------------------------------------------------------------+\n"
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
    governor = RiskGovernor()
    
    # Connect to MT5
    mt5_connected = False
    try:
        if mt5.connect():
            mt5_connected = True
        else:
            logger.error("Failed to connect to MT5.")
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")

    if not mt5_connected:
        if settings.dry_run:
            logger.warning(
                "MT5 not available but DRY_RUN=true. "
                "Switching to SimBroker for offline dry-run mode."
            )
            # Load synthetic/historical data into SimBroker for offline operation
            from ml.data_loader import KaggleDataLoader
            try:
                loader = KaggleDataLoader(data_dir="data")
                tf_map = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
                          "H1": "1h", "H4": "4h", "D1": "1d"}
                tf_key = tf_map.get(settings.timeframe, "1h")
                offline_data = loader.load_data(tf_key)
                # Start cursor at bar 600 so there's enough warmup history
                mt5 = SimBroker(initial_balance=10_000.0, data=offline_data, start_cursor=600)
                logger.info(f"SimBroker loaded {len(offline_data)} bars of offline {tf_key} data")
            except Exception as e:
                logger.warning(f"Could not load offline data: {e}. SimBroker will have no data.")
                mt5 = SimBroker(initial_balance=10_000.0)
            mt5.connect()
            mt5_connected = True
        else:
            logger.error("MT5 required for live trading. Exiting.")
            return
    
    # Initialize strategy with MT5 reference for HTF data
    strategy = XAUUSDStrategy(mt5=mt5 if mt5_connected else None)

    # Reconcile DB with broker positions in case of crash recovery (REQ-P1-09).
    if mt5_connected:
        try:
            reconciler = Reconciler(broker=mt5, db=db)
            report = reconciler.reconcile_on_startup(symbol=settings.symbol)
            if report.has_drift or report.errors:
                logger.warning(f"Startup reconciliation drift detected: {report}")
                if notifier.is_enabled:
                    notifier.send_error_alert(str(report), "Startup reconciliation")
            else:
                logger.info(f"Startup reconciliation clean: {report.in_sync_count} positions in sync")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Startup reconciliation crashed (continuing without it): {exc}")
            if notifier.is_enabled:
                notifier.send_error_alert(str(exc), "Reconciler")
    else:
        logger.info("Skipping reconciliation (MT5 not connected)")

    # Send startup notification
    if notifier.is_enabled:
        balance = mt5.get_account_balance() if mt5_connected else 0.0
        notifier.send_startup_message(balance)
    
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

                                # Notify the governor so loss-streak / DD math stays current
                                governor.record_trade_result(profit)

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
                    # Risk governor gate (REQ-P1-08)
                    decision = governor.can_open(
                        AccountState(
                            equity=mt5.get_account_equity(),
                            balance=mt5.get_account_balance(),
                            open_positions=len(positions),
                        )
                    )
                    if not decision.allow:
                        logger.warning(f"Risk governor blocked order: {decision.reason}")
                        if notifier.is_enabled:
                            notifier.send_error_alert(
                                f"Order blocked: {decision.reason}", "RiskGovernor"
                            )
                        time.sleep(settings.loop_interval_seconds)
                        continue

                    current_price = xau_df['close'].iloc[-1]
                    sl_pips = abs(current_price - signal.stop_loss) / 0.1  # Convert to pips

                    lot_size = mt5.calculate_lot_size(settings.symbol, sl_pips)
                    # Apply governor's soft-scale (e.g. 0.5x risk inside a drawdown)
                    if decision.risk_scale < 1.0:
                        original_lot = lot_size
                        lot_size = max(0.01, round(lot_size * decision.risk_scale, 2))
                        logger.info(
                            f"Governor scaled lot size {original_lot} -> {lot_size} "
                            f"({decision.reason})"
                        )
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

                            # Tell the governor an order went out so the rate limiter ticks
                            governor.record_order_sent()

                            # Record in database
                            db.record_trade_entry(
                                ticket=ticket,
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
                # In dry-run with SimBroker, cycle fast (1s) to replay history quickly.
                # In live mode, respect the configured interval.
                interval = 1 if settings.dry_run and isinstance(mt5, SimBroker) else settings.loop_interval_seconds
                sleep_time = max(0, interval - elapsed)
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
