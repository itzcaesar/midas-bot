"""
MT5 Manager - MetaTrader 5 Connection and Order Management
Enhanced with logging, trailing stops, and improved error handling.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple



from config import settings
from core.logger import get_logger
from core.exceptions import ConnectionError, OrderError, SymbolError

# Module logger
logger = get_logger("mt5bot.broker")


class MT5Manager:
    """Handles all MetaTrader 5 interactions."""
    
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    
    # Magic number for bot identification
    MAGIC_NUMBER = 123456
    
    def __init__(self):
        self.connected = False
        self._account_info = None
    
    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        logger.info("Initializing MT5 connection...")
        
        # Check if custom login credentials are provided
        if settings.mt5_login and settings.mt5_password and settings.mt5_server:
            if not mt5.initialize(
                login=int(settings.mt5_login),
                password=settings.mt5_password,
                server=settings.mt5_server
            ):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                raise ConnectionError(f"MT5 initialization failed", {'error': error})
        else:
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                raise ConnectionError(f"MT5 initialization failed", {'error': error})
        
        self.connected = True
        self._account_info = mt5.account_info()
        
        logger.info(
            f"Connected to MT5 - Account: {self._account_info.login}, "
            f"Balance: ${self._account_info.balance:,.2f}, "
            f"Broker: {self._account_info.company}"
        )
        return True
    
    def disconnect(self):
        """Shutdown MT5 connection."""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def get_ohlcv(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: Timeframe string (e.g., "M15")
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        tf = self.TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol {symbol}")
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df[['open', 'high', 'low', 'close', 'tick_volume']].rename(
            columns={'tick_volume': 'volume'}
        )
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol trading specifications."""
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol info not available for {symbol}")
            return None
        return {
            'point': info.point,
            'digits': info.digits,
            'min_lot': info.volume_min,
            'max_lot': info.volume_max,
            'lot_step': info.volume_step,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'currency_profit': info.currency_profit,
        }
    
    def get_account_balance(self) -> float:
        """Get current account balance."""
        info = mt5.account_info()
        return info.balance if info else 0.0
    
    def get_account_equity(self) -> float:
        """Get current account equity."""
        info = mt5.account_info()
        return info.equity if info else 0.0
    
    def get_pip_value(self, symbol: str, lot_size: float = 1.0) -> float:
        """
        Calculate pip value for a symbol dynamically.
        
        Args:
            symbol: Trading symbol
            lot_size: Position size in lots
            
        Returns:
            Value per pip in account currency
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Using default pip value for {symbol}")
            return 0.1 * lot_size * 100  # Default fallback
        
        # For metals like XAUUSD: 1 pip = $0.01 movement
        # Contract size is typically 100 oz for gold
        point = info.point
        contract_size = info.trade_contract_size
        
        # Calculate: pip_value = contract_size * lot_size * point * 10
        # For XAUUSD: 100 * 1.0 * 0.01 * 10 = $10 per pip for 1 lot
        pip_value = contract_size * lot_size * point * 10
        
        return pip_value
    
    def calculate_lot_size(self, symbol: str, stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk percentage.
        
        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss distance in pips
            
        Returns:
            Lot size
        """
        balance = self.get_account_balance()
        risk_amount = balance * (settings.risk_percent / 100)
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            # Return safe default minimum lot size
            logger.warning(f"Symbol info unavailable for {symbol}, using default min lot 0.01")
            return 0.01
        
        # Calculate pip value for minimum lot
        min_lot = symbol_info['min_lot']
        pip_value_per_min_lot = self.get_pip_value(symbol, min_lot)
        
        # Calculate risk per pip
        risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
        
        # Calculate lot size
        if pip_value_per_min_lot > 0:
            lot_size = (risk_per_pip / pip_value_per_min_lot) * min_lot
        else:
            lot_size = min_lot
        
        # Round to lot step
        lot_step = symbol_info['lot_step']
        lot_size = max(min_lot, round(lot_size / lot_step) * lot_step)
        lot_size = min(lot_size, symbol_info['max_lot'])
        
        logger.debug(
            f"Calculated lot size: {lot_size} for {symbol} "
            f"(Risk: ${risk_amount:.2f}, SL: {stop_loss_pips} pips)"
        )
        
        return lot_size
    
    def place_order(self, symbol: str, order_type: str, lot: float, 
                    sl_pips: float = None, tp_pips: float = None,
                    sl_price: float = None, tp_price: float = None,
                    comment: str = "MT5Bot") -> Optional[int]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            lot: Position size
            sl_pips: Stop loss in pips (optional, used if sl_price not set)
            tp_pips: Take profit in pips (optional, used if tp_price not set)
            sl_price: Stop loss price (optional, overrides sl_pips)
            tp_price: Take profit price (optional, overrides tp_pips)
            comment: Order comment
            
        Returns:
            Order ticket number or None if failed
        """
        if settings.dry_run:
            logger.info(f"[DRY RUN] Would place {order_type} {lot} lots on {symbol}")
            return -1
        
        # Validate symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            raise SymbolError(symbol)
        
        point = symbol_info.point
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == "BUY" else tick.bid
        
        # Calculate SL/TP prices if not provided directly
        if sl_price is None and sl_pips:
            sl_distance = sl_pips * point * 10
            sl_price = price - sl_distance if order_type == "BUY" else price + sl_distance
        
        if tp_price is None and tp_pips:
            tp_distance = tp_pips * point * 10
            tp_price = price + tp_distance if order_type == "BUY" else price - tp_distance
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price or 0.0,
            "tp": tp_price or 0.0,
            "deviation": 20,
            "magic": self.MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        logger.info(
            f"Sending order: {order_type} {lot} {symbol} @ {price:.2f} "
            f"SL: {sl_price:.2f if sl_price else 'None'} "
            f"TP: {tp_price:.2f if tp_price else 'None'}"
        )
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            raise OrderError(
                f"Order failed: {result.comment}",
                order_type=order_type,
                retcode=result.retcode
            )
        
        logger.info(f"Order executed! Ticket: {result.order}")
        return result.order
    
    def get_open_positions(self, symbol: str = None) -> list:
        """Get list of open positions."""
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        
        return [
            {
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == 0 else 'SELL',
                'volume': p.volume,
                'price_open': p.price_open,
                'price_current': p.price_current,
                'sl': p.sl,
                'tp': p.tp,
                'profit': p.profit,
                'swap': p.swap,
                'time': datetime.fromtimestamp(p.time),
                'magic': p.magic,
            }
            for p in positions
        ]
    
    def get_bot_positions(self, symbol: str = None) -> list:
        """Get positions opened by this bot (matching magic number)."""
        all_positions = self.get_open_positions(symbol)
        return [p for p in all_positions if p.get('magic') == self.MAGIC_NUMBER]
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket number."""
        if settings.dry_run:
            logger.info(f"[DRY RUN] Would close position {ticket}")
            return True
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} not found")
            return False
        
        position = position[0]
        symbol = position.symbol
        volume = position.volume
        order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if position.type == 0 else mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.MAGIC_NUMBER,
            "comment": "Close by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed for {ticket}: {result.comment}")
            return False
        
        logger.info(f"Position {ticket} closed at {price:.2f}")
        return True
    
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """
        Modify stop loss and/or take profit of an existing position.
        
        Args:
            ticket: Position ticket number
            sl: New stop loss price (None to keep current)
            tp: New take profit price (None to keep current)
            
        Returns:
            True if modification successful
        """
        if settings.dry_run:
            logger.info(f"[DRY RUN] Would modify position {ticket} - SL: {sl}, TP: {tp}")
            return True
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} not found for modification")
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify failed for {ticket}: {result.comment}")
            return False
        
        logger.info(f"Position {ticket} modified - SL: {sl}, TP: {tp}")
        return True
    
    def update_trailing_stop(self, ticket: int, trailing_pips: float = None) -> bool:
        """
        Move stop loss to lock in profits as price moves favorably.
        
        Args:
            ticket: Position ticket number
            trailing_pips: Trailing distance in pips (uses settings if not provided)
            
        Returns:
            True if stop loss was updated
        """
        trailing_pips = trailing_pips or settings.trailing_stop_pips
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        pos = position[0]
        symbol = pos.symbol
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return False
        
        point = symbol_info.point
        trailing_distance = trailing_pips * point * 10
        
        tick = mt5.symbol_info_tick(symbol)
        current_price = tick.bid if pos.type == 0 else tick.ask  # BUY uses bid, SELL uses ask
        
        if pos.type == 0:  # BUY position
            new_sl = current_price - trailing_distance
            # Only move SL up, never down
            if new_sl > pos.sl and new_sl < current_price:
                profit_pips = (current_price - pos.price_open) / (point * 10)
                if profit_pips > trailing_pips:  # Only trail when in sufficient profit
                    logger.info(
                        f"Trailing stop for BUY {ticket}: "
                        f"Moving SL from {pos.sl:.2f} to {new_sl:.2f}"
                    )
                    return self.modify_position(ticket, sl=new_sl)
        else:  # SELL position
            new_sl = current_price + trailing_distance
            # Only move SL down, never up
            if new_sl < pos.sl and new_sl > current_price:
                profit_pips = (pos.price_open - current_price) / (point * 10)
                if profit_pips > trailing_pips:
                    logger.info(
                        f"Trailing stop for SELL {ticket}: "
                        f"Moving SL from {pos.sl:.2f} to {new_sl:.2f}"
                    )
                    return self.modify_position(ticket, sl=new_sl)
        
        return False
    
    def close_all_positions(self, symbol: str = None) -> int:
        """
        Close all open positions.
        
        Args:
            symbol: If specified, only close positions for this symbol
            
        Returns:
            Number of positions closed
        """
        positions = self.get_open_positions(symbol)
        closed = 0
        
        for pos in positions:
            if self.close_position(pos['ticket']):
                closed += 1
        
        logger.info(f"Closed {closed} of {len(positions)} positions")
        return closed
    
    def get_daily_profit(self) -> float:
        """Get total profit/loss for today."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get closed deals today
        deals = mt5.history_deals_get(today, datetime.now())
        if deals is None:
            return 0.0
        
        return sum(deal.profit + deal.swap + deal.commission for deal in deals)
