"""
XAUUSD Multi-Factor Strategy
Enhanced with MTF analysis, session filtering, and news avoidance.
Combines: Liquidity, MACD, Breakout/Consolidation, Price Structure, DXY Correlation
"""
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass, field



from config import settings
from core.logger import get_logger
from analysis import indicators, liquidity, dxy
from analysis.mtf_analysis import MultiTimeframeAnalyzer
from analysis.session_filter import SessionFilter
from analysis.news_filter import NewsFilter

logger = get_logger("mt5bot.strategy")


@dataclass
class Signal:
    """Trading signal with metadata."""
    direction: str  # "BUY" | "SELL" | "HOLD"
    confidence: float  # 0.0 - 1.0
    reasons: list = field(default_factory=list)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    htf_bias: str = ""
    session_info: str = ""
    filtered: bool = False
    filter_reason: str = ""


class XAUUSDStrategy:
    """
    Multi-factor strategy for XAUUSD trading.
    
    Entry conditions (need 3+ factors aligned):
    1. Liquidity sweep or near liquidity zone
    2. MACD confirmation
    3. Breakout from consolidation OR price structure alignment
    4. DXY correlation support
    5. [NEW] HTF trend alignment
    6. [NEW] Valid trading session
    7. [NEW] No high-impact news
    """
    
    def __init__(self, mt5_manager=None):
        """
        Initialize strategy.
        
        Args:
            mt5_manager: Optional MT5Manager instance for MTF data
        """
        self.min_factors = settings.min_factors
        self.consolidation_state = {'active': False, 'high': 0, 'low': 0}
        
        # Initialize filters
        self.mtf_analyzer = MultiTimeframeAnalyzer(mt5_manager)
        self.session_filter = SessionFilter()
        self.news_filter = NewsFilter()
        
        # Filter toggles from settings
        self.htf_enabled = settings.htf_enabled
        self.session_filter_enabled = settings.session_filter_enabled
        self.news_filter_enabled = settings.news_filter_enabled
    
    def check_filters(self) -> Tuple[bool, str]:
        """
        Check all pre-trade filters.
        
        Returns:
            Tuple of (passed, reason)
        """
        # Session filter
        if self.session_filter_enabled:
            is_valid, reason = self.session_filter.is_valid_session()
            if not is_valid:
                return False, f"Session filter: {reason}"
        
        # News filter
        if self.news_filter_enabled:
            is_safe, reason = self.news_filter.is_safe_to_trade()
            if not is_safe:
                return False, f"News filter: {reason}"
        
        return True, "All filters passed"
    
    def analyze(
        self, 
        xau_df: pd.DataFrame, 
        dxy_df: Optional[pd.DataFrame] = None,
        htf_df: Optional[pd.DataFrame] = None
    ) -> Signal:
        """
        Analyze market conditions and generate trading signal.
        
        Args:
            xau_df: XAUUSD OHLCV data
            dxy_df: DXY OHLCV data (optional)
            htf_df: Higher timeframe data (optional)
            
        Returns:
            Signal object with direction and metadata
        """
        # Check pre-trade filters first
        filters_passed, filter_reason = self.check_filters()
        if not filters_passed:
            logger.debug(f"Trade filtered: {filter_reason}")
            return Signal(
                direction="HOLD",
                confidence=0.0,
                reasons=[filter_reason],
                filtered=True,
                filter_reason=filter_reason
            )
        
        factors_buy = []
        factors_sell = []
        
        current_price = xau_df['close'].iloc[-1]
        
        # --- 1. MACD Analysis ---
        xau_df = indicators.calculate_macd(xau_df)
        macd_signal = indicators.get_macd_signal(xau_df)
        
        if macd_signal == "BUY":
            factors_buy.append("MACD bullish crossover")
        elif macd_signal == "SELL":
            factors_sell.append("MACD bearish crossover")
        
        # Check MACD momentum (histogram direction)
        if xau_df['macd_hist'].iloc[-1] > xau_df['macd_hist'].iloc[-2] > 0:
            factors_buy.append("MACD momentum increasing")
        elif xau_df['macd_hist'].iloc[-1] < xau_df['macd_hist'].iloc[-2] < 0:
            factors_sell.append("MACD momentum decreasing")
        
        # --- 2. Liquidity Analysis ---
        liq_signal = liquidity.get_liquidity_signal(xau_df)
        
        if liq_signal == "BUY":
            factors_buy.append("Bullish liquidity sweep")
        elif liq_signal == "SELL":
            factors_sell.append("Bearish liquidity sweep")
        
        # Check proximity to liquidity zones
        zones = liquidity.identify_liquidity_zones(xau_df)
        atr = indicators.calculate_atr(xau_df).iloc[-1]
        
        for level in zones['sell_side_liquidity']:
            if abs(current_price - level) < atr * 0.5:
                factors_buy.append(f"Near sell-side liquidity {level:.2f}")
                break
        
        for level in zones['buy_side_liquidity']:
            if abs(current_price - level) < atr * 0.5:
                factors_sell.append(f"Near buy-side liquidity {level:.2f}")
                break
        
        # --- 3. Consolidation & Breakout ---
        is_consolidating, range_high, range_low = indicators.detect_consolidation(xau_df)
        
        if is_consolidating:
            self.consolidation_state = {'active': True, 'high': range_high, 'low': range_low}
        elif self.consolidation_state['active']:
            breakout = indicators.detect_breakout(
                xau_df, 
                self.consolidation_state['high'], 
                self.consolidation_state['low']
            )
            if breakout == "BULLISH":
                factors_buy.append("Bullish breakout from consolidation")
                self.consolidation_state['active'] = False
            elif breakout == "BEARISH":
                factors_sell.append("Bearish breakout from consolidation")
                self.consolidation_state['active'] = False
        
        # --- 4. Price Structure ---
        structure = indicators.detect_structure(xau_df)
        
        if structure['trend'] == 'UP':
            factors_buy.append("Uptrend structure (HH/HL)")
        elif structure['trend'] == 'DOWN':
            factors_sell.append("Downtrend structure (LH/LL)")
        
        if structure['bos'] == 'BULLISH':
            factors_buy.append("Bullish Break of Structure")
        elif structure['bos'] == 'BEARISH':
            factors_sell.append("Bearish Break of Structure")
        
        # --- 5. DXY Correlation ---
        if dxy_df is not None and len(dxy_df) > 0:
            dxy_signal, dxy_strength = dxy.get_dxy_signal(xau_df, dxy_df)
            
            if dxy_signal == "BUY":
                factors_buy.append(f"DXY correlation ({dxy_strength})")
            elif dxy_signal == "SELL":
                factors_sell.append(f"DXY correlation ({dxy_strength})")
        
        # --- 6. HTF Bias Confirmation ---
        htf_bias = "NEUTRAL"
        htf_info = ""
        
        if self.htf_enabled:
            bias, details = self.mtf_analyzer.get_htf_bias(
                settings.symbol, 
                settings.timeframe,
                htf_df
            )
            htf_bias = bias
            htf_info = f"HTF: {bias}"
            
            # Add as factor if confirmation required
            if settings.htf_confirmation_required:
                if bias == 'BULLISH':
                    factors_buy.append(f"HTF bullish bias ({details.get('htf', 'H4')})")
                elif bias == 'BEARISH':
                    factors_sell.append(f"HTF bearish bias ({details.get('htf', 'H4')})")
        
        # --- Decision Logic ---
        buy_score = len(factors_buy)
        sell_score = len(factors_sell)
        
        # Get session info for logging
        session_info = ""
        if self.session_filter_enabled:
            current_session, _ = self.session_filter.get_current_session()
            session_info = f"Session: {current_session or 'Unknown'}"
        
        # Check HTF alignment if required
        htf_aligned_buy = not self.htf_enabled or not settings.htf_confirmation_required or htf_bias != 'BEARISH'
        htf_aligned_sell = not self.htf_enabled or not settings.htf_confirmation_required or htf_bias != 'BULLISH'
        
        if buy_score >= self.min_factors and buy_score > sell_score and htf_aligned_buy:
            # Use swing low as SL for buys
            sl_price = structure['last_swing_low'] - (atr * 0.5)
            tp_price = current_price + (current_price - sl_price) * 2  # 2:1 RR
            
            confidence = min(buy_score / 5, 1.0)
            
            logger.info(
                f"BUY signal: confidence={confidence:.0%}, factors={buy_score}, "
                f"htf={htf_bias}, {session_info}"
            )
            
            return Signal(
                direction="BUY",
                confidence=confidence,
                reasons=factors_buy,
                stop_loss=sl_price,
                take_profit=tp_price,
                htf_bias=htf_bias,
                session_info=session_info
            )
        
        elif sell_score >= self.min_factors and sell_score > buy_score and htf_aligned_sell:
            # Use swing high as SL for sells
            sl_price = structure['last_swing_high'] + (atr * 0.5)
            tp_price = current_price - (sl_price - current_price) * 2  # 2:1 RR
            
            confidence = min(sell_score / 5, 1.0)
            
            logger.info(
                f"SELL signal: confidence={confidence:.0%}, factors={sell_score}, "
                f"htf={htf_bias}, {session_info}"
            )
            
            return Signal(
                direction="SELL",
                confidence=confidence,
                reasons=factors_sell,
                stop_loss=sl_price,
                take_profit=tp_price,
                htf_bias=htf_bias,
                session_info=session_info
            )
        
        # No clear signal
        logger.debug(f"HOLD: buy_factors={buy_score}, sell_factors={sell_score}, htf={htf_bias}")
        
        return Signal(
            direction="HOLD",
            confidence=0.0,
            reasons=["Insufficient aligned factors"],
            htf_bias=htf_bias,
            session_info=session_info
        )
    
    def should_close_position(self, position: dict, xau_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if an existing position should be closed.
        
        Args:
            position: Position dict from MT5Manager
            xau_df: Current OHLCV data
            
        Returns:
            (should_close, reason)
        """
        structure = indicators.detect_structure(xau_df)
        
        # Close long if bearish BOS
        if position['type'] == 'BUY' and structure['bos'] == 'BEARISH':
            return True, "Bearish Break of Structure"
        
        # Close short if bullish BOS
        if position['type'] == 'SELL' and structure['bos'] == 'BULLISH':
            return True, "Bullish Break of Structure"
        
        # Check for reversal MACD
        xau_df = indicators.calculate_macd(xau_df)
        macd_signal = indicators.get_macd_signal(xau_df)
        
        if position['type'] == 'BUY' and macd_signal == 'SELL':
            return True, "MACD bearish crossover"
        
        if position['type'] == 'SELL' and macd_signal == 'BUY':
            return True, "MACD bullish crossover"
        
        return False, ""
    
    def get_status(self) -> dict:
        """Get current strategy status and filter states."""
        session_valid, session_reason = self.session_filter.is_valid_session()
        news_safe, news_reason = self.news_filter.is_safe_to_trade()
        
        return {
            'htf_enabled': self.htf_enabled,
            'session_filter_enabled': self.session_filter_enabled,
            'session_valid': session_valid,
            'session_reason': session_reason,
            'news_filter_enabled': self.news_filter_enabled,
            'news_safe': news_safe,
            'news_reason': news_reason,
            'min_factors': self.min_factors,
        }
