"""
Multi-Timeframe Analysis Module
Provides higher timeframe (HTF) bias confirmation for trading decisions.
"""
import pandas as pd
from typing import Optional, Tuple
import sys
sys.path.append('../..')

from config import settings
from core.logger import get_logger
from analysis import indicators

logger = get_logger("mt5bot.mtf")


class MultiTimeframeAnalyzer:
    """
    Implements Higher Timeframe (HTF) bias confirmation.
    Only allows trades in the direction of the HTF trend.
    """
    
    # Map lower timeframe to higher timeframe for confirmation
    HTF_MAP = {
        "M1": "M15",
        "M5": "H1",
        "M15": "H4",
        "M30": "H4",
        "H1": "D1",
        "H4": "D1",
        "D1": "W1",
    }
    
    # Timeframe multipliers for bar calculation
    TF_MINUTES = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
    }
    
    def __init__(self, mt5_manager=None):
        """
        Initialize MTF analyzer.
        
        Args:
            mt5_manager: Optional MT5Manager instance for fetching HTF data
        """
        self.mt5 = mt5_manager
    
    def get_htf_for_ltf(self, ltf: str) -> Optional[str]:
        """Get the corresponding higher timeframe for a lower timeframe."""
        return self.HTF_MAP.get(ltf)
    
    def get_htf_data(self, symbol: str, ltf: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch higher timeframe data for the given symbol and lower timeframe.
        
        Args:
            symbol: Trading symbol
            ltf: Lower timeframe (e.g., "M15")
            bars: Number of HTF bars to fetch
            
        Returns:
            DataFrame with HTF OHLCV data
        """
        if not self.mt5:
            logger.warning("MT5 manager not set, cannot fetch HTF data")
            return None
        
        htf = self.get_htf_for_ltf(ltf)
        if not htf:
            logger.warning(f"No HTF mapping for {ltf}")
            return None
        
        htf_df = self.mt5.get_ohlcv(symbol, htf, bars)
        if htf_df is None or len(htf_df) < 20:
            logger.warning(f"Insufficient HTF data for {symbol} {htf}")
            return None
        
        return htf_df
    
    def get_htf_bias(self, symbol: str, ltf: str, htf_df: pd.DataFrame = None) -> Tuple[str, dict]:
        """
        Determine the higher timeframe bias.
        
        Args:
            symbol: Trading symbol
            ltf: Lower timeframe being traded
            htf_df: Optional pre-fetched HTF data
            
        Returns:
            Tuple of (bias, details) where:
            - bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            - details: Dictionary with analysis details
        """
        # Fetch HTF data if not provided
        if htf_df is None:
            htf_df = self.get_htf_data(symbol, ltf)
        
        if htf_df is None:
            return 'NEUTRAL', {'reason': 'No HTF data available'}
        
        htf = self.get_htf_for_ltf(ltf) or 'Unknown'
        
        # Analyze structure on HTF
        structure = indicators.detect_structure(htf_df, lookback=20)
        
        # Check MACD on HTF
        htf_df = indicators.calculate_macd(htf_df)
        macd_hist = htf_df['macd_hist'].iloc[-1]
        macd_direction = 'BULLISH' if macd_hist > 0 else 'BEARISH'
        
        # Calculate EMAs for trend
        ema_20 = htf_df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = htf_df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        current_price = htf_df['close'].iloc[-1]
        
        # Determine bias based on multiple factors
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # Structure trend
        if structure['trend'] == 'UP':
            bullish_score += 2
            reasons.append(f"{htf} showing uptrend (HH/HL)")
        elif structure['trend'] == 'DOWN':
            bearish_score += 2
            reasons.append(f"{htf} showing downtrend (LH/LL)")
        
        # MACD
        if macd_direction == 'BULLISH':
            bullish_score += 1
            reasons.append(f"{htf} MACD bullish")
        else:
            bearish_score += 1
            reasons.append(f"{htf} MACD bearish")
        
        # EMA alignment
        if current_price > ema_20 > ema_50:
            bullish_score += 1
            reasons.append(f"{htf} price > EMA20 > EMA50")
        elif current_price < ema_20 < ema_50:
            bearish_score += 1
            reasons.append(f"{htf} price < EMA20 < EMA50")
        
        # Price relative to EMAs
        if current_price > ema_50:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # Determine final bias
        if bullish_score >= 3 and bullish_score > bearish_score:
            bias = 'BULLISH'
        elif bearish_score >= 3 and bearish_score > bullish_score:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'
        
        logger.debug(
            f"HTF Bias for {symbol}: {bias} "
            f"(Bullish: {bullish_score}, Bearish: {bearish_score})"
        )
        
        return bias, {
            'htf': htf,
            'trend': structure['trend'],
            'macd': macd_direction,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'reasons': reasons,
            'last_swing_high': structure['last_swing_high'],
            'last_swing_low': structure['last_swing_low'],
        }
    
    def is_signal_aligned(self, signal_direction: str, symbol: str, ltf: str,
                           htf_df: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Check if a trading signal aligns with HTF bias.
        
        Args:
            signal_direction: 'BUY' or 'SELL'
            symbol: Trading symbol
            ltf: Lower timeframe
            htf_df: Optional pre-fetched HTF data
            
        Returns:
            Tuple of (is_aligned, reason)
        """
        bias, details = self.get_htf_bias(symbol, ltf, htf_df)
        
        if signal_direction == 'BUY':
            if bias == 'BULLISH':
                return True, f"BUY aligned with {details['htf']} bullish bias"
            elif bias == 'BEARISH':
                return False, f"BUY conflicts with {details['htf']} bearish bias"
            else:
                return True, f"BUY with {details['htf']} neutral bias (allowed)"
        
        elif signal_direction == 'SELL':
            if bias == 'BEARISH':
                return True, f"SELL aligned with {details['htf']} bearish bias"
            elif bias == 'BULLISH':
                return False, f"SELL conflicts with {details['htf']} bullish bias"
            else:
                return True, f"SELL with {details['htf']} neutral bias (allowed)"
        
        return True, "No alignment check for HOLD signal"
    
    def get_htf_key_levels(self, symbol: str, ltf: str, 
                           htf_df: pd.DataFrame = None) -> dict:
        """
        Get key support/resistance levels from HTF.
        
        Args:
            symbol: Trading symbol
            ltf: Lower timeframe
            htf_df: Optional pre-fetched HTF data
            
        Returns:
            Dictionary with key levels
        """
        if htf_df is None:
            htf_df = self.get_htf_data(symbol, ltf)
        
        if htf_df is None:
            return {}
        
        structure = indicators.detect_structure(htf_df, lookback=30)
        
        # Find major swing points
        from analysis import liquidity
        swing_highs, swing_lows = liquidity.find_swing_points(htf_df)
        
        current_price = htf_df['close'].iloc[-1]
        
        # Find nearest resistance (swing high above current price)
        resistance_levels = [sh['price'] for sh in swing_highs if sh['price'] > current_price]
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        
        # Find nearest support (swing low below current price)
        support_levels = [sl['price'] for sl in swing_lows if sl['price'] < current_price]
        nearest_support = max(support_levels) if support_levels else None
        
        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'last_swing_high': structure['last_swing_high'],
            'last_swing_low': structure['last_swing_low'],
            'major_resistance': sorted(resistance_levels)[:3] if resistance_levels else [],
            'major_support': sorted(support_levels, reverse=True)[:3] if support_levels else [],
        }
