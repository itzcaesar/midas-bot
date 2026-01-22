"""
Technical Indicators - MACD, Consolidation/Breakout, Price Structure
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


from config import settings


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD indicator.
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with macd, macd_signal, macd_hist columns added
    """
    fast = settings.MACD_FAST
    slow = settings.MACD_SLOW
    signal = settings.MACD_SIGNAL
    
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def detect_consolidation(df: pd.DataFrame) -> Tuple[bool, float, float]:
    """
    Detect if price is in a consolidation range.
    
    Returns:
        (is_consolidating, range_high, range_low)
    """
    lookback = settings.CONSOLIDATION_LOOKBACK
    threshold = settings.CONSOLIDATION_THRESHOLD
    
    recent = df.tail(lookback)
    atr = calculate_atr(df).iloc[-1]
    
    range_high = recent['high'].max()
    range_low = recent['low'].min()
    range_size = range_high - range_low
    
    # Consolidation = tight range relative to ATR
    is_consolidating = range_size < (atr * threshold * lookback)
    
    return is_consolidating, range_high, range_low


def detect_breakout(df: pd.DataFrame, range_high: float, range_low: float) -> Optional[str]:
    """
    Detect breakout from consolidation range.
    
    Args:
        df: OHLCV DataFrame
        range_high: Upper boundary of range
        range_low: Lower boundary of range
        
    Returns:
        "BULLISH", "BEARISH", or None
    """
    current_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    
    # Bullish breakout: close above range high
    if current_close > range_high and prev_close <= range_high:
        return "BULLISH"
    
    # Bearish breakout: close below range low
    if current_close < range_low and prev_close >= range_low:
        return "BEARISH"
    
    return None


def detect_structure(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect price structure: Higher Highs/Lows (uptrend) or Lower Highs/Lows (downtrend).
    Also detects Break of Structure (BOS).
    
    Returns:
        {
            'trend': 'UP' | 'DOWN' | 'NEUTRAL',
            'last_swing_high': float,
            'last_swing_low': float,
            'bos': 'BULLISH' | 'BEARISH' | None
        }
    """
    recent = df.tail(lookback)
    
    # Find swing highs and lows using rolling window
    swing_strength = settings.SWING_STRENGTH
    
    highs = []
    lows = []
    
    for i in range(swing_strength, len(recent) - swing_strength):
        window_high = recent['high'].iloc[i-swing_strength:i+swing_strength+1]
        window_low = recent['low'].iloc[i-swing_strength:i+swing_strength+1]
        
        if recent['high'].iloc[i] == window_high.max():
            highs.append((recent.index[i], recent['high'].iloc[i]))
        if recent['low'].iloc[i] == window_low.min():
            lows.append((recent.index[i], recent['low'].iloc[i]))
    
    result = {
        'trend': 'NEUTRAL',
        'last_swing_high': highs[-1][1] if highs else df['high'].max(),
        'last_swing_low': lows[-1][1] if lows else df['low'].min(),
        'bos': None
    }
    
    if len(highs) >= 2 and len(lows) >= 2:
        # Check for higher highs and higher lows (uptrend)
        hh = highs[-1][1] > highs[-2][1]
        hl = lows[-1][1] > lows[-2][1]
        
        # Check for lower highs and lower lows (downtrend)
        lh = highs[-1][1] < highs[-2][1]
        ll = lows[-1][1] < lows[-2][1]
        
        if hh and hl:
            result['trend'] = 'UP'
        elif lh and ll:
            result['trend'] = 'DOWN'
        
        # Break of Structure detection
        current_close = df['close'].iloc[-1]
        
        # Bullish BOS: price breaks above last lower high in downtrend
        if result['trend'] == 'DOWN' and current_close > highs[-1][1]:
            result['bos'] = 'BULLISH'
        
        # Bearish BOS: price breaks below last higher low in uptrend
        if result['trend'] == 'UP' and current_close < lows[-1][1]:
            result['bos'] = 'BEARISH'
    
    return result


def get_macd_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Get trading signal from MACD.
    
    Returns:
        "BUY" if MACD crosses above signal
        "SELL" if MACD crosses below signal
        None otherwise
    """
    if 'macd' not in df.columns:
        df = calculate_macd(df)
    
    current_hist = df['macd_hist'].iloc[-1]
    prev_hist = df['macd_hist'].iloc[-2]
    
    # Bullish crossover
    if prev_hist < 0 and current_hist > 0:
        return "BUY"
    
    # Bearish crossover
    if prev_hist > 0 and current_hist < 0:
        return "SELL"
    
    return None
