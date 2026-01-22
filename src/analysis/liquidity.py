"""
Liquidity Analysis - Swing Points, Liquidity Pools, and Sweep Detection
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


from config import settings


def find_swing_points(df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    """
    Identify swing highs and swing lows as potential liquidity pools.
    
    Returns:
        (swing_highs, swing_lows) - Lists of dicts with 'time', 'price', 'touched' keys
    """
    lookback = settings.LIQUIDITY_LOOKBACK
    strength = settings.SWING_STRENGTH
    
    recent = df.tail(lookback)
    swing_highs = []
    swing_lows = []
    
    for i in range(strength, len(recent) - strength):
        idx = recent.index[i]
        high = recent['high'].iloc[i]
        low = recent['low'].iloc[i]
        
        # Check if this is a swing high
        left_highs = recent['high'].iloc[i-strength:i]
        right_highs = recent['high'].iloc[i+1:i+strength+1]
        
        if high > left_highs.max() and high > right_highs.max():
            swing_highs.append({
                'time': idx,
                'price': high,
                'touched': False,
                'type': 'equal_high' if len(swing_highs) > 0 and abs(high - swing_highs[-1]['price']) < 0.5 else 'swing_high'
            })
        
        # Check if this is a swing low
        left_lows = recent['low'].iloc[i-strength:i]
        right_lows = recent['low'].iloc[i+1:i+strength+1]
        
        if low < left_lows.min() and low < right_lows.min():
            swing_lows.append({
                'time': idx,
                'price': low,
                'touched': False,
                'type': 'equal_low' if len(swing_lows) > 0 and abs(low - swing_lows[-1]['price']) < 0.5 else 'swing_low'
            })
    
    return swing_highs, swing_lows


def detect_liquidity_sweep(df: pd.DataFrame, swing_highs: List[dict], swing_lows: List[dict]) -> Optional[dict]:
    """
    Detect if price swept a liquidity pool and reversed.
    A sweep occurs when price spikes through a swing point but closes back.
    
    Returns:
        {
            'type': 'BULLISH_SWEEP' | 'BEARISH_SWEEP',
            'level': float,
            'signal': 'BUY' | 'SELL'
        }
        or None
    """
    if len(df) < 2:
        return None
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Check for bullish sweep (wick below swing low, close above)
    for sl in swing_lows[-5:]:  # Check recent swing lows
        if not sl['touched']:
            # Price wicked below the swing low
            if current['low'] < sl['price'] and current['close'] > sl['price']:
                # Strong reversal candle
                if current['close'] > current['open']:  # Bullish candle
                    return {
                        'type': 'BULLISH_SWEEP',
                        'level': sl['price'],
                        'signal': 'BUY'
                    }
    
    # Check for bearish sweep (wick above swing high, close below)
    for sh in swing_highs[-5:]:  # Check recent swing highs
        if not sh['touched']:
            # Price wicked above the swing high
            if current['high'] > sh['price'] and current['close'] < sh['price']:
                # Strong reversal candle
                if current['close'] < current['open']:  # Bearish candle
                    return {
                        'type': 'BEARISH_SWEEP',
                        'level': sh['price'],
                        'signal': 'SELL'
                    }
    
    return None


def identify_liquidity_zones(df: pd.DataFrame) -> dict:
    """
    Identify key liquidity zones where stop losses likely cluster.
    
    Returns:
        {
            'buy_side_liquidity': List[float],  # Above price (sell stops)
            'sell_side_liquidity': List[float], # Below price (buy stops)
        }
    """
    swing_highs, swing_lows = find_swing_points(df)
    current_price = df['close'].iloc[-1]
    
    # Buy-side liquidity = swing highs above current price (where sell stops cluster)
    buy_side = [sh['price'] for sh in swing_highs if sh['price'] > current_price]
    
    # Sell-side liquidity = swing lows below current price (where buy stops cluster)
    sell_side = [sl['price'] for sl in swing_lows if sl['price'] < current_price]
    
    return {
        'buy_side_liquidity': sorted(buy_side)[:3],  # Nearest 3 levels
        'sell_side_liquidity': sorted(sell_side, reverse=True)[:3],
    }


def get_liquidity_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Get trading signal based on liquidity analysis.
    
    Returns:
        "BUY" | "SELL" | None
    """
    swing_highs, swing_lows = find_swing_points(df)
    sweep = detect_liquidity_sweep(df, swing_highs, swing_lows)
    
    if sweep:
        return sweep['signal']
    
    return None
