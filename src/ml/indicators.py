"""
Advanced Technical Indicators Module
Additional indicators for enhanced ML signal analysis.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Ichimoku Cloud indicators.
    - Tenkan-sen (Conversion Line): 9-period
    - Kijun-sen (Base Line): 26-period
    - Senkou Span A & B (Cloud)
    - Chikou Span (Lagging Span)
    """
    # Tenkan-sen (9-period)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['ichimoku_tenkan'] = (high_9 + low_9) / 2
    
    # Kijun-sen (26-period)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['ichimoku_kijun'] = (high_26 + low_26) / 2
    
    # Senkou Span A (shifted 26 periods)
    df['ichimoku_span_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
    
    # Senkou Span B (52-period, shifted 26 periods)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['ichimoku_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    # Cloud features
    df['ichimoku_above_cloud'] = (
        (df['close'] > df['ichimoku_span_a']) & 
        (df['close'] > df['ichimoku_span_b'])
    ).astype(int)
    
    df['ichimoku_below_cloud'] = (
        (df['close'] < df['ichimoku_span_a']) & 
        (df['close'] < df['ichimoku_span_b'])
    ).astype(int)
    
    df['ichimoku_cloud_thickness'] = abs(df['ichimoku_span_a'] - df['ichimoku_span_b']) / df['close']
    
    # TK Cross (bullish when Tenkan crosses above Kijun)
    df['ichimoku_tk_cross'] = (df['ichimoku_tenkan'] > df['ichimoku_kijun']).astype(int)
    
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average Directional Index (ADX) for trend strength.
    - ADX: Trend strength (>25 = trending)
    - +DI: Bullish pressure
    - -DI: Bearish pressure
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # Trend signals
    df['adx_trending'] = (df['adx'] > 25).astype(int)
    df['adx_strong_trend'] = (df['adx'] > 40).astype(int)
    df['adx_bullish'] = ((df['plus_di'] > df['minus_di']) & (df['adx'] > 20)).astype(int)
    df['adx_bearish'] = ((df['minus_di'] > df['plus_di']) & (df['adx'] > 20)).astype(int)
    
    return df


def add_cci(df: pd.DataFrame, periods: List[int] = [14, 20]) -> pd.DataFrame:
    """
    Add Commodity Channel Index (CCI).
    - Measures price deviation from statistical mean
    - >100 = overbought, <-100 = oversold
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    for period in periods:
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'cci_{period}'] = (typical_price - sma) / (0.015 * mad + 1e-10)
    
    # CCI signals
    df['cci_overbought'] = (df['cci_14'] > 100).astype(int)
    df['cci_oversold'] = (df['cci_14'] < -100).astype(int)
    df['cci_extreme_ob'] = (df['cci_14'] > 200).astype(int)
    df['cci_extreme_os'] = (df['cci_14'] < -200).astype(int)
    
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add On-Balance Volume (OBV) for volume-price relationship.
    """
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        # No volume data
        df['obv'] = 0
        df['obv_sma'] = 0
        df['obv_trend'] = 0
        return df
    
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    df['obv'] = obv
    df['obv_sma'] = df['obv'].rolling(window=20).mean()
    df['obv_trend'] = (df['obv'] > df['obv_sma']).astype(int)
    
    return df


def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Pivot Points (standard formula).
    Uses previous bar's high, low, close.
    """
    # Previous bar values
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Pivot Point
    df['pivot'] = (prev_high + prev_low + prev_close) / 3
    
    # Support levels
    df['support_1'] = 2 * df['pivot'] - prev_high
    df['support_2'] = df['pivot'] - (prev_high - prev_low)
    
    # Resistance levels
    df['resistance_1'] = 2 * df['pivot'] - prev_low
    df['resistance_2'] = df['pivot'] + (prev_high - prev_low)
    
    # Price relative to levels
    df['above_pivot'] = (df['close'] > df['pivot']).astype(int)
    df['near_support'] = (df['close'] < df['support_1'] * 1.005).astype(int)
    df['near_resistance'] = (df['close'] > df['resistance_1'] * 0.995).astype(int)
    
    return df


def add_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Add Fibonacci retracement levels.
    """
    # Rolling high and low
    rolling_high = df['high'].rolling(window=lookback).max()
    rolling_low = df['low'].rolling(window=lookback).min()
    
    price_range = rolling_high - rolling_low
    
    # Fibonacci levels
    df['fib_236'] = rolling_high - 0.236 * price_range
    df['fib_382'] = rolling_high - 0.382 * price_range
    df['fib_500'] = rolling_high - 0.500 * price_range
    df['fib_618'] = rolling_high - 0.618 * price_range
    df['fib_786'] = rolling_high - 0.786 * price_range
    
    # Position relative to Fib levels
    df['fib_position'] = (df['close'] - rolling_low) / (price_range + 1e-10)
    
    # Near key levels
    df['near_fib_382'] = (abs(df['close'] - df['fib_382']) / df['close'] < 0.005).astype(int)
    df['near_fib_618'] = (abs(df['close'] - df['fib_618']) / df['close'] < 0.005).astype(int)
    
    return df


def add_divergences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect RSI and MACD divergences.
    - Bullish divergence: Price lower low, RSI higher low
    - Bearish divergence: Price higher high, RSI lower high
    """
    lookback = 14
    
    # Price swing detection
    df['price_higher_high'] = (
        (df['high'] > df['high'].shift(lookback)) & 
        (df['high'].shift(1) < df['high'].shift(lookback + 1))
    ).astype(int)
    
    df['price_lower_low'] = (
        (df['low'] < df['low'].shift(lookback)) & 
        (df['low'].shift(1) > df['low'].shift(lookback + 1))
    ).astype(int)
    
    # RSI divergence
    if 'rsi_14' in df.columns:
        df['rsi_higher_low'] = (df['rsi_14'] > df['rsi_14'].shift(lookback)).astype(int)
        df['rsi_lower_high'] = (df['rsi_14'] < df['rsi_14'].shift(lookback)).astype(int)
        
        # Bullish: Price lower low + RSI higher low
        df['rsi_bullish_div'] = (df['price_lower_low'] & df['rsi_higher_low']).astype(int)
        # Bearish: Price higher high + RSI lower high
        df['rsi_bearish_div'] = (df['price_higher_high'] & df['rsi_lower_high']).astype(int)
    
    # MACD divergence
    if 'macd_hist' in df.columns:
        df['macd_higher_low'] = (df['macd_hist'] > df['macd_hist'].shift(lookback)).astype(int)
        df['macd_lower_high'] = (df['macd_hist'] < df['macd_hist'].shift(lookback)).astype(int)
        
        df['macd_bullish_div'] = (df['price_lower_low'] & df['macd_higher_low']).astype(int)
        df['macd_bearish_div'] = (df['price_higher_high'] & df['macd_lower_high']).astype(int)
    
    return df


def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect market structure: Higher Highs, Higher Lows, etc.
    """
    lookback = 5
    
    # Swing highs and lows
    df['swing_high'] = (
        (df['high'] > df['high'].shift(1)) & 
        (df['high'] > df['high'].shift(-1))
    ).astype(int)
    
    df['swing_low'] = (
        (df['low'] < df['low'].shift(1)) & 
        (df['low'] < df['low'].shift(-1))
    ).astype(int)
    
    # Higher highs / Lower lows
    rolling_high = df['high'].rolling(window=lookback).max()
    rolling_low = df['low'].rolling(window=lookback).min()
    
    df['higher_high'] = (df['high'] > rolling_high.shift(lookback)).astype(int)
    df['lower_low'] = (df['low'] < rolling_low.shift(lookback)).astype(int)
    df['higher_low'] = (df['low'] > rolling_low.shift(lookback)).astype(int)
    df['lower_high'] = (df['high'] < rolling_high.shift(lookback)).astype(int)
    
    # Trend identification
    df['uptrend_structure'] = (df['higher_high'] & df['higher_low']).astype(int)
    df['downtrend_structure'] = (df['lower_low'] & df['lower_high']).astype(int)
    
    return df


def add_support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect dynamic support and resistance levels.
    """
    # Rolling support/resistance
    df['resistance_level'] = df['high'].rolling(window=lookback).max()
    df['support_level'] = df['low'].rolling(window=lookback).min()
    
    # Distance to levels
    df['dist_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
    df['dist_to_support'] = (df['close'] - df['support_level']) / df['close']
    
    # Breaking levels
    df['breaking_resistance'] = (df['close'] > df['resistance_level'].shift(1)).astype(int)
    df['breaking_support'] = (df['close'] < df['support_level'].shift(1)).astype(int)
    
    # Near levels
    df['near_resistance_level'] = (df['dist_to_resistance'] < 0.005).astype(int)
    df['near_support_level'] = (df['dist_to_support'] < 0.005).astype(int)
    
    return df


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect candlestick patterns.
    """
    # Body and wick calculations
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    # Doji (small body)
    df['doji'] = (body / (total_range + 1e-10) < 0.1).astype(int)
    
    # Hammer (long lower wick, small body at top)
    df['hammer'] = (
        (lower_wick > 2 * body) & 
        (upper_wick < body) &
        (df['close'] > df['open'])
    ).astype(int)
    
    # Shooting Star (long upper wick, small body at bottom)
    df['shooting_star'] = (
        (upper_wick > 2 * body) & 
        (lower_wick < body) &
        (df['close'] < df['open'])
    ).astype(int)
    
    # Engulfing patterns
    prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
    
    df['bullish_engulfing'] = (
        (df['close'] > df['open']) &  # Current bullish
        (df['close'].shift(1) < df['open'].shift(1)) &  # Previous bearish
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    ).astype(int)
    
    df['bearish_engulfing'] = (
        (df['close'] < df['open']) &  # Current bearish
        (df['close'].shift(1) > df['open'].shift(1)) &  # Previous bullish
        (df['close'] < df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1))
    ).astype(int)
    
    # Three white soldiers / Three black crows
    df['three_white_soldiers'] = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'].shift(2) > df['open'].shift(2)) &
        (df['close'] > df['close'].shift(1)) &
        (df['close'].shift(1) > df['close'].shift(2))
    ).astype(int)
    
    df['three_black_crows'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'].shift(2) < df['open'].shift(2)) &
        (df['close'] < df['close'].shift(1)) &
        (df['close'].shift(1) < df['close'].shift(2))
    ).astype(int)
    
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all advanced technical indicators.
    """
    df = add_ichimoku(df)
    df = add_adx(df)
    df = add_cci(df)
    df = add_obv(df)
    df = add_pivot_points(df)
    df = add_fibonacci_levels(df)
    df = add_divergences(df)
    df = add_market_structure(df)
    df = add_support_resistance(df)
    df = add_candle_patterns(df)
    
    return df
