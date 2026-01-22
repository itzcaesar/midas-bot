"""
DXY Correlation Analysis - USD Index correlation with XAUUSD
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import sys
sys.path.append('../..')
from config import settings

# Optional: fallback to yfinance if DXY not available on MT5
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def calculate_correlation(xau_df: pd.DataFrame, dxy_df: pd.DataFrame) -> pd.Series:
    """
    Calculate rolling correlation between XAUUSD and DXY.
    
    Args:
        xau_df: XAUUSD OHLCV DataFrame
        dxy_df: DXY OHLCV DataFrame
        
    Returns:
        Series of rolling correlation values
    """
    period = settings.DXY_CORRELATION_PERIOD
    
    # Align dataframes by time
    xau_returns = xau_df['close'].pct_change()
    dxy_returns = dxy_df['close'].pct_change()
    
    # Resample if needed to align timestamps
    combined = pd.concat([xau_returns, dxy_returns], axis=1, keys=['xau', 'dxy'])
    combined = combined.dropna()
    
    if len(combined) < period:
        return pd.Series([np.nan])
    
    correlation = combined['xau'].rolling(window=period).corr(combined['dxy'])
    return correlation


def get_dxy_from_yfinance(period: str = "5d", interval: str = "15m") -> Optional[pd.DataFrame]:
    """
    Fallback: Fetch DXY data from Yahoo Finance.
    
    Note: yfinance DXY is delayed and less precise than MT5 feed.
    """
    if not YFINANCE_AVAILABLE:
        print("yfinance not installed. Cannot fetch DXY from external source.")
        return None
    
    try:
        ticker = yf.Ticker("DX-Y.NYB")  # DXY futures
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None
        
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Failed to fetch DXY from yfinance: {e}")
        return None


def analyze_dxy_divergence(xau_df: pd.DataFrame, dxy_df: pd.DataFrame) -> dict:
    """
    Analyze divergence between XAUUSD and DXY.
    Gold typically moves inversely to USD.
    
    Returns:
        {
            'correlation': float,  # Current correlation
            'divergence': 'BULLISH' | 'BEARISH' | None,
            'signal_strength': 'STRONG' | 'MODERATE' | 'WEAK'
        }
    """
    correlation = calculate_correlation(xau_df, dxy_df)
    current_corr = correlation.iloc[-1] if len(correlation) > 0 else np.nan
    
    result = {
        'correlation': current_corr,
        'divergence': None,
        'signal_strength': 'WEAK'
    }
    
    if pd.isna(current_corr):
        return result
    
    # Recent price movements
    xau_change = (xau_df['close'].iloc[-1] / xau_df['close'].iloc[-5] - 1) * 100
    dxy_change = (dxy_df['close'].iloc[-1] / dxy_df['close'].iloc[-5] - 1) * 100
    
    threshold = settings.DXY_THRESHOLD
    
    # Strong negative correlation (normal) - trade with the inverse
    if current_corr < threshold:
        # DXY falling = Gold bullish
        if dxy_change < -0.1:
            result['divergence'] = 'BULLISH'
            result['signal_strength'] = 'STRONG' if dxy_change < -0.3 else 'MODERATE'
        # DXY rising = Gold bearish
        elif dxy_change > 0.1:
            result['divergence'] = 'BEARISH'
            result['signal_strength'] = 'STRONG' if dxy_change > 0.3 else 'MODERATE'
    
    # Positive correlation (divergence from norm) - potential reversal
    elif current_corr > 0:
        # Both rising - Gold may reverse down
        if xau_change > 0 and dxy_change > 0:
            result['divergence'] = 'BEARISH'
            result['signal_strength'] = 'WEAK'
        # Both falling - Gold may reverse up
        elif xau_change < 0 and dxy_change < 0:
            result['divergence'] = 'BULLISH'
            result['signal_strength'] = 'WEAK'
    
    return result


def get_dxy_signal(xau_df: pd.DataFrame, dxy_df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Get trading signal based on DXY correlation.
    
    Returns:
        (signal, strength) where signal is "BUY" | "SELL" | None
    """
    analysis = analyze_dxy_divergence(xau_df, dxy_df)
    
    if analysis['divergence'] == 'BULLISH':
        return 'BUY', analysis['signal_strength']
    elif analysis['divergence'] == 'BEARISH':
        return 'SELL', analysis['signal_strength']
    
    return None, 'WEAK'
