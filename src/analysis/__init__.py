"""
Analysis Module - Technical Analysis and Market Filters
"""
from .indicators import (
    calculate_macd,
    calculate_atr,
    detect_consolidation,
    detect_breakout,
    detect_structure,
    get_macd_signal
)
from .liquidity import (
    find_swing_points,
    detect_liquidity_sweep,
    identify_liquidity_zones,
    get_liquidity_signal
)
from .dxy import (
    calculate_correlation,
    get_dxy_from_yfinance,
    analyze_dxy_divergence,
    get_dxy_signal
)
from .mtf_analysis import MultiTimeframeAnalyzer
from .session_filter import SessionFilter, TradingSessionPresets
from .news_filter import NewsFilter, ManualNewsFilter

__all__ = [
    # Indicators
    'calculate_macd',
    'calculate_atr',
    'detect_consolidation',
    'detect_breakout',
    'detect_structure',
    'get_macd_signal',
    # Liquidity
    'find_swing_points',
    'detect_liquidity_sweep',
    'identify_liquidity_zones',
    'get_liquidity_signal',
    # DXY
    'calculate_correlation',
    'get_dxy_from_yfinance',
    'analyze_dxy_divergence',
    'get_dxy_signal',
    # MTF
    'MultiTimeframeAnalyzer',
    # Session
    'SessionFilter',
    'TradingSessionPresets',
    # News
    'NewsFilter',
    'ManualNewsFilter',
]
