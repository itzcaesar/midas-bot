"""
Trading Styles Module
Different trading approaches based on timeframes and risk profiles.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class TradingStyle(Enum):
    """Available trading styles."""
    SCALPING = "scalping"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


@dataclass
class StyleConfig:
    """Configuration for a trading style."""
    name: str
    description: str
    timeframe: str
    htf_timeframe: str  # Higher timeframe for confirmation
    
    # Target thresholds
    target_threshold: float  # % move for BUY/SELL signal
    min_confidence: float
    
    # Risk management
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    risk_reward_ratio: float
    max_hold_bars: int  # Max bars to hold position
    
    # Feature weights (which indicators matter most)
    momentum_weight: float
    trend_weight: float
    volatility_weight: float
    
    # Filters
    use_session_filter: bool
    use_news_filter: bool
    allowed_sessions: List[str]


# Pre-configured trading styles
styles: Dict[TradingStyle, StyleConfig] = {
    
    TradingStyle.SCALPING: StyleConfig(
        name="Scalping",
        description="Quick trades targeting small moves (5-15 pips). High frequency, low hold time.",
        timeframe="5m",
        htf_timeframe="15m",
        target_threshold=0.001,  # 0.1% move
        min_confidence=0.75,
        sl_atr_multiplier=1.0,
        tp_atr_multiplier=1.5,
        risk_reward_ratio=1.5,
        max_hold_bars=12,  # 1 hour max
        momentum_weight=0.5,
        trend_weight=0.2,
        volatility_weight=0.3,
        use_session_filter=True,
        use_news_filter=True,
        allowed_sessions=["LONDON", "NEW_YORK", "OVERLAP"],
    ),
    
    TradingStyle.INTRADAY: StyleConfig(
        name="Intraday",
        description="Day trades closed before market close. Medium frequency, hours hold time.",
        timeframe="15m",
        htf_timeframe="1h",
        target_threshold=0.003,  # 0.3% move
        min_confidence=0.70,
        sl_atr_multiplier=1.5,
        tp_atr_multiplier=2.5,
        risk_reward_ratio=1.67,
        max_hold_bars=24,  # 6 hours max
        momentum_weight=0.4,
        trend_weight=0.4,
        volatility_weight=0.2,
        use_session_filter=True,
        use_news_filter=True,
        allowed_sessions=["LONDON", "NEW_YORK", "OVERLAP"],
    ),
    
    TradingStyle.SWING: StyleConfig(
        name="Swing Trading",
        description="Multi-day trades capturing larger moves. Low frequency, days hold time.",
        timeframe="1h",
        htf_timeframe="4h",
        target_threshold=0.005,  # 0.5% move
        min_confidence=0.65,
        sl_atr_multiplier=2.0,
        tp_atr_multiplier=4.0,
        risk_reward_ratio=2.0,
        max_hold_bars=72,  # 3 days max
        momentum_weight=0.3,
        trend_weight=0.5,
        volatility_weight=0.2,
        use_session_filter=False,
        use_news_filter=True,
        allowed_sessions=["ALL"],
    ),
    
    TradingStyle.POSITION: StyleConfig(
        name="Position Trading",
        description="Long-term trades following major trends. Very low frequency, weeks hold time.",
        timeframe="4h",
        htf_timeframe="1d",
        target_threshold=0.01,  # 1% move
        min_confidence=0.60,
        sl_atr_multiplier=3.0,
        tp_atr_multiplier=6.0,
        risk_reward_ratio=2.0,
        max_hold_bars=42,  # 1 week max (4h bars)
        momentum_weight=0.2,
        trend_weight=0.6,
        volatility_weight=0.2,
        use_session_filter=False,
        use_news_filter=False,
        allowed_sessions=["ALL"],
    ),
}


def get_style_config(style: str) -> StyleConfig:
    """
    Get configuration for a trading style.
    
    Args:
        style: Style name ('scalping', 'intraday', 'swing', 'position')
        
    Returns:
        StyleConfig for the requested style
    """
    try:
        trading_style = TradingStyle(style.lower())
        return styles[trading_style]
    except (ValueError, KeyError):
        raise ValueError(f"Unknown trading style: {style}. Valid options: {[s.value for s in TradingStyle]}")


def get_all_styles() -> Dict[str, StyleConfig]:
    """Get all available trading styles."""
    return {style.value: config for style, config in styles.items()}


def print_style_info(style: str = None):
    """Print information about trading style(s)."""
    if style:
        config = get_style_config(style)
        styles = {style: config}
    else:
        styles = get_all_styles()
    
    for name, config in styles.items():
        print(f"\n{'='*50}")
        print(f"📊 {config.name.upper()}")
        print(f"{'='*50}")
        print(f"Description: {config.description}")
        print(f"Timeframe: {config.timeframe} (HTF: {config.htf_timeframe})")
        print(f"Target: {config.target_threshold*100:.1f}% | Confidence: {config.min_confidence*100:.0f}%")
        print(f"SL: {config.sl_atr_multiplier}x ATR | TP: {config.tp_atr_multiplier}x ATR")
        print(f"R:R Ratio: 1:{config.risk_reward_ratio:.1f}")
        print(f"Max Hold: {config.max_hold_bars} bars")
        print(f"Filters: Session={config.use_session_filter}, News={config.use_news_filter}")


# Timeframe to style mapping
TIMEFRAME_STYLE_MAP = {
    '1m': TradingStyle.SCALPING,
    '5m': TradingStyle.SCALPING,
    '15m': TradingStyle.INTRADAY,
    '30m': TradingStyle.INTRADAY,
    '1h': TradingStyle.SWING,
    '4h': TradingStyle.SWING,
    '1d': TradingStyle.POSITION,
    '1w': TradingStyle.POSITION,
}


def get_style_for_timeframe(timeframe: str) -> StyleConfig:
    """Automatically select trading style based on timeframe."""
    style = TIMEFRAME_STYLE_MAP.get(timeframe, TradingStyle.SWING)
    return styles[style]
