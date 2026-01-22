"""
Strategy Module
Trading strategies, styles, and signal generation.
"""
from .styles import (
    TradingStyle,
    StyleConfig,
    get_style_config,
    get_all_styles,
    get_style_for_timeframe,
    TRADING_STYLES,
)
from .signals import (
    TradingSignal,
    SignalGenerator,
    SignalDirection,
    generate_signal,
    # Backward compatibility
    MLSignal,
    StyledSignal,
    MLSignalGenerator,
    StyledSignalGenerator,
)

__all__ = [
    # Styles
    "TradingStyle",
    "StyleConfig",
    "get_style_config",
    "get_all_styles",
    "get_style_for_timeframe",
    "TRADING_STYLES",
    # Signals
    "TradingSignal",
    "SignalGenerator",
    "SignalDirection",
    "generate_signal",
    # Backward compatibility aliases
    "MLSignal",
    "StyledSignal",
    "MLSignalGenerator",
    "StyledSignalGenerator",
]
