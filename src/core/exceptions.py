"""
Custom Exceptions for MT5Bot
Provides specific error types for better error handling and debugging.
"""


class MT5BotError(Exception):
    """Base exception for all MT5Bot errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConnectionError(MT5BotError):
    """Raised when MT5 connection fails."""
    
    def __init__(self, message: str = "Failed to connect to MT5 terminal", details: dict = None):
        super().__init__(message, details)


class OrderError(MT5BotError):
    """Raised when order placement/modification fails."""
    
    def __init__(self, message: str, order_type: str = None, 
                 retcode: int = None, details: dict = None):
        details = details or {}
        if order_type:
            details['order_type'] = order_type
        if retcode:
            details['retcode'] = retcode
        super().__init__(message, details)


class InsufficientDataError(MT5BotError):
    """Raised when there's not enough data for analysis."""
    
    def __init__(self, required: int, available: int, symbol: str = None):
        details = {
            'required_bars': required,
            'available_bars': available
        }
        if symbol:
            details['symbol'] = symbol
        super().__init__(
            f"Insufficient data: need {required} bars, got {available}",
            details
        )


class ConfigurationError(MT5BotError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, param: str = None, value: any = None):
        details = {}
        if param:
            details['parameter'] = param
        if value is not None:
            details['value'] = value
        super().__init__(message, details)


class SymbolError(MT5BotError):
    """Raised when symbol operations fail."""
    
    def __init__(self, symbol: str, message: str = None):
        message = message or f"Symbol '{symbol}' not found or unavailable"
        super().__init__(message, {'symbol': symbol})


class SessionError(MT5BotError):
    """Raised when trading outside allowed sessions."""
    
    def __init__(self, current_session: str = None, allowed_sessions: list = None):
        details = {}
        if current_session:
            details['current_session'] = current_session
        if allowed_sessions:
            details['allowed_sessions'] = allowed_sessions
        super().__init__("Trading not allowed in current session", details)


class NewsFilterError(MT5BotError):
    """Raised when news filter blocks trading."""
    
    def __init__(self, event: str = None, time_until: int = None):
        details = {}
        if event:
            details['upcoming_event'] = event
        if time_until is not None:
            details['minutes_until'] = time_until
        super().__init__("High-impact news event approaching", details)


class RiskLimitError(MT5BotError):
    """Raised when risk limits are exceeded."""
    
    def __init__(self, limit_type: str, current: float, maximum: float):
        details = {
            'limit_type': limit_type,
            'current_value': current,
            'maximum_allowed': maximum
        }
        super().__init__(
            f"Risk limit exceeded: {limit_type} ({current:.2f} > {maximum:.2f})",
            details
        )


class BacktestError(MT5BotError):
    """Raised when backtesting encounters an error."""
    
    def __init__(self, message: str, bar_index: int = None, details: dict = None):
        details = details or {}
        if bar_index is not None:
            details['bar_index'] = bar_index
        super().__init__(message, details)
