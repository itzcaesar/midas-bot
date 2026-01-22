"""
Core Module - Logging, Database, and Utilities
"""
from .logger import setup_logger, get_logger
from .database import Database, Trade, Signal as SignalRecord
from .exceptions import (
    MT5BotError,
    ConnectionError,
    OrderError,
    InsufficientDataError,
    ConfigurationError
)

__all__ = [
    'setup_logger',
    'get_logger',
    'Database',
    'Trade',
    'SignalRecord',
    'MT5BotError',
    'ConnectionError',
    'OrderError',
    'InsufficientDataError',
    'ConfigurationError'
]
