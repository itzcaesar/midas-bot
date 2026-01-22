"""
Professional Logging System for MT5Bot
Provides structured logging with console and file output, rotation, and formatting.
"""
import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional


# Global loggers cache
_loggers: dict[str, logging.Logger] = {}

# Log format templates
CONSOLE_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-15s │ %(message)s"
FILE_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-15s │ %(funcName)s:%(lineno)d │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Add color codes
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Create colored level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.
    
    Args:
        name: Logger name (e.g., 'mt5bot.broker', 'mt5bot.strategy')
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console: Enable console output
        file: Enable file output
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Check cache first
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(CONSOLE_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with date
        log_file = log_path / f"{name.replace('.', '_')}_{datetime.now():%Y%m%d}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Cache and return
    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with defaults.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class TradeLogger:
    """Specialized logger for trade events with structured output."""
    
    def __init__(self, log_dir: str = "logs"):
        self.logger = setup_logger("mt5bot.trades", log_dir=log_dir)
        self.trade_log_path = Path(log_dir) / "trades.csv"
        self._init_trade_log()
    
    def _init_trade_log(self):
        """Initialize CSV trade log if it doesn't exist."""
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, 'w') as f:
                f.write("timestamp,ticket,symbol,direction,lot,entry_price,sl,tp,exit_price,profit,factors\n")
    
    def log_entry(self, ticket: int, symbol: str, direction: str, lot: float, 
                  entry_price: float, sl: float, tp: float, factors: list):
        """Log trade entry."""
        self.logger.info(
            f"ENTRY: {direction} {symbol} @ {entry_price:.2f} | "
            f"Lot: {lot} | SL: {sl:.2f} | TP: {tp:.2f} | Ticket: {ticket}"
        )
        
        # Append to CSV
        with open(self.trade_log_path, 'a') as f:
            factors_str = ';'.join(factors).replace(',', '|')
            f.write(f"{datetime.now()},{ticket},{symbol},{direction},{lot},{entry_price},{sl},{tp},,,\"{factors_str}\"\n")
    
    def log_exit(self, ticket: int, symbol: str, direction: str, 
                 exit_price: float, profit: float, reason: str):
        """Log trade exit."""
        emoji = "✅" if profit > 0 else "❌"
        self.logger.info(
            f"EXIT {emoji}: {direction} {symbol} @ {exit_price:.2f} | "
            f"P/L: ${profit:+.2f} | Reason: {reason} | Ticket: {ticket}"
        )
    
    def log_signal(self, direction: str, confidence: float, factors: list):
        """Log signal generation."""
        self.logger.info(
            f"SIGNAL: {direction} | Confidence: {confidence:.0%} | "
            f"Factors: {', '.join(factors)}"
        )
