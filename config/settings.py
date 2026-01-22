"""
MT5Bot Configuration Settings
Pydantic-based configuration with environment variable support and validation.
"""
import os
from typing import Literal, Optional
from pathlib import Path

# Try to load pydantic, fall back to simple class if not available
try:
    from pydantic import BaseSettings, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


if PYDANTIC_AVAILABLE:
    class Settings(BaseSettings):
        """Configuration settings with validation."""
        
        # MT5 Connection
        mt5_login: Optional[int] = Field(None, env='MT5_LOGIN')
        mt5_password: Optional[str] = Field(None, env='MT5_PASSWORD')
        mt5_server: Optional[str] = Field(None, env='MT5_SERVER')
        
        # Symbol Settings
        symbol: str = Field("XAUUSD", env='SYMBOL')
        dxy_symbol: str = Field("DXY", env='DXY_SYMBOL')
        timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"] = Field("M15", env='TIMEFRAME')
        
        # Risk Management
        risk_percent: float = Field(1.0, ge=0.1, le=5.0, env='RISK_PERCENT')
        max_positions: int = Field(1, ge=1, le=10, env='MAX_POSITIONS')
        stop_loss_pips: float = Field(50.0, ge=10.0, le=200.0)
        take_profit_pips: float = Field(100.0, ge=20.0, le=500.0)
        trailing_stop_pips: float = Field(30.0, ge=10.0, le=100.0)
        
        # Strategy Parameters - MACD
        macd_fast: int = Field(12, ge=5, le=20)
        macd_slow: int = Field(26, ge=15, le=50)
        macd_signal: int = Field(9, ge=5, le=20)
        
        # Strategy Parameters - Consolidation
        consolidation_lookback: int = Field(20, ge=10, le=50)
        consolidation_threshold: float = Field(0.3, ge=0.1, le=1.0)
        
        # Strategy Parameters - Liquidity
        liquidity_lookback: int = Field(50, ge=20, le=100)
        swing_strength: int = Field(3, ge=2, le=10)
        
        # Strategy Parameters - DXY
        dxy_correlation_period: int = Field(20, ge=10, le=50)
        dxy_threshold: float = Field(-0.5, ge=-1.0, le=0.0)
        
        # Strategy Parameters - General
        min_confidence: float = Field(0.8, ge=0.4, le=1.0, env='MIN_CONFIDENCE')
        min_factors: int = Field(3, ge=2, le=5)
        
        # Multi-Timeframe
        htf_enabled: bool = Field(True)
        htf_confirmation_required: bool = Field(True)
        
        # Session Filter
        session_filter_enabled: bool = Field(True)
        allowed_sessions: str = Field("LONDON,NEW_YORK,OVERLAP")
        
        # News Filter
        news_filter_enabled: bool = Field(True)
        news_buffer_minutes: int = Field(30, ge=10, le=120)
        
        # Execution
        dry_run: bool = Field(True, env='DRY_RUN')
        loop_interval_seconds: int = Field(60, ge=10, le=300)
        
        # Notifications
        telegram_enabled: bool = Field(False)
        telegram_bot_token: Optional[str] = Field(None, env='TELEGRAM_BOT_TOKEN')
        telegram_chat_id: Optional[str] = Field(None, env='TELEGRAM_CHAT_ID')
        
        # Database
        database_url: str = Field("sqlite:///data/mt5bot.db", env='DATABASE_URL')
        
        # Logging
        log_level: str = Field("INFO")
        log_dir: str = Field("logs")
        
        @validator('macd_slow')
        def validate_macd_slow(cls, v, values):
            if 'macd_fast' in values and v <= values['macd_fast']:
                raise ValueError('macd_slow must be greater than macd_fast')
            return v
        
        def get_allowed_sessions_list(self) -> list:
            """Parse allowed sessions string to list."""
            return [s.strip().upper() for s in self.allowed_sessions.split(',')]
        
        class Config:
            env_file = '.env'
            env_file_encoding = 'utf-8'
            case_sensitive = False
    
    # Create singleton settings instance
    settings = Settings()

else:
    # Fallback simple settings class
    class Settings:
        """Simple settings without pydantic validation."""
        
        def __init__(self):
            # MT5 Connection
            self.mt5_login = os.getenv('MT5_LOGIN')
            self.mt5_password = os.getenv('MT5_PASSWORD')
            self.mt5_server = os.getenv('MT5_SERVER')
            
            # Symbol Settings
            self.symbol = os.getenv('SYMBOL', 'XAUUSD')
            self.dxy_symbol = os.getenv('DXY_SYMBOL', 'DXY')
            self.timeframe = os.getenv('TIMEFRAME', 'M15')
            
            # Risk Management
            self.risk_percent = float(os.getenv('RISK_PERCENT', '1.0'))
            self.max_positions = int(os.getenv('MAX_POSITIONS', '1'))
            self.stop_loss_pips = 50.0
            self.take_profit_pips = 100.0
            self.trailing_stop_pips = 30.0
            
            # Strategy Parameters - MACD
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            
            # Strategy Parameters - Consolidation
            self.consolidation_lookback = 20
            self.consolidation_threshold = 0.3
            
            # Strategy Parameters - Liquidity
            self.liquidity_lookback = 50
            self.swing_strength = 3
            
            # Strategy Parameters - DXY
            self.dxy_correlation_period = 20
            self.dxy_threshold = -0.5
            
            # Strategy Parameters - General
            self.min_confidence = 0.6
            self.min_factors = 3
            
            # Multi-Timeframe
            self.htf_enabled = True
            self.htf_confirmation_required = True
            
            # Session Filter
            self.session_filter_enabled = True
            self.allowed_sessions = "LONDON,NEW_YORK,OVERLAP"
            
            # News Filter
            self.news_filter_enabled = True
            self.news_buffer_minutes = 30
            
            # Execution
            self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
            self.loop_interval_seconds = 60
            
            # Notifications
            self.telegram_enabled = False
            self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            # Database
            self.database_url = os.getenv('DATABASE_URL', 'sqlite:///data/mt5bot.db')
            
            # Logging
            self.log_level = 'INFO'
            self.log_dir = 'logs'
        
        def get_allowed_sessions_list(self) -> list:
            """Parse allowed sessions string to list."""
            return [s.strip().upper() for s in self.allowed_sessions.split(',')]
    
    settings = Settings()


# Export for backwards compatibility
SYMBOL = settings.symbol
DXY_SYMBOL = settings.dxy_symbol
TIMEFRAME = settings.timeframe
RISK_PERCENT = settings.risk_percent
MAX_POSITIONS = settings.max_positions
STOP_LOSS_PIPS = settings.stop_loss_pips
TAKE_PROFIT_PIPS = settings.take_profit_pips
MACD_FAST = settings.macd_fast
MACD_SLOW = settings.macd_slow
MACD_SIGNAL = settings.macd_signal
CONSOLIDATION_LOOKBACK = settings.consolidation_lookback
CONSOLIDATION_THRESHOLD = settings.consolidation_threshold
LIQUIDITY_LOOKBACK = settings.liquidity_lookback
SWING_STRENGTH = settings.swing_strength
DXY_CORRELATION_PERIOD = settings.dxy_correlation_period
DXY_THRESHOLD = settings.dxy_threshold
DRY_RUN = settings.dry_run
LOOP_INTERVAL_SECONDS = settings.loop_interval_seconds
