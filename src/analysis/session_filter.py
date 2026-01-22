"""
Session Filter Module
Filter trades based on market sessions for optimal liquidity and volatility.
"""
from datetime import datetime, time, timedelta
from typing import Tuple, List, Optional



from config import settings
from core.logger import get_logger

logger = get_logger("mt5bot.session")


# Try to import zoneinfo (Python 3.9+) or fall back to pytz
try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from pytz import timezone as ZoneInfo
    except ImportError:
        ZoneInfo = None


class SessionFilter:
    """
    Filter trades based on major forex market sessions.
    
    Gold (XAUUSD) is most active during:
    - London session: 08:00-16:00 UTC
    - New York session: 13:00-21:00 UTC
    - London/NY overlap: 13:00-16:00 UTC (highest volatility)
    """
    
    # Session times in UTC
    SESSIONS = {
        'SYDNEY': {
            'start': time(21, 0),  # 21:00 UTC (Sunday night)
            'end': time(6, 0),     # 06:00 UTC
            'description': 'Sydney/Tokyo Asian session',
            'volatility': 'LOW',
        },
        'TOKYO': {
            'start': time(0, 0),   # 00:00 UTC
            'end': time(9, 0),     # 09:00 UTC
            'description': 'Tokyo session',
            'volatility': 'LOW',
        },
        'LONDON': {
            'start': time(8, 0),   # 08:00 UTC
            'end': time(16, 0),    # 16:00 UTC
            'description': 'London session',
            'volatility': 'HIGH',
        },
        'NEW_YORK': {
            'start': time(13, 0),  # 13:00 UTC
            'end': time(21, 0),    # 21:00 UTC
            'description': 'New York session',
            'volatility': 'HIGH',
        },
        'OVERLAP': {
            'start': time(13, 0),  # 13:00 UTC
            'end': time(16, 0),    # 16:00 UTC
            'description': 'London/NY overlap - highest liquidity',
            'volatility': 'VERY_HIGH',
        },
    }
    
    # Days when forex market is closed (no trading)
    WEEKEND = [5, 6]  # Saturday=5, Sunday=6
    
    def __init__(self, allowed_sessions: List[str] = None, timezone: str = 'UTC'):
        """
        Initialize session filter.
        
        Args:
            allowed_sessions: List of allowed session names
            timezone: Timezone for time calculations (default: UTC)
        """
        self.allowed_sessions = allowed_sessions or settings.get_allowed_sessions_list()
        self.timezone = timezone
    
    def get_utc_now(self) -> datetime:
        """Get current UTC time."""
        if ZoneInfo:
            return datetime.now(ZoneInfo('UTC'))
        return datetime.utcnow()
    
    def get_current_session(self) -> Tuple[Optional[str], dict]:
        """
        Determine the current active session(s).
        
        Returns:
            Tuple of (primary_session, session_details)
        """
        utc_now = self.get_utc_now()
        current_time = utc_now.time() if hasattr(utc_now, 'time') else time(
            utc_now.hour, utc_now.minute
        )
        
        active_sessions = []
        
        for session_name, session_info in self.SESSIONS.items():
            start = session_info['start']
            end = session_info['end']
            
            # Handle sessions that span midnight
            if start > end:
                # Session spans midnight (e.g., Sydney)
                if current_time >= start or current_time <= end:
                    active_sessions.append(session_name)
            else:
                if start <= current_time <= end:
                    active_sessions.append(session_name)
        
        if not active_sessions:
            return None, {'description': 'Market closed or between sessions'}
        
        # Prioritize overlap, then London, then New York
        priority = ['OVERLAP', 'LONDON', 'NEW_YORK', 'TOKYO', 'SYDNEY']
        for session in priority:
            if session in active_sessions:
                return session, self.SESSIONS[session]
        
        return active_sessions[0], self.SESSIONS[active_sessions[0]]
    
    def is_market_open(self) -> bool:
        """Check if the forex market is open (not weekend)."""
        utc_now = self.get_utc_now()
        weekday = utc_now.weekday() if hasattr(utc_now, 'weekday') else datetime.utcnow().weekday()
        
        # Market closes Friday ~21:00 UTC and opens Sunday ~21:00 UTC
        if weekday == 4:  # Friday
            current_time = utc_now.time() if hasattr(utc_now, 'time') else time(utc_now.hour, utc_now.minute)
            if current_time > time(21, 0):
                return False
        elif weekday == 5:  # Saturday
            return False
        elif weekday == 6:  # Sunday
            current_time = utc_now.time() if hasattr(utc_now, 'time') else time(utc_now.hour, utc_now.minute)
            if current_time < time(21, 0):
                return False
        
        return True
    
    def is_valid_session(self) -> Tuple[bool, str]:
        """
        Check if current time is within allowed trading sessions.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if market is open
        if not self.is_market_open():
            return False, "Market closed (weekend)"
        
        # Get current session
        current_session, details = self.get_current_session()
        
        if current_session is None:
            return False, "No active session"
        
        # Check if current session is in allowed list
        if current_session in self.allowed_sessions:
            return True, f"Trading in {current_session} session ({details['description']})"
        
        # Check if we're in an allowed session (might be multiple active)
        for session in self.allowed_sessions:
            session_info = self.SESSIONS.get(session)
            if session_info:
                start = session_info['start']
                end = session_info['end']
                utc_now = self.get_utc_now()
                current_time = utc_now.time() if hasattr(utc_now, 'time') else time(utc_now.hour, utc_now.minute)
                
                if start > end:
                    if current_time >= start or current_time <= end:
                        return True, f"Trading in {session} session"
                else:
                    if start <= current_time <= end:
                        return True, f"Trading in {session} session"
        
        return False, f"Current session '{current_session}' not in allowed sessions: {self.allowed_sessions}"
    
    def time_until_next_session(self) -> Tuple[str, timedelta]:
        """
        Calculate time until the next allowed trading session.
        
        Returns:
            Tuple of (next_session_name, time_until)
        """
        utc_now = self.get_utc_now()
        current_time = utc_now.time() if hasattr(utc_now, 'time') else time(utc_now.hour, utc_now.minute)
        
        min_wait = timedelta(days=7)
        next_session = None
        
        for session_name in self.allowed_sessions:
            session_info = self.SESSIONS.get(session_name)
            if not session_info:
                continue
            
            start = session_info['start']
            
            # Calculate time until this session starts
            if current_time < start:
                # Session starts later today
                wait = datetime.combine(datetime.today(), start) - datetime.combine(datetime.today(), current_time)
            else:
                # Session starts tomorrow
                wait = timedelta(days=1) - (
                    datetime.combine(datetime.today(), current_time) - 
                    datetime.combine(datetime.today(), start)
                )
            
            if wait < min_wait:
                min_wait = wait
                next_session = session_name
        
        return next_session, min_wait
    
    def get_session_info(self) -> dict:
        """
        Get comprehensive session information.
        
        Returns:
            Dictionary with session details
        """
        current_session, details = self.get_current_session()
        is_valid, reason = self.is_valid_session()
        next_session, time_until = self.time_until_next_session()
        
        return {
            'market_open': self.is_market_open(),
            'current_session': current_session,
            'session_details': details,
            'is_valid_session': is_valid,
            'reason': reason,
            'next_session': next_session,
            'time_until_next': str(time_until),
            'allowed_sessions': self.allowed_sessions,
        }
    
    def get_volatility_level(self) -> str:
        """
        Get current market volatility level based on session.
        
        Returns:
            'LOW', 'MEDIUM', 'HIGH', or 'VERY_HIGH'
        """
        current_session, details = self.get_current_session()
        
        if current_session is None:
            return 'LOW'
        
        return details.get('volatility', 'MEDIUM')


# Pre-configured session filters
class TradingSessionPresets:
    """Pre-configured session filter presets for different trading styles."""
    
    @staticmethod
    def gold_optimal() -> SessionFilter:
        """Optimal sessions for XAUUSD trading (London/NY overlap)."""
        return SessionFilter(allowed_sessions=['OVERLAP', 'LONDON', 'NEW_YORK'])
    
    @staticmethod
    def london_only() -> SessionFilter:
        """London session only."""
        return SessionFilter(allowed_sessions=['LONDON', 'OVERLAP'])
    
    @staticmethod
    def new_york_only() -> SessionFilter:
        """New York session only."""
        return SessionFilter(allowed_sessions=['NEW_YORK', 'OVERLAP'])
    
    @staticmethod
    def all_sessions() -> SessionFilter:
        """Trade in all sessions (24/5)."""
        return SessionFilter(allowed_sessions=['SYDNEY', 'TOKYO', 'LONDON', 'NEW_YORK', 'OVERLAP'])
    
    @staticmethod
    def high_volatility_only() -> SessionFilter:
        """Only trade during high volatility periods."""
        return SessionFilter(allowed_sessions=['OVERLAP'])
