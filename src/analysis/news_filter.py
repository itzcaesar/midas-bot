"""
News Filter Module
Avoid trading during high-impact economic news events.
"""
import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import sys
sys.path.append('../..')

from config import settings
from core.logger import get_logger

logger = get_logger("mt5bot.news")

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not installed, news filter will use fallback")


class NewsEvent:
    """Represents an economic news event."""
    
    def __init__(self, title: str, country: str, impact: str, 
                 event_time: datetime, forecast: str = None, previous: str = None):
        self.title = title
        self.country = country
        self.impact = impact  # 'Low', 'Medium', 'High'
        self.event_time = event_time
        self.forecast = forecast
        self.previous = previous
    
    def __repr__(self):
        return f"NewsEvent({self.title}, {self.country}, {self.impact}, {self.event_time})"
    
    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'country': self.country,
            'impact': self.impact,
            'time': self.event_time.isoformat(),
            'forecast': self.forecast,
            'previous': self.previous,
        }


class NewsFilter:
    """
    Filter out trading during high-impact economic news events.
    Uses ForexFactory calendar or similar data sources.
    """
    
    # ForexFactory calendar API (unofficial)
    FOREXFACTORY_API = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    
    # Alternative sources
    BACKUP_APIS = [
        "https://www.forexfactory.com/calendar/week",
    ]
    
    # Countries/currencies relevant for XAUUSD
    RELEVANT_CURRENCIES = ['USD', 'EUR', 'GBP', 'CHF', 'JPY']
    
    # Known high-impact events to always avoid
    HIGH_IMPACT_EVENTS = [
        'Non-Farm Payrolls',
        'NFP',
        'FOMC',
        'Fed Interest Rate Decision',
        'CPI',
        'Core CPI',
        'GDP',
        'Initial Jobless Claims',
        'Retail Sales',
        'ISM Manufacturing PMI',
        'ISM Services PMI',
    ]
    
    def __init__(self, buffer_minutes: int = None, currencies: List[str] = None):
        """
        Initialize news filter.
        
        Args:
            buffer_minutes: Minutes before/after event to avoid trading
            currencies: List of currencies to filter on
        """
        self.buffer_minutes = buffer_minutes or settings.news_buffer_minutes
        self.currencies = currencies or self.RELEVANT_CURRENCIES
        self._cache = None
        self._cache_time = None
        self._cache_duration = timedelta(hours=4)
    
    def _fetch_calendar(self) -> List[dict]:
        """Fetch economic calendar from API."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Cannot fetch news: requests library not available")
            return []
        
        try:
            response = requests.get(self.FOREXFACTORY_API, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch news calendar: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse news calendar: {e}")
            return []
    
    def _parse_event(self, event_data: dict) -> Optional[NewsEvent]:
        """Parse event data into NewsEvent object."""
        try:
            # Parse datetime from ForexFactory format
            date_str = event_data.get('date', '')
            
            # Handle different date formats
            event_time = None
            for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                try:
                    event_time = datetime.strptime(date_str[:19], fmt[:len(date_str)-1] if len(fmt) > len(date_str) else fmt)
                    break
                except ValueError:
                    continue
            
            if event_time is None:
                return None
            
            return NewsEvent(
                title=event_data.get('title', 'Unknown Event'),
                country=event_data.get('country', 'Unknown'),
                impact=event_data.get('impact', 'Low'),
                event_time=event_time,
                forecast=event_data.get('forecast'),
                previous=event_data.get('previous'),
            )
        except Exception as e:
            logger.debug(f"Failed to parse event: {e}")
            return None
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """
        Get upcoming economic events.
        
        Args:
            hours_ahead: How many hours ahead to look
            
        Returns:
            List of NewsEvent objects
        """
        now = datetime.utcnow()
        
        # Check cache
        if self._cache and self._cache_time and (now - self._cache_time) < self._cache_duration:
            events = self._cache
        else:
            raw_events = self._fetch_calendar()
            events = []
            for event_data in raw_events:
                parsed = self._parse_event(event_data)
                if parsed:
                    events.append(parsed)
            self._cache = events
            self._cache_time = now
        
        # Filter events
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []
        
        for event in events:
            # Check if event is in the future and within range
            if now <= event.event_time <= cutoff:
                # Check if event is relevant
                if event.country in self.currencies:
                    upcoming.append(event)
        
        # Sort by time
        upcoming.sort(key=lambda e: e.event_time)
        
        return upcoming
    
    def get_high_impact_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """Get only high-impact upcoming events."""
        events = self.get_upcoming_events(hours_ahead)
        return [e for e in events if e.impact.lower() == 'high']
    
    def is_near_high_impact_event(self) -> Tuple[bool, Optional[NewsEvent], int]:
        """
        Check if we're within the buffer zone of a high-impact event.
        
        Returns:
            Tuple of (is_near_event, event, minutes_until)
        """
        now = datetime.utcnow()
        buffer = timedelta(minutes=self.buffer_minutes)
        
        # Check upcoming high-impact events
        high_impact_events = self.get_high_impact_events(hours_ahead=2)
        
        for event in high_impact_events:
            time_until_event = event.event_time - now
            
            # Check if we're within the buffer before the event
            if timedelta(0) <= time_until_event <= buffer:
                minutes = int(time_until_event.total_seconds() / 60)
                logger.info(f"High-impact event in {minutes} minutes: {event.title}")
                return True, event, minutes
            
            # Check if we're within the buffer after the event
            if -buffer <= time_until_event < timedelta(0):
                minutes = int(abs(time_until_event.total_seconds()) / 60)
                logger.info(f"High-impact event occurred {minutes} minutes ago: {event.title}")
                return True, event, -minutes
        
        return False, None, 0
    
    def is_known_high_impact(self, event_title: str) -> bool:
        """Check if event title matches known high-impact events."""
        title_lower = event_title.lower()
        return any(known.lower() in title_lower for known in self.HIGH_IMPACT_EVENTS)
    
    def is_safe_to_trade(self) -> Tuple[bool, str]:
        """
        Check if it's safe to trade (no imminent high-impact news).
        
        Returns:
            Tuple of (is_safe, reason)
        """
        is_near, event, minutes = self.is_near_high_impact_event()
        
        if is_near and event:
            if minutes > 0:
                reason = f"High-impact event '{event.title}' in {minutes} minutes"
            else:
                reason = f"High-impact event '{event.title}' occurred {abs(minutes)} minutes ago"
            return False, reason
        
        return True, "No high-impact events nearby"
    
    def get_next_event(self) -> Optional[NewsEvent]:
        """Get the next upcoming event."""
        events = self.get_upcoming_events(hours_ahead=48)
        return events[0] if events else None
    
    def get_filter_status(self) -> dict:
        """Get comprehensive news filter status."""
        is_safe, reason = self.is_safe_to_trade()
        upcoming = self.get_high_impact_events(hours_ahead=24)
        
        return {
            'is_safe_to_trade': is_safe,
            'reason': reason,
            'buffer_minutes': self.buffer_minutes,
            'upcoming_high_impact': [e.to_dict() for e in upcoming[:5]],
            'next_event': self.get_next_event().to_dict() if self.get_next_event() else None,
        }


class ManualNewsFilter:
    """
    Manual news filter for when API is unavailable.
    Uses a predefined schedule of known recurring events.
    """
    
    # Weekly recurring events (day_of_week, hour_utc, title)
    RECURRING_EVENTS = [
        (3, 13, 30, "Initial Jobless Claims"),     # Thursday 13:30 UTC
        (4, 13, 30, "Non-Farm Payrolls"),          # First Friday of month
        (2, 15, 0, "FOMC Meeting Minutes"),        # Wednesday (after FOMC)
    ]
    
    # Monthly events (approximate dates)
    MONTHLY_EVENTS = {
        'NFP': {'week': 1, 'day': 4, 'hour': 13, 'minute': 30},
        'CPI': {'week': 2, 'day': 2, 'hour': 13, 'minute': 30},
        'Retail Sales': {'week': 2, 'day': 3, 'hour': 13, 'minute': 30},
    }
    
    def __init__(self, buffer_minutes: int = 30):
        self.buffer_minutes = buffer_minutes
    
    def is_nfp_day(self) -> bool:
        """Check if today is NFP (Non-Farm Payrolls) day (first Friday of month)."""
        now = datetime.utcnow()
        if now.weekday() != 4:  # Not Friday
            return False
        if now.day > 7:  # Not first week
            return False
        return True
    
    def is_near_nfp(self) -> bool:
        """Check if we're near NFP release time."""
        if not self.is_nfp_day():
            return False
        
        now = datetime.utcnow()
        nfp_time = now.replace(hour=13, minute=30, second=0)
        buffer = timedelta(minutes=self.buffer_minutes)
        
        return (nfp_time - buffer) <= now <= (nfp_time + buffer)
    
    def is_safe_to_trade(self) -> Tuple[bool, str]:
        """Simple check for known high-impact events."""
        if self.is_near_nfp():
            return False, "Near NFP release (first Friday high-impact)"
        
        return True, "No known high-impact events (manual filter)"
