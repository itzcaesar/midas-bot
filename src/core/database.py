"""
Database Module for MT5Bot
SQLAlchemy-based persistence for trades, signals, and performance metrics.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()


class Trade(Base):
    """Trade record model."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(Integer, unique=True, index=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(4), nullable=False)  # BUY/SELL
    lot_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    factors = Column(Text)  # JSON string of triggering factors
    confidence = Column(Float)
    status = Column(String(10), default='OPEN')  # OPEN, CLOSED, CANCELLED
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'direction': self.direction,
            'lot_size': self.lot_size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'profit': self.profit,
            'status': self.status,
            'factors': json.loads(self.factors) if self.factors else []
        }


class Signal(Base):
    """Signal record for analysis."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(5), nullable=False)
    direction = Column(String(4), nullable=False)  # BUY/SELL/HOLD
    confidence = Column(Float)
    factors = Column(Text)  # JSON string
    executed = Column(Boolean, default=False)
    result = Column(String(10), nullable=True)  # WIN/LOSS/PENDING


class DailyStats(Base):
    """Daily performance statistics."""
    __tablename__ = 'daily_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, unique=True, index=True)
    starting_balance = Column(Float)
    ending_balance = Column(Float)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    gross_profit = Column(Float, default=0.0)
    gross_loss = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)


class Database:
    """Database manager for MT5Bot."""
    
    def __init__(self, db_path: str = "data/mt5bot.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # Trade operations
    def record_trade_entry(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        lot_size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        factors: list,
        confidence: float
    ) -> Trade:
        """Record a new trade entry."""
        with self.get_session() as session:
            trade = Trade(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                lot_size=lot_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                factors=json.dumps(factors),
                confidence=confidence,
                status='OPEN'
            )
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
    
    def record_trade_exit(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        commission: float = 0.0,
        swap: float = 0.0
    ) -> Optional[Trade]:
        """Record trade exit."""
        with self.get_session() as session:
            trade = session.query(Trade).filter(Trade.ticket == ticket).first()
            if trade:
                trade.exit_time = datetime.utcnow()
                trade.exit_price = exit_price
                trade.profit = profit
                trade.commission = commission
                trade.swap = swap
                trade.status = 'CLOSED'
                session.commit()
                session.refresh(trade)
            return trade
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades from database."""
        with self.get_session() as session:
            return session.query(Trade).filter(Trade.status == 'OPEN').all()
    
    def get_trade_by_ticket(self, ticket: int) -> Optional[Trade]:
        """Get trade by MT5 ticket number."""
        with self.get_session() as session:
            return session.query(Trade).filter(Trade.ticket == ticket).first()
    
    def get_trades_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Trade]:
        """Get trades within a date range."""
        with self.get_session() as session:
            return session.query(Trade).filter(
                Trade.entry_time >= start_date,
                Trade.entry_time <= end_date
            ).all()
    
    # Signal operations
    def record_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        confidence: float,
        factors: list,
        executed: bool = False
    ) -> Signal:
        """Record a trading signal."""
        with self.get_session() as session:
            signal = Signal(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                confidence=confidence,
                factors=json.dumps(factors),
                executed=executed
            )
            session.add(signal)
            session.commit()
            session.refresh(signal)
            return signal
    
    # Statistics operations
    def get_performance_stats(self, days: int = 30) -> dict:
        """Calculate performance statistics."""
        with self.get_session() as session:
            start_date = datetime.utcnow().replace(
                hour=0, minute=0, second=0
            )
            
            trades = session.query(Trade).filter(
                Trade.status == 'CLOSED',
                Trade.exit_time >= start_date
            ).all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'avg_profit': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'profit_factor': 0.0
                }
            
            wins = [t for t in trades if t.profit and t.profit > 0]
            losses = [t for t in trades if t.profit and t.profit <= 0]
            
            total_profit = sum(t.profit or 0 for t in trades)
            gross_profit = sum(t.profit for t in wins) if wins else 0
            gross_loss = abs(sum(t.profit for t in losses)) if losses else 0
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_profit': total_profit,
                'avg_profit': total_profit / len(trades) if trades else 0,
                'max_profit': max((t.profit for t in wins), default=0),
                'max_loss': min((t.profit for t in losses), default=0),
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf')
            }
