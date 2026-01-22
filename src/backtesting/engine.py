"""
Backtesting Engine
Historical simulation engine for strategy validation and performance analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field



from config import settings
from core.logger import get_logger

logger = get_logger("mt5bot.backtest")


@dataclass
class BacktestTrade:
    """Represents a trade in backtesting."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    lot_size: float
    profit: float = 0.0
    factors: List[str] = field(default_factory=list)
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED_OUT, TP_HIT
    
    @property
    def pips(self) -> float:
        """Calculate profit in pips."""
        if self.exit_price is None:
            return 0.0
        
        if self.direction == "BUY":
            return (self.exit_price - self.entry_price) * 10
        else:
            return (self.entry_price - self.exit_price) * 10
    
    @property
    def risk_reward_actual(self) -> float:
        """Calculate actual risk/reward achieved."""
        if self.stop_loss == 0:
            return 0.0
        
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        
        reward = abs(self.profit) if self.profit > 0 else 0
        return reward / risk if reward else 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[BacktestTrade]
    equity_curve: List[float]
    initial_balance: float
    final_balance: float
    start_date: datetime
    end_date: datetime
    params: Dict = field(default_factory=dict)
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if t.profit > 0])
    
    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if t.profit <= 0])
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def total_profit(self) -> float:
        return sum(t.profit for t in self.trades)
    
    @property
    def gross_profit(self) -> float:
        return sum(t.profit for t in self.trades if t.profit > 0)
    
    @property
    def gross_loss(self) -> float:
        return abs(sum(t.profit for t in self.trades if t.profit < 0))
    
    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0
        
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.profit / self.initial_balance for t in self.trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming ~250 trading days)
        return mean_return / std_return * np.sqrt(250)
    
    @property
    def avg_win(self) -> float:
        wins = [t.profit for t in self.trades if t.profit > 0]
        return np.mean(wins) if wins else 0.0
    
    @property
    def avg_loss(self) -> float:
        losses = [t.profit for t in self.trades if t.profit < 0]
        return np.mean(losses) if losses else 0.0
    
    @property
    def expectancy(self) -> float:
        """Calculate trade expectancy."""
        if self.total_trades == 0:
            return 0.0
        return (self.win_rate * self.avg_win) + ((1 - self.win_rate) * self.avg_loss)
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100
    
    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'total_profit': round(self.total_profit, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown * 100, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'expectancy': round(self.expectancy, 2),
            'return_pct': round(self.return_pct, 2),
            'initial_balance': self.initial_balance,
            'final_balance': round(self.final_balance, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
        }
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.final_balance:,.2f}")
        print(f"Return: {self.return_pct:+.2f}%")
        print("-" * 50)
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.win_rate * 100:.1f}%")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print(f"Max Drawdown: {self.max_drawdown * 100:.1f}%")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Expectancy: ${self.expectancy:.2f}")
        print("=" * 50)


class BacktestEngine:
    """
    Historical simulation engine for strategy validation.
    """
    
    def __init__(self, initial_balance: float = 10000.0, commission_pips: float = 0.0):
        """
        Initialize backtest engine.
        
        Args:
            initial_balance: Starting account balance
            commission_pips: Commission per trade in pips
        """
        self.initial_balance = initial_balance
        self.commission_pips = commission_pips
        self.balance = initial_balance
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.position: Optional[BacktestTrade] = None
    
    def reset(self):
        """Reset engine state for new backtest."""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.position = None
    
    def calculate_profit(self, entry_price: float, exit_price: float, 
                          direction: str, lot_size: float) -> float:
        """
        Calculate profit from a trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: 'BUY' or 'SELL'
            lot_size: Position size in lots
            
        Returns:
            Profit in account currency
        """
        # For XAUUSD: 1 pip = $0.01 movement, 1 lot = 100 oz
        # Pip value = $10 per pip for 1 lot
        pip_value = 10.0 * lot_size
        
        if direction == "BUY":
            pips = (exit_price - entry_price) * 10
        else:
            pips = (entry_price - exit_price) * 10
        
        # Subtract commission
        net_pips = pips - (self.commission_pips * 2)  # Entry and exit commission
        
        return net_pips * pip_value
    
    def check_sl_tp(self, bar: pd.Series, position: BacktestTrade) -> Tuple[bool, Optional[float], str]:
        """
        Check if stop loss or take profit was hit.
        
        Args:
            bar: Current OHLCV bar
            position: Current position
            
        Returns:
            Tuple of (was_hit, exit_price, reason)
        """
        if position.direction == "BUY":
            # Check SL (price went below SL)
            if bar['low'] <= position.stop_loss:
                return True, position.stop_loss, "STOPPED_OUT"
            # Check TP (price went above TP)
            if bar['high'] >= position.take_profit:
                return True, position.take_profit, "TP_HIT"
        else:  # SELL
            # Check SL (price went above SL)
            if bar['high'] >= position.stop_loss:
                return True, position.stop_loss, "STOPPED_OUT"
            # Check TP (price went below TP)
            if bar['low'] <= position.take_profit:
                return True, position.take_profit, "TP_HIT"
        
        return False, None, ""
    
    def run(
        self,
        historical_data: pd.DataFrame,
        strategy,
        dxy_data: Optional[pd.DataFrame] = None,
        warmup_bars: int = 100,
        progress_callback: Callable[[int, int], None] = None
    ) -> BacktestResult:
        """
        Run backtest over historical data.
        
        Args:
            historical_data: DataFrame with OHLCV data
            strategy: Strategy instance with analyze() method
            dxy_data: Optional DXY correlation data
            warmup_bars: Number of bars for indicator warmup
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        
        logger.info(f"Starting backtest with {len(historical_data)} bars")
        
        if len(historical_data) < warmup_bars + 10:
            logger.error("Insufficient data for backtest")
            return BacktestResult(
                trades=[],
                equity_curve=[self.initial_balance],
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                start_date=historical_data.index[0],
                end_date=historical_data.index[-1]
            )
        
        start_date = historical_data.index[warmup_bars]
        end_date = historical_data.index[-1]
        total_bars = len(historical_data) - warmup_bars
        
        for i in range(warmup_bars, len(historical_data)):
            # Progress update
            if progress_callback and i % 100 == 0:
                progress_callback(i - warmup_bars, total_bars)
            
            # Get data window
            window = historical_data.iloc[:i+1].copy()
            current_bar = historical_data.iloc[i]
            current_time = historical_data.index[i]
            
            # Get DXY window if available
            dxy_window = None
            if dxy_data is not None:
                dxy_window = dxy_data[dxy_data.index <= current_time].tail(100)
            
            # Check existing position
            if self.position:
                hit, exit_price, reason = self.check_sl_tp(current_bar, self.position)
                
                if hit:
                    # Close position
                    self.position.exit_time = current_time
                    self.position.exit_price = exit_price
                    self.position.profit = self.calculate_profit(
                        self.position.entry_price,
                        exit_price,
                        self.position.direction,
                        self.position.lot_size
                    )
                    self.position.status = reason
                    self.balance += self.position.profit
                    self.trades.append(self.position)
                    self.position = None
            
            # Generate signal if no position
            if not self.position:
                try:
                    signal = strategy.analyze(window, dxy_window)
                    
                    if signal.direction in ["BUY", "SELL"] and signal.confidence >= settings.min_confidence:
                        # Calculate lot size based on risk
                        current_price = current_bar['close']
                        sl_distance = abs(current_price - signal.stop_loss)
                        risk_amount = self.balance * (settings.risk_percent / 100)
                        
                        # Pip value = $10 per lot for XAUUSD
                        sl_pips = sl_distance * 10
                        if sl_pips > 0:
                            lot_size = risk_amount / (sl_pips * 10)
                            lot_size = max(0.01, min(lot_size, 10.0))  # Clamp
                        else:
                            lot_size = 0.01
                        
                        # Open position
                        self.position = BacktestTrade(
                            entry_time=current_time,
                            exit_time=None,
                            direction=signal.direction,
                            entry_price=current_price,
                            exit_price=None,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            lot_size=lot_size,
                            factors=signal.reasons.copy(),
                            status="OPEN"
                        )
                except Exception as e:
                    logger.debug(f"Strategy error at bar {i}: {e}")
            
            # Update equity curve
            current_equity = self.balance
            if self.position:
                unrealized = self.calculate_profit(
                    self.position.entry_price,
                    current_bar['close'],
                    self.position.direction,
                    self.position.lot_size
                )
                current_equity += unrealized
            self.equity_curve.append(current_equity)
        
        # Close any open position at end
        if self.position:
            last_bar = historical_data.iloc[-1]
            self.position.exit_time = historical_data.index[-1]
            self.position.exit_price = last_bar['close']
            self.position.profit = self.calculate_profit(
                self.position.entry_price,
                last_bar['close'],
                self.position.direction,
                self.position.lot_size
            )
            self.position.status = "CLOSED"
            self.balance += self.position.profit
            self.trades.append(self.position)
            self.position = None
        
        result = BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"Win rate: {result.win_rate:.1%}, "
            f"Profit: ${result.total_profit:.2f}"
        )
        
        return result
    
    def run_walk_forward(
        self,
        historical_data: pd.DataFrame,
        strategy,
        dxy_data: Optional[pd.DataFrame] = None,
        in_sample_pct: float = 0.7,
        n_periods: int = 5
    ) -> List[BacktestResult]:
        """
        Run walk-forward analysis.
        
        Args:
            historical_data: Full historical dataset
            strategy: Strategy instance
            dxy_data: Optional DXY data
            in_sample_pct: Percentage of each period for optimization
            n_periods: Number of walk-forward periods
            
        Returns:
            List of BacktestResult for each out-of-sample period
        """
        total_bars = len(historical_data)
        period_size = total_bars // n_periods
        results = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = min((i + 1) * period_size, total_bars)
            
            period_data = historical_data.iloc[start_idx:end_idx]
            
            # Split into in-sample and out-of-sample
            split_idx = int(len(period_data) * in_sample_pct)
            out_sample = period_data.iloc[split_idx:]
            
            # Get corresponding DXY data
            dxy_period = None
            if dxy_data is not None:
                dxy_period = dxy_data[
                    (dxy_data.index >= out_sample.index[0]) &
                    (dxy_data.index <= out_sample.index[-1])
                ]
            
            # Run backtest on out-of-sample
            self.reset()
            result = self.run(out_sample, strategy, dxy_period)
            result.params = {'period': i + 1, 'type': 'out_of_sample'}
            results.append(result)
        
        return results
