"""
Parameter Optimizer
Grid search and optimization for strategy parameters.
"""
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import copy



from config import settings
from core.logger import get_logger
from backtesting.engine import BacktestEngine, BacktestResult

logger = get_logger("mt5bot.optimizer")


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, Any]
    best_result: BacktestResult
    all_results: List[Dict]
    total_combinations: int
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "=" * 50)
        print("OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"Total combinations tested: {self.total_combinations}")
        print(f"\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest Performance:")
        for key, value in self.best_result.to_dict().items():
            print(f"  {key}: {value}")
        print("=" * 50)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        return pd.DataFrame(self.all_results)


class ParameterOptimizer:
    """
    Optimize strategy parameters through grid search.
    """
    
    def __init__(
        self,
        strategy_class,
        initial_balance: float = 10000.0,
        metric: str = 'profit_factor'
    ):
        """
        Initialize optimizer.
        
        Args:
            strategy_class: Strategy class to instantiate
            initial_balance: Starting balance for backtests
            metric: Metric to optimize ('profit_factor', 'sharpe_ratio', 'return_pct', 'win_rate')
        """
        self.strategy_class = strategy_class
        self.initial_balance = initial_balance
        self.metric = metric
        self.engine = BacktestEngine(initial_balance=initial_balance)
    
    def _apply_params(self, params: Dict[str, Any]):
        """Apply parameters to settings."""
        for key, value in params.items():
            if hasattr(settings, key.lower()):
                setattr(settings, key.lower(), value)
    
    def _restore_params(self, original: Dict[str, Any]):
        """Restore original parameters."""
        for key, value in original.items():
            if hasattr(settings, key.lower()):
                setattr(settings, key.lower(), value)
    
    def _get_metric_value(self, result: BacktestResult) -> float:
        """Extract the optimization metric from results."""
        metric_map = {
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'return_pct': result.return_pct,
            'win_rate': result.win_rate,
            'total_profit': result.total_profit,
            'expectancy': result.expectancy,
        }
        return metric_map.get(self.metric, result.profit_factor)
    
    def grid_search(
        self,
        historical_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        dxy_data: Optional[pd.DataFrame] = None,
        min_trades: int = 10,
        progress_callback: Callable[[int, int, Dict], None] = None
    ) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            historical_data: Historical OHLCV data
            param_grid: Dictionary of parameter names to list of values to test
                Example: {
                    'MACD_FAST': [8, 12, 16],
                    'MACD_SLOW': [21, 26, 30],
                    'MIN_CONFIDENCE': [0.5, 0.6, 0.7]
                }
            dxy_data: Optional DXY correlation data
            min_trades: Minimum trades required for valid result
            progress_callback: Optional callback(current, total, params)
            
        Returns:
            OptimizationResult with best parameters and all results
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        total = len(combinations)
        
        logger.info(f"Starting grid search with {total} combinations")
        
        # Store original settings
        original_settings = {key: getattr(settings, key.lower(), None) for key in keys}
        
        all_results = []
        best_result = None
        best_params = None
        best_metric = float('-inf')
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total, params)
            
            # Apply parameters
            self._apply_params(params)
            
            try:
                # Create fresh strategy instance
                strategy = self.strategy_class()
                
                # Run backtest
                self.engine.reset()
                result = self.engine.run(historical_data, strategy, dxy_data)
                
                # Check minimum trades
                if result.total_trades < min_trades:
                    logger.debug(f"Skipping {params}: only {result.total_trades} trades")
                    continue
                
                # Get metric value
                metric_value = self._get_metric_value(result)
                
                # Store result
                result_dict = {
                    **params,
                    **result.to_dict(),
                    'metric_value': metric_value
                }
                all_results.append(result_dict)
                
                # Check if best
                if metric_value > best_metric and result.total_trades >= min_trades:
                    # Additional validation: max drawdown shouldn't be too high
                    if result.max_drawdown < 0.5:  # Less than 50% drawdown
                        best_metric = metric_value
                        best_result = result
                        best_params = params
                        logger.info(
                            f"New best: {self.metric}={metric_value:.2f}, params={params}"
                        )
            
            except Exception as e:
                logger.warning(f"Error testing {params}: {e}")
                continue
        
        # Restore original settings
        self._restore_params(original_settings)
        
        if best_result is None:
            logger.warning("No valid results found during optimization")
            # Return empty result
            return OptimizationResult(
                best_params={},
                best_result=BacktestResult(
                    trades=[],
                    equity_curve=[],
                    initial_balance=self.initial_balance,
                    final_balance=self.initial_balance,
                    start_date=historical_data.index[0],
                    end_date=historical_data.index[-1]
                ),
                all_results=all_results,
                total_combinations=total
            )
        
        best_result.params = best_params
        
        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            all_results=all_results,
            total_combinations=total
        )
    
    def random_search(
        self,
        historical_data: pd.DataFrame,
        param_distributions: Dict[str, tuple],
        n_iter: int = 50,
        dxy_data: Optional[pd.DataFrame] = None,
        min_trades: int = 10
    ) -> OptimizationResult:
        """
        Run random search optimization.
        
        Args:
            historical_data: Historical OHLCV data
            param_distributions: Dictionary of parameter names to (min, max) tuples
                Example: {
                    'MACD_FAST': (5, 20),
                    'MACD_SLOW': (15, 35),
                    'MIN_CONFIDENCE': (0.4, 0.8)
                }
            n_iter: Number of random iterations
            dxy_data: Optional DXY data
            min_trades: Minimum trades required
            
        Returns:
            OptimizationResult
        """
        logger.info(f"Starting random search with {n_iter} iterations")
        
        # Store original settings
        keys = list(param_distributions.keys())
        original_settings = {key: getattr(settings, key.lower(), None) for key in keys}
        
        all_results = []
        best_result = None
        best_params = None
        best_metric = float('-inf')
        
        for i in range(n_iter):
            # Generate random parameters
            params = {}
            for key, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[key] = np.random.randint(min_val, max_val + 1)
                else:
                    params[key] = np.random.uniform(min_val, max_val)
            
            # Apply and test
            self._apply_params(params)
            
            try:
                strategy = self.strategy_class()
                self.engine.reset()
                result = self.engine.run(historical_data, strategy, dxy_data)
                
                if result.total_trades < min_trades:
                    continue
                
                metric_value = self._get_metric_value(result)
                
                result_dict = {
                    **params,
                    **result.to_dict(),
                    'metric_value': metric_value
                }
                all_results.append(result_dict)
                
                if metric_value > best_metric and result.max_drawdown < 0.5:
                    best_metric = metric_value
                    best_result = result
                    best_params = params
                    logger.info(f"New best at iteration {i+1}: {self.metric}={metric_value:.2f}")
            
            except Exception as e:
                logger.debug(f"Error at iteration {i+1}: {e}")
                continue
        
        # Restore settings
        self._restore_params(original_settings)
        
        if best_result is None:
            return OptimizationResult(
                best_params={},
                best_result=BacktestResult(
                    trades=[],
                    equity_curve=[],
                    initial_balance=self.initial_balance,
                    final_balance=self.initial_balance,
                    start_date=historical_data.index[0],
                    end_date=historical_data.index[-1]
                ),
                all_results=all_results,
                total_combinations=n_iter
            )
        
        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            all_results=all_results,
            total_combinations=n_iter
        )
    
    def analyze_parameter_sensitivity(
        self,
        results: List[Dict],
        param_name: str
    ) -> pd.DataFrame:
        """
        Analyze how a parameter affects performance.
        
        Args:
            results: List of result dictionaries from optimization
            param_name: Parameter to analyze
            
        Returns:
            DataFrame with parameter values and average metrics
        """
        df = pd.DataFrame(results)
        
        if param_name not in df.columns:
            return pd.DataFrame()
        
        grouped = df.groupby(param_name).agg({
            'total_profit': 'mean',
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'max_drawdown': 'mean',
            'total_trades': 'mean',
        }).reset_index()
        
        return grouped.sort_values(param_name)
