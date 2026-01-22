"""
Feature Engineering Module
Create technical indicators and features for ML models.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple



from core.logger import get_logger

logger = get_logger("mt5bot.ml.features")


class FeatureEngineer:
    """
    Create technical indicators and features for ML trading models.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_lags: bool = True,
        include_targets: bool = True,
        include_advanced: bool = True,
        target_horizon: int = 1,
        target_threshold: float = 0.005  # 0.5% for signal generation
    ) -> pd.DataFrame:
        """
        Create all features including technical indicators, lags, and targets.
        
        Args:
            df: DataFrame with OHLCV data
            include_lags: Include lag features
            include_targets: Include target variables
            include_advanced: Include advanced indicators (Ichimoku, ADX, etc.)
            target_horizon: Bars ahead for target calculation
            target_threshold: Threshold for BUY/SELL classification
            
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        # Basic technical indicators
        df = self.add_moving_averages(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_stochastic(df)
        df = self.add_momentum(df)
        df = self.add_volatility_features(df)
        df = self.add_price_features(df)
        
        # Advanced indicators
        if include_advanced:
            from ml.advanced_indicators import add_all_advanced_indicators
            df = add_all_advanced_indicators(df)
        
        # Lag features
        if include_lags:
            df = self.add_lag_features(df)
        
        # Target variables
        if include_targets:
            df = self.add_targets(df, horizon=target_horizon, threshold=target_threshold)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in 
                              ['open', 'high', 'low', 'close', 'volume', 
                               'target_return', 'target_direction', 'target_signal']]
        
        # Drop NaN rows from indicator calculations
        df = df.dropna()
        
        logger.info(f"Created {len(self.feature_names)} features, {len(df)} rows remaining")
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        # SMA
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # EMA
        for period in [9, 12, 21, 26, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Price relative to MAs
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        df['price_ema21_ratio'] = df['close'] / df['ema_21']
        df['sma20_sma50_ratio'] = df['sma_20'] / df['sma_50']
        
        # MA crossover signals
        df['ema9_ema21_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['sma20_sma50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD crossover
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_hist_positive'] = (df['macd_hist'] > 0).astype(int)
        
        # MACD momentum
        df['macd_hist_change'] = df['macd_hist'].diff()
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """Add RSI indicator for multiple periods."""
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI overbought/oversold
        df['rsi_14_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_14_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # BB width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price outside bands
        df['bb_above_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_below_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        return df
    
    def add_atr(self, df: pd.DataFrame, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """Add Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for period in periods:
            df[f'atr_{period}'] = tr.rolling(window=period).mean()
        
        # Normalized ATR
        df['atr_14_pct'] = df['atr_14'] / df['close'] * 100
        
        return df
    
    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic signals
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        
        return df
    
    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Williams %R
        period = 14
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Rolling volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std() * np.sqrt(252)
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_sma'] = df['hl_range'].rolling(window=10).mean()
        
        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * (np.log(df['high'] / df['low']) ** 2).rolling(window=20).mean()
        )
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        
        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Distance from high/low
        period = 20
        df['dist_from_high'] = (df['high'].rolling(period).max() - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['low'].rolling(period).min()) / df['close']
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Add lagged features."""
        lag_cols = ['close', 'return_1', 'rsi_14', 'macd_hist', 'bb_position']
        
        for col in lag_cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_targets(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.005
    ) -> pd.DataFrame:
        """
        Add target variables for supervised learning.
        
        Args:
            df: DataFrame with price data
            horizon: Number of bars ahead for prediction
            threshold: Threshold for BUY/SELL classification (e.g., 0.005 = 0.5%)
            
        Returns:
            DataFrame with target columns
        """
        # Future return (regression target)
        df['target_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Direction (binary classification)
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        # Signal (3-class: BUY=1, SELL=-1, HOLD=0)
        df['target_signal'] = 0
        df.loc[df['target_return'] > threshold, 'target_signal'] = 1   # BUY
        df.loc[df['target_return'] < -threshold, 'target_signal'] = -1  # SELL
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_names
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for importance analysis (excluding lags)."""
        return [f for f in self.feature_names if '_lag_' not in f]


def prepare_ml_data(
    df: pd.DataFrame,
    target_col: str = 'target_signal',
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for ML training.
    
    Args:
        df: DataFrame with features and targets
        target_col: Target column name
        test_size: Test set size ratio
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in 
                    ['open', 'high', 'low', 'close', 'volume',
                     'target_return', 'target_direction', 'target_signal']]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Time-series split (no shuffling)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols
