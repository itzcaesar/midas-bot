"""
Kaggle Data Loader
Download and manage XAUUSD historical data from Kaggle.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import sys
sys.path.append('../..')

from core.logger import get_logger

logger = get_logger("mt5bot.ml.data")


class KaggleDataLoader:
    """
    Load and manage XAUUSD historical data from Kaggle.
    Dataset: novandraanugrah/xauusd-gold-price-historical-data-2004-2024
    """
    
    TIMEFRAME_FILES = {
        '1m': 'XAU_1m_data.csv',
        '5m': 'XAU_5m_data.csv',
        '15m': 'XAU_15m_data.csv',
        '30m': 'XAU_30m_data.csv',
        '1h': 'XAU_1h_data.csv',
        '4h': 'XAU_4h_data.csv',
        '1d': 'XAU_1d_data.csv',
        '1w': 'XAU_1w_data.csv',
        '1M': 'XAU_1Month_data.csv',
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV data files
        """
        self.data_dir = Path(data_dir)
        self._cache = {}
    
    def download_dataset(self, kaggle_key: str = None, kaggle_username: str = None) -> bool:
        """
        Download dataset from Kaggle (requires kaggle API credentials).
        
        Args:
            kaggle_key: Kaggle API key (or set KAGGLE_KEY env var)
            kaggle_username: Kaggle username (or set KAGGLE_USERNAME env var)
            
        Returns:
            True if download successful
        """
        try:
            import subprocess
            
            env = os.environ.copy()
            if kaggle_key:
                env['KAGGLE_KEY'] = kaggle_key
            if kaggle_username:
                env['KAGGLE_USERNAME'] = kaggle_username
            
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run([
                'kaggle', 'datasets', 'download', '-d',
                'novandraanugrah/xauusd-gold-price-historical-data-2004-2024',
                '-p', str(self.data_dir), '--unzip'
            ], env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Dataset downloaded successfully")
                return True
            else:
                logger.error(f"Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def load_data(
        self,
        timeframe: str = '1h',
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load historical data for specified timeframe.
        
        Args:
            timeframe: Timeframe to load ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M')
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{timeframe}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        filename = self.TIMEFRAME_FILES.get(timeframe)
        if not filename:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid options: {list(self.TIMEFRAME_FILES.keys())}")
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}. Run download_dataset() first.")
        
        logger.info(f"Loading {timeframe} data from {filepath}")
        
        # Load CSV (handle semicolon delimiter)
        try:
            df = pd.read_csv(filepath, sep=';')
        except:
            df = pd.read_csv(filepath)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Parse datetime
        df = self._parse_datetime(df)
        
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Validate data
        df = self._validate_data(df)
        
        logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        # Cache
        if use_cache:
            self._cache[cache_key] = df.copy()
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        column_mapping = {
            'Time': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Tick Volume': 'volume',
            'tick_volume': 'volume',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df
    
    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime column and set as index."""
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        elif 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time')
        else:
            # Try first column
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col])
                df = df.set_index(first_col)
            except:
                pass
        
        df = df.sort_index()
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data."""
        required_cols = ['open', 'high', 'low', 'close']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove rows with invalid prices
        df = df[df['close'] > 0]
        df = df[df['high'] >= df['low']]
        
        # Forward fill missing values
        df = df.ffill()
        
        # Drop remaining NaN
        df = df.dropna(subset=required_cols)
        
        return df
    
    def get_train_test_split(
        self,
        timeframe: str = '1h',
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (time-series aware).
        
        Args:
            timeframe: Timeframe to load
            train_ratio: Ratio for training (default 80%)
            validation_ratio: Ratio for validation (default 10%)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.load_data(timeframe)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_available_timeframes(self) -> list:
        """Get list of available timeframes based on existing files."""
        available = []
        for tf, filename in self.TIMEFRAME_FILES.items():
            if (self.data_dir / filename).exists():
                available.append(tf)
        return available
    
    def get_data_info(self, timeframe: str = '1h') -> dict:
        """Get information about the dataset."""
        df = self.load_data(timeframe)
        
        return {
            'timeframe': timeframe,
            'rows': len(df),
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'columns': df.columns.tolist(),
            'price_range': {
                'min': df['low'].min(),
                'max': df['high'].max(),
                'latest': df['close'].iloc[-1]
            }
        }
