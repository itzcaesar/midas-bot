"""
ML Signal Predictor
Machine learning model for enhancing signal confidence prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys
sys.path.append('../..')

from config import settings
from core.logger import get_logger

logger = get_logger("mt5bot.ml")

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML features disabled.")


class FeatureExtractor:
    """Extract ML features from market data."""
    
    FEATURE_NAMES = [
        'macd_hist',
        'macd_hist_change',
        'rsi',
        'atr_normalized',
        'price_ema20_ratio',
        'price_ema50_ratio',
        'ema20_ema50_ratio',
        'volume_sma_ratio',
        'distance_to_high',
        'distance_to_low',
        'body_ratio',
        'upper_wick_ratio',
        'lower_wick_ratio',
        'trend_strength',
        'hour_sin',
        'hour_cos',
        'day_of_week'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._fitted = False
    
    def calculate_rsi(self, closes: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def extract_features(self, df: pd.DataFrame, dxy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            dxy_df: Optional DXY correlation data
            
        Returns:
            DataFrame with extracted features
        """
        if len(df) < 50:
            logger.warning("Insufficient data for feature extraction")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        # MACD features
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd_hist'] = macd - signal
        features['macd_hist_change'] = features['macd_hist'].diff()
        
        # RSI
        features['rsi'] = self.calculate_rsi(df['close'])
        
        # ATR normalized
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        features['atr_normalized'] = atr / df['close']
        
        # EMA ratios
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        features['price_ema20_ratio'] = df['close'] / ema20
        features['price_ema50_ratio'] = df['close'] / ema50
        features['ema20_ema50_ratio'] = ema20 / ema50
        
        # Volume ratio
        if 'volume' in df.columns:
            vol_sma = df['volume'].rolling(window=20).mean()
            features['volume_sma_ratio'] = df['volume'] / vol_sma
        else:
            features['volume_sma_ratio'] = 1.0
        
        # Distance to recent high/low
        rolling_high = df['high'].rolling(window=20).max()
        rolling_low = df['low'].rolling(window=20).min()
        price_range = rolling_high - rolling_low
        features['distance_to_high'] = (rolling_high - df['close']) / price_range
        features['distance_to_low'] = (df['close'] - rolling_low) / price_range
        
        # Candle body ratios
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        features['body_ratio'] = body / total_range.replace(0, 1)
        
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        features['upper_wick_ratio'] = upper_wick / total_range.replace(0, 1)
        features['lower_wick_ratio'] = lower_wick / total_range.replace(0, 1)
        
        # Trend strength (slope of EMA20)
        features['trend_strength'] = (ema20 - ema20.shift(5)) / ema20.shift(5)
        
        # Time features
        if hasattr(df.index, 'hour'):
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            features['day_of_week'] = df.index.dayofweek
        else:
            features['hour_sin'] = 0
            features['hour_cos'] = 0
            features['day_of_week'] = 0
        
        # DXY correlation
        if dxy_df is not None and len(dxy_df) > 0:
            try:
                xau_returns = df['close'].pct_change()
                dxy_returns = dxy_df['close'].pct_change()
                aligned = pd.concat([xau_returns, dxy_returns], axis=1).dropna()
                if len(aligned) >= 20:
                    features['dxy_correlation'] = aligned.iloc[:, 0].rolling(20).corr(aligned.iloc[:, 1])
                else:
                    features['dxy_correlation'] = 0
            except:
                features['dxy_correlation'] = 0
        else:
            features['dxy_correlation'] = 0
        
        return features.dropna()
    
    def fit_scaler(self, features: pd.DataFrame):
        """Fit the feature scaler."""
        if self.scaler is not None:
            self.scaler.fit(features)
            self._fitted = True
    
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.scaler is None or not self._fitted:
            return features.values
        return self.scaler.transform(features)


class MLSignalPredictor:
    """
    Machine learning model for predicting trade success probability.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ML predictor.
        
        Args:
            model_dir: Directory for saving/loading models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.model_path = self.model_dir / "signal_predictor.pkl"
        self.scaler_path = self.model_dir / "feature_scaler.pkl"
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained model if exists."""
        if not SKLEARN_AVAILABLE:
            return
        
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Loaded ML model from disk")
                
                if self.scaler_path.exists():
                    self.feature_extractor.scaler = joblib.load(self.scaler_path)
                    self.feature_extractor._fitted = True
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def _save_model(self):
        """Save trained model to disk."""
        if not SKLEARN_AVAILABLE or self.model is None:
            return
        
        try:
            joblib.dump(self.model, self.model_path)
            if self.feature_extractor._fitted:
                joblib.dump(self.feature_extractor.scaler, self.scaler_path)
            logger.info("Saved ML model to disk")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def train(
        self,
        historical_data: pd.DataFrame,
        trade_outcomes: pd.DataFrame,
        dxy_data: Optional[pd.DataFrame] = None,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Train the ML model on historical trade data.
        
        Args:
            historical_data: OHLCV data
            trade_outcomes: DataFrame with columns ['timestamp', 'direction', 'outcome']
                           outcome: 1 for profitable, 0 for loss
            dxy_data: Optional DXY correlation data
            model_type: 'random_forest' or 'gradient_boosting'
            
        Returns:
            Training metrics dictionary
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for training")
            return {'error': 'sklearn not installed'}
        
        logger.info(f"Training ML model with {len(trade_outcomes)} trade outcomes")
        
        # Extract features
        features = self.feature_extractor.extract_features(historical_data, dxy_data)
        
        # Align features with trade outcomes
        X = []
        y = []
        
        for _, trade in trade_outcomes.iterrows():
            trade_time = trade['timestamp']
            if trade_time in features.index:
                X.append(features.loc[trade_time].values)
                y.append(trade['outcome'])
        
        if len(X) < 20:
            logger.error("Insufficient aligned data for training")
            return {'error': 'insufficient data', 'samples': len(X)}
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit scaler
        self.feature_extractor.fit_scaler(pd.DataFrame(X))
        X_scaled = self.feature_extractor.transform(pd.DataFrame(X))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_extractor.FEATURE_NAMES[:len(self.model.feature_importances_)],
                self.model.feature_importances_
            ))
            metrics['feature_importance'] = importance
        
        logger.info(f"Training complete - Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}")
        
        # Save model
        self._save_model()
        
        return metrics
    
    def predict_confidence(
        self,
        df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame] = None,
        signal_direction: str = None
    ) -> float:
        """
        Predict trade success probability.
        
        Args:
            df: Current OHLCV data
            dxy_df: Optional DXY data
            signal_direction: 'BUY' or 'SELL' (not currently used, for future enhancement)
            
        Returns:
            Probability of successful trade (0-1)
        """
        if self.model is None:
            logger.debug("No ML model available, returning default confidence")
            return 0.5
        
        try:
            # Extract features for latest bar
            features = self.feature_extractor.extract_features(df, dxy_df)
            
            if len(features) == 0:
                return 0.5
            
            latest_features = features.iloc[-1:].values
            
            # Scale features
            if self.feature_extractor._fitted:
                latest_features = self.feature_extractor.transform(
                    pd.DataFrame(latest_features)
                )
            
            # Predict probability
            probability = self.model.predict_proba(latest_features)[0][1]
            
            return float(probability)
        
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.5
    
    def enhance_signal_confidence(
        self,
        base_confidence: float,
        df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame] = None,
        ml_weight: float = 0.3
    ) -> Tuple[float, str]:
        """
        Enhance signal confidence using ML prediction.
        
        Args:
            base_confidence: Original strategy confidence (0-1)
            df: Current OHLCV data
            dxy_df: Optional DXY data
            ml_weight: Weight of ML prediction in final confidence
            
        Returns:
            Tuple of (enhanced_confidence, explanation)
        """
        ml_confidence = self.predict_confidence(df, dxy_df)
        
        # Weighted combination
        enhanced = (1 - ml_weight) * base_confidence + ml_weight * ml_confidence
        
        # Generate explanation
        if ml_confidence > base_confidence + 0.1:
            explanation = f"ML boost (+{(ml_confidence - base_confidence):.0%})"
        elif ml_confidence < base_confidence - 0.1:
            explanation = f"ML caution ({(ml_confidence - base_confidence):.0%})"
        else:
            explanation = "ML neutral"
        
        return enhanced, explanation
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained and available."""
        return self.model is not None
