"""
Signal Generator
Real-time ML signal generation pipeline.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path



from config import settings
from core.logger import get_logger
from ml.data_loader import KaggleDataLoader
from ml.features import FeatureEngineer
from ml.models import (
    LogisticRegressionModel, RandomForestModel, 
    XGBoostModel, GradientBoostingModel, ModelEnsemble
)

logger = get_logger("mt5bot.ml.signal_generator")


@dataclass
class MLSignal:
    """ML-generated trading signal."""
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    model_name: str
    probabilities: Dict[str, float]  # {'BUY': 0.x, 'SELL': 0.x, 'HOLD': 0.x}
    factors: List[str]
    timestamp: datetime
    timeframe: str


class MLSignalGenerator:
    """
    Generate trading signals using trained ML models.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        data_dir: str = "data",
        timeframe: str = '1h'
    ):
        """
        Initialize signal generator.
        
        Args:
            model_dir: Directory containing trained models
            data_dir: Directory containing data
            timeframe: Timeframe for analysis
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.timeframe = timeframe
        
        self.data_loader = KaggleDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        
        self.models = {}
        self.ensemble = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        model_classes = {
            'logistic_regression': LogisticRegressionModel,
            'random_forest': RandomForestModel,
            'gradient_boosting': GradientBoostingModel,
        }
        
        try:
            from ml.models import XGBoostModel
            model_classes['xgboost'] = XGBoostModel
        except:
            pass
        
        for name, cls in model_classes.items():
            try:
                model = cls(model_dir=str(self.model_dir))
                if model.load():
                    self.models[name] = model
                    logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.debug(f"Could not load {name}: {e}")
        
        if len(self.models) >= 2:
            self.ensemble = ModelEnsemble(list(self.models.values()))
            logger.info(f"Created ensemble with {len(self.models)} models")
    
    def generate_signal(
        self,
        df: pd.DataFrame = None,
        use_ensemble: bool = True,
        min_confidence: float = 0.6
    ) -> MLSignal:
        """
        Generate a trading signal from current market data.
        
        Args:
            df: DataFrame with OHLCV data (uses latest Kaggle data if None)
            use_ensemble: Use ensemble prediction
            min_confidence: Minimum confidence for BUY/SELL
            
        Returns:
            MLSignal object
        """
        # Load latest data if not provided
        if df is None:
            df = self.data_loader.load_data(self.timeframe)
        
        # Create features
        df_features = self.feature_engineer.create_all_features(df, include_targets=False)
        
        if len(df_features) == 0:
            logger.warning("No features generated")
            return self._create_hold_signal(df)
        
        # Get feature columns
        feature_cols = [col for col in df_features.columns if col not in 
                        ['open', 'high', 'low', 'close', 'volume']]
        
        # Get latest row
        X = df_features[feature_cols].iloc[-1:].values
        current_price = df_features['close'].iloc[-1]
        
        # Make prediction
        if use_ensemble and self.ensemble:
            prediction = self.ensemble.predict(X)[0]
            probabilities = self.ensemble.predict_proba(X)[0]
            model_name = "Ensemble"
        elif self.models:
            # Use best performing model
            best_model = list(self.models.values())[0]
            prediction = best_model.predict(X)[0]
            probabilities = best_model.predict_proba(X)[0]
            model_name = best_model.name
        else:
            logger.warning("No models available")
            return self._create_hold_signal(df)
        
        # Map prediction to direction
        direction_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        direction = direction_map.get(prediction, 'HOLD')
        
        # Calculate confidence
        confidence = float(max(probabilities))
        
        # Check minimum confidence
        if confidence < min_confidence:
            direction = 'HOLD'
        
        # Create probability dict
        prob_dict = {
            'SELL': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'HOLD': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'BUY': float(probabilities[2]) if len(probabilities) > 2 else 0.0,
        }
        
        # Calculate SL/TP
        atr = self._calculate_atr(df_features)
        stop_loss, take_profit = self._calculate_sl_tp(current_price, direction, atr)
        
        # Get top factors
        factors = self._get_signal_factors(df_features, direction)
        
        return MLSignal(
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model_name=model_name,
            probabilities=prob_dict,
            factors=factors,
            timestamp=datetime.now(),
            timeframe=self.timeframe
        )
    
    def _create_hold_signal(self, df: pd.DataFrame) -> MLSignal:
        """Create a HOLD signal."""
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0.0
        return MLSignal(
            direction='HOLD',
            confidence=0.0,
            entry_price=current_price,
            stop_loss=current_price,
            take_profit=current_price,
            model_name='None',
            probabilities={'BUY': 0.0, 'SELL': 0.0, 'HOLD': 1.0},
            factors=['No signal generated'],
            timestamp=datetime.now(),
            timeframe=self.timeframe
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for SL/TP."""
        if 'atr_14' in df.columns:
            return df['atr_14'].iloc[-1]
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 10.0  # Default ATR for gold
    
    def _calculate_sl_tp(
        self,
        price: float,
        direction: str,
        atr: float,
        sl_multiplier: float = 1.5,
        rr_ratio: float = 2.0
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit.
        
        Args:
            price: Current price
            direction: 'BUY' or 'SELL'
            atr: Average True Range
            sl_multiplier: ATR multiplier for SL
            rr_ratio: Risk/Reward ratio
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        sl_distance = atr * sl_multiplier
        tp_distance = sl_distance * rr_ratio
        
        if direction == 'BUY':
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        elif direction == 'SELL':
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
        else:
            stop_loss = price
            take_profit = price
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    def _get_signal_factors(self, df: pd.DataFrame, direction: str, top_n: int = 5) -> List[str]:
        """Get top contributing factors for the signal."""
        factors = []
        
        latest = df.iloc[-1]
        
        # MACD
        if 'macd_hist' in df.columns:
            macd_hist = latest['macd_hist']
            if direction == 'BUY' and macd_hist > 0:
                factors.append(f"MACD bullish ({macd_hist:.2f})")
            elif direction == 'SELL' and macd_hist < 0:
                factors.append(f"MACD bearish ({macd_hist:.2f})")
        
        # RSI
        if 'rsi_14' in df.columns:
            rsi = latest['rsi_14']
            if direction == 'BUY' and rsi < 40:
                factors.append(f"RSI oversold ({rsi:.0f})")
            elif direction == 'SELL' and rsi > 60:
                factors.append(f"RSI overbought ({rsi:.0f})")
            else:
                factors.append(f"RSI neutral ({rsi:.0f})")
        
        # Bollinger
        if 'bb_position' in df.columns:
            bb_pos = latest['bb_position']
            if direction == 'BUY' and bb_pos < 0.2:
                factors.append("Near lower Bollinger band")
            elif direction == 'SELL' and bb_pos > 0.8:
                factors.append("Near upper Bollinger band")
        
        # EMA trend
        if 'ema9_ema21_cross' in df.columns:
            ema_cross = latest['ema9_ema21_cross']
            if direction == 'BUY' and ema_cross == 1:
                factors.append("EMA9 > EMA21 (bullish)")
            elif direction == 'SELL' and ema_cross == 0:
                factors.append("EMA9 < EMA21 (bearish)")
        
        # Momentum
        if 'roc_10' in df.columns:
            roc = latest['roc_10']
            if direction == 'BUY' and roc > 0:
                factors.append(f"10-period ROC positive ({roc:.1f}%)")
            elif direction == 'SELL' and roc < 0:
                factors.append(f"10-period ROC negative ({roc:.1f}%)")
        
        return factors[:top_n]
    
    def get_model_comparison(self, df: pd.DataFrame = None) -> Dict:
        """
        Get predictions from all models for comparison.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of model predictions
        """
        if df is None:
            df = self.data_loader.load_data(self.timeframe)
        
        df_features = self.feature_engineer.create_all_features(df, include_targets=False)
        feature_cols = [col for col in df_features.columns if col not in 
                        ['open', 'high', 'low', 'close', 'volume']]
        X = df_features[feature_cols].iloc[-1:].values
        
        results = {}
        direction_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                
                results[name] = {
                    'direction': direction_map.get(pred, 'HOLD'),
                    'confidence': float(max(proba)),
                    'probabilities': {
                        'SELL': float(proba[0]) if len(proba) > 0 else 0.0,
                        'HOLD': float(proba[1]) if len(proba) > 1 else 0.0,
                        'BUY': float(proba[2]) if len(proba) > 2 else 0.0,
                    }
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results


def run_signal_pipeline(
    timeframe: str = '1h',
    send_to_discord: bool = True,
    discord_webhook_url: str = None
) -> MLSignal:
    """
    Run the complete signal generation pipeline.
    
    Args:
        timeframe: Timeframe for analysis
        send_to_discord: Send signal to Discord
        discord_webhook_url: Discord webhook URL
        
    Returns:
        Generated MLSignal
    """
    # Generate signal
    generator = MLSignalGenerator(timeframe=timeframe)
    signal = generator.generate_signal()
    
    logger.info(
        f"Generated signal: {signal.direction} | "
        f"Confidence: {signal.confidence:.0%} | "
        f"Entry: ${signal.entry_price:.2f}"
    )
    
    # Send to Discord
    if send_to_discord and signal.direction != 'HOLD':
        from notifications.discord import DiscordNotifier
        notifier = DiscordNotifier(discord_webhook_url)
        notifier.send_signal(
            direction=signal.direction,
            symbol='XAUUSD',
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            factors=signal.factors,
            model_name=signal.model_name,
            timeframe=signal.timeframe
        )
    
    return signal
