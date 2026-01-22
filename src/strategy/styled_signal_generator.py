"""
Style-Based Signal Generator
Generate signals using trading-style-specific configurations.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/', 2)[0])

from ml.kaggle_loader import KaggleDataLoader
from ml.feature_engineering import FeatureEngineer
from ml.models import RandomForestModel, LogisticRegressionModel, GradientBoostingModel, ModelEnsemble
from strategy.trading_styles import TradingStyle, StyleConfig, get_style_config, get_all_styles
from core.logger import get_logger

logger = get_logger("mt5bot.style_signal")


@dataclass
class StyledSignal:
    """Signal with trading style context."""
    direction: str  # BUY, SELL, HOLD
    confidence: float
    style: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    factors: list
    model_name: str
    timestamp: datetime
    max_hold_hours: float


class StyledSignalGenerator:
    """
    Generate trading signals based on selected trading style.
    """
    
    def __init__(
        self,
        style: str = "swing",
        data_dir: str = "data",
        model_dir: str = "models"
    ):
        """
        Initialize styled signal generator.
        
        Args:
            style: Trading style ('scalping', 'intraday', 'swing', 'position')
            data_dir: Data directory path
            model_dir: Model directory path
        """
        self.style_config = get_style_config(style)
        self.data_loader = KaggleDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.ensemble = None
        
        self._load_models(model_dir)
        
        logger.info(f"Initialized {self.style_config.name} signal generator")
    
    def _load_models(self, model_dir: str):
        """Load trained models."""
        model_classes = [
            ('logistic_regression', LogisticRegressionModel),
            ('random_forest', RandomForestModel),
            ('gradient_boosting', GradientBoostingModel),
        ]
        
        for name, cls in model_classes:
            try:
                model = cls(model_dir=model_dir)
                if model.load():
                    self.models[name] = model
            except Exception as e:
                logger.debug(f"Could not load {name}: {e}")
        
        if len(self.models) >= 2:
            self.ensemble = ModelEnsemble(list(self.models.values()))
    
    def generate_signal(self, df: pd.DataFrame = None) -> StyledSignal:
        """
        Generate a signal based on the configured trading style.
        
        Args:
            df: Price data (loads from Kaggle if None)
            
        Returns:
            StyledSignal with style-specific parameters
        """
        config = self.style_config
        
        # Load data for the style's timeframe
        if df is None:
            df = self.data_loader.load_data(config.timeframe)
        
        # Create features with style-specific threshold
        df_features = self.feature_engineer.create_all_features(
            df,
            include_targets=False,
            target_threshold=config.target_threshold
        )
        
        if len(df_features) == 0:
            return self._create_hold_signal(df)
        
        # Get feature columns
        feature_cols = [col for col in df_features.columns 
                        if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Get latest data point
        X = df_features[feature_cols].iloc[-1:].values
        current_price = df_features['close'].iloc[-1]
        
        # Make prediction
        if self.ensemble:
            prediction = self.ensemble.predict(X)[0]
            probabilities = self.ensemble.predict_proba(X)[0]
            model_name = "Ensemble"
        elif self.models:
            model = list(self.models.values())[0]
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            model_name = model.name
        else:
            return self._create_hold_signal(df)
        
        # Map prediction to direction
        direction_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        direction = direction_map.get(prediction, 'HOLD')
        
        # Calculate confidence with style weighting
        confidence = self._calculate_weighted_confidence(
            df_features.iloc[-1], probabilities, config
        )
        
        # Apply minimum confidence filter
        if confidence < config.min_confidence:
            direction = 'HOLD'
        
        # Calculate SL/TP based on style
        atr = self._calculate_atr(df_features)
        stop_loss, take_profit = self._calculate_sl_tp(
            current_price, direction, atr, config
        )
        
        # Get factors
        factors = self._get_style_factors(df_features, direction, config)
        
        # Calculate max hold time
        if config.timeframe == '5m':
            bar_minutes = 5
        elif config.timeframe == '15m':
            bar_minutes = 15
        elif config.timeframe == '1h':
            bar_minutes = 60
        elif config.timeframe == '4h':
            bar_minutes = 240
        else:
            bar_minutes = 60
        
        max_hold_hours = (config.max_hold_bars * bar_minutes) / 60
        
        return StyledSignal(
            direction=direction,
            confidence=confidence,
            style=config.name,
            timeframe=config.timeframe,
            entry_price=round(current_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward=config.risk_reward_ratio,
            factors=factors,
            model_name=model_name,
            timestamp=datetime.now(),
            max_hold_hours=max_hold_hours
        )
    
    def _calculate_weighted_confidence(
        self,
        latest: pd.Series,
        probabilities: np.ndarray,
        config: StyleConfig
    ) -> float:
        """Calculate confidence with style-specific weighting."""
        base_confidence = float(max(probabilities))
        
        # Adjust based on indicator alignment with style
        adjustments = 0.0
        
        # Momentum indicators (RSI, MACD)
        if 'rsi_14' in latest.index:
            rsi = latest['rsi_14']
            if 30 < rsi < 70:  # Neutral zone
                adjustments += 0.02 * config.momentum_weight
        
        # Trend indicators (EMA crossovers)
        if 'ema9_ema21_cross' in latest.index:
            if latest['ema9_ema21_cross'] == 1:  # Bullish
                adjustments += 0.03 * config.trend_weight
        
        # Volatility (ATR relative)
        if 'atr_14_pct' in latest.index:
            atr_pct = latest['atr_14_pct']
            if config.name == "Scalping" and atr_pct > 0.5:
                adjustments += 0.02 * config.volatility_weight
            elif config.name == "Swing" and atr_pct < 1.0:
                adjustments += 0.02 * config.volatility_weight
        
        return min(1.0, base_confidence + adjustments)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if 'atr_14' in df.columns:
            return df['atr_14'].iloc[-1]
        return 10.0  # Default for gold
    
    def _calculate_sl_tp(
        self,
        price: float,
        direction: str,
        atr: float,
        config: StyleConfig
    ) -> Tuple[float, float]:
        """Calculate SL/TP based on trading style."""
        sl_distance = atr * config.sl_atr_multiplier
        tp_distance = atr * config.tp_atr_multiplier
        
        if direction == 'BUY':
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        elif direction == 'SELL':
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
        else:
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit
    
    def _get_style_factors(
        self,
        df: pd.DataFrame,
        direction: str,
        config: StyleConfig,
        top_n: int = 5
    ) -> list:
        """Get factors relevant to the trading style."""
        factors = []
        latest = df.iloc[-1]
        
        # Style indicator
        factors.append(f"Style: {config.name} ({config.timeframe})")
        
        # Momentum (important for scalping)
        if config.momentum_weight > 0.3 and 'rsi_14' in latest.index:
            rsi = latest['rsi_14']
            if rsi < 30:
                factors.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                factors.append(f"RSI overbought ({rsi:.0f})")
            else:
                factors.append(f"RSI neutral ({rsi:.0f})")
        
        # Trend (important for swing/position)
        if config.trend_weight > 0.3:
            if 'ema9_ema21_cross' in latest.index:
                cross = latest['ema9_ema21_cross']
                factors.append("EMA9 > EMA21 (bullish)" if cross else "EMA9 < EMA21 (bearish)")
        
        # MACD
        if 'macd_hist' in latest.index:
            macd_hist = latest['macd_hist']
            if macd_hist > 0:
                factors.append(f"MACD bullish ({macd_hist:.2f})")
            else:
                factors.append(f"MACD bearish ({macd_hist:.2f})")
        
        # Volatility
        if 'atr_14_pct' in latest.index:
            atr_pct = latest['atr_14_pct']
            vol_level = "high" if atr_pct > 1.0 else "normal" if atr_pct > 0.5 else "low"
            factors.append(f"Volatility: {vol_level} ({atr_pct:.1f}%)")
        
        return factors[:top_n]
    
    def _create_hold_signal(self, df: pd.DataFrame) -> StyledSignal:
        """Create a HOLD signal."""
        price = df['close'].iloc[-1] if len(df) > 0 else 0.0
        config = self.style_config
        
        return StyledSignal(
            direction='HOLD',
            confidence=0.0,
            style=config.name,
            timeframe=config.timeframe,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            risk_reward=0.0,
            factors=['No signal'],
            model_name='None',
            timestamp=datetime.now(),
            max_hold_hours=0
        )


def generate_all_style_signals() -> Dict[str, StyledSignal]:
    """Generate signals for all trading styles."""
    results = {}
    
    for style_name in get_all_styles().keys():
        try:
            generator = StyledSignalGenerator(style=style_name)
            signal = generator.generate_signal()
            results[style_name] = signal
            
            print(f"\n📊 {signal.style.upper()} ({signal.timeframe})")
            print(f"   Direction: {signal.direction} ({signal.confidence:.0%})")
            print(f"   Entry: ${signal.entry_price:.2f}")
            print(f"   SL: ${signal.stop_loss:.2f} | TP: ${signal.take_profit:.2f}")
            print(f"   R:R: 1:{signal.risk_reward:.1f}")
            print(f"   Max Hold: {signal.max_hold_hours:.0f}h")
            
        except Exception as e:
            print(f"Error with {style_name}: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', '-s', default='swing',
                        choices=['scalping', 'intraday', 'swing', 'position', 'all'])
    args = parser.parse_args()
    
    if args.style == 'all':
        generate_all_style_signals()
    else:
        gen = StyledSignalGenerator(style=args.style)
        signal = gen.generate_signal()
        
        print(f"\n{'='*50}")
        print(f"📊 {signal.style.upper()} Signal")
        print(f"{'='*50}")
        print(f"Direction: {signal.direction}")
        print(f"Confidence: {signal.confidence:.0%}")
        print(f"Timeframe: {signal.timeframe}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Risk/Reward: 1:{signal.risk_reward:.1f}")
        print(f"Max Hold: {signal.max_hold_hours:.0f} hours")
        print(f"\nFactors:")
        for f in signal.factors:
            print(f"  • {f}")
