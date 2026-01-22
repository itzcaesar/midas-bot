"""
Unified Signal Generation Module
Consolidates all signal generation functionality.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from core.logger import get_logger
from ml.data_loader import KaggleDataLoader
from ml.features import FeatureEngineer
from ml.models import (
    LogisticRegressionModel, RandomForestModel,
    GradientBoostingModel, ModelEnsemble
)
from strategy.styles import TradingStyle, StyleConfig, get_style_config

logger = get_logger("midas.signals")


class SignalDirection(Enum):
    """Signal direction enum."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """
    Unified trading signal with all relevant information.
    Replaces MLSignal and StyledSignal.
    """
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Style info
    style: str = "swing"
    timeframe: str = "1h"
    risk_reward: float = 2.0
    max_hold_hours: float = 72.0
    
    # Model info
    model_name: str = "Ensemble"
    probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Analysis
    factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not HOLD)."""
        return self.direction in ["BUY", "SELL"]
    
    @property
    def pip_sl(self) -> float:
        """Stop loss in pips/points."""
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def pip_tp(self) -> float:
        """Take profit in pips/points."""
        return abs(self.take_profit - self.entry_price)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "style": self.style,
            "timeframe": self.timeframe,
            "risk_reward": self.risk_reward,
            "model": self.model_name,
            "factors": self.factors,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalGenerator:
    """
    Unified signal generator combining all signal generation logic.
    
    Features:
    - Multiple trading styles (scalping, intraday, swing, position)
    - Ensemble model predictions
    - Configurable confidence thresholds
    - Discord notification support
    """
    
    def __init__(
        self,
        style: str = "swing",
        model_dir: str = "models",
        data_dir: str = "data",
    ):
        """
        Initialize signal generator.
        
        Args:
            style: Trading style ('scalping', 'intraday', 'swing', 'position')
            model_dir: Directory containing trained models
            data_dir: Directory containing data
        """
        self.style_config = get_style_config(style)
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        self.data_loader = KaggleDataLoader(str(data_dir))
        self.feature_engineer = FeatureEngineer()
        
        self.models: Dict[str, object] = {}
        self.ensemble: Optional[ModelEnsemble] = None
        
        self._load_models()
        
        logger.info(f"Initialized {self.style_config.name} signal generator")
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        model_classes = [
            ("logistic_regression", LogisticRegressionModel),
            ("random_forest", RandomForestModel),
            ("gradient_boosting", GradientBoostingModel),
        ]
        
        for name, cls in model_classes:
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
    
    def generate(
        self,
        df: pd.DataFrame = None,
        min_confidence: float = None,
    ) -> TradingSignal:
        """
        Generate a trading signal.
        
        Args:
            df: Price data (loads from data source if None)
            min_confidence: Minimum confidence for BUY/SELL (uses style default if None)
            
        Returns:
            TradingSignal object
        """
        config = self.style_config
        min_conf = min_confidence or config.min_confidence
        
        # Load data
        if df is None:
            df = self.data_loader.load_data(config.timeframe)
        
        # Create features
        df_features = self.feature_engineer.create_all_features(
            df, include_targets=False
        )
        
        if len(df_features) == 0:
            return self._create_hold_signal(df)
        
        # Get feature columns
        feature_cols = [
            col for col in df_features.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        
        # Get latest data
        X = df_features[feature_cols].iloc[-1:].values
        current_price = float(df_features["close"].iloc[-1])
        
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
        direction_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        direction = direction_map.get(prediction, "HOLD")
        
        # Calculate confidence
        confidence = float(max(probabilities))
        
        # Apply minimum confidence filter
        if confidence < min_conf:
            direction = "HOLD"
        
        # Calculate SL/TP
        atr = self._calculate_atr(df_features)
        stop_loss, take_profit = self._calculate_sl_tp(
            current_price, direction, atr
        )
        
        # Get factors
        factors = self._get_factors(df_features, direction)
        
        # Calculate max hold time
        timeframe_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        bar_minutes = timeframe_minutes.get(config.timeframe, 60)
        max_hold_hours = (config.max_hold_bars * bar_minutes) / 60
        
        return TradingSignal(
            direction=direction,
            confidence=confidence,
            entry_price=round(current_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            style=config.name,
            timeframe=config.timeframe,
            risk_reward=config.risk_reward_ratio,
            max_hold_hours=max_hold_hours,
            model_name=model_name,
            probabilities={
                "sell": float(probabilities[0]),
                "hold": float(probabilities[1]),
                "buy": float(probabilities[2]) if len(probabilities) > 2 else 0,
            },
            factors=factors,
            timestamp=datetime.now(),
        )
    
    def _create_hold_signal(self, df: pd.DataFrame) -> TradingSignal:
        """Create a HOLD signal."""
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        config = self.style_config
        
        return TradingSignal(
            direction="HOLD",
            confidence=0.0,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            style=config.name,
            timeframe=config.timeframe,
            risk_reward=0.0,
            max_hold_hours=0,
            model_name="None",
            factors=["No signal"],
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if "atr_14" in df.columns:
            return float(df["atr_14"].iloc[-1])
        return 10.0  # Default for gold
    
    def _calculate_sl_tp(
        self,
        price: float,
        direction: str,
        atr: float,
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit."""
        config = self.style_config
        sl_distance = atr * config.sl_atr_multiplier
        tp_distance = atr * config.tp_atr_multiplier
        
        if direction == "BUY":
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        elif direction == "SELL":
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
        else:
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit
    
    def _get_factors(
        self,
        df: pd.DataFrame,
        direction: str,
        top_n: int = 5,
    ) -> List[str]:
        """Get contributing factors for the signal."""
        factors = []
        latest = df.iloc[-1]
        config = self.style_config
        
        # Style
        factors.append(f"Style: {config.name} ({config.timeframe})")
        
        # RSI
        if "rsi_14" in latest.index:
            rsi = latest["rsi_14"]
            if rsi < 30:
                factors.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                factors.append(f"RSI overbought ({rsi:.0f})")
            else:
                factors.append(f"RSI neutral ({rsi:.0f})")
        
        # EMA cross
        if "ema9_ema21_cross" in latest.index:
            cross = latest["ema9_ema21_cross"]
            factors.append("EMA9 > EMA21 (bullish)" if cross else "EMA9 < EMA21 (bearish)")
        
        # MACD
        if "macd_hist" in latest.index:
            macd = latest["macd_hist"]
            factors.append(f"MACD {'bullish' if macd > 0 else 'bearish'} ({macd:.2f})")
        
        # ATR
        if "atr_14_pct" in latest.index:
            atr_pct = latest["atr_14_pct"]
            vol_level = "high" if atr_pct > 1.0 else "normal" if atr_pct > 0.5 else "low"
            factors.append(f"Volatility: {vol_level}")
        
        return factors[:top_n]
    
    def set_style(self, style: str) -> None:
        """Change trading style."""
        self.style_config = get_style_config(style)
        logger.info(f"Changed style to {self.style_config.name}")


def generate_signal(
    style: str = "swing",
    send_to_discord: bool = False,
    webhook_url: str = None,
) -> TradingSignal:
    """
    Convenience function to generate a signal.
    
    Args:
        style: Trading style
        send_to_discord: Send signal to Discord
        webhook_url: Discord webhook URL
        
    Returns:
        TradingSignal
    """
    import os
    
    generator = SignalGenerator(style=style)
    signal = generator.generate()
    
    if send_to_discord and signal.is_actionable:
        url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        if url:
            try:
                from notifications.discord import DiscordNotifier
                notifier = DiscordNotifier(url)
                notifier.send_signal(
                    direction=signal.direction,
                    symbol="XAUUSD",
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    factors=signal.factors,
                    model_name=signal.model_name,
                    timeframe=signal.timeframe,
                )
            except Exception as e:
                logger.error(f"Failed to send Discord notification: {e}")
    
    return signal


# Aliases for backward compatibility
MLSignal = TradingSignal
StyledSignal = TradingSignal
MLSignalGenerator = SignalGenerator
StyledSignalGenerator = SignalGenerator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading signals")
    parser.add_argument("--style", "-s", default="swing",
                        choices=["scalping", "intraday", "swing", "position"])
    parser.add_argument("--discord", "-d", action="store_true")
    args = parser.parse_args()
    
    signal = generate_signal(style=args.style, send_to_discord=args.discord)
    
    print(f"\n{'='*50}")
    print(f"📊 {signal.style.upper()} Signal")
    print(f"{'='*50}")
    print(f"Direction: {signal.direction}")
    print(f"Confidence: {signal.confidence:.0%}")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Take Profit: ${signal.take_profit:.2f}")
    print(f"R:R: 1:{signal.risk_reward:.1f}")
    print(f"Max Hold: {signal.max_hold_hours:.0f}h")
    print(f"\nFactors:")
    for f in signal.factors:
        print(f"  • {f}")
