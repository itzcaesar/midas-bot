"""
ML Training Script - GPU Optimized
Train ML models on XAUUSD Kaggle data.
"""

from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime

from ml.kaggle_loader import KaggleDataLoader
from ml.feature_engineering import FeatureEngineer, prepare_ml_data
from ml.models import train_all_models, RandomForestModel
from core.logger import get_logger

logger = get_logger("mt5bot.ml.train")

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.set_default_device('cuda')
except ImportError:
    GPU_AVAILABLE = False


def train_models(timeframe: str = '1h', test_size: float = 0.2, train_dl: bool = False):
    """Train all ML models on historical data."""
    
    print("=" * 60)
    print("ML MODEL TRAINING PIPELINE")
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if GPU_AVAILABLE else '❌ CPU only'}")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    loader = KaggleDataLoader(data_dir="data")
    df = loader.load_data(timeframe)
    print(f"   {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}")
    
    # Feature engineering
    print("\n🔧 Creating features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df, target_threshold=0.005)
    print(f"   {len(engineer.get_feature_columns())} features, {len(df_features)} samples")
    
    # Target distribution
    dist = df_features['target_signal'].value_counts()
    print(f"\n📈 Targets: SELL={dist.get(-1,0)}, HOLD={dist.get(0,0)}, BUY={dist.get(1,0)}")
    
    # Prepare data
    print("\n📦 Train/Test split...")
    X_train, X_test, y_train, y_test, features = prepare_ml_data(
        df_features, target_col='target_signal', test_size=test_size
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    print("\n🤖 Training...")
    results = train_all_models(X_train, y_train, X_test, y_test, features)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"❌ {name}: {result['error']}")
            continue
        
        test = result['test_metrics']
        print(f"\n✅ {name}")
        print(f"   Accuracy: {test['accuracy']:.1%} | F1: {test['f1_weighted']:.1%}")
        
        if result.get('feature_importance'):
            top_features = list(result['feature_importance'].items())[:3]
            print(f"   Top: {', '.join([f[0] for f in top_features])}")
    
    # Train LSTM if requested
    if train_dl and GPU_AVAILABLE:
        print("\n🧠 Training LSTM (GPU)...")
        from ml.deep_learning import DeepLearningTrainer
        
        trainer = DeepLearningTrainer(model_type='lstm', epochs=50)
        history = trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        print(f"   LSTM: Accuracy={metrics['accuracy']:.1%}, F1={metrics['f1_weighted']:.1%}")
        trainer.save()
    
    print("\n✅ Done! Models saved to 'models/'")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', '-t', default='1h')
    parser.add_argument('--quick', '-q', action='store_true')
    parser.add_argument('--deep', '-d', action='store_true')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test with Random Forest only
        print("Quick test...")
        loader = KaggleDataLoader(data_dir="data")
        df = loader.load_data('1h').tail(5000)
        
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        X_train, X_test, y_train, y_test, features = prepare_ml_data(df_features.tail(2000))
        
        model = RandomForestModel()
        model.fit(X_train, y_train, feature_names=features)
        metrics = model.evaluate(X_test, y_test)
        model.save()
        
        print(f"\n✅ RF: Accuracy={metrics['accuracy']:.1%}, F1={metrics['f1_weighted']:.1%}")
    else:
        train_models(timeframe=args.timeframe, train_dl=args.deep)
