"""
ML Training Script - GPU Optimized
Train ML models on XAUUSD Kaggle data.
"""
import sys

from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime

from ml.data_loader import KaggleDataLoader
from ml.features import FeatureEngineer, prepare_ml_data
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
    GPU_AVAILABLE = True


def _print_detailed_metrics(results: dict, y_test: np.ndarray) -> None:
    """Print per-class precision/recall and trade-only metrics (REQ-P1-03).

    This replaces the misleading weighted-accuracy headline with metrics that
    actually matter for a trading system:
      - Per-class precision/recall (especially BUY and SELL classes)
      - Trade-only precision: of all BUY+SELL predictions, how many were correct?
      - Expected R per signal: (trade_precision * avg_TP_R) - ((1-trade_precision) * avg_SL_R)
    """
    from sklearn.metrics import classification_report, confusion_matrix

    target_names = ["SELL (-1)", "HOLD (0)", "BUY (+1)"]

    for name, result in results.items():
        if "error" in result or "model" not in result:
            continue

        model = result["model"]
        try:
            y_pred = model.predict(
                # We need X_test but it's not passed here; use the stored test metrics
                # Instead, reconstruct from confusion matrix
                None  # placeholder
            )
        except Exception:
            pass

        # Use the confusion matrix already stored in test_metrics
        cm = result["test_metrics"].get("confusion_matrix")
        if cm is None:
            continue

        cm = np.array(cm)
        print(f"\n--- {name} ---")
        print(f"Confusion Matrix (rows=actual, cols=predicted):")
        print(f"         SELL  HOLD  BUY")
        labels = ["SELL", "HOLD", " BUY"]
        for i, row in enumerate(cm):
            print(f"  {labels[i]}  {row}")

        # Per-class precision/recall from confusion matrix
        print(f"\n  Per-class metrics:")
        for i, cls_name in enumerate(target_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = cm[i, :].sum()
            print(f"    {cls_name:12s}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  n={support}")

        # Trade-only metrics: precision on BUY+SELL predictions only
        # BUY predictions = column 2, SELL predictions = column 0
        buy_preds_total = cm[:, 2].sum() if cm.shape[1] > 2 else 0
        buy_preds_correct = cm[2, 2] if cm.shape[0] > 2 and cm.shape[1] > 2 else 0
        sell_preds_total = cm[:, 0].sum()
        sell_preds_correct = cm[0, 0]

        trade_preds = buy_preds_total + sell_preds_total
        trade_correct = buy_preds_correct + sell_preds_correct
        trade_precision = trade_correct / trade_preds if trade_preds > 0 else 0.0

        print(f"\n  TRADE-ONLY (BUY+SELL predictions):")
        print(f"    Signals issued: {trade_preds}")
        print(f"    Correct:        {trade_correct}")
        print(f"    Trade precision: {trade_precision:.1%}")
        if trade_precision > 0:
            # Assuming 2:1 R:R (default in the strategy)
            expected_r = trade_precision * 2.0 - (1 - trade_precision) * 1.0
            print(f"    Expected R (at 2:1 R:R): {expected_r:+.2f}R per signal")
            if expected_r <= 0:
                print(f"    ⚠️  Negative expectancy — this model loses money at 2:1 R:R")
        else:
            print(f"    ⚠️  No trade signals issued — model predicts HOLD for everything")


def train_models(timeframe: str = '1h', test_size: float = 0.2, train_dl: bool = False):
    """Train all ML models on historical data."""
    
    print("=" * 60)
    print("ML MODEL TRAINING PIPELINE")
    gpu_str = f"GPU: {torch.cuda.get_device_name(0)}" if GPU_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available() else "GPU: CPU only"
    print(gpu_str)
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
    # Pass purge_gap=horizon so train labels cannot peek into the test slice.
    # `create_all_features` uses the default target_horizon=1, so purge_gap=1.
    X_train, X_test, y_train, y_test, features = prepare_ml_data(
        df_features, target_col='target_signal', test_size=test_size, purge_gap=1
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

    # REQ-P1-03: Per-class metrics + trade-only metrics
    # This is the honest reporting that replaces "92.7% accuracy" theater.
    print("\n" + "=" * 60)
    print("PER-CLASS & TRADE-ONLY METRICS (REQ-P1-03)")
    print("=" * 60)
    _print_detailed_metrics(results, y_test)
    
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
