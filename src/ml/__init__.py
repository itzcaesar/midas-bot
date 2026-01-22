"""
ML Module - Machine Learning Signal Enhancement
"""
from .data_loader import KaggleDataLoader
from .features import FeatureEngineer, prepare_ml_data
from .models import (
    BaseModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    ModelEnsemble,
    train_all_models
)
from .signal_generator import MLSignalGenerator, MLSignal, run_signal_pipeline

# Optional imports
try:
    from .models import XGBoostModel
except ImportError:
    XGBoostModel = None

try:
    from .deep_learning import LSTMModel, GRUModel, DeepLearningTrainer
except ImportError:
    LSTMModel = None
    GRUModel = None
    DeepLearningTrainer = None

__all__ = [
    # Data
    'KaggleDataLoader',
    'FeatureEngineer',
    'prepare_ml_data',
    # Models
    'BaseModel',
    'LogisticRegressionModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'XGBoostModel',
    'ModelEnsemble',
    'train_all_models',
    # Deep Learning
    'LSTMModel',
    'GRUModel',
    'DeepLearningTrainer',
    # Signal Generation
    'MLSignalGenerator',
    'MLSignal',
    'run_signal_pipeline',
]
