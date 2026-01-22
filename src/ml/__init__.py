"""
ML Module
Machine learning components for trading signal prediction.
"""
from .data_loader import KaggleDataLoader
from .features import FeatureEngineer, prepare_ml_data
from .models import (
    BaseModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    ModelEnsemble,
    train_all_models,
)

# Conditionally import XGBoost
try:
    from .models import XGBoostModel
except ImportError:
    XGBoostModel = None

# Conditionally import deep learning
try:
    from .deep_learning import LSTMModel, GRUModel, DeepLearningTrainer
except ImportError:
    LSTMModel = None
    GRUModel = None
    DeepLearningTrainer = None

__all__ = [
    # Data
    "KaggleDataLoader",
    # Features
    "FeatureEngineer",
    "prepare_ml_data",
    # Models
    "BaseModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "XGBoostModel",
    "ModelEnsemble",
    "train_all_models",
    # Deep Learning
    "LSTMModel",
    "GRUModel",
    "DeepLearningTrainer",
]
