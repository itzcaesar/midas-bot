"""
ML Models for Trading Signal Prediction
Traditional ML models: Logistic Regression, Random Forest, XGBoost
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import sys
sys.path.append('../..')

from core.logger import get_logger

logger = get_logger("mt5bot.ml.models")

# Import ML libraries
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not installed")


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, name: str, model_dir: str = "models"):
        self.name = name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        self.feature_names = []
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying model."""
        pass
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str] = None,
        scale_features: bool = True,
        **kwargs
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            scale_features: Whether to scale features
            
        Returns:
            Training metrics
        """
        self.feature_names = feature_names or []
        
        # Scale features
        if scale_features and self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = self._create_model(**kwargs)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)
        
        logger.info(f"{self.name} trained - Accuracy: {metrics['accuracy']:.2%}")
        
        return metrics
    
    def predict(self, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            scale_features: Whether to scale features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if scale_features and self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features
            scale_features: Whether to scale features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if scale_features and self.scaler:
            X = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Return one-hot encoded predictions
            preds = self.model.predict(X)
            return np.eye(len(np.unique(preds)))[preds]
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scale_features: bool = True
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            scale_features: Whether to scale features
            
        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X_test, scale_features)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"{self.name} evaluation - Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1_weighted']:.2%}")
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        scale_features: bool = True
    ) -> Dict:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            scale_features: Whether to scale features
            
        Returns:
            CV metrics
        """
        # Use a temporary scaler for CV (don't modify the fitted one)
        temp_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        if scale_features and temp_scaler:
            X = temp_scaler.fit_transform(X)
        
        # Use a temporary model for CV (don't overwrite fitted model)
        cv_model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = cross_val_score(cv_model, X, y, cv=tscv, scoring='accuracy')
        
        return {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist()
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (if available)."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance = self.model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importance):
            return dict(sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            ))
        return dict(enumerate(importance))
    
    def save(self, filename: str = None) -> str:
        """Save model to disk."""
        filename = filename or f"{self.name.lower().replace(' ', '_')}.pkl"
        filepath = self.model_dir / filename
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load(self, filename: str = None) -> bool:
        """Load model from disk."""
        filename = filename or f"{self.name.lower().replace(' ', '_')}.pkl"
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return True


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""
    
    def __init__(self, model_dir: str = "models"):
        super().__init__("Logistic Regression", model_dir)
    
    def _create_model(self, **kwargs):
        return LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            C=kwargs.get('C', 1.0),
            class_weight=kwargs.get('class_weight', 'balanced'),
            random_state=42
        )


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self, model_dir: str = "models"):
        super().__init__("Random Forest", model_dir)
    
    def _create_model(self, **kwargs):
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 5),
            class_weight=kwargs.get('class_weight', 'balanced'),
            random_state=42,
            n_jobs=-1
        )


class XGBoostModel(BaseModel):
    """XGBoost classifier."""
    
    def __init__(self, model_dir: str = "models"):
        super().__init__("XGBoost", model_dir)
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost package not installed")
    
    def _create_model(self, **kwargs):
        return xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier (lightweight for speed)."""
    
    def __init__(self, model_dir: str = "models"):
        super().__init__("Gradient Boosting", model_dir)
    
    def _create_model(self, **kwargs):
        return GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 50),  # Reduced for speed
            max_depth=kwargs.get('max_depth', 4),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=0.8,
            random_state=42
        )


class ModelEnsemble:
    """Ensemble of multiple models."""
    
    def __init__(self, models: List[BaseModel], weights: List[float] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for voting (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted voting.
        
        Args:
            X: Features
            
        Returns:
            Ensemble predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Weighted voting
        weighted_preds = np.zeros((X.shape[0], 3))  # Assuming 3 classes: -1, 0, 1
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            for j, p in enumerate(pred):
                class_idx = int(p) + 1  # Map -1,0,1 to 0,1,2
                weighted_preds[j, class_idx] += weight
        
        return np.argmax(weighted_preds, axis=1) - 1  # Map back to -1,0,1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble probabilities.
        
        Args:
            X: Features
            
        Returns:
            Average probabilities
        """
        all_proba = []
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            all_proba.append(proba * weight)
        
        return np.sum(all_proba, axis=0)
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction confidence (max probability).
        
        Args:
            X: Features
            
        Returns:
            Confidence scores
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str] = None
) -> Dict[str, Dict]:
    """
    Train and evaluate all available models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        
    Returns:
        Dictionary of model results
    """
    results = {}
    
    models = [
        LogisticRegressionModel(),
        RandomForestModel(),
        GradientBoostingModel(),
    ]
    
    if XGBOOST_AVAILABLE:
        models.append(XGBoostModel())
    
    for model in models:
        logger.info(f"Training {model.name}...")
        
        try:
            # Train
            train_metrics = model.fit(X_train, y_train, feature_names=feature_names)
            
            # Evaluate
            test_metrics = model.evaluate(X_test, y_test)
            
            # CV
            cv_metrics = model.cross_validate(X_train, y_train)
            
            # Feature importance
            importance = model.get_feature_importance()
            
            # Save model
            model.save()
            
            results[model.name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'feature_importance': dict(list(importance.items())[:10]),  # Top 10
                'model': model
            }
        except Exception as e:
            logger.error(f"Error training {model.name}: {e}")
            results[model.name] = {'error': str(e)}
    
    return results
