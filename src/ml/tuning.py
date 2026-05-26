"""
Hyperparameter Tuning with Optuna + PurgedKFold (REQ-P1-07).

Provides a proper time-series-aware cross-validation splitter that respects
the purge gap (no label leakage across folds) and an Optuna-based tuner that
searches over model hyperparameters using this splitter.

Usage:
    from ml.tuning import tune_model
    best_params, study = tune_model(X_train, y_train, model_type='lightgbm', n_trials=50)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from core.logger import get_logger

logger = get_logger("midas.ml.tuning")

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.info("optuna not installed — hyperparameter tuning unavailable")

try:
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.metrics import f1_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# PurgedKFold — time-series CV with a gap between train and validation
# ---------------------------------------------------------------------------


class PurgedKFold(BaseCrossValidator):
    """Time-series K-Fold with a purge gap between train and validation.

    Unlike sklearn's TimeSeriesSplit, this splitter drops ``purge_gap`` rows
    from the end of each training fold so that forward-looking labels in the
    training set cannot peek into the validation fold.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    purge_gap : int
        Number of rows to drop from the end of each training fold (default 1).
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 1) -> None:
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n)

            if val_start >= n or val_end <= val_start:
                continue

            train_idx = np.arange(0, max(0, train_end - self.purge_gap))
            val_idx = np.arange(val_start, val_end)

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Optuna tuner
# ---------------------------------------------------------------------------


def _lightgbm_objective(trial, X, y, cv):
    """Optuna objective for LightGBM."""
    import lightgbm as lgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))

    return np.mean(scores)


def _random_forest_objective(trial, X, y, cv):
    """Optuna objective for Random Forest."""
    from sklearn.ensemble import RandomForestClassifier

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))

    return np.mean(scores)


_OBJECTIVES = {
    "lightgbm": _lightgbm_objective,
    "random_forest": _random_forest_objective,
}


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "lightgbm",
    n_trials: int = 50,
    n_splits: int = 5,
    purge_gap: int = 1,
    timeout: Optional[int] = None,
) -> Tuple[Dict, "optuna.Study"]:
    """Run Optuna hyperparameter search with PurgedKFold CV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: 'lightgbm' or 'random_forest'.
        n_trials: Number of Optuna trials.
        n_splits: Number of CV folds.
        purge_gap: Rows to purge between train/val in each fold.
        timeout: Optional timeout in seconds.

    Returns:
        Tuple of (best_params dict, optuna Study object).
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for tuning. pip install optuna")

    if model_type not in _OBJECTIVES:
        raise ValueError(f"Unknown model_type: {model_type}. Options: {list(_OBJECTIVES)}")

    cv = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap)
    objective_fn = _OBJECTIVES[model_type]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"midas_{model_type}_tuning",
    )

    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, cv),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    logger.info(
        f"Tuning complete: best f1_macro={study.best_value:.4f} "
        f"in {len(study.trials)} trials"
    )
    logger.info(f"Best params: {study.best_params}")

    return study.best_params, study
