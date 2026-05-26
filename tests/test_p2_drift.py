"""REQ-P2-06: Drift monitoring via PSI."""
from __future__ import annotations

import numpy as np
import pytest


def test_no_drift_same_distribution() -> None:
    from ml.drift import DriftMonitor

    rng = np.random.default_rng(1)
    X_train = rng.normal(0, 1, (1000, 5))
    X_live = rng.normal(0, 1, (200, 5))
    names = [f"f{i}" for i in range(5)]

    monitor = DriftMonitor.from_training_data(X_train, names)
    report = monitor.check(X_live)

    assert not report.has_drift
    assert report.max_psi < 0.10


def test_drift_detected_shifted_distribution() -> None:
    from ml.drift import DriftMonitor

    rng = np.random.default_rng(2)
    X_train = rng.normal(0, 1, (1000, 3))
    # Shift feature 1 by 3 standard deviations
    X_live = rng.normal(0, 1, (200, 3))
    X_live[:, 1] += 3.0

    names = ["stable_a", "shifted_b", "stable_c"]
    monitor = DriftMonitor.from_training_data(X_train, names, threshold=0.20)
    report = monitor.check(X_live)

    assert report.has_drift
    assert "shifted_b" in report.drifted_features
    assert report.feature_psi["shifted_b"] > 0.20
    assert report.feature_psi["stable_a"] < 0.20


def test_save_and_load(tmp_path) -> None:
    from ml.drift import DriftMonitor

    rng = np.random.default_rng(3)
    X_train = rng.normal(0, 1, (500, 4))
    names = ["a", "b", "c", "d"]

    monitor = DriftMonitor.from_training_data(X_train, names)
    path = str(tmp_path / "ref.npz")
    monitor.save(path)

    loaded = DriftMonitor.load(path)
    report = loaded.check(rng.normal(0, 1, (100, 4)))
    assert not report.has_drift
