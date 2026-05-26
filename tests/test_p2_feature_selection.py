"""REQ-P2-03: Feature clustering/reduction."""
from __future__ import annotations

import numpy as np
import pytest


def test_cluster_reduces_correlated_features() -> None:
    from ml.feature_selection import cluster_features

    rng = np.random.default_rng(42)
    n = 500

    # Create 3 independent signals + 7 copies/near-copies
    base_a = rng.normal(0, 1, n)
    base_b = rng.normal(0, 1, n)
    base_c = rng.normal(0, 1, n)

    X = np.column_stack([
        base_a,                          # 0: independent
        base_a + rng.normal(0, 0.01, n), # 1: near-copy of 0
        base_a + rng.normal(0, 0.05, n), # 2: near-copy of 0
        base_b,                          # 3: independent
        base_b + rng.normal(0, 0.02, n), # 4: near-copy of 3
        base_c,                          # 5: independent
        base_c + rng.normal(0, 0.01, n), # 6: near-copy of 5
        base_c + rng.normal(0, 0.03, n), # 7: near-copy of 5
        rng.normal(0, 1, n),             # 8: independent noise
        rng.normal(0, 1, n),             # 9: independent noise
    ])

    names = [f"f{i}" for i in range(10)]
    selected, info = cluster_features(X, names, corr_threshold=0.7)

    # Should keep ~5 features (3 base clusters + 2 independent noise)
    assert 4 <= len(selected) <= 7
    # The near-copies should be dropped
    assert len(selected) < 10


def test_cluster_with_target_picks_best() -> None:
    from ml.feature_selection import cluster_features

    rng = np.random.default_rng(7)
    n = 500

    # Feature 0 correlates with target; feature 1 is a copy but noisier
    target = rng.choice([-1, 0, 1], n).astype(float)
    f0 = target * 0.5 + rng.normal(0, 0.1, n)  # high MI with target
    f1 = f0 + rng.normal(0, 0.01, n)            # copy of f0 (lower MI)

    X = np.column_stack([f0, f1])
    names = ["good_feature", "copy_feature"]

    selected, info = cluster_features(X, names, y=target, corr_threshold=0.7)

    # Should keep the one with higher target correlation
    assert "good_feature" in selected
    assert len(selected) == 1


def test_select_features_pipeline() -> None:
    from ml.feature_selection import select_features_pipeline

    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, (200, 20))
    y = rng.choice([-1, 0, 1], 200)
    names = [f"feat_{i}" for i in range(20)]

    selected_names, X_reduced = select_features_pipeline(X, y, names, corr_threshold=0.9)

    assert X_reduced.shape[0] == 200
    assert X_reduced.shape[1] == len(selected_names)
    assert X_reduced.shape[1] <= 20
