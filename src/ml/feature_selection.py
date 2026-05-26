"""
Feature Selection via Hierarchical Clustering (REQ-P2-03).

Reduces multicollinearity by:
  1. Computing the correlation matrix of all features
  2. Hierarchical clustering on (1 - |corr|) distance
  3. Cutting the dendrogram at a threshold to form clusters
  4. Keeping one representative per cluster (highest mutual information with target)

This replaces the naive "use all 162 features" approach with a principled
reduction that preserves information while eliminating redundancy.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from core.logger import get_logger

logger = get_logger("midas.feature_selection")


def cluster_features(
    X: np.ndarray,
    feature_names: List[str],
    y: Optional[np.ndarray] = None,
    corr_threshold: float = 0.7,
    method: str = "average",
) -> Tuple[List[str], dict]:
    """Reduce features by clustering correlated ones and keeping the best per cluster.

    Args:
        X: Feature matrix (n_samples, n_features).
        feature_names: Names corresponding to columns of X.
        y: Target array (used to pick the best feature per cluster via MI).
        corr_threshold: Correlation threshold for clustering. Features with
            |corr| > threshold are grouped together.
        method: Linkage method ('average', 'complete', 'ward').

    Returns:
        Tuple of (selected_feature_names, cluster_info_dict).
    """
    n_features = X.shape[1]
    if n_features != len(feature_names):
        raise ValueError(f"X has {n_features} cols but {len(feature_names)} names")

    # Correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    # Replace NaN correlations (constant features) with 0
    corr = np.nan_to_num(corr, nan=0.0)

    # Distance = 1 - |correlation|
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Ensure symmetry and non-negative
    dist = np.clip((dist + dist.T) / 2, 0, None)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)

    # Cut at threshold (distance = 1 - corr_threshold)
    cut_distance = 1 - corr_threshold
    clusters = fcluster(Z, t=cut_distance, criterion="distance")

    # Group features by cluster
    cluster_map: dict = {}
    for idx, cluster_id in enumerate(clusters):
        cluster_map.setdefault(int(cluster_id), []).append(idx)

    # Select best feature per cluster
    selected_indices = []
    cluster_info = {}

    for cluster_id, indices in cluster_map.items():
        if len(indices) == 1:
            selected_indices.append(indices[0])
            cluster_info[feature_names[indices[0]]] = {
                "cluster": cluster_id,
                "cluster_size": 1,
                "dropped": [],
            }
        else:
            # Pick the feature with highest variance (or MI if target provided)
            if y is not None:
                # Mutual information proxy: correlation with target
                target_corrs = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in indices]
                target_corrs = [0.0 if np.isnan(c) else c for c in target_corrs]
                best_local = indices[int(np.argmax(target_corrs))]
            else:
                # Fallback: highest variance
                variances = [X[:, i].var() for i in indices]
                best_local = indices[int(np.argmax(variances))]

            selected_indices.append(best_local)
            dropped = [feature_names[i] for i in indices if i != best_local]
            cluster_info[feature_names[best_local]] = {
                "cluster": cluster_id,
                "cluster_size": len(indices),
                "dropped": dropped,
            }

    selected_names = [feature_names[i] for i in sorted(selected_indices)]

    logger.info(
        f"Feature clustering: {n_features} -> {len(selected_names)} features "
        f"({n_features - len(selected_names)} dropped, "
        f"threshold={corr_threshold}, clusters={len(cluster_map)})"
    )

    return selected_names, cluster_info


def select_features_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    corr_threshold: float = 0.7,
) -> Tuple[List[str], np.ndarray]:
    """Full feature selection pipeline: cluster + filter.

    Returns:
        Tuple of (selected_names, X_train_reduced).
    """
    selected_names, info = cluster_features(
        X_train, feature_names, y=y_train, corr_threshold=corr_threshold
    )

    # Get column indices for selected features
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_idx = [name_to_idx[n] for n in selected_names]

    X_reduced = X_train[:, selected_idx]

    return selected_names, X_reduced
