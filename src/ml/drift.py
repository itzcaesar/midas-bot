"""
Feature Drift Monitoring (REQ-P2-06).

Computes Population Stability Index (PSI) between the training distribution
and the live inference distribution for each feature. When PSI > threshold,
the model's assumptions about the data have shifted and retraining is needed.

PSI interpretation:
  - PSI < 0.10: No significant shift
  - 0.10 <= PSI < 0.20: Moderate shift — monitor closely
  - PSI >= 0.20: Significant shift — retrain recommended

Usage:
    from ml.drift import DriftMonitor
    monitor = DriftMonitor.from_training_data(X_train, feature_names)
    report = monitor.check(X_live)
    if report.has_drift:
        send_alert(report.summary())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.logger import get_logger

logger = get_logger("midas.drift")


def _psi_single(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI for a single feature using equal-frequency binning.

    Uses the training distribution's quantile boundaries to bin both arrays,
    then computes the PSI formula: sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct)).
    """
    # Compute bin edges from expected (training) distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    # Bin both distributions
    expected_counts = np.histogram(expected, bins=bin_edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=bin_edges)[0].astype(float)

    # Convert to proportions (add small epsilon to avoid log(0))
    eps = 1e-6
    expected_pct = expected_counts / expected_counts.sum() + eps
    actual_pct = actual_counts / actual_counts.sum() + eps

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


@dataclass
class DriftReport:
    """Results of a drift check."""
    feature_psi: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.20
    n_features_checked: int = 0
    n_features_drifted: int = 0

    @property
    def has_drift(self) -> bool:
        return self.n_features_drifted > 0

    @property
    def max_psi(self) -> float:
        return max(self.feature_psi.values()) if self.feature_psi else 0.0

    @property
    def drifted_features(self) -> List[str]:
        return [f for f, psi in self.feature_psi.items() if psi >= self.threshold]

    def summary(self) -> str:
        lines = [
            f"Drift report: {self.n_features_drifted}/{self.n_features_checked} features drifted "
            f"(threshold={self.threshold:.2f}, max_psi={self.max_psi:.3f})"
        ]
        if self.has_drift:
            top = sorted(self.feature_psi.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("Top drifted features:")
            for name, psi in top:
                flag = " ⚠️" if psi >= self.threshold else ""
                lines.append(f"  {name}: PSI={psi:.3f}{flag}")
        return "\n".join(lines)


class DriftMonitor:
    """Monitors feature distribution drift between training and live data."""

    def __init__(
        self,
        reference_stats: Dict[str, np.ndarray],
        feature_names: List[str],
        threshold: float = 0.20,
        n_bins: int = 10,
    ) -> None:
        self._reference = reference_stats
        self._feature_names = feature_names
        self._threshold = threshold
        self._n_bins = n_bins

    @classmethod
    def from_training_data(
        cls,
        X_train: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.20,
        n_bins: int = 10,
    ) -> "DriftMonitor":
        """Create a monitor from the training feature matrix."""
        reference = {}
        for i, name in enumerate(feature_names):
            col = X_train[:, i]
            # Store the raw column for PSI computation
            reference[name] = col.copy()
        return cls(reference, feature_names, threshold, n_bins)

    def check(self, X_live: np.ndarray, top_k: Optional[int] = None) -> DriftReport:
        """Check live data against training reference.

        Args:
            X_live: Live feature matrix (n_samples, n_features).
            top_k: Only check the top-K most important features (None = all).

        Returns:
            DriftReport with per-feature PSI values.
        """
        names = self._feature_names[:top_k] if top_k else self._feature_names
        report = DriftReport(threshold=self._threshold)

        for i, name in enumerate(names):
            if i >= X_live.shape[1]:
                break
            if name not in self._reference:
                continue

            ref = self._reference[name]
            live = X_live[:, i]

            # Skip if insufficient data
            if len(live) < 30 or len(ref) < 30:
                continue

            psi = _psi_single(ref, live, self._n_bins)
            report.feature_psi[name] = psi

        report.n_features_checked = len(report.feature_psi)
        report.n_features_drifted = sum(
            1 for psi in report.feature_psi.values() if psi >= self._threshold
        )

        if report.has_drift:
            logger.warning(report.summary())
        else:
            logger.debug(f"Drift check clean: max PSI={report.max_psi:.3f}")

        return report

    def save(self, path: str = "models/drift_reference.npz") -> None:
        """Save reference distributions to disk."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            feature_names=np.array(self._feature_names),
            **{f"ref_{name}": arr for name, arr in self._reference.items()},
        )
        logger.info(f"Drift reference saved to {filepath}")

    @classmethod
    def load(cls, path: str = "models/drift_reference.npz", threshold: float = 0.20) -> "DriftMonitor":
        """Load reference distributions from disk."""
        data = np.load(path, allow_pickle=True)
        feature_names = list(data["feature_names"])
        reference = {name: data[f"ref_{name}"] for name in feature_names if f"ref_{name}" in data}
        return cls(reference, feature_names, threshold)
