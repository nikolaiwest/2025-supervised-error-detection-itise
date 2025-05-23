from typing import Any, Dict, Optional

import numpy as np


class FoldResult:
    """Results from evaluating a model on a single cross-validation fold."""

    def __init__(
        self,
        fold_index: int,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        training_time: float = 0.0,
        prediction_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize fold result."""
        if fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if not metrics:
            raise ValueError("metrics cannot be empty")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        self.fold_index = fold_index
        self.metrics = self._validate_metrics(metrics)
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.training_time = max(0.0, training_time)
        self.prediction_time = max(0.0, prediction_time)
        self.metadata = metadata or {}

    def _validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean metrics."""
        validated = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                validated[key] = float(value)
            else:
                # Log warning but don't fail
                validated[key] = 0.0
        return validated

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for this fold."""
        from sklearn.metrics import confusion_matrix

        return confusion_matrix(self.y_true, self.y_pred)

    def get_mlflow_params(self) -> Dict[str, Any]:
        """Get parameters suitable for MLflow logging."""
        params = {
            "fold_index": self.fold_index,
            "train_samples": len(self.y_true) if hasattr(self, "y_true") else 0,
            "test_samples": len(self.y_pred) if hasattr(self, "y_pred") else 0,
        }

        # Add safe metadata (only basic types)
        safe_metadata = {
            k: v
            for k, v in self.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
        params.update(safe_metadata)

        return params

    def get_mlflow_metrics(self) -> Dict[str, float]:
        """Get metrics suitable for MLflow logging."""
        metrics = {}

        # Add performance metrics
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics[key] = float(value)

        # Add timing metrics
        metrics.update(
            {
                "training_time_seconds": self.training_time,
                "prediction_time_seconds": self.prediction_time,
            }
        )

        return metrics

    def __repr__(self) -> str:
        f1 = self.metrics.get("f1_score", 0)
        acc = self.metrics.get("accuracy", 0)
        return f"FoldResult(fold={self.fold_index}, f1={f1:.3f}, acc={acc:.3f})"
