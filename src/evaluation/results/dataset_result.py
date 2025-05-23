from typing import Any, Dict, List, Optional

import numpy as np

from .model_result import ModelResult


class DatasetResult:
    """Results from evaluating all models on a single dataset."""

    def __init__(self, dataset_name: str, dataset_tags: Dict[str, Any]):
        """Initialize dataset result to hold multiple model results."""
        # Get name and tags for the dataset
        self.dataset_name = dataset_name
        self.dataset_tags = dataset_tags

        # Initialize an empty list to hold all model results
        self.model_results: List[ModelResult] = []

    def add_result(self, model_result: ModelResult) -> None:
        """Add a model result to this dataset."""
        if not isinstance(model_result, ModelResult):
            raise TypeError("model_result must be a ModelResult instance")
        self.model_results.append(model_result)

    def get_tags(self) -> Dict[str, Any]:
        """Get tags for MLflow dataset logging."""
        return self.dataset_tags.copy()

    def get_best_model(self, metric: str = "f1_score") -> Optional[ModelResult]:
        """Get the best performing model for this dataset."""
        if not self.model_results:
            return None

        valid_results = [
            result
            for result in self.model_results
            if metric in result.mean_metrics
            and not np.isnan(result.mean_metrics[metric])
        ]

        if not valid_results:
            return None

        return max(valid_results, key=lambda x: x.mean_metrics[metric])

    def get_model_ranking(self, metric: str = "f1_score") -> List[ModelResult]:
        """Get models ranked by performance on this dataset."""
        valid_results = [
            result
            for result in self.model_results
            if metric in result.mean_metrics
            and not np.isnan(result.mean_metrics[metric])
        ]

        return sorted(valid_results, key=lambda x: x.mean_metrics[metric], reverse=True)

    def get_performance_summary(self, metric: str = "f1_score") -> Dict[str, float]:
        """Get performance summary across all models for this dataset."""
        if not self.model_results:
            return {"count": 0}

        # Get metric values from all models
        values = [
            result.get_mean_metric(metric)
            for result in self.model_results
            if result.mean_metrics
        ]

        if not values:
            return {"count": 0}

        import numpy as np

        return {
            "count": len(values),
            "best": max(values),
            "worst": min(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
        }

    def __repr__(self) -> str:
        return f"DatasetResult({self.dataset_name}, {len(self.model_results)} model results)"
