from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .dataset_result import DatasetResult
from .model_result import ModelResult


class ExperimentResult:
    """Results from running a complete experiment across multiple datasets."""

    def __init__(self, experiment_name: str, model_selection: str, scenario_id: str):
        """Initialize experiment result class to hold multiple dataset results."""
        # Core configuration (same as in ExperimentRunner)
        self.experiment_name = experiment_name
        self.model_selection = model_selection
        self.scenario_id = scenario_id

        # Initialize an empty list to hold all dataset results
        self.dataset_results: List[DatasetResult] = []

        # Set up variables to tag the experiment time
        self.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.finish_time: Optional[str] = None

    def add_result(self, dataset_result: DatasetResult) -> None:
        """Add a dataset result to this experiment."""
        if not isinstance(dataset_result, DatasetResult):
            raise TypeError("'dataset_result' must be a DatasetResult instance")
        self.dataset_results.append(dataset_result)

    def get_tags(self) -> Dict[str, Any]:
        """Get tags for MLflow experiment logging."""
        tags = {
            "experiment_type": self.experiment_name,
            "scenario_id": self.scenario_id,
            "model_selection": self.model_selection,
            "start_time": self.start_time,
            "status": "running",  # Will be updated by MLflowManager
        }

        if self.finish_time:
            tags["finish_time"] = self.finish_time
            tags["status"] = "completed"

        return tags

    def get_model_selection(self) -> str:
        """Get the model selection strategy used."""
        return self.model_selection

    def get_scenario_id(self) -> str:
        """Get the scenario ID."""
        return self.scenario_id

    def finalize(self) -> None:
        """Mark experiment as complete."""
        self.finish_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    def get_overall_best_models(
        self, metric: str = "f1_score"
    ) -> Dict[str, "ModelResult"]:
        """Get the best model for each dataset across the experiment."""
        best_models = {}

        for dataset_result in self.dataset_results:
            best_model = dataset_result.get_best_model(metric)
            if best_model:
                best_models[dataset_result.dataset_name] = best_model

        return best_models

    def get_model_performance_summary(
        self, metric: str = "f1_score"
    ) -> Dict[str, Dict[str, float]]:
        """Get average performance of each model across all datasets."""
        model_scores = {}

        for dataset_result in self.dataset_results:
            for model_result in dataset_result.model_results:
                model_name = model_result.model_name
                score = model_result.get_mean_metric(metric)

                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(score)

        # Calculate statistics for each model
        import numpy as np

        summary = {}
        for model_name, scores in model_scores.items():
            summary[model_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "count": len(scores),
            }

        return summary

    def get_dataset_difficulty_ranking(
        self, metric: str = "f1_score"
    ) -> List[Dict[str, Any]]:
        """Get datasets ranked by difficulty (based on average model performance)."""
        dataset_difficulties = []

        for dataset_result in self.dataset_results:
            summary = dataset_result.get_performance_summary(metric)
            if summary.get("count", 0) > 0:
                dataset_difficulties.append(
                    {
                        "dataset_name": dataset_result.dataset_name,
                        "avg_performance": summary["mean"],
                        "best_performance": summary["best"],
                        "worst_performance": summary["worst"],
                        "performance_std": summary["std"],
                        "model_count": summary["count"],
                    }
                )

        # Sort by average performance (ascending = hardest first)
        return sorted(dataset_difficulties, key=lambda x: x["avg_performance"])

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        total_datasets = len(self.dataset_results)
        total_models = sum(len(dr.model_results) for dr in self.dataset_results)
        total_folds = sum(
            len(mr.fold_results)
            for dr in self.dataset_results
            for mr in dr.model_results
        )

        return {
            "experiment_name": self.experiment_name,
            "scenario_id": self.scenario_id,
            "model_selection": self.model_selection,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "total_datasets": total_datasets,
            "total_models": total_models,
            "total_folds": total_folds,
            "status": "completed" if self.finish_time else "running",
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        import pandas as pd

        rows = []
        for dataset_result in self.dataset_results:
            for model_result in dataset_result.model_results:
                # Base row information
                row = {
                    "experiment_name": self.experiment_name,
                    "scenario_id": self.scenario_id,
                    "model_selection": self.model_selection,
                    "dataset": dataset_result.dataset_name,
                    "model": model_result.model_name,
                    "n_folds": len(model_result.fold_results),
                }

                # Add mean metrics
                row.update(model_result.mean_metrics)

                # Add standard deviations with _std suffix
                for metric, value in model_result.std_metrics.items():
                    row[f"{metric}_std"] = value

                rows.append(row)

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        status = "completed" if self.finish_time else "running"
        return f"ExperimentResult({self.experiment_name}, {len(self.dataset_results)} datasets, {status})"
