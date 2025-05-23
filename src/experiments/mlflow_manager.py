import os
from typing import Optional

import mlflow
import pandas as pd

from src.evaluation.results import DatasetResult, ExperimentResult, ModelResult
from src.utils import get_logger

from .exceptions import FatalExperimentError


class MLflowManager:
    """Handles MLflow tracking for hierarchical experiment runs."""

    def __init__(self, port: int = 5000):
        self.port = port
        self.logger = get_logger(__name__)

    def setup_tracking(self, experiment_name: str) -> None:
        """Initialize MLflow tracking connection."""
        try:
            mlflow.set_tracking_uri(f"http://localhost:{self.port}")
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow tracking initialized for '{experiment_name}'")
        except Exception as e:
            raise FatalExperimentError(
                f"MLflow initialization failed: {str(e)}. "
                "Please ensure MLflow server is running."
            ) from e

    def log_experiment_start(self, experiment_result: ExperimentResult) -> str:
        """Start main experiment run and log configuration."""
        # Generate run name from config
        scenario_id = experiment_result.experiment_config.get("scenario_id", "unknown")
        model_selection = experiment_result.experiment_config.get(
            "model_selection", "unknown"
        )
        run_name = f"{scenario_id}_{model_selection}"

        run = mlflow.start_run(run_name=run_name)

        # Log experiment-level data
        mlflow.log_params(experiment_result.get_mlflow_params())
        mlflow.log_metrics(experiment_result.get_mlflow_metrics())

        for key, value in experiment_result.get_mlflow_tags().items():
            mlflow.set_tag(key, value)

        self.logger.info(f"Started experiment run: {run_name}")
        return run.info.run_id

    def log_dataset_start(self, dataset_result: DatasetResult) -> str:
        """Start nested dataset run and log dataset metadata."""
        run = mlflow.start_run(run_name=dataset_result.dataset_name, nested=True)

        # Log dataset characteristics
        mlflow.log_params(dataset_result.get_mlflow_params())
        mlflow.log_metrics(dataset_result.get_mlflow_metrics())

        for key, value in dataset_result.get_mlflow_tags().items():
            mlflow.set_tag(key, value)

        self.logger.debug(f"Started dataset run: {dataset_result.dataset_name}")
        return run.info.run_id

    def log_model_evaluation(self, model_result: ModelResult) -> Optional[str]:
        """Log model evaluation with metrics and artifacts."""
        try:
            run = mlflow.start_run(run_name=model_result.model_name, nested=True)

            # Log model metadata (safe parameters only)
            if model_result.metadata:
                safe_params = {
                    k: v
                    for k, v in model_result.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                if safe_params:
                    mlflow.log_params(safe_params)

            # Log basic performance metrics
            basic_metrics = {
                k: v
                for k, v in model_result.metrics.items()
                if not any(
                    k.endswith(suffix) for suffix in ["_std", "_var", "_min", "_max"]
                )
            }
            if basic_metrics:
                mlflow.log_metrics(basic_metrics)

            # Log statistical information with stats_ prefix
            stats_metrics = {
                f"stats_{k}": v
                for k, v in model_result.metrics.items()
                if any(
                    k.endswith(suffix) for suffix in ["_std", "_var", "_min", "_max"]
                )
            }
            if stats_metrics:
                mlflow.log_metrics(stats_metrics)

            # Log confusion matrix as artifact
            if model_result.confusion_matrix is not None:
                self._log_confusion_matrix(model_result)

            self.logger.debug(f"Logged evaluation: {model_result.model_name}")
            return run.info.run_id

        except Exception as e:
            self.logger.warning(f"Failed to log {model_result.model_name}: {str(e)}")
            return None

    def _log_confusion_matrix(self, model_result: ModelResult) -> None:
        """Log confusion matrix as MLflow artifact."""
        cm_path = f"{model_result.model_name}_cm.csv"
        try:
            pd.DataFrame(model_result.confusion_matrix).to_csv(cm_path, index=False)
            mlflow.log_artifact(cm_path)
            self.logger.debug(f"Logged confusion matrix for {model_result.model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to log confusion matrix: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(cm_path):
                os.remove(cm_path)

    def end_run(self) -> None:
        """End current MLflow run safely."""
        try:
            mlflow.end_run()
        except Exception as e:
            self.logger.warning(f"Error ending MLflow run: {str(e)}")
