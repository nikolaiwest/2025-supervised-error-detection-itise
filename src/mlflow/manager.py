import os
from typing import Optional

import mlflow
import pandas as pd

from src.evaluation.results import (
    DatasetResult,
    ExperimentResult,
    FoldResult,
    ModelResult,
)
from src.utils import get_logger
from src.utils.exceptions import FatalExperimentError


class MLflowManager:
    """Handles MLflow tracking for hierarchical experiment runs with real-time fold logging."""

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

    def start_main_run(self, experiment_result: ExperimentResult) -> None:
        """Start main experiment run and log configuration."""

        # Generate run name from inital experiment result
        scenario_id = experiment_result.get_scenario_id()
        model_selection = experiment_result.get_model_selection()
        run_name = f"run_{scenario_id}_{model_selection}"

        # Start the mlflow run (at root level aka not nested)
        run_description = "Parent run for all datasets and models in the experiment."
        mlflow.start_run(run_name=run_name, description=run_description)
        mlflow.set_tags(experiment_result.get_tags())

        self.logger.info(f"Started main run for the experiment: '{run_name}'")

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

    def log_model_start(self, model_name: str, dataset_name: str) -> str:
        """Start nested model run for fold logging."""
        run = mlflow.start_run(run_name=model_name, nested=True)

        # Log basic model info
        mlflow.log_params({"model_name": model_name, "dataset_name": dataset_name})

        mlflow.set_tag("evaluation_phase", "fold_by_fold")

        self.logger.debug(f"Started model run: {model_name}")
        return run.info.run_id

    def log_fold_result(
        self, fold_result: FoldResult, model_name: str, dataset_name: str
    ) -> Optional[str]:
        """
        Log individual fold result in real-time.

        This creates a nested run under the current model run and logs all fold metrics.
        """
        try:
            fold_run_name = f"fold_{fold_result.fold_index}"

            with mlflow.start_run(run_name=fold_run_name, nested=True):
                # Log fold parameters
                mlflow.log_params(fold_result.get_mlflow_params())

                # Log fold metrics
                mlflow.log_metrics(fold_result.get_mlflow_metrics())

                # Log fold tags
                mlflow.set_tag("fold_index", str(fold_result.fold_index))
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("dataset_name", dataset_name)

                # Log confusion matrix if available
                try:
                    confusion_matrix = fold_result.get_confusion_matrix()
                    if confusion_matrix is not None:
                        cm_path = f"fold_{fold_result.fold_index}_cm.csv"
                        pd.DataFrame(confusion_matrix).to_csv(cm_path, index=False)
                        mlflow.log_artifact(cm_path)
                        os.remove(cm_path)  # Clean up
                except Exception as e:
                    self.logger.warning(
                        f"Failed to log confusion matrix for fold {fold_result.fold_index}: {str(e)}"
                    )

                run_id = mlflow.active_run().info.run_id
                self.logger.debug(
                    f"Logged fold {fold_result.fold_index} for {model_name}"
                )
                return run_id

        except Exception as e:
            self.logger.warning(
                f"Failed to log fold {fold_result.fold_index} for {model_name}: {str(e)}"
            )
            return None

    def log_model_evaluation(self, model_result: ModelResult) -> Optional[str]:
        """Log aggregated model evaluation results."""
        try:
            # Log model metadata (safe parameters only)
            if model_result.model_metadata:
                safe_params = {
                    k: v
                    for k, v in model_result.model_metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                if safe_params:
                    mlflow.log_params(safe_params)

            # Log aggregated performance metrics
            metrics = model_result.get_mlflow_metrics()
            if metrics:
                mlflow.log_metrics(metrics)

            # Log model tags
            tags = model_result.get_mlflow_tags()
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            # Log aggregated confusion matrix as artifact
            if model_result.confusion_matrix is not None:
                self._log_confusion_matrix(model_result)

            # Log fold stability analysis
            self._log_fold_stability_analysis(model_result)

            self.logger.debug(
                f"Logged aggregated evaluation: {model_result.model_name}"
            )
            return mlflow.active_run().info.run_id if mlflow.active_run() else None

        except Exception as e:
            self.logger.warning(
                f"Failed to log aggregated results for {model_result.model_name}: {str(e)}"
            )
            return None

    def _log_fold_stability_analysis(self, model_result: ModelResult) -> None:
        """Log fold stability analysis as metrics and tags."""
        try:
            # Log stability for key metrics
            for metric_name in ["f1_score", "accuracy", "precision", "recall"]:
                if metric_name in model_result.mean_metrics:
                    stability = model_result.get_fold_stability(metric_name)
                    mlflow.set_tag(f"{metric_name}_stability", stability)

                    # Log coefficient of variation as a metric
                    summary = model_result.get_metric_summary(metric_name)
                    mlflow.log_metric(f"{metric_name}_cv", summary["cv"])

        except Exception as e:
            self.logger.warning(f"Failed to log stability analysis: {str(e)}")

    def _log_confusion_matrix(self, model_result: ModelResult) -> None:
        """Log aggregated confusion matrix as MLflow artifact."""
        cm_path = f"{model_result.model_name}_aggregated_cm.csv"
        try:
            pd.DataFrame(model_result.confusion_matrix).to_csv(cm_path, index=False)
            mlflow.log_artifact(cm_path)
            self.logger.debug(
                f"Logged aggregated confusion matrix for {model_result.model_name}"
            )
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
