from typing import Any, Dict, List, Tuple, Union

from sklearn.base import BaseEstimator
from sktime.classification.base import BaseClassifier

from src.data import load_data, process_data
from src.evaluation.results import DatasetResult, ExperimentResult, ModelResult
from src.experiments.server import launch_server
from src.modeling import apply_model, get_classifier_dict
from src.utils import get_logger

from .exceptions import (
    DatasetPreparationError,
    FatalExperimentError,
    ModelEvaluationError,
)
from .mlflow_manager import MLflowManager
from .sampling import get_sampling_data


class ExperimentRunner:
    """
    Orchestrates ML experiments on time series data with MLflow tracking.

    Supports binary/multiclass classification with cross-validation and
    hierarchical result tracking (Experiment -> Dataset -> Model -> Fold).
    Handles partial failures gracefully and provides comprehensive logging.
    """

    # Class variables to track MLflow server state
    _mlflow_server_running = False
    _MLFLOW_PORT = 5000

    def __init__(
        self,
        experiment_type: str,
        model_selection: str,
        scenario_id: str = "s04",
        target_length: int = 2000,
        screw_positions: str = "left",
        cv_folds: int = 5,
        random_seed: int = 42,
        n_jobs: int = -1,
        stratify: bool = True,
        log_level: str = "INFO",
    ):
        """Initialize experiment runner with configuration parameters."""
        # Core configuration
        self.experiment_type = experiment_type
        self.model_selection = model_selection
        self.scenario_id = scenario_id

        # Data processing parameters
        self.target_length = target_length
        self.screw_positions = screw_positions

        # Model evaluation parameters
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.stratify = stratify

        # Initialize logging
        self.logger = get_logger(__name__, log_level)

        # Initialize MLflow manager
        self.mlflow_manager = MLflowManager(port=self._MLFLOW_PORT)

        # Initialize experiment result container
        self.experiment_result = ExperimentResult(
            experiment_type=experiment_type,
            experiment_config={
                "model_selection": model_selection,
                "scenario_id": scenario_id,
                "target_length": target_length,
                "screw_positions": screw_positions,
                "cv_folds": cv_folds,
                "random_seed": random_seed,
                "n_jobs": n_jobs,
                "stratify": stratify,
            },
        )

        # Instance variables for stateful design
        self.datasets: List[Dict[str, Any]] = []
        self.models: Dict[str, Any] = {}

        self._log_initialization()

    @classmethod
    def _ensure_mlflow_server(cls) -> None:
        """Start MLflow server if not already running."""
        if cls._mlflow_server_running:
            return

        try:
            launch_server(port=cls._MLFLOW_PORT)
            cls._mlflow_server_running = True
        except Exception as e:
            raise FatalExperimentError(
                f"Could not start MLflow server: {str(e)}. "
                "Please start the MLflow server manually or check the configuration."
            ) from e

    def _log_initialization(self) -> None:
        """Log experiment configuration for debugging."""
        self.logger.info(f"Initializing {self.experiment_type} experiment")
        self.logger.info(
            f"  Dataset: {self.scenario_id}, Models: {self.model_selection}"
        )
        self.logger.info(
            f"  CV folds: {self.cv_folds}, Random seed: {self.random_seed}"
        )
        self.logger.info(f"  Stratify: {self.stratify}, n_jobs: {self.n_jobs}")

    def _setup_experiment(self) -> None:
        """Set up MLflow server and tracking."""
        self.logger.info("Setting up experiment environment...")

        # Ensure MLflow server is running
        self._ensure_mlflow_server()

        # Setup MLflow tracking
        self.mlflow_manager.setup_tracking(self.experiment_type)

    def _setup_datasets(self) -> None:
        """Load data, apply preprocessing, and initialize models."""
        try:
            self.logger.info("Loading and preprocessing data...")

            # Load and preprocess time series data
            raw_data = load_data(
                self.scenario_id, self.target_length, self.screw_positions
            )
            processed_data = process_data(raw_data, target_length=200)

            # Generate datasets based on experiment type (binary/multiclass)
            self.datasets = get_sampling_data(
                processed_data, self.experiment_type, self.scenario_id
            )

            # Initialize ML models with experiment configuration
            self.models = get_classifier_dict(
                self.model_selection, self.random_seed, self.n_jobs
            )

            self.logger.info(
                f"Prepared {len(self.datasets)} datasets and {len(self.models)} models"
            )

        except Exception as e:
            raise DatasetPreparationError(f"Failed to prepare data: {str(e)}") from e

    def _process_experiment(self) -> None:
        """Execute main experiment loop with MLflow tracking."""
        self.logger.info("Starting experiment execution...")

        # Start main experiment run
        main_run_id = self.mlflow_manager.log_experiment_start(self.experiment_result)

        try:
            # Process each dataset with all models
            for dataset_idx, dataset_dict in enumerate(self.datasets):
                self.logger.info(
                    f"Processing dataset {dataset_idx + 1}/{len(self.datasets)}: {dataset_dict['name']}"
                )

                dataset_result = self._process_single_dataset(dataset_dict)
                self.experiment_result.add_dataset_result(dataset_result)

        finally:
            self.mlflow_manager.end_run()

    def _process_single_dataset(self, dataset_dict: Dict[str, Any]) -> DatasetResult:
        """Process all models on a single dataset with error recovery."""

        # Create DatasetResult with metadata (exclude actual data arrays)
        dataset_result = DatasetResult(
            dataset_name=dataset_dict["name"],
            dataset_info={
                k: v
                for k, v in dataset_dict.items()
                if k not in ["x_values", "y_values"]  # Exclude data arrays
            },
        )

        # Start dataset-level MLflow run
        dataset_run_id = self.mlflow_manager.log_dataset_start(dataset_result)

        try:
            # Process all models, handling individual failures gracefully
            successful_models, failed_models = self._process_all_models(dataset_dict)

            # Add successful results to dataset
            for model_result in successful_models:
                dataset_result.add_model_result(model_result)

            # Log any model failures (but continue experiment)
            if failed_models:
                self.logger.warning(
                    f"Failed models on {dataset_dict['name']}: "
                    f"{[name for name, _ in failed_models]}"
                )

        finally:
            self.mlflow_manager.end_run()

        return dataset_result

    def _process_all_models(
        self, dataset_dict: Dict[str, Any]
    ) -> Tuple[List[ModelResult], List[Tuple[str, str]]]:
        """Process all models on dataset, collecting successes and failures."""

        successful_results = []
        failed_models = []

        for model_name, model in self.models.items():
            try:
                self.logger.info(f"  Evaluating model: {model_name}")

                # Evaluate single model with cross-validation
                model_result = self._evaluate_single_model(
                    model_name, model, dataset_dict
                )
                successful_results.append(model_result)

                # Log results to MLflow
                self.mlflow_manager.log_model_evaluation(model_result)

            except ModelEvaluationError as e:
                self.logger.error(f"Model {model_name} failed: {str(e)}")
                failed_models.append((model_name, str(e)))
                # Continue with remaining models

        return successful_results, failed_models

    def _evaluate_single_model(
        self,
        model_name: str,
        model: Union[BaseEstimator, BaseClassifier],
        dataset_dict: Dict[str, Any],
    ) -> ModelResult:
        """Evaluate model with cross-validation, handling failures gracefully."""

        try:
            # Apply model with cross-validation (returns ModelResult)
            model_result = apply_model(
                model_name=model_name,
                model=model,
                x_values=dataset_dict["x_values"],
                y_values=dataset_dict["y_values"],
                dataset_name=dataset_dict["name"],
                cv_folds=self.cv_folds,
                stratify=self.stratify,
                random_state=self.random_seed,
            )

            # Log performance for quick monitoring
            f1_score = model_result.metrics.get("f1_score", 0)
            self.logger.info(f"    {model_name}: f1_score = {f1_score:.3f}")

            return model_result

        except Exception as e:
            raise ModelEvaluationError(
                f"Failed to evaluate {model_name} on {dataset_dict['name']}: {str(e)}"
            ) from e

    def _evaluate_experiment(self) -> None:
        """Generate experiment summary and log best performers."""
        self.logger.info("Evaluating experiment results...")

        # Calculate success statistics
        total_evaluations = sum(
            len(dr.model_results) for dr in self.experiment_result.dataset_results
        )

        successful_evaluations = sum(
            1
            for dr in self.experiment_result.dataset_results
            for mr in dr.model_results
            if mr.metrics  # Has metrics = successful evaluation
        )

        # Log experiment summary
        self.logger.info(f"Experiment completed:")
        self.logger.info(
            f"  Total datasets: {len(self.experiment_result.dataset_results)}"
        )
        self.logger.info(
            f"  Successful evaluations: {successful_evaluations}/{total_evaluations}"
        )

        # Log best performing models for each dataset
        self._log_best_performers()

    def _log_best_performers(self) -> None:
        """Log the best performing model for each dataset."""

        for dataset_result in self.experiment_result.dataset_results:
            if not dataset_result.model_results:
                continue

            # Find best model by F1 score
            best_model = max(
                dataset_result.model_results,
                key=lambda mr: mr.metrics.get("f1_score", 0),
            )

            f1_score = best_model.metrics.get("f1_score", 0)
            self.logger.info(
                f"  {dataset_result.dataset_name}: {best_model.model_name} (f1_score: {f1_score:.3f})"
            )

    def run(self) -> ExperimentResult:
        """
        Execute complete experiment workflow.

        Returns hierarchical results with statistical analysis across
        all models and datasets.
        """
        try:
            # Setup datasets and models
            self._setup_datasets()

            # Setup experiment environment
            self._setup_experiment()

            # Execute experiment
            self._process_experiment()

            # Generate summary and analysis
            self._evaluate_experiment()

            return self.experiment_result

        except FatalExperimentError:
            raise  # Re-raise fatal errors
        except Exception as e:
            # Wrap unexpected errors as fatal
            raise FatalExperimentError(
                f"Unexpected error in experiment: {str(e)}"
            ) from e
