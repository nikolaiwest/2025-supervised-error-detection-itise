import os
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sktime.classification.base import BaseClassifier

from src.data import load_data, process_data
from src.experiments.server import launch_server
from src.modeling import apply_model, get_classifier_dict
from src.utils import get_logger

from .sampling import get_sampling_data


class ExperimentRunner:
    """
    Base class for running machine learning experiments on time series data.

    This class manages the entire experimental workflow, including:
    - Data loading and preprocessing
    - Model training and evaluation using cross-validation
    - Result collection and storage

    It supports various experiment types like binary and multiclass classification
    with different sampling strategies and model selections.
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
        save_results: bool = True,
        cv_folds: int = 5,
        random_seed: int = 42,
        n_jobs: int = -1,
        stratify: bool = True,
        output_dir: str = "results",
        log_level: str = "INFO",
    ):
        """
        Initialize the experiment runner with configuration parameters.

        Parameters:
        -----------
        experiment_type : str
            Type of experiment to run. Options include:
            - 'binary_vs_ref': Binary classification comparing to reference
            - 'binary_vs_all': Binary classification of one class vs. all others
            - 'multiclass_all': Multiclass classification using all classes
            - 'multiclass_group': Multiclass classification within groups
        model_selection : str
            Model selection strategy. Options include:
            - 'fast': Quick models for rapid experimentation
            - 'paper': Models used in the referenced paper
            - 'full': Comprehensive set of classifier models
        scenario_id : str, optional
            Dataset scenario identifier (default: "s04")
        target_length : int, optional
            Target length of time series for PAA preprocessing (default: 2000)
        screw_positions : str, optional
            Screw positions to use in the dataset (default: "left")
        save_results : bool, optional
            Whether to save results to disk (default: True)
        cv_folds : int, optional
            Number of cross-validation folds for model evaluation (default: 5)
        random_seed : int, optional
            Random seed for reproducibility in models and CV (default: 42)
        n_jobs : int, optional
            Number of parallel jobs for models that support it (default: -1)
            -1 means using all available processors
        stratify : bool, optional
            Whether to use stratified cross-validation (default: True)
        output_dir : str, optional
            Directory path to save results (default: "results")
        log_level: str, optional
            Logging level for debug information (default: "INFO")
        """
        # Experiment configuration
        self.experiment_type = experiment_type
        self.model_selection = model_selection
        self.scenario_id = scenario_id
        # Processing parameters
        self.target_length = target_length
        self.screw_positions = screw_positions
        # Modeling parameters
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.stratify = stratify
        # Storage configuration
        self._save_results = save_results
        self._output_dir = output_dir
        # Set up logger for tracking experiment progress
        self.logger = get_logger(__name__, log_level)
        self.results: List[Dict[str, Any]] = []
        # Log experiment configuration
        self.logger.info(f"Initializing {experiment_type} experiment")
        self.logger.info(f"  Dataset: {scenario_id}, Models: {model_selection}")
        self.logger.info(
            f"  CV folds: {cv_folds}, Random seed: {random_seed}, n_jobs: {n_jobs}"
        )
        self.logger.info(f"  Stratify: {stratify}, Save results: {save_results}")

    @classmethod
    def _ensure_mlflow_server(cls):
        """Ensure MLflow server is running, start it if needed."""
        if cls._mlflow_server_running:
            return

        try:
            # Start the MLflow server using the imported function
            launch_server(port=cls._MLFLOW_PORT)
            cls._mlflow_server_running = True
        except Exception as e:
            print(f"Warning: Could not start MLflow server: {str(e)}")
            print("Will try to connect to existing server...")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data based on the experiment configuration.

        This method retrieves the raw time series data along with class labels
        and scenario condition information using the configured scenario ID,
        target length, and screw positions.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - torque_values: Time series data with shape (n_samples, n_timestamps)
            - class_values: Class labels for each sample
            - scenario_condition: Additional categorical information for each sample
        """
        self.logger.info(f"Load: scenario={self.scenario_id}, len={self.target_length}")
        return load_data(self.scenario_id, self.target_length, self.screw_positions)

    def _process_data(
        self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply preprocessing to the time series data.

        This method applies a preprocessing pipeline to the raw time series data,
        including:
        1. Dimensionality reduction using Piecewise Aggregate Approximation (PAA)
        2. Normalization to standardize the scale of features

        The preprocessing helps improve model performance by reducing computational
        complexity and preserving important patterns while standardizing feature scales.

        Parameters:
        -----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Raw data tuple containing (torque_values, class_values, scenario_condition)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Processed data tuple with the same structure but reduced dimensionality
        """
        return process_data(data, target_length=200)

    def _apply_model(
        self,
        data: Dict[str, Any],
        model_name: str,
        model: Union[BaseEstimator, BaseClassifier],
    ) -> Dict[str, Any]:
        """
        Apply a machine learning model to a dataset using cross-validation.

        This is a wrapper around the apply_model function in src.modeling.model
        that provides the experiment runner's configuration.

        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary containing dataset information
        model_name : str
            Name of the machine learning model
        model : Union[BaseEstimator, BaseClassifier]
            Machine learning classifier model

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing evaluation results
        """
        x_values = data["x_values"]
        y_values = data["y_values"]

        return apply_model(
            model_name=model_name,
            model=model,
            x_values=x_values,
            y_values=y_values,
            dataset_name=data["name"],
            cv_folds=self.cv_folds,
            stratify=self.stratify,
            random_state=self.random_seed,
        )

    def _save_confusion_matrix(self, confusion_matrix, model_name):
        """
        Save a confusion matrix as a CSV file.

        Parameters:
        -----------
        confusion_matrix : array-like
            The confusion matrix to save
        model_name : str
            Name of the model for the filename
        """
        if confusion_matrix is None:
            return

        # Create the full path for the confusion matrix directory
        cm_dir = os.path.join(
            self._output_dir,
            self.scenario_id,
            self.experiment_type,
            "confusion_matrix",
        )
        # Ensure the directory exists
        os.makedirs(cm_dir, exist_ok=True)

        # Create the full path for the CSV file
        cm_path = os.path.join(cm_dir, f"{model_name}_cm.csv")

        # Save the confusion matrix
        pd.DataFrame(confusion_matrix).to_csv(cm_path, index=False)
        self.logger.info(f"Confusion matrix saved to {cm_path}")

    def _save_results_csv(
        self, results: List[Dict[str, Any]], output_path: Optional[str] = None
    ) -> None:
        """
        Save experiment results to disk in CSV format.

        This method organizes results into a structured directory hierarchy
        based on the experiment configuration and writes them to a CSV file.
        The directory structure follows the pattern:
        {output_dir}/{scenario_id}/{experiment_type}/results.csv

        Parameters:
        -----------
        results : List[Dict[str, Any]]
            List of dictionaries containing evaluation results for each
            model-dataset combination
        output_path : str, optional
            Custom path to save results to. If None, a default path is generated
            based on experiment configuration.
        """
        if not self._save_results:
            self.logger.info("Skipping result saving (_save_results=False)")
            return

        if output_path is None:
            # Create default path based on experiment attributes
            output_path = os.path.join(
                self._output_dir, self.scenario_id, self.experiment_type, "results.csv"
            )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save results as CSV
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the complete experiment workflow with MLflow tracking.

        This method orchestrates the entire experiment pipeline:
        1. Loads and preprocesses the time series data
        2. Prepares dataset configurations based on the experiment type
        3. Initializes the selected machine learning models
        4. Evaluates each model on each dataset using cross-validation
        5. Collects and stores the results
        6. Tracks everything in MLflow

        Returns:
        --------
        List[Dict[str, Any]]
            List of dictionaries containing evaluation results for all
            model-dataset combinations in the experiment

        Raises:
        -------
        RuntimeError
            If MLflow server is not available
        """

        # Ensure MLflow server is running
        self._ensure_mlflow_server()

        # Load and preprocess data
        data = self._load_data()
        data = self._process_data(data)

        # Get sampling configuration and model dictionary
        experiment_data = get_sampling_data(
            data, self.experiment_type, self.scenario_id
        )
        experiment_models = get_classifier_dict(
            self.model_selection, random_seed=self.random_seed, n_jobs=self.n_jobs
        )

        # Set up MLflow tracking - using the class-defined port
        try:
            mlflow.set_tracking_uri(f"http://localhost:{self._MLFLOW_PORT}")
            mlflow.set_experiment(f"{self.experiment_type}")
            self.logger.info("MLflow tracking initialized successfully")
        except Exception as e:
            self.logger.error(f"MLflow initialization failed: {str(e)}")
            raise RuntimeError(
                "MLflow server is not available. Please start the MLflow server before running experiments."
            ) from e

        # Initialize results list
        self.results = []

        # Create a parent run for the entire experiment

        # E X P E R I M E N T (first layer)

        with mlflow.start_run(
            run_name=f"{self.scenario_id}_{self.model_selection}"
        ) as main_run:
            # Log experiment configuration parameters
            mlflow.log_params(
                {
                    "experiment_type": self.experiment_type,
                    "model_selection": self.model_selection,
                    "scenario_id": self.scenario_id,
                    "target_length": self.target_length,
                    "screw_positions": self.screw_positions,
                    "cv_folds": self.cv_folds,
                    "random_seed": self.random_seed,
                    "n_jobs": self.n_jobs,
                    "stratify": self.stratify,
                }
            )

        # Process each dataset
        for data_index, dataset in enumerate(experiment_data):
            self.logger.info(
                f"Processing {dataset['name']} ({data_index+1}/{len(experiment_data)})"
            )

            # D A T A S E T (second layer)

            # Create a nested run for this dataset
            with mlflow.start_run(run_name=dataset["name"], nested=True):

                # Log dataset characteristics
                mlflow.log_params(
                    {
                        "dataset_name": dataset["name"],
                        "n_samples": len(dataset["x_values"]),
                        "n_features": (
                            dataset["x_values"].shape[1]
                            if hasattr(dataset["x_values"], "shape")
                            else None
                        ),
                        "class_balance": str(
                            np.unique(dataset["y_values"], return_counts=True)[
                                1
                            ].tolist()
                        ),
                    }
                )

                # Process each model for this dataset
                for model_name, model in experiment_models.items():
                    self.logger.info(f"  Evaluating model: {model_name}")

                    # M O D E L

                    # Model evaluation with MLflow tracking
                    with mlflow.start_run(run_name=model_name, nested=True):
                        # Apply the model and get results
                        result, confusion_matrix = self._apply_model(
                            dataset, model_name, model
                        )

                        # Log the model and its metrics
                        try:
                            # Log model type/family information
                            mlflow.log_param(
                                "model_family", result.get("model_type", "unknown")
                            )

                            # Try to log the model itself if possible
                            mlflow.sklearn.log_model(model, "model")
                        except Exception as e:
                            self.logger.warning(
                                f"Could not log model artifact: {str(e)}"
                            )

                        # Log metrics
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and not isinstance(
                                value, bool
                            ):
                                mlflow.log_metric(key, value)
                            elif isinstance(value, (str, bool)):
                                mlflow.log_param(key, value)

                        # Store confusion matrix as an artifact
                        if confusion_matrix is not None:
                            cm_path = f"{model_name}_cm.csv"
                            pd.DataFrame(confusion_matrix).to_csv(cm_path, index=False)
                            mlflow.log_artifact(cm_path)
                            os.remove(cm_path)  # Clean up temp file

                        # Add to legacy results collection
                        self.results.append(result)

        # Save consolidated results to CSV
        self._save_results_csv(self.results)

        return self.results
