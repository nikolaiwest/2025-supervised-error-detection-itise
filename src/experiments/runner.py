import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sktime.classification.base import BaseClassifier

from src.data import load_data, process_data
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
        Execute the complete experiment workflow.

        This method orchestrates the entire experiment pipeline:
        1. Loads and preprocesses the time series data
        2. Prepares dataset configurations based on the experiment type
        3. Initializes the selected machine learning models
        4. Evaluates each model on each dataset using cross-validation
        5. Collects and stores the results

        The method handles the coordination between all components of the
        experiment, ensuring proper data flow and result collection.

        Returns:
        --------
        List[Dict[str, Any]]
            List of dictionaries containing evaluation results for all
            model-dataset combinations in the experiment
        """
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

        # Run experiments for each combination of dataset and model
        self.results = []
        for data in experiment_data:
            for model_name, model in experiment_models.items():
                result, confusion_matrix = self._apply_model(data, model_name, model)
                self.results.append(result)

                # Save the confusion matrix if available
                if confusion_matrix is not None and True:  # disable during debugging
                    self._save_confusion_matrix(confusion_matrix, model_name)

        # Store results
        self._save_results_csv(self.results)

        return self.results
