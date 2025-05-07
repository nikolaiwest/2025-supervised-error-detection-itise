import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.evaluation import evaluate_model
from src.data import load_data, process_data
from src.models import get_classifier_dict
from src.utils import get_logger

from .sampling import get_sampling_data


class ExperimentRunner:
    """Base class for running experiments."""

    def __init__(
        self,
        experiment_type: str,
        model_selection: str,
        scenario_id: str = "s04",
        target_length: int = 2000,
        screw_positions: str = "left",
        save_results: bool = True,
        cv_folds: int = 5,
        output_dir: str = "results",
        log_level: str = "INFO",
    ):
        """
        Initialize the experiment runner.
        Parameters:
        -----------
        experiment_type : str
            Type of experiment to run (binary_vs_ref, binary_vs_all, multiclass_all, multiclass_group)
        model_selection : str
            Model selection strategy (fast, paper, full)
        scenario_id : str, optional
            Dataset scenario ID (default: "s04")
        target_length : int, optional
            Target length of time series (default: 2000)
        screw_positions : str, optional
            Screw positions to use (default: "left")
        save_results : bool, optional
            Whether to save results (default: True)
        cv_folds : int, optional
            Number of cross-validation folds (default: 5)
        output_dir : str, optional
            Directory to save results (default: "results")
        log_level: str, optional
            Logging level (default: "INFO")
        """
        # Experiment
        self.experiment_type = experiment_type
        self.model_selection = model_selection
        self.scenario_id = scenario_id
        # Processing
        self.target_length = target_length
        self.screw_positions = screw_positions
        # Modeling
        self.cv_folds = cv_folds
        # Other
        self._save_results = save_results
        self._output_dir = output_dir
        # Set up logger
        self.logger = get_logger(__name__, log_level)
        self.results = []
        # Log configuration
        self.logger.info(f"Initializing {experiment_type} experiment")
        self.logger.info(f"  Dataset: {scenario_id}, Models: {model_selection}")
        self.logger.info(f"  CV folds: {cv_folds}, Save results: {save_results}")

        self.results = []

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data based on configuration."""
        self.logger.info(f"Load: scenario={self.scenario_id}, len={self.target_length}")
        return load_data(self.scenario_id, self.target_length, self.screw_positions)

    def _process_data(
        self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply preprocessing to the data."""
        return process_data(data)

    def _apply_model(self, data: Tuple, model_name: str, model) -> Dict[str, Any]:
        """Apply a model to a dataset using cross-validation."""
        x_values = data["x_values"]
        y_values = data["y_values"]

        self.logger.info(f"Evaluating {model_name} on dataset: {data['name']}")

        try:
            # Set up cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

            scores = []
            for train_idx, test_idx in cv.split(x_values, y_values):
                x_train, x_test = x_values[train_idx], x_values[test_idx]
                y_train, y_test = y_values[train_idx], y_values[test_idx]
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                scores.append(evaluate_model(y_test, y_pred))

            # Mittelwerte berechnen
            result = {
                "dataset": data["name"],
                "model": model_name,
                **{
                    metric: np.mean([fold[metric] for fold in scores])
                    for metric in scores[0]
                },
                **{k: v for k, v in data.items() if k not in ["X", "y"]},
            }

            self.logger.info(
                f"{model_name}: Evaluation complete (f1_score = f1: {result['f1_score']:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            # Return error result
            result = {
                "dataset": data,
                "model": model_name,
                "error": str(e),
                **{k: v for k, v in data.items() if k not in ["X", "y"]},
            }

        return result

    def _store_results(
        self, results: List[Dict[str, Any]], output_path: Optional[str] = None
    ) -> None:
        """
        Save experiment results.

        Parameters:
        -----------
        results : List[Dict[str, Any]]
            Results to save
        output_path : str, optional
            Path to save results to
        """
        if not self._save_results:
            self.logger.info("Skipping result saving (_save_results=False)")
            return

        if output_path is None:
            # Create default path based on attributes
            output_path = os.path.join(
                self._output_dir, self.scenario_id, self.experiment_type, "results.csv"
            )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save results
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")

    def run(self) -> List[Dict[str, Any]]:
        """
        Run the experiment.
        Returns:
        --------
        List[Dict[str, Any]]: Experiment results
        """
        # Load and preprocess data
        data = self._load_data()
        data = self._process_data(data)

        # Get sampling config and model dict
        experiment_data = get_sampling_data(data, self.experiment_type)
        experiment_models = get_classifier_dict(self.model_selection, 42, -1)

        # Run experiments for each combination of dataset and model
        self.results = []
        for data in experiment_data:
            for model_name, model in experiment_models.items():
                result = self._apply_model(data, model_name, model)
                self.results.append(result)

        # Store results
        self._store_results(self.results)
