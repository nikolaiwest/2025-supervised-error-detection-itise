from typing import Any, Dict, Generator, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from src.evaluation import evaluate_model
from src.utils import get_logger

# Set up module logger
logger = get_logger(__name__)

# Type aliases for better readability
Features = Union[np.ndarray, pd.DataFrame]
Labels = Union[np.ndarray, pd.Series]
CVSplit = Tuple[Features, Features, Labels, Labels]


def apply_model(
    model_name: str,
    model: Union[BaseEstimator, BaseClassifier],
    x_values: Union[np.ndarray, pd.DataFrame],
    y_values: Union[np.ndarray, pd.Series],
    dataset_name: str,
    cv_folds: int = 5,
    stratify: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Apply a machine learning model to a dataset using cross-validation.

    This function handles the entire workflow of applying a model:
    1. Formatting data appropriately based on model type
    2. Performing cross-validation splits
    3. Training and evaluating the model
    4. Aggregating and returning results

    Parameters:
    -----------
    model_name : str
        Name of the machine learning model
    model : Union[BaseEstimator, BaseClassifier]
        Machine learning classifier model
    x_values : Union[np.ndarray, pd.DataFrame]
        Feature matrix with shape (n_samples, n_features) or (n_samples, n_timesteps)
    y_values : Union[np.ndarray, pd.Series]
        Target values with shape (n_samples,)
    dataset_name : str
        Name of the dataset for result identification
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)
    stratify : bool, optional
        Whether to use stratified cross-validation (default: True)
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing evaluation results with metrics like accuracy,
        precision, recall, and F1 score, or error information if an exception occurs
    """
    # Determine if this is a sktime model using improved function
    is_sktime_model = _is_sktime_classifier(model)

    # Format input data for this specific model type
    try:
        x_formatted = _format_data_for_modeling(x_values, model, is_sktime_model)
    except Exception as e:
        logger.error(f"Error formatting data for {model_name}: {str(e)}")
        return {
            "dataset": dataset_name,
            "model": model_name,
            "error": f"Data formatting error: {str(e)}",
        }

    logger.info(f"Evaluating {model_name} on dataset: {dataset_name}")

    try:
        scores = []
        all_y_test = []
        all_y_pred = []

        # Use CV generator to get properly aligned splits
        for x_train, x_test, y_train, y_test in _cross_validation_split(
            x_formatted,
            y_values,
            n_splits=cv_folds,
            is_sktime_model=is_sktime_model,
            stratify=stratify,
            random_state=random_state,
        ):
            # Train the model
            model.fit(x_train, y_train)

            # Get predictions
            y_pred = model.predict(x_test)

            # Standardize prediction format for evaluation
            y_pred = _format_pred_for_evaluation(y_pred)

            # Get y_test in appropriate format for evaluation
            y_test_values = (
                y_test.values
                if isinstance(y_test, (pd.Series, pd.DataFrame))
                else y_test
            )

            # Store for confusion matrix calculation
            all_y_test.append(y_test_values)
            all_y_pred.append(y_pred)

            # Evaluate
            scores.append(evaluate_model(y_test_values, y_pred))

        # Calculate average metrics across all folds
        result = {
            "dataset": dataset_name,
            "model": model_name,
            "model_type": "sktime" if is_sktime_model else "sklearn",
            "stratify": stratify,
            **{
                metric: np.mean([fold[metric] for fold in scores])
                for metric in scores[0]
            },
        }

        logger.info(
            f"{model_name}: Evaluation complete (f1_score: {result.get('f1_score', 0):.2f})"
        )

        # Calculate confusion matrix across all folds
        all_y_test_flat = np.concatenate(all_y_test)
        all_y_pred_flat = np.concatenate(all_y_pred)
        cm = confusion_matrix(all_y_test_flat, all_y_pred_flat)

        # Return the result dict and confusion matrix
        return result, cm

    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {str(e)}")
        return {
            "dataset": dataset_name,
            "model": model_name,
            "error": str(e),
        }, None


def _is_sktime_classifier(model: Union[BaseEstimator, BaseClassifier]) -> bool:
    """
    More robustly determine if a model is a sktime classifier.

    Parameters:
    -----------
    model : Union[BaseEstimator, BaseClassifier]
        The model to check

    Returns:
    --------
    bool
        True if the model is a sktime classifier, False otherwise
    """
    # Check if it's an instance of sktime's BaseClassifier
    if isinstance(model, BaseClassifier):
        return True

    # Check for sktime-specific attributes
    if hasattr(model, "_is_sktime_classifier") or hasattr(
        model, "_is_sktime_estimator"
    ):
        return True

    # Check the module path (less reliable but useful as fallback)
    if hasattr(model, "__module__") and any(
        mod in model.__module__ for mod in ["sktime", "sktime_dl", "sktime_forest"]
    ):
        return True

    # Not a sktime classifier
    return False


def _format_data_for_modeling(
    x_values: Union[np.ndarray, pd.DataFrame],
    model: Union[BaseEstimator, BaseClassifier],
    is_sktime_model: bool = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Format data to be compatible with the specific model type.

    Parameters:
    -----------
    x_values : Union[np.ndarray, pd.DataFrame]
        Feature matrix
    model : Union[BaseEstimator, BaseClassifier]
        Machine learning model
    is_sktime_model : bool, optional
        Whether the model is a sktime model. If None, will be determined.

    Returns:
    --------
    Union[np.ndarray, pd.DataFrame]
        Data formatted for the specific model type
    """
    if is_sktime_model is None:
        is_sktime_model = _is_sktime_classifier(model)

    if not is_sktime_model:
        # For sklearn models: ensure numpy array format
        return x_values
    else:
        # Convert to nested format for sktime models
        return from_2d_array_to_nested(x_values)


def _format_pred_for_evaluation(
    y_pred: Union[np.ndarray, pd.Series, pd.DataFrame],
) -> np.ndarray:
    """
    Format model predictions to a standard format for evaluation.

    Parameters:
    -----------
    y_pred : Union[np.ndarray, pd.Series, pd.DataFrame]
        Predictions from a model

    Returns:
    --------
    np.ndarray
        Predictions in a standardized numpy array format
    """
    # For pandas objects
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        return y_pred.values

    # For any other case where we don't have a numpy array
    if not isinstance(y_pred, np.ndarray):
        return np.array(y_pred)

    # Already in the correct format
    return y_pred


def _cross_validation_split(
    x_values: Features,
    y_values: Labels,
    n_splits: int = 5,
    is_sktime_model: bool = False,
    stratify: bool = True,
    random_state: int = 42,
) -> Generator[CVSplit, None, None]:
    """
    Generate cross-validation splits with properly aligned indices.

    This generator function yields training and testing sets with indices
    properly aligned for both sklearn and sktime models.

    Parameters:
    -----------
    x_values : Union[np.ndarray, pd.DataFrame]
        Feature matrix or nested DataFrame for time series
    y_values : Union[np.ndarray, pd.Series]
        Target values
    n_splits : int, optional
        Number of CV folds (default: 5)
    is_sktime_model : bool, optional
        Whether the splits are for a sktime model (default: False)
    stratify : bool, optional
        Whether to use stratified cross-validation (default: True)
        If True, uses StratifiedKFold to maintain class distribution
        If False, uses regular KFold for random splits
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Yields:
    -------
    Tuple[x_train, x_test, y_train, y_test]
        Cross-validation splits with properly aligned indices
    """
    # Set up cross-validation with appropriate split strategy
    if stratify:
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_method = lambda: cv.split(x_values, y_values)
        logger.debug(f"Using StratifiedKFold with {n_splits} splits")
    else:
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_method = lambda: cv.split(x_values)
        logger.debug(f"Using regular KFold with {n_splits} splits")

    # Handle differently based on model type
    if is_sktime_model:
        # For sktime models (using pandas objects)
        for train_idx, test_idx in split_method():
            # Reset indices to ensure alignment
            x_train = x_values.iloc[train_idx].reset_index(drop=True)
            x_test = x_values.iloc[test_idx].reset_index(drop=True)

            # Create Series with aligned indices for sktime models
            if isinstance(y_values, (pd.Series, pd.DataFrame)):
                y_train = pd.Series(
                    y_values.iloc[train_idx].values, index=x_train.index
                )
                y_test = pd.Series(y_values.iloc[test_idx].values, index=x_test.index)
            else:
                y_train = pd.Series(y_values[train_idx], index=x_train.index)
                y_test = pd.Series(y_values[test_idx], index=x_test.index)

            yield x_train, x_test, y_train, y_test
    else:
        # For sklearn models (using numpy arrays)
        for train_idx, test_idx in split_method():
            x_train, x_test = x_values[train_idx], x_values[test_idx]

            if isinstance(y_values, (pd.Series, pd.DataFrame)):
                y_train = y_values.iloc[train_idx].values
                y_test = y_values.iloc[test_idx].values
            else:
                y_train = y_values[train_idx]
                y_test = y_values[test_idx]

            yield x_train, x_test, y_train, y_test
