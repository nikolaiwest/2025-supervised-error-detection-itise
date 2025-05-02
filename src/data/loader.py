"""
Data handling module for pyscrew analysis
"""

import numpy as np
import pandas as pd
import pyscrew
from sktime.datatypes._panel._convert import from_2d_array_to_nested


def load_data(dataset_id="s04"):
    """Load and return the pyscrew dataset.

    Parameters:
    -----------
    dataset_id : str
        Dataset ID to load from pyscrew

    Returns:
    --------
    tuple: (torque_values, class_values, label_values)
    """
    data = pyscrew.get_data(dataset_id)  #  screw_positions="left"
    torque_values = np.array(data["torque values"])
    class_values = np.array(data["class values"])
    # TODO: Load information using pyscrew (requires an upate)
    label_values = np.array(pd.read_csv("data/labels.csv")["scenario_condition"])
    return torque_values, class_values, label_values


def prepare_binary_dataset(torque_values, label_values, normal_indices, faulty_indices):
    """Prepare a binary dataset from specified normal and faulty indices.

    Parameters:
    -----------
    torque_values : ndarray
        Array of torque time series
    label_values : ndarray
        Array of label values
    normal_indices : ndarray
        Indices of normal samples
    faulty_indices : ndarray
        Indices of faulty samples

    Returns:
    --------
    tuple: (X, y) in sktime format
    """
    # Combine indices and extract data
    combined_indices = np.concatenate([normal_indices, faulty_indices])
    X_values = torque_values[combined_indices]
    y_values = np.concatenate(
        [np.zeros(len(normal_indices)), np.ones(len(faulty_indices))]
    )

    # Convert to sktime format
    X = from_2d_array_to_nested(X_values)
    y = pd.Series(y_values)

    return X, y


def prepare_multiclass_dataset(
    torque_values, class_values, label_values, class_filter_func
):
    """Prepare a multiclass dataset based on a filtering function.

    Parameters:
    -----------
    torque_values : ndarray
        Array of torque time series
    class_values : ndarray
        Array of class values
    label_values : ndarray
        Array of label values
    class_filter_func : callable
        Function that takes class_values and returns a boolean mask

    Returns:
    --------
    tuple: (X, y) in sktime format
    """
    # Apply filter to get relevant data
    class_mask = class_filter_func(class_values)
    x_values = torque_values[class_mask]
    y_values = label_values[class_mask]

    # Convert to sktime format
    X = from_2d_array_to_nested(x_values)
    y = pd.Series(y_values)

    return X, y
