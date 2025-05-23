"""
Result handling module for machine learning experiments.

This module provides classes for organizing and analyzing results at different
levels of the experiment hierarchy:

- FoldResult: Results for a single cross-validation fold
- ModelResult: Results for a model applied to a dataset (across CV folds)
- DatasetResult: Results for multiple models applied to a dataset
- ExperimentResult: Results for an entire experiment across datasets

These classes help organize and analyze results, track performance metrics
at different levels, and integrate with MLflow for experiment tracking.
"""

from .dataset_result import DatasetResult
from .experiment_result import ExperimentResult
from .fold_result import FoldResult
from .model_result import ModelResult

__all__ = ["FoldResult", "ModelResult", "DatasetResult", "ExperimentResult"]
