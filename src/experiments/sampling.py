"""
Sampling strategies for different experiment types.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SamplingError(Exception):
    """Raised when a dataset cannot be properly created due to sampling issues."""

    pass


@dataclass
class ExperimentDataset:
    """Standardized dataset representation for all experiment types."""

    # Required core fields (always present)
    name: str
    experiment_type: str
    x_values: np.ndarray  # Consistent naming for model input
    y_values: np.ndarray  # Consistent naming for target labels

    # Metadata fields (can be None if not applicable)
    class_name: Optional[str] = None  # For all experiment types
    group_name: Optional[str] = None  # For group-based experiments

    # Class information
    num_classes: int = 2  # Default for binary, overridden for multiclass
    class_mapping: Optional[Dict] = None  # Maps original class values to numeric IDs
    class_names: Optional[Dict] = None  # Maps numeric IDs to human-readable names

    # Sample counts
    normal_samples: Optional[int] = None
    faulty_samples: Optional[int] = None
    class_distribution: Optional[Dict] = None  # Count of samples per class
    imbalance_ratio: Optional[float] = None

    # Additional metadata
    description: Optional[str] = None  # Human-readable description
    additional_info: Optional[Dict] = None  # Any other experiment-specific info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result


def get_sampling_data(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray], experiment_type: str
) -> List[Dict[str, Any]]:
    """
    Generate datasets for different experiment configurations based on experiment type.

    Parameters:
    -----------
    data : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (torque_values, class_values, scenario_condition)
    experiment_type : str
        Type of experiment to run (binary_vs_ref, binary_vs_all, multiclass_all, multiclass_group)

    Returns:
    --------
    List[Dict[str, Any]]: List of dataset configurations, each containing:
        - 'name': Unique identifier for the dataset
        - 'x_values': Feature data (torque values)
        - 'y_values': Target labels
        - Additional metadata specific to the experiment type
    """
    # Dispatch to the appropriate sampling strategy based on experiment type
    if experiment_type == "binary_vs_ref":
        return _get_binary_vs_ref_data(data[0], data[1], data[2])
    elif experiment_type == "binary_vs_all":
        return _get_binary_vs_all_data(data[0], data[1], data[2])
    elif experiment_type == "multiclass_with_groups":
        return _get_multiclass_with_groups(data[0], data[1], data[2])
    elif experiment_type == "multiclass_with_all":
        return _get_multiclass_with_all(data[0], data[1], data[2])
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def _get_binary_vs_ref_data(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[Dict[str, Any]]:
    """Generate datasets for binary classification of errors vs reference in one class."""
    datasets = []

    for class_value in sorted(set(class_values)):
        # Filter data for this class
        class_mask = class_values == class_value
        filtered_torque_values = torque_values[class_mask]
        filtered_condition = scenario_condition[class_mask]

        # Get indices of normal and faulty samples
        normal_mask = filtered_condition == "normal"
        faulty_mask = filtered_condition != "normal"  # == "faulty"
        n_normal = np.sum(normal_mask)
        n_faulty = np.sum(faulty_mask)

        # Check for classes with insufficient samples
        if n_normal == 0 or n_faulty == 0:
            raise SamplingError(
                f"Could not create vs_all dataset for class {class_value} due to missing samples "
                f"(n_normal={n_normal}, n_faulty={n_faulty})"
            )

        # Create binary labels (0 for normal, 1 for faulty) for modeling
        y_values = np.zeros(len(filtered_condition), dtype=int)
        y_values[faulty_mask] = 1

        # Calculate a few simple metrics for logging
        class_ratio = n_normal / n_faulty if n_faulty > 0 else float("inf")
        class_distribution = {"normal": int(n_normal), "faulty": int(n_faulty)}
        class_names = {0: "normal", 1: "faulty"}

        # Create dataset using standardized format
        dataset = ExperimentDataset(
            name=f"binary_vs_ref_{class_value}",
            experiment_type="binary_vs_ref",
            x_values=filtered_torque_values,  # Consistent field name
            y_values=y_values,  # Consistent field name
            class_name=class_value,  # Specific class being analyzed
            num_classes=2,  # Binary classification
            class_names=class_names,  # Human-readable class names
            normal_samples=n_normal,
            faulty_samples=n_faulty,
            class_distribution=class_distribution,
            imbalance_ratio=class_ratio,
            description=f"Binary classification of normal vs. faulty samples for class {class_value}",
        )

        datasets.append(dataset.to_dict())

    # Check if we have any datasets
    if not datasets:
        raise SamplingError(
            f"No valid datasets could be created for binary_vs_ref experiment"
        )

    return datasets


def _get_binary_vs_all_data(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[Dict[str, Any]]:
    """Generate datasets for binary classification comparing each class's faulty samples vs ALL normal samples."""
    datasets = []

    # Find all normal samples across all classes in the scenario
    normal_mask = scenario_condition == "normal"
    normal_indices = np.where(normal_mask)[0]
    normal_torque = torque_values[normal_indices]
    n_normal = len(normal_indices)

    # Process each class separately for its faulty samples
    for class_value in sorted(set(class_values)):
        # Filter to get only faulty samples for this class
        class_mask = class_values == class_value
        class_condition = scenario_condition[class_mask]
        faulty_mask = class_condition != "normal"
        n_faulty = np.sum(faulty_mask)

        # Check for classes with insufficient samples
        if n_normal == 0 or n_faulty == 0:
            raise SamplingError(
                f"Could not create vs_all dataset for class {class_value} due to missing samples "
                f"(n_normal={n_normal}, n_faulty={n_faulty})"
            )

        # Get faulty samples for this class
        class_torque = torque_values[class_mask]
        faulty_torque = class_torque[faulty_mask]

        # Combine normal samples from all classes with faulty samples from this class
        x_combined = np.vstack([normal_torque, faulty_torque])

        # Create binary labels (0 for normal, 1 for faulty)
        y_values = np.zeros(len(x_combined), dtype=int)
        y_values[n_normal:] = 1

        # Calculate class ratio and distribution
        class_ratio = n_normal / n_faulty if n_faulty > 0 else float("inf")
        class_distribution = {"normal": int(n_normal), "faulty": int(n_faulty)}
        class_names = {0: "normal", 1: "faulty"}

        # Create dataset using standardized format
        dataset = ExperimentDataset(
            name=f"binary_vs_all_{class_value}",
            experiment_type="binary_vs_all",
            x_values=x_combined,
            y_values=y_values,
            class_name=class_value,
            num_classes=2,
            class_names=class_names,
            normal_samples=n_normal,
            faulty_samples=n_faulty,
            class_distribution=class_distribution,
            imbalance_ratio=class_ratio,
            description=f"Binary classification of ALL normal samples vs faulty samples for class {class_value}",
        )

        datasets.append(dataset.to_dict())

    return datasets


def _get_multiclass_with_groups(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[Dict[str, Any]]:
    """Generate datasets for multi-class classification within error groups."""
    datasets = []
    # Define groups
    # TODO: move to a separate file or so for multiple scenarios (not just s04...)
    error_groups = {
        # Variations in screw thread quality
        "group_1": [
            "101_deformed-thread",
            "102_filed-screw-tip",
            "103_glued-screw-tip",
            "104_coated-screw",
            "105_worn-out-screw",
        ],
        # Variation in workpiece behavior
        "group_2": [
            "201_damaged-contact-surface",
            "202_broken-contact-surface",
            "203_metal-ring-upper-part",
            "204_rubber-ring-upper-part",
            "205_different-material",
        ],
        # Variation in the screw hole
        "group_3": [
            "301_plastic-pin-screw-hole",
            "302_enlarged-screw-hole",
            "303_less-glass-fiber",
            "304_glued-screw-hole",
            "305_gap-between-parts",
        ],
        # Environmental variations
        "group_4": [
            "401_surface-lubricant",
            "402_surface-moisture",
            "403_plastic-chip",
            "404_increased-temperature",
            "405_decreased-temperature",
        ],
        # Variations in process parameters
        "group_5": [
            "001_control-group",
            "501_increased-ang-velocity",
            "502_decreased-ang-velocity",
            "503_increased-torque",
            "504_decreased-torque",
        ],
    }

    # Process each group
    for group_name, group_errors in error_groups.items():
        # Filter data for this group's classes
        class_mask = np.isin(class_values, group_errors)
        filtered_torque_values = torque_values[class_mask]
        filtered_class_values = class_values[class_mask]
        filtered_condition = scenario_condition[class_mask]

        # Create mapping for class labels:
        # 0 = normal (regardless of class)
        # 1-5 = faulty samples from each class
        class_mapping = {}
        for i, class_val in enumerate(sorted(group_errors)):
            class_mapping[class_val] = i + 1  # Faulty samples get 1-5

        # Create reverse mapping for readability
        class_names = {0: "normal"}  # Start with normal class
        for class_val, idx in class_mapping.items():
            class_names[idx] = f"faulty_{class_val}"

        # Initialize y_values with zeros (normal)
        y_values = np.zeros(len(filtered_torque_values), dtype=int)

        # Set faulty samples to their respective class values
        for i in range(len(filtered_torque_values)):
            if filtered_condition[i] != "normal":
                y_values[i] = class_mapping[filtered_class_values[i]]

        # Verify we have samples for each class
        unique_y, counts = np.unique(y_values, return_counts=True)
        class_distribution = {
            class_names[y_val]: int(count) for y_val, count in zip(unique_y, counts)
        }

        # Calculate normal and faulty counts
        n_normal = np.sum(y_values == 0)
        n_faulty = len(y_values) - n_normal

        # Skip if we don't have both normal and faulty samples
        if n_normal == 0 or n_faulty == 0:
            continue

        # Create dataset using standardized format
        dataset = ExperimentDataset(
            name=f"multiclass_{group_name}",
            experiment_type="multiclass_group",
            x_values=filtered_torque_values,
            y_values=y_values,
            group_name=group_name,
            num_classes=len(group_errors) + 1,  # Normal + all classes in group
            class_mapping=class_mapping,
            class_names=class_names,
            normal_samples=n_normal,
            faulty_samples=n_faulty,
            class_distribution=class_distribution,
            imbalance_ratio=n_normal / n_faulty if n_faulty > 0 else float("inf"),
            description=f"Multi-class classification within error group '{group_name}' (0=normal, 1-{len(group_errors)}=faulty class)",
            additional_info={"group_errors": group_errors},
        )

        datasets.append(dataset.to_dict())

    # Check if we have any datasets
    if not datasets:
        raise SamplingError(
            f"No valid datasets could be created for multiclass_group experiment"
        )

    return datasets


def _get_multiclass_with_all(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[Dict[str, Any]]:
    """Generate dataset for multi-class classification with one class for normals and N classes for errors."""
    # Map class values to integers (1-25 or however many classes)
    unique_classes = sorted(set(class_values))
    class_mapping = {class_val: idx + 1 for idx, class_val in enumerate(unique_classes)}

    # Check that each class has samples
    for class_val in unique_classes:
        class_mask = class_values == class_val
        normal_mask = (class_values == class_val) & (scenario_condition == "normal")
        faulty_mask = (class_values == class_val) & (scenario_condition != "normal")
        n_normal = np.sum(normal_mask)
        n_faulty = np.sum(faulty_mask)

        # Check for classes with insufficient samples
        if n_normal == 0 or n_faulty == 0:
            raise SamplingError(
                f"Could not create multiclass_all dataset: class {class_val} has missing samples "
                f"(n_normal={n_normal}, n_faulty={n_faulty})"
            )

    # Create reverse mapping for readability
    class_names = {0: "normal"}  # Start with normal class
    for class_val, idx in class_mapping.items():
        class_names[idx] = class_val

    # Create multiclass labels (0 for normal, 1-N for error classes)
    y_values = np.zeros(len(torque_values), dtype=int)
    for i in range(len(torque_values)):
        if scenario_condition[i] != "normal":
            y_values[i] = class_mapping[class_values[i]]

    # Count samples per class for metadata
    unique_y, counts = np.unique(y_values, return_counts=True)
    class_distribution = {
        class_names[y_val]: int(count) for y_val, count in zip(unique_y, counts)
    }

    # Calculate normal vs faulty ratio for reference
    n_normal = class_distribution.get("normal", 0)
    n_faulty = sum(v for k, v in class_distribution.items() if k != "normal")
    class_ratio = n_normal / n_faulty if n_faulty > 0 else float("inf")

    # Final check for at least some samples in both categories
    if n_normal == 0 or n_faulty == 0:
        raise SamplingError(
            f"Could not create multiclass_all dataset: missing samples "
            f"(n_normal={n_normal}, n_faulty={n_faulty})"
        )

    # Create dataset using standardized format
    dataset = ExperimentDataset(
        name="multiclass_all_errors",
        experiment_type="multiclass_all",
        x_values=torque_values,
        y_values=y_values,
        num_classes=len(unique_classes) + 1,  # Normal + all error classes
        class_mapping=class_mapping,
        class_names=class_names,
        class_distribution=class_distribution,
        normal_samples=n_normal,
        faulty_samples=n_faulty,
        imbalance_ratio=class_ratio,
        description=f"Multi-class classification with class 0 for normal samples and {len(unique_classes)} classes for different error types",
    )

    return [dataset.to_dict()]  # Return as a list for consistent interface
