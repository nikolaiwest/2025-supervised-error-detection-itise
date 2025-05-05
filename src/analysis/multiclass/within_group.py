import os
import sys

# Temp: Add parent directory to path for imports while testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from src.data.loader import load_data
from src.models.classifiers import get_model_dict
from src.plots.confusion_matrix import plot_confusion_matrix
from src.plots.utils import ensure_directory, save_results_with_plots
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_multiclass_within_group(model_selection="fast", save_results=True):
    """
    Run multi-class classification within error groups:
    - Groups the 25 error classes into 5 groups
    - Creates a 6-class classification problem for each group:
        - Class 0: All normal samples from the five classes in the group
        - Classes 1-5: The five error classes in the group

    Data was collected in alternating batches of 5 reference (OK) and 5 faulty (NOK) samples
    for each error class, with a total of 100 reference and 100 faulty samples per class.
    This structured collection pattern may introduce time-dependent patterns that could
    affect model performance.

    Parameters:
    -----------
    model_selection : str
        Which model set to use ('paper', 'full', or 'fast')
    save_results : bool
        Whether to save results to disk

    Returns:
    --------
    DataFrame with classification results
    """
    logger.info(
        f"Running multi-class classification within groups (model selection: {model_selection})"
    )

    # Create results directory if saving results
    if save_results:
        ensure_directory("results/multiclass/within_group")
        ensure_directory("results/multiclass/within_group/images")
        logger.debug("Created results directories")

    # Load data
    logger.info("Loading data...")
    torque_values, class_values, label_values = load_data()
    logger.debug(
        f"Loaded {len(torque_values)} samples with {len(set(class_values))} classes"
    )

    # Get models
    logger.info(f"Getting models for selection: {model_selection}")
    model_dict = get_model_dict(model_selection)
    logger.debug(f"Loaded {len(model_dict)} models: {', '.join(model_dict.keys())}")

    # Define the error class groups
    # Assuming class values follow a pattern where groups can be identified
    # by the first digit or similar characteristic
    logger.info("Defining error class groups")
    all_classes = sorted(set(class_values))

    # Example grouping - adjust according to your actual class naming scheme
    # This example assumes classes can be grouped like: [100s, 200s, 300s, 400s, 500s]
    groups = {}
    for i in range(5):
        group_name = f"Group_{i+1}"
        start_idx = i * 5
        end_idx = start_idx + 5
        if end_idx <= len(all_classes):
            groups[group_name] = all_classes[start_idx:end_idx]

    logger.debug(f"Created {len(groups)} groups: {groups}")

    # Store all results
    all_results = []

    # Process each group
    for group_name, group_classes in groups.items():
        logger.info(f"Processing {group_name} with classes: {group_classes}")

        # Filter data for this group
        group_mask = np.isin(class_values, group_classes)
        group_torque_values = torque_values[group_mask]
        group_class_values = class_values[group_mask]
        group_label_values = label_values[group_mask]

        logger.debug(f"Filtered {len(group_torque_values)} samples for {group_name}")

        # Create class mapping for this group
        class_to_id = {
            class_val: idx + 1 for idx, class_val in enumerate(group_classes)
        }

        # Create reverse mapping (id to name) for readability
        id_to_name = {0: "normal"}  # Start with normal class
        for class_val, idx in class_to_id.items():
            id_to_name[idx] = class_val

        logger.debug(f"Class mapping for {group_name}: {class_to_id}")
        logger.debug(f"Reverse mapping for {group_name}: {id_to_name}")

        # Save mapping to file for future reference
        if save_results:
            mapping_df = pd.DataFrame(
                {
                    "class_id": list(id_to_name.keys()),
                    "class_name": list(id_to_name.values()),
                    "group": group_name,
                }
            )
            mapping_path = (
                f"results/multiclass/within_group/{group_name}_class_mapping.csv"
            )
            mapping_df.to_csv(mapping_path, index=False)
            logger.debug(f"Saved class mapping to {mapping_path}")

        # Create multi-class labels:
        # - 0 for all normal samples across all classes in this group
        # - 1-5 for each error class in this group
        y_values = np.zeros(len(group_torque_values), dtype=int)
        for i in range(len(group_torque_values)):
            if group_label_values[i] != "normal":
                y_values[i] = class_to_id[group_class_values[i]]

        # Log class distribution with readable names
        unique_y, counts = np.unique(y_values, return_counts=True)
        class_dist_readable = {}
        for y_val, count in zip(unique_y, counts):
            class_name = id_to_name[y_val]
            class_dist_readable[f"{y_val} ({class_name})"] = count

        logger.info(f"Class distribution for {group_name}: {class_dist_readable}")

        # Convert to sktime format for sktime models
        logger.debug("Converting to sktime format")
        X_sktime = from_2d_array_to_nested(group_torque_values)
        y = pd.Series(y_values)

        # Also prepare flattened format for sklearn models
        logger.debug("Preparing flattened format for sklearn models")
        X_sklearn = group_torque_values.reshape(group_torque_values.shape[0], -1)

        # Split into train/test sets
        logger.debug("Splitting data into train/test sets")
        (
            X_sktime_train,
            X_sktime_test,
            X_sklearn_train,
            X_sklearn_test,
            y_train,
            y_test,
        ) = train_test_split(
            X_sktime, X_sklearn, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.debug(
            f"Train set: {len(y_train)} samples, Test set: {len(y_test)} samples"
        )

        # Run all models for this group
        for model_name, model in model_dict.items():
            logger.info(f"Training {model_name} on {group_name}...")
            try:
                # Check if it's a sklearn model or sktime model
                if model.__module__.startswith("sklearn"):
                    # Use flattened data for sklearn models
                    logger.debug(
                        f"Using sklearn format for {model_name} (shape: {X_sklearn_train.shape})"
                    )
                    model.fit(X_sklearn_train, y_train)
                    y_pred = model.predict(X_sklearn_test)
                else:
                    # Use nested format for sktime models
                    logger.debug(f"Using sktime format for {model_name}")
                    model.fit(X_sktime_train, y_train)
                    y_pred = model.predict(X_sktime_test)

                # Evaluate
                report = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )

                # Store overall results
                result_entry = {
                    "group": group_name,
                    "model": model_name,
                    "accuracy": report["accuracy"],
                    "macro_precision": report["macro avg"]["precision"],
                    "macro_recall": report["macro avg"]["recall"],
                    "macro_f1": report["macro avg"]["f1-score"],
                    "weighted_precision": report["weighted avg"]["precision"],
                    "weighted_recall": report["weighted avg"]["recall"],
                    "weighted_f1": report["weighted avg"]["f1-score"],
                }

                # Add per-class metrics for each class in the group
                for class_idx in range(6):  # 0-5 classes
                    if str(class_idx) in report:
                        result_entry[f"precision_{class_idx}"] = report[str(class_idx)][
                            "precision"
                        ]
                        result_entry[f"recall_{class_idx}"] = report[str(class_idx)][
                            "recall"
                        ]
                        result_entry[f"f1_{class_idx}"] = report[str(class_idx)][
                            "f1-score"
                        ]
                        result_entry[f"support_{class_idx}"] = report[str(class_idx)][
                            "support"
                        ]

                all_results.append(result_entry)

                # Log performance metrics
                logger.info(
                    f"Performance for {model_name} on {group_name}: "
                    f"Accuracy={report['accuracy']:.4f}, "
                    f"Macro F1={report['macro avg']['f1-score']:.4f}"
                )

                # Generate confusion matrix visualization
                if save_results:
                    save_path = f"results/multiclass/within_group/images/confusion_matrix_{group_name}_{model_name}.png"
                    plot_confusion_matrix(
                        y_test, y_pred, group_name, model_name, save_path
                    )
                    logger.debug(f"Saved confusion matrix to {save_path}")

                logger.info(f"{model_name} on {group_name} completed successfully.")
            except Exception as e:
                logger.error(
                    f"Error with {model_name} on {group_name}: {str(e)}", exc_info=True
                )

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)

    # Log summary statistics
    if not results_df.empty:
        logger.info(f"Experiment completed with {len(results_df)} model evaluations")

        # Calculate average metrics across all models and groups
        avg_metrics = results_df[["accuracy", "macro_f1", "weighted_f1"]].mean()
        logger.info(
            f"Average metrics across all models and groups: "
            f"Accuracy={avg_metrics['accuracy']:.4f}, "
            f"Macro F1={avg_metrics['macro_f1']:.4f}"
        )

        # Find best performing model by macro F1 score
        best_model_idx = results_df["macro_f1"].idxmax()
        best_model = results_df.iloc[best_model_idx]
        logger.info(
            f"Best performing model: {best_model['model']} on {best_model['group']} "
            f"with Macro F1={best_model['macro_f1']:.4f}"
        )

        # Report best performance by group
        logger.info("Best model performance by group:")
        for group_name in groups.keys():
            group_df = results_df[results_df["group"] == group_name]
            if not group_df.empty:
                best_group_idx = group_df["macro_f1"].idxmax()
                best_group_model = group_df.iloc[best_group_idx]
                logger.info(
                    f"{group_name}: {best_group_model['model']} "
                    f"with Macro F1={best_group_model['macro_f1']:.4f}"
                )
    else:
        logger.warning("No results generated during experiment")

    # Save results if requested
    if save_results and not results_df.empty:
        logger.info("Saving results and visualizations")
        save_path = "results/multiclass/within_group/results.csv"
        results_df.to_csv(save_path, index=False)
        logger.info(f"Results saved to {save_path}")

    return results_df


if __name__ == "__main__":
    # Run the experiment
    logger.info("Starting experiment run_multiclass_within_group")
    results = run_multiclass_within_group(model_selection="paper", save_results=True)
    logger.info("Experiment completed")
