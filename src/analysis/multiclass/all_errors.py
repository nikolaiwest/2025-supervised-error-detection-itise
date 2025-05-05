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


def run_multiclass_all_errors(model_selection="fast", save_results=True):
    """
    Run multi-class classification: One class for all normal samples across all classes,
    and 25 separate classes for each error type.

    This creates a 26-class classification problem (1 normal + 25 error classes).

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
        f"Running multi-class classification: All errors (model selection: {model_selection})"
    )

    # Create results directory if saving results
    if save_results:
        ensure_directory("results/multiclass/all_errors")
        ensure_directory("results/multiclass/all_errors/images")
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

    # Create multi-class labels:
    # - 0 for all normal samples across all classes
    # - 1-25 for each error class (using the class_values as identifiers)
    logger.info("Preparing multi-class labels for all errors classification")

    # Map class values to integers (1-25)
    unique_class_values = sorted(set(class_values))
    class_to_id = {
        class_val: idx + 1 for idx, class_val in enumerate(unique_class_values)
    }

    # Create reverse mapping (id to name) for readability
    id_to_name = {0: "normal"}  # Start with normal class
    for class_val, idx in class_to_id.items():
        id_to_name[idx] = class_val

    logger.debug(f"Created class mapping: {class_to_id}")
    logger.debug(f"Created reverse mapping: {id_to_name}")

    # Save mapping to file for future reference
    if save_results:
        mapping_df = pd.DataFrame(
            {
                "class_id": list(id_to_name.keys()),
                "class_name": list(id_to_name.values()),
            }
        )
        mapping_df.to_csv(
            "results/multiclass/all_errors/class_mapping.csv", index=False
        )
        logger.debug(
            "Saved class mapping to results/multiclass/all_errors/class_mapping.csv"
        )

    # Create y values (0 for normal, 1-25 for error classes)
    y_values = np.zeros(len(torque_values), dtype=int)
    for i in range(len(torque_values)):
        if label_values[i] != "normal":
            y_values[i] = class_to_id[class_values[i]]

    # Log class distribution with readable names
    unique_y, counts = np.unique(y_values, return_counts=True)
    class_dist_readable = {}
    for y_val, count in zip(unique_y, counts):
        class_name = id_to_name[y_val]
        class_dist_readable[f"{y_val} ({class_name})"] = count

    logger.info(f"Class distribution: {class_dist_readable}")

    # Convert to sktime format for sktime models
    logger.debug("Converting to sktime format")
    X_sktime = from_2d_array_to_nested(torque_values)
    y = pd.Series(y_values)

    # Also prepare flattened format for sklearn models
    logger.debug("Preparing flattened format for sklearn models")
    X_sklearn = torque_values.reshape(torque_values.shape[0], -1)

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
    logger.debug(f"Train set: {len(y_train)} samples, Test set: {len(y_test)} samples")

    # Run all models
    all_results = []
    for model_name, model in model_dict.items():
        logger.info(f"Training {model_name}...")
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
                "model": model_name,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_precision": report["weighted avg"]["precision"],
                "weighted_recall": report["weighted avg"]["recall"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }

            # Add per-class metrics for each class
            for class_idx in range(26):  # 0-25 classes
                if str(class_idx) in report:
                    result_entry[f"precision_{class_idx}"] = report[str(class_idx)][
                        "precision"
                    ]
                    result_entry[f"recall_{class_idx}"] = report[str(class_idx)][
                        "recall"
                    ]
                    result_entry[f"f1_{class_idx}"] = report[str(class_idx)]["f1-score"]
                    result_entry[f"support_{class_idx}"] = report[str(class_idx)][
                        "support"
                    ]

            all_results.append(result_entry)

            # Log performance metrics
            logger.info(
                f"Performance for {model_name}: "
                f"Accuracy={report['accuracy']:.4f}, "
                f"Macro F1={report['macro avg']['f1-score']:.4f}"
            )

            # Generate confusion matrix visualization
            if save_results:
                save_path = f"results/multiclass/all_errors/images/confusion_matrix_{model_name}.png"
                plot_confusion_matrix(
                    y_test, y_pred, "all_errors", model_name, save_path
                )
                logger.debug(f"Saved confusion matrix to {save_path}")

            logger.info(f"{model_name} completed successfully.")
        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}", exc_info=True)

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)

    # Log summary statistics
    if not results_df.empty:
        logger.info(f"Experiment completed with {len(results_df)} model evaluations")

        # Calculate average metrics across all models
        avg_metrics = results_df[["accuracy", "macro_f1", "weighted_f1"]].mean()
        logger.info(
            f"Average metrics across all models: "
            f"Accuracy={avg_metrics['accuracy']:.4f}, "
            f"Macro F1={avg_metrics['macro_f1']:.4f}"
        )

        # Find best performing model by macro F1 score
        best_model_idx = results_df["macro_f1"].idxmax()
        best_model = results_df.iloc[best_model_idx]
        logger.info(
            f"Best performing model: {best_model['model']} "
            f"with Macro F1={best_model['macro_f1']:.4f}"
        )
    else:
        logger.warning("No results generated during experiment")

    # Save results if requested
    if save_results and not results_df.empty:
        logger.info("Saving results and visualizations")
        # Modified to work with multiclass results
        save_path = "results/multiclass/all_errors/results.csv"
        results_df.to_csv(save_path, index=False)
        logger.info(f"Results saved to {save_path}")

    return results_df


if __name__ == "__main__":
    # Run the experiment
    logger.info("Starting experiment run_multiclass_all_errors")
    results = run_multiclass_all_errors(model_selection="paper", save_results=True)
    logger.info("Experiment completed")
