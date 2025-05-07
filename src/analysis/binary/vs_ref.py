import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from src.data.loader import load_data
from src.evaluation import evaluate_model
from src.models import get_classifier_dict
from src.plots.confusion_matrix import plot_confusion_matrix
from src.plots.utils import ensure_directory, save_results_with_plots
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_binary_vs_ref(model_selection="paper", save_results=False):
    """
    Run binary classification: All normal (reference) vs all faulty samples for each class value.

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
        f"Running binary classification:"
        f"Reference vs faulty (model selection: {model_selection})"
    )

    # Create results directory if saving results
    if save_results:
        ensure_directory("results/binary/vs_ref")
        ensure_directory("results/binary/vs_ref/images")
        logger.debug("Created results directories")

    # Load data
    logger.info("Loading data...")
    torque_values, class_values, scenario_condition = load_data()
    logger.debug(
        f"Loaded {len(torque_values)} samples with {len(set(class_values))} classes"
    )

    # Get models
    logger.info(f"Getting models for selection: {model_selection}")
    model_dict = get_classifier_dict(model_selection)
    logger.debug(f"Loaded {len(model_dict)} models: {', '.join(model_dict.keys())}")

    # Run classification for each class value
    all_results = []

    for class_value in sorted(set(class_values)):
        logger.info(f"Processing class: {class_value}")

        # Filter data for this class
        class_mask = class_values == class_value
        filtered_torque_values = torque_values[class_mask]
        filtered_label_values = scenario_condition[class_mask]

        # Print class distribution
        unique_labels, counts = np.unique(filtered_label_values, return_counts=True)
        class_dist = {str(k): int(v) for k, v in zip(unique_labels, counts)}
        logger.debug(f"Class distribution: {class_dist}")

        # Skip classes with only one label
        if len(unique_labels) < 2:
            logger.warning(f"Skip {class_value}: only one label '{unique_labels[0]}'.")
            continue

        # Get indices of normal (reference) and faulty samples within this class
        normal_idx = np.where(filtered_label_values == "normal")[0]
        faulty_idx = np.where(filtered_label_values != "normal")[0]

        # Skip if insufficient samples
        if len(normal_idx) < 1 or len(faulty_idx) < 1:
            logger.warning(f"Skipping {class_value}: insufficient samples in classes.")
            continue

        # Use all samples (no subsampling)
        logger.info(
            f"- Using {len(normal_idx)} normal and {len(faulty_idx)} faulty runs"
        )

        # Calculate class weights due to potential imbalance
        class_ratio = len(normal_idx) / len(faulty_idx)
        logger.info(f"- Class imbalance ratio (normal:faulty): {class_ratio:.2f}:1.00")

        # Combine indices (all samples)
        all_indices = np.concatenate([normal_idx, faulty_idx])
        x_values = filtered_torque_values[all_indices]

        # Create binary labels using list comprehension
        y_raw = filtered_label_values[all_indices]
        y_values = np.array([0 if label == "normal" else 1 for label in y_raw])
        y = pd.Series(y_values)

        # Convert to sktime format for sktime models
        x_sktime = from_2d_array_to_nested(x_values)
        # Also prepare flattened format for sklearn models
        x_sklearn = x_values.reshape(x_values.shape[0], -1)

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
            x_sktime,
            x_sklearn,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
        logger.debug(
            f"Train set: {len(y_train)} samples, Test set: {len(y_test)} samples"
        )

        # Run all models for this class
        for model_name, model in model_dict.items():
            logger.info(f"Training {model_name} on {class_value}...")
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
                result_dict = evaluate_model(y_test, y_pred)

                # Store results
                result_entry = {
                    "class": class_value,
                    "model": model_name,
                    "accuracy": result_dict["accuracy"],
                    "precision": result_dict["precision"],
                    "recall": result_dict["recall"],
                    "f1-score": result_dict["f1-score"],
                    "reference_samples": len(normal_idx),
                    "faulty_samples": len(faulty_idx),
                    "imbalance_ratio": class_ratio,
                }
                all_results.append(result_entry)

                # Log performance metrics
                logger.info(
                    f"- Performance for {model_name} on {class_value}: "
                    f"Accuracy={result_dict['accuracy']:.4f}, "
                    f"F1-Score={result_dict['f1-score']:.4f}"
                )

                # Generate confusion matrix visualization
                if save_results:
                    save_path = f"results/binary/vs_ref/images/confusion_matrix/{class_value}_{model_name}.png"
                    plot_confusion_matrix(
                        y_test, y_pred, class_value, model_name, save_path
                    )
                    logger.debug(f"Saved confusion matrix to {save_path}")

                logger.info(f"{model_name} on {class_value} completed successfully.")
            except Exception as e:
                logger.error(
                    f"Error with {model_name} on {class_value}: {str(e)}", exc_info=True
                )

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)

    # Log summary statistics
    if not results_df.empty:
        logger.info(f"Experiment completed with {len(results_df)} model evaluations")

        # Calculate average metrics across all models
        avg_metrics = results_df[["accuracy", "precision", "recall", "f1-score"]].mean()
        logger.info(
            f"Average metrics across all models: "
            f"Accuracy={avg_metrics['accuracy']:.4f}, "
            f"F1-Score={avg_metrics['f1-score']:.4f}"
        )

        # Find best performing model by F1 score
        best_model_idx = results_df["f1-score"].idxmax()
        best_model = results_df.iloc[best_model_idx]
        logger.info(
            f"Best performing model: {best_model['model']} on {best_model['class']} "
            f"with F1-Score={best_model['f1-score']:.4f}"
        )
    else:
        logger.warning("No results generated during experiment")

    # Save results if requested
    if save_results and not results_df.empty:
        logger.info("Saving results and visualizations")
        save_results_with_plots(results_df, "results/binary", "vs_ref")
        logger.info("Results saved successfully")

    return results_df
