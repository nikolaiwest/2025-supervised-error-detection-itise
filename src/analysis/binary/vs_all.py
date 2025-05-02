"""
Binary classification: 50 faulty vs all normal samples for each class value.
Updated version using the plots module.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from src.plots.confusion_matrix import plot_confusion_matrix
from src.plots.utils import ensure_directory, save_results_with_plots
from src.data.loader import load_data
from src.models.classifiers import get_model_dict


def run_binary_vs_all(model_selection="paper", save_results=True):
    """
    Run binary classification: 50 faulty vs all normal samples for each class value.

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
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Running binary classification: 50 faulty vs all normal"
    )

    # Create results directory if saving results
    if save_results:
        ensure_directory("results/binary/50vsAll")
        ensure_directory("results/binary/50vsAll/images")

    # Load data
    torque_values, class_values, label_values = load_data()

    # Get models
    model_dict = get_model_dict(model_selection)

    # Run classification for each class value
    all_results = []

    for class_value in sorted(set(class_values)):
        print(
            f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Running: {class_value}"
        )

        # Filter data for this class
        class_mask = class_values == class_value
        filtered_torque_values = torque_values[class_mask]
        filtered_label_values = label_values[class_mask]

        # Print class distribution
        unique_labels, counts = np.unique(filtered_label_values, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")

        # Skip classes with only one label
        if len(unique_labels) < 2:
            print(
                f"⚠️ Skipping {class_value}: only one label '{unique_labels[0]}' present."
            )
            continue

        # Get indices of normal and faulty samples within this class
        normal_indices = np.where(filtered_label_values == "normal")[0]
        faulty_indices = np.where(filtered_label_values != "normal")[0]

        # Skip if insufficient samples
        if len(normal_indices) < 1 or len(faulty_indices) < 1:
            print(
                f"⚠️ Skipping {class_value}: insufficient samples in one or both classes."
            )
            continue

        # Sample 50 faulty examples (or all if less than 50)
        np.random.seed(42)  # For reproducibility
        sampled_faulty = np.random.choice(
            faulty_indices, min(50, len(faulty_indices)), replace=False
        )

        # Use all normal samples
        print(
            f"Using {len(normal_indices)} normal samples and {len(sampled_faulty)} faulty samples"
        )

        # Combine indices (all normal + sampled faulty)
        all_indices = np.concatenate([normal_indices, sampled_faulty])
        x_values = filtered_torque_values[all_indices]

        # Create binary labels using list comprehension
        y_raw = filtered_label_values[all_indices]
        y_values = np.array([0 if label == "normal" else 1 for label in y_raw])

        # Convert to sktime format
        X = from_2d_array_to_nested(x_values)
        y = pd.Series(y_values)

        # Calculate class weights due to imbalance
        class_ratio = len(normal_indices) / len(sampled_faulty)
        print(f"Class imbalance ratio (normal:faulty): {class_ratio:.2f}:1")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Run all models for this class
        for model_name, model in model_dict.items():
            print(f"Training {model_name} on {class_value}...")
            try:
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Evaluate
                report = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )

                # Store results
                result_entry = {
                    "class": class_value,
                    "model": model_name,
                    "accuracy": report["accuracy"],
                    "precision": report.get("1", {}).get("precision", 0),
                    "recall": report.get("1", {}).get("recall", 0),
                    "f1-score": report.get("1", {}).get("f1-score", 0),
                    "normal_samples": len(normal_indices),
                    "faulty_samples": len(sampled_faulty),
                    "imbalance_ratio": class_ratio,
                }
                all_results.append(result_entry)

                # Generate confusion matrix visualization
                if save_results:
                    save_path = f"results/binary/50vsAll/images/confusion_matrix_{class_value}_{model_name}.png"
                    plot_confusion_matrix(
                        y_test, y_pred, class_value, model_name, save_path
                    )

                print(f"✓ {model_name} on {class_value} completed successfully.")
            except Exception as e:
                print(f"❌ Error with {model_name} on {class_value}: {e}")

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)

    # Save results if requested
    if save_results and not results_df.empty:
        save_results_with_plots(results_df, "results/binary", "50vsAll")

    return results_df


if __name__ == "__main__":
    # Run the experiment
    results = run_binary_vs_all(model_selection="paper", save_results=True)
    print("Experiment completed.")
