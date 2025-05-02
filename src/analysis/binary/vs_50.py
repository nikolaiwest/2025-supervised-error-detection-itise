import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from src.data.loader import load_data
from src.models.classifiers import get_model_dict


def run_binary_vs_50(model_selection="paper", save_results=True):
    """
    Run binary classification: 50 normal vs 50 faulty samples for each class value.

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
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Running binary classification: 50 normal vs 50 faulty"
    )

    # Create results directory if saving results
    if save_results:
        os.makedirs("results/binary/50vs50", exist_ok=True)

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

        # Sample 50 from each class (or all if less than 50)
        np.random.seed(42)  # For reproducibility
        sampled_normal = np.random.choice(
            normal_indices, min(50, len(normal_indices)), replace=False
        )
        sampled_faulty = np.random.choice(
            faulty_indices, min(50, len(faulty_indices)), replace=False
        )

        # Combine indices
        all_indices = np.concatenate([sampled_normal, sampled_faulty])
        x_values = filtered_torque_values[all_indices]

        # Create binary labels using list comprehension
        y_raw = filtered_label_values[all_indices]
        y_values = np.array([0 if label == "normal" else 1 for label in y_raw])

        # Convert to sktime format
        X = from_2d_array_to_nested(x_values)
        y = pd.Series(y_values)

        # Split into train/test
        # Note: In future releases, we'll implement cross-validation here
        # for more robust model evaluation
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
                    "normal_samples": len(sampled_normal),
                    "faulty_samples": len(sampled_faulty),
                }
                all_results.append(result_entry)

                # Generate confusion matrix visualization
                if save_results:
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Normal", "Faulty"],
                        yticklabels=["Normal", "Faulty"],
                    )
                    plt.title(f"Confusion Matrix - {class_value} - {model_name}")
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.tight_layout()
                    plt.savefig(
                        f"results/binary/50vs50/images/confusion_matrix_{class_value}_{model_name}.png"
                    )
                    plt.close()

                print(f"✓ {model_name} on {class_value} completed successfully.")
            except Exception as e:
                print(f"❌ Error with {model_name} on {class_value}: {e}")

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)

    # Save results if requested
    if save_results and not results_df.empty:
        results_df.to_csv("results/binary/50vs50/results.csv", index=False)

        # Create summary visualization per model
        for model_name in results_df["model"].unique():
            model_results = results_df[results_df["model"] == model_name]

            plt.figure(figsize=(10, 6))
            plt.bar(model_results["class"], model_results["f1-score"])
            plt.title(f"F1-Score by Class - {model_name}")
            plt.xlabel("Class")
            plt.ylabel("F1-Score")
            plt.ylim(0, 1)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"results/binary/50vs50/images/f1_by_class_{model_name}.png")
            plt.close()

        # Create overall metrics comparison
        plt.figure(figsize=(12, 8))
        model_metrics = (
            results_df.groupby("model")[["accuracy", "precision", "recall", "f1-score"]]
            .mean()
            .reset_index()
        )

        metrics = ["accuracy", "precision", "recall", "f1-score"]
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            plt.bar(model_metrics["model"], model_metrics[metric])
            plt.title(f"Average {metric}")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig("results/binary/50vs50/images/overall_metrics_comparison.png")
        plt.close()

        print(f"Results saved to results/binary/50vs50/")

    return results_df
