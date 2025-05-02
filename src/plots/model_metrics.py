"""
Functions for generating model performance metrics visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics_comparison(results_df, save_path=None):
    """
    Generate and optionally save a comparison of model metrics.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing model results with metrics.
        Should include columns: 'model', 'accuracy', 'precision', 'recall', 'f1-score'.
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed instead.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object.
    """
    if results_df.empty:
        print("Warning: Empty results DataFrame.")
        return None

    # Compute average metrics per model
    model_metrics = (
        results_df.groupby("model")[["accuracy", "precision", "recall", "f1-score"]]
        .mean()
        .reset_index()
    )

    # Create subplot for each metric
    plt.figure(figsize=(12, 8))
    metrics = ["accuracy", "precision", "recall", "f1-score"]

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar(model_metrics["model"], model_metrics[metric])
        plt.title(f"Average {metric}")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        return plt.gcf()
