"""
Functions for generating class-specific performance visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_class_performance(
    results_df, model_name=None, metric="f1-score", save_path=None
):
    """
    Generate and optionally save a visualization of model performance by class.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing model results with metrics.
        Should include columns: 'class', 'model', and the specified metric.
    model_name : str, optional
        If provided, only plot results for this model. Otherwise, plot all models.
    metric : str, default="f1-score"
        Metric to visualize. Default is "f1-score".
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed instead.

    Returns:
    --------
    matplotlib.figure.Figure or dict
        If model_name is specified, returns the generated figure object.
        If model_name is None, returns a dictionary of model names to figure objects.
    """
    if results_df.empty:
        print("Warning: Empty results DataFrame.")
        return None

    # Filter for specified model if provided
    if model_name:
        model_results = results_df[results_df["model"] == model_name]
        models_to_plot = [model_name]
    else:
        model_results = results_df
        models_to_plot = results_df["model"].unique()

    results = {}

    # Create a plot for each model
    for model in models_to_plot:
        current_model_results = model_results[model_results["model"] == model]

        plt.figure(figsize=(10, 6))
        plt.bar(current_model_results["class"], current_model_results[metric])
        plt.title(f"{metric.capitalize()} by Class - {model}")
        plt.xlabel("Class")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_path:
            if model_name:
                # Single model case
                plt.savefig(save_path)
            else:
                # Multiple models case
                model_save_path = save_path.replace(".png", f"_{model}.png")
                plt.savefig(model_save_path)
            plt.close()
        else:
            results[model] = plt.gcf()

    if model_name:
        return results.get(model_name) if not save_path else None
    else:
        return results if not save_path else None
