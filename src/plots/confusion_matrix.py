"""
Functions for generating confusion matrix visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_value, model_name, save_path=None):
    """
    Generate and optionally save a confusion matrix visualization.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    class_value : str
        Class value used for the title.
    model_name : str
        Model name used for the title.
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed instead.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
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

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        return plt.gcf()
