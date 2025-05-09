# %%
# Model Performance Visualization

# This notebook generates the visualization figures for the paper "Supervised Error Detection Using Machine Learning", comparing different classifier performance across various datasets.
# 1. Setup and Imports
# 2. Utility Functions
# 3. Binary Classification Visualization
# 4. Multiclass Classification Visualization

# %% 1. Setup and Imports

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set global plot parameters for a consistent look
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# Set the mapping for model names to ensure consistent labeling
MODEL_MAPPING_BINARY = {
    "DummyClassifier": "Random",
    "RandomForestClassifier": "Random Forest",
    "SVC": "SVM-Classifier",
    "TimeSeriesForestClassifier": "TS-Forest",
    "ROCKET": "ROCKET",
}

MODEL_MAPPING_MULTICLASS = {
    "DummyClassifier": "Baseline",
    "RandomForestClassifier": "Random Forest",
    "SVC": "SVM",
    "TimeSeriesForestClassifier": "TS Forest",
    "ROCKET": "ROCKET",
}

# Set the order of models for visualization
MODEL_ORDER_BINARY = [
    "Random",
    "SVM-Classifier",
    "Random Forest",
    "TS-Forest",
    "ROCKET",
]
MODEL_ORDER_MULTICLASS = ["Baseline", "SVM", "Random Forest", "TS Forest", "ROCKET"]

# %% 4. Utility Functions
# This section contains the visualization functions used to generate the plots.


def plot_results_as_matrix(
    df,
    plot_title,
    metric="f1_score",
    figsize=(15, 5),
    font_size=10,
    cmap="YlGnBu",
    model_name_mapping=None,
    model_order=None,
):
    """
    Creates a heatmap visualization of model performance across different datasets.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing results with at least 'dataset', 'model', and performance metrics
    plot_title : str
        Title to be displayed on the plot
    metric : str, default='f1_score'
        The metric to visualize (must be a column in df)
    figsize : tuple, default=(15, 5)
        Figure size as (width, height)
    font_size : int, default=10
        Base font size for labels and annotations
    cmap : str, default="YlGnBu"
        Colormap for the heatmap
    model_name_mapping : dict, default=None
        Dictionary mapping original model names to display names
    model_order : list, default=None
        List defining the order of models on the y-axis (from top to bottom)

    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects from matplotlib
    """
    # Validate input
    required_cols = ["dataset", "model", metric]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Extract dataset name without the full prefix for cleaner labels
    df = df.copy()  # Make a copy to avoid modifying the original

    # Extract a more readable dataset name using the last part after underscore
    df["dataset_name"] = (
        df["dataset"]
        .str.split("_")
        .apply(lambda x: "_".join(x[3:]) if len(x) > 3 else x[-1])
    )

    # Apply model name mapping if provided
    if model_name_mapping:
        df["model_display_name"] = df["model"].map(
            lambda x: model_name_mapping.get(x, x)
        )
    else:
        df["model_display_name"] = df["model"]

    # Create pivot table
    pivot_df = df.pivot(
        index="model_display_name", columns="dataset_name", values=metric
    )

    # Reorder the index based on model_order if provided
    if model_order:
        # Verify all models in model_order exist in the pivot_df
        missing_models = [model for model in model_order if model not in pivot_df.index]
        if missing_models:
            print(
                f"Warning: The following models in model_order were not found: {missing_models}"
            )

        # Get valid models that exist in both model_order and pivot_df.index
        valid_models = [model for model in model_order if model in pivot_df.index]

        # Get models in pivot_df.index that aren't in model_order (to append at the end if needed)
        remaining_models = [
            model for model in pivot_df.index if model not in model_order
        ]

        # Reorder with specified order first, then any remaining models
        new_order = valid_models + remaining_models
        pivot_df = pivot_df.reindex(new_order)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        pivot_df,
        cmap=cmap,
        annot=True,  # Show values in cells
        fmt=".2f",  # Format to 2 decimal places
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
        ax=ax,
    )

    # Style the plot
    ax.set_title(
        f'{metric.replace("_", " ").title()} by Model and Dataset for "{plot_title}"',
        fontsize=font_size + 2,
    )
    ax.set_xlabel("Dataset", fontsize=font_size + 2)
    ax.set_ylabel("Model", fontsize=font_size + 2)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    return fig, ax


# %%
def plot_combined_results(
    df_list,
    metric="f1_score",
    figsize=(8, 5),
    font_size=10,
    cmap="YlGnBu",
    model_name_mapping=None,
    model_order=None,
    title=None,
    add_gap=True,
    remove_prefix=None,
):
    """
    Creates a heatmap visualization combining multiple result dataframes with gap columns.
    Uses a simple approach by combining pivoted dataframes side by side.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame or single pandas.DataFrame
        DataFrames containing results with at least 'dataset', 'model', and performance metrics
    metric : str, default='f1_score'
        The metric to visualize (must be a column in df)
    figsize : tuple, default=(8, 5)
        Figure size as (width, height)
    font_size : int, default=10
        Base font size for labels and annotations
    cmap : str, default="YlGnBu"
        Colormap for the heatmap
    model_name_mapping : dict, default=None
        Dictionary mapping original model names to display names
    model_order : list, default=None
        List defining the order of models on the y-axis (from top to bottom)
    title : str, default=None
        Title for the figure. If None, a default title will be created.
    add_gap : bool, default=True
        Whether to add gap columns between different dataframes
    remove_prefix : str, default=None
        Prefix to remove from dataset names, e.g., "multiclass_"

    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects from matplotlib
    """
    # Convert to list if a single DataFrame is passed
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    # Process each dataframe to get pivoted versions
    pivoted_dfs = []

    for df_idx, df in enumerate(df_list):
        df_copy = df.copy()

        # Apply model name mapping if provided
        if model_name_mapping:
            df_copy["model_display_name"] = df_copy["model"].map(
                lambda x: model_name_mapping.get(x, x)
            )
        else:
            df_copy["model_display_name"] = df_copy["model"]

        # Remove prefix from dataset names if specified
        if remove_prefix and isinstance(remove_prefix, str):
            df_copy["dataset_display_name"] = df_copy["dataset"].apply(
                lambda x: (
                    x.replace(remove_prefix, "")
                    if isinstance(x, str) and x.startswith(remove_prefix)
                    else x
                )
            )
        else:
            df_copy["dataset_display_name"] = df_copy["dataset"]

        # Create pivot table with cleaned dataset names
        pivot_df = df_copy.pivot(
            index="model_display_name", columns="dataset_display_name", values=metric
        )

        # Add to list of pivoted dataframes
        pivoted_dfs.append(pivot_df)

    # Combine all pivoted dataframes side by side with gaps if requested
    combined_df = None

    for i, pivot_df in enumerate(pivoted_dfs):
        if combined_df is None:
            combined_df = pivot_df.copy()
        else:
            # Add a gap column if requested
            if add_gap:
                # Create gap column with NaN values
                gap_name = f"_gap_{i}"
                combined_df[gap_name] = np.nan

            # Add the next dataframe's columns
            for col in pivot_df.columns:
                combined_df[col] = pivot_df[col]

    # Reorder the index based on model_order if provided
    if model_order:
        # Get valid models that exist in both model_order and combined_df.index
        valid_models = [model for model in model_order if model in combined_df.index]

        # Get models in combined_df.index that aren't in model_order (to append at the end)
        remaining_models = [
            model for model in combined_df.index if model not in model_order
        ]

        # Reorder with specified order first, then any remaining models
        new_order = valid_models + remaining_models
        combined_df = combined_df.reindex(new_order)

    # Sort columns alphabetically within each group, keeping gap columns in place
    if add_gap:
        # Identify gap columns
        gap_cols = [
            col
            for col in combined_df.columns
            if isinstance(col, str) and col.startswith("_gap_")
        ]

        # Group columns by their position relative to gap columns
        grouped_cols = []
        current_group = []

        for col in combined_df.columns:
            if col in gap_cols:
                if current_group:
                    grouped_cols.append(sorted(current_group))
                grouped_cols.append([col])
                current_group = []
            else:
                current_group.append(col)

        if current_group:
            grouped_cols.append(sorted(current_group))

        # Flatten the grouped columns
        sorted_cols = []
        for group in grouped_cols:
            sorted_cols.extend(group)

        # Reorder columns
        combined_df = combined_df[sorted_cols]

    # Create figure and plot - explicitly set figsize to ensure it takes effect
    fig, ax = plt.subplots(figsize=figsize)

    # Create a mask to hide the gap columns in the heatmap
    if add_gap:
        mask = pd.DataFrame(False, index=combined_df.index, columns=combined_df.columns)
        for col in combined_df.columns:
            if isinstance(col, str) and col.startswith("_gap_"):
                mask[col] = True
    else:
        mask = None

    # Create heatmap
    cbar_kws = {"label": metric.replace("_", " ").title()}

    # Create the heatmap with all the settings
    sns.heatmap(
        combined_df,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws=cbar_kws,
        ax=ax,
        mask=mask,
    )

    # Set title
    if title is None:
        title = f'{metric.replace("_", " ").title()} by Model and Dataset'
    ax.set_title(title, fontsize=font_size + 2)

    # Style the plot
    ax.set_xlabel("Dataset", fontsize=font_size)
    ax.set_ylabel("Model", fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # FIX: Only hide gap columns without disrupting other labels
    if add_gap:
        # Get all the current tick positions and labels
        positions = np.arange(len(combined_df.columns)) + 0.5
        labels = list(combined_df.columns)

        # Create lists for non-gap positions and labels
        non_gap_positions = []
        non_gap_labels = []

        # Filter out gap columns
        for i, (pos, label) in enumerate(zip(positions, labels)):
            if not (isinstance(label, str) and label.startswith("_gap_")):
                non_gap_positions.append(pos)
                non_gap_labels.append(label)

        # Set the positions and labels
        ax.set_xticks(non_gap_positions)
        ax.set_xticklabels(non_gap_labels, rotation=45, ha="right")

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    return fig, ax


# %% 2. Binary Classification Visualization
# This section generates visualizations for the binary classification experiments,
# comparing different models on anomaly detection tasks.

# Load binary classification results
df_binary_vs_ref = pd.read_csv("../results/s04/binary_vs_ref/results.csv")

# Generate visualization for binary classification against reference observations
title_ref = "Binary Classification vs Reference Normal Observations (#OK=50)"
fig_ref, ax_ref = plot_results_as_matrix(
    df_binary_vs_ref,
    title_ref,
    metric="f1_score",
    model_name_mapping=MODEL_MAPPING_BINARY,
    model_order=MODEL_ORDER_BINARY,
    figsize=(15, 5),
)
plt.savefig("matrix_binary_vs_ref.png", dpi=300, bbox_inches="tight")
plt.show()

# Load binary classification results (all observations)
df_binary_vs_all = pd.read_csv("../results/s04/binary_vs_all/results.csv")

# Generate visualization for binary classification against all normal observations
title_all = "Binary Classification vs All Normal Observations (#OK=1215)"
fig_all, ax_all = plot_results_as_matrix(
    df_binary_vs_all,
    title_all,
    metric="f1_score",
    model_name_mapping=MODEL_MAPPING_BINARY,
    model_order=MODEL_ORDER_BINARY,
    figsize=(15, 5),
)
plt.savefig("matrix_binary_vs_all.png", dpi=300, bbox_inches="tight")
plt.show()

# %% 3. Multiclass Classification Visualization
# This section generates visualizations for the multiclass classification experiments,
# comparing model performance across different error group classifications.

# Load multiclass classification results
df_groups = pd.read_csv("../results/s04/multiclass_with_groups/results.csv")
df_all = pd.read_csv("../results/s04/multiclass_with_all/results.csv")

# Generate visualization for multiclass classification
fig_multi, ax_multi = plot_combined_results(
    [df_groups, df_all],
    metric="f1_score",
    model_name_mapping=MODEL_MAPPING_MULTICLASS,
    model_order=MODEL_ORDER_MULTICLASS,
    title="F1-Score by Model and Dataset for Multiclass Classification",
    figsize=(8, 5),
    remove_prefix="multiclass_",
)
plt.savefig("multiclass_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %%


def plot_confusion_matrix(
    cm_path,
    figsize=(12, 10),
    font_size=10,
    cmap="YlGnBu",
    normalize=True,
    title=None,
    class_names=None,
    save_path=None,
):
    """
    Plots a confusion matrix using a similar style to the other visualizations.

    Parameters:
    -----------
    cm_path : str
        Path to the CSV file containing the confusion matrix
    figsize : tuple, default=(12, 10)
        Figure size as (width, height)
    font_size : int, default=10
        Base font size for labels and annotations
    cmap : str, default="YlGnBu"
        Colormap for the heatmap
    normalize : bool, default=True
        Whether to normalize the confusion matrix by row (true labels)
    title : str, default=None
        Title for the figure. If None, a default title will be created.
    class_names : list, default=None
        List of class names to use as labels. If None, uses indices.
    save_path : str, default=None
        Path to save the figure. If None, the figure is not saved.

    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects from matplotlib
    """
    # Load the confusion matrix from CSV
    cm = pd.read_csv(cm_path, header=0, index_col=None).values

    # Create class names if not provided
    if class_names is None:
        class_names = [f"{i}" for i in range(cm.shape[0])]

    # Create figure with specified font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize the confusion matrix if requested
    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1)
        # Replace zeros with ones to avoid division by zero
        row_sums[row_sums == 0] = 1
        cm_normalized = cm / row_sums[:, np.newaxis]

        # Create heatmap with normalized values
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show original counts as annotations
            fmt="d",  # Format as integers
            cmap=cmap,
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Normalized Frequency"},
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
        )
    else:
        # Create heatmap with raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",  # Format as integers
            cmap=cmap,
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Count"},
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
        )

    # Set title
    if title is None:
        title = "Confusion Matrix"
    ax.set_title(title, fontsize=font_size + 4)

    # Style the plot
    ax.set_xlabel("Predicted Label", fontsize=font_size + 2)
    ax.set_ylabel("True Label", fontsize=font_size + 2)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    # Adjust the layout for better display of all elements
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig, ax


# Example usage
fig, ax = plot_confusion_matrix(
    cm_path="results/multiclass_with_all/multiclass_all_errors_ROCKET_cm.csv",
    figsize=(12, 10),
    normalize=True,  # Normalize by row (true labels)
    title="Confusion Matrix: ROCKET Model on Multiclass All Errors",
    save_path="confusion_matrix_rocket.png",
)
plt.show()
