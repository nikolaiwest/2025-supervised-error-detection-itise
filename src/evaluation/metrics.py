from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(y_true, y_pred, experiment_type=None):
    """
    Evaluate model performance for classification using direct metrics.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    experiment_type : str, optional
        Type of experiment ('binary_vs_ref', 'binary_vs_all', 'multiclass_group', 'multiclass_all')
        Used to determine appropriate metric parameters

    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    # Determine if binary or multiclass based on experiment type
    is_binary = len(set(y_true)) == 2

    # Set appropriate parameters based on classification type
    if is_binary:
        # Binary classification
        pos_label = 1
        average = "binary"
    else:
        # Multiclass classification
        pos_label = None
        average = "weighted"  # Good default for imbalanced datasets

    # Calculate metrics with appropriate parameters
    result_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    # Add precision, recall, and F1 with appropriate parameters
    if is_binary:
        # Binary metrics with pos_label
        result_dict.update(
            {
                "precision": precision_score(
                    y_true, y_pred, pos_label=pos_label, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, pos_label=pos_label, zero_division=0
                ),
                "f1_score": f1_score(
                    y_true, y_pred, pos_label=pos_label, zero_division=0
                ),
            }
        )
    else:
        # Multiclass metrics with average parameter
        result_dict.update(
            {
                "precision": precision_score(
                    y_true, y_pred, average=average, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, average=average, zero_division=0
                ),
                "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
            }
        )

    return result_dict
