from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(y_true, y_pred, pos_label=1):
    """
    Evaluate model performance for binary classification using direct metrics.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    pos_label : int or str, default=1
        The label of the positive class

    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    # Calculate metrics directly using sklearn functions
    result_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1-score": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }

    return result_dict
