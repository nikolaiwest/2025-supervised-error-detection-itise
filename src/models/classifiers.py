"""
Model definitions for pyscrew analysis
"""


def get_model_dict(selection="paper"):
    """Return dictionary of time series classifiers based on selection.

    Parameters:
    -----------
    selection : str
        "paper" - Models used in the paper (fewer but well-tested)
        "full" - All available models
        "fast" - Only fast models for quick experiments

    Returns:
    --------
    Dictionary of initialized model instances
    """
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.classification.dictionary_based import IndividualBOSS, BOSSEnsemble
    from sktime.classification.shapelet_based import ShapeletTransformClassifier

    if selection == "paper":
        return {
            "TSF": TimeSeriesForestClassifier(n_estimators=100, random_state=42),
            "KNN-DTW": KNeighborsTimeSeriesClassifier(),
            "IndividualBOSS": IndividualBOSS(random_state=42),
        }
    elif selection == "full":
        return {
            "TSF": TimeSeriesForestClassifier(n_estimators=100, random_state=42),
            "KNN-DTW": KNeighborsTimeSeriesClassifier(),
            "IndividualBOSS": IndividualBOSS(random_state=42),
            "BOSS": BOSSEnsemble(random_state=42),
            "Shapelet": ShapeletTransformClassifier(random_state=42),
        }
    elif selection == "fast":
        return {
            "TSF": TimeSeriesForestClassifier(n_estimators=10, random_state=42),
            "KNN-DTW": KNeighborsTimeSeriesClassifier(n_neighbors=1),
        }
    else:
        raise ValueError(f"Unknown model selection: {selection}")
