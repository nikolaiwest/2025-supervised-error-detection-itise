"""
Model definitions for time series classification tasks.
Provides a variety of models from both sklearn and sktime libraries.
"""


def get_model_dict(selection="paper"):
    """
    Return dictionary of classifiers based on selection.

    Parameters:
    -----------
    selection : str
        "fast" - Fastest models for quick testing and development (4 models)
        "paper" - Balanced selection for publication (8 models)
        "full" - All available models (20 models)

    Returns:
    --------
    Dictionary of initialized model instances
    """
    # Import sktime models
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.classification.dictionary_based import (
        IndividualBOSS,
        BOSSEnsemble,
        WEASEL,
    )
    from sktime.classification.shapelet_based import ShapeletTransformClassifier
    from sktime.classification.deep_learning import CNNClassifier
    from sktime.classification.feature_based import FreshPRINCE
    from sktime.classification.ensemble import ComposableTimeSeriesForestClassifier
    from sktime.classification.hybrid import HIVECOTEV2

    # Import scikit-learn models
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Define fast models for rapid prototyping
    if selection == "fast":
        return {
            # Fast sktime models
            "TSF-fast": TimeSeriesForestClassifier(n_estimators=10, random_state=42),
            "KNN-DTW-fast": KNeighborsTimeSeriesClassifier(n_neighbors=1),
            # Fast sklearn models (without adapter)
            "RF-fast": RandomForestClassifier(n_estimators=10, random_state=42),
            "DT-fast": DecisionTreeClassifier(random_state=42),
        }

    # Define balanced selection for paper
    elif selection == "paper":
        return {
            # sktime models
            "TSF": TimeSeriesForestClassifier(n_estimators=100, random_state=42),
            "KNN-DTW": KNeighborsTimeSeriesClassifier(n_neighbors=3),
            "IndividualBOSS": IndividualBOSS(random_state=42),
            "WEASEL": WEASEL(random_state=42),
            # sklearn models (without adapter)
            "RF": RandomForestClassifier(n_estimators=100, random_state=42),
            "GBM": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            ),
        }

    # Define full model suite
    elif selection == "full":
        return {
            # All sktime models
            "TSF": TimeSeriesForestClassifier(n_estimators=100, random_state=42),
            "KNN-DTW": KNeighborsTimeSeriesClassifier(n_neighbors=3),
            "IndividualBOSS": IndividualBOSS(random_state=42),
            "BOSS": BOSSEnsemble(random_state=42),
            "WEASEL": WEASEL(random_state=42),
            "Shapelet": ShapeletTransformClassifier(
                random_state=42,
                n_shapelet_samples=100,  # Reduced for speed
                max_shapelets=20,  # Reduced for speed
            ),
            "CNN": CNNClassifier(
                n_epochs=20,  # Reduced for speed
                batch_size=16,
                random_state=42,
            ),
            "FreshPRINCE": FreshPRINCE(random_state=42),
            "CTSF": ComposableTimeSeriesForestClassifier(
                n_estimators=100, random_state=42
            ),
            "HIVECOTE": HIVECOTEV2(
                random_state=42,
                n_jobs=1,  # Set based on your available CPU cores
                stc_params={
                    "n_shapelet_samples": 100,  # Reduced for speed
                    "max_shapelets": 20,  # Reduced for speed
                },
            ),
            # All sklearn models (without adapter)
            "RF": RandomForestClassifier(n_estimators=100, random_state=42),
            "GBM": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "DT": DecisionTreeClassifier(random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "LogReg": LogisticRegression(max_iter=1000, random_state=42),
            "GNB": GaussianNB(),
            "LDA": LinearDiscriminantAnalysis(),
            "HistGBM": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        }

    else:
        raise ValueError(
            f"Unknown model selection: {selection}. "
            f"Choose from 'fast', 'paper', or 'full'."
        )
