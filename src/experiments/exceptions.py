class ExperimentError(Exception):
    """Base class for experiment errors with optional context."""

    pass


class FatalExperimentError(ExperimentError):
    """
    Critical errors that stop the entire experiment.

    Examples: MLflow server unreachable, data files missing,
    invalid experiment configuration.
    """

    pass


class ModelEvaluationError(ExperimentError):
    """
    Errors during individual model evaluation.

    The experiment continues with remaining models, but tracks
    which models failed and why for debugging.
    """

    pass


class DatasetPreparationError(ExperimentError):
    """
    Errors during data loading or preprocessing.

    Examples: corrupted files, incompatible data formats,
    insufficient samples for cross-validation.
    """

    pass
