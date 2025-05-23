import os


def get_tracking_uri():
    """Return the MLflow tracking URI."""
    # Using a relative path within the experiments directory
    db_path = os.path.join(os.path.dirname(__file__), "experiments.db")
    return f"sqlite:///{db_path}"


def setup_mlflow_tracking():
    """
    Configure MLflow to use the local tracking server.
    Call this in your training scripts.
    """
    import mlflow

    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    return tracking_uri
