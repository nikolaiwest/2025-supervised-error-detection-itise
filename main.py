from src.experiments import ExperimentRunner


DEFAULT_SETTINGS = {
    "scenario_id": "s04",
    "target_length": 2000,
    "screw_positions": "left",
}

OPTIONS = [
    "binary_vs_ref",
    "binary_vs_all",
    "multiclass_with_groups",
    "multiclass_with_all",
]

for option in OPTIONS:
    runner = ExperimentRunner(
        experiment_name=option,
        model_selection="paper",
        **DEFAULT_SETTINGS,
    )
    runner.run()


# sampling now returns ExperimentDataset instead of dicts
# simplified the ExperimentDataset to better fit the pipeline (now also a separate file)
# moved the mlflow manager to server/ and renamed the module to mlflow/
