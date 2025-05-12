from src.experiments import ExperimentRunner


DEFAULT_SETTINGS = {
    "scenario_id": "s04",
    "target_length": 2000,
    "screw_positions": "left",
}

OPTIONS = [
    "binary_vs_ref",
    # "binary_vs_all",
    # "multiclass_with_groups",
    # "multiclass_with_all",
]


for option in OPTIONS:
    runner = ExperimentRunner(
        experiment_type=option,
        model_selection="sklearn",
        **DEFAULT_SETTINGS,
    )
    runner.run()
