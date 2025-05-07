import warnings

from sklearn.exceptions import ConvergenceWarning

from src.experiments import ExperimentRunner

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


runner = ExperimentRunner(
    experiment_type=OPTIONS[0],
    model_selection="fast",
    **DEFAULT_SETTINGS,
)
runner.run()
