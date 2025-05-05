"""
Multi-Class Classification Experiment Runner for PyScrew Project
==============================================================

This script executes comprehensive multi-class classification experiments for the
PyScrew project using time series classification models. The script supports
two distinct experimental approaches:

1. All Errors (all_errors):
   Creates a 26-class classification problem:
   - 1 class for all normal samples across all experiments
   - 25 classes for each individual error type

2. Within Group (within_group):
   Groups the 25 error classes into 5 logical groups and creates a 6-class
   classification problem within each group:
   - 1 class for all normal samples from the 5 classes in that group
   - 5 classes for each error class in that group

Results are saved as CSV files and visualized through multiple plots to
facilitate analysis and interpretation.

Usage:
    python scripts/run_multiclass.py [--fast|--paper|--full]
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from src.analysis.multiclass import run_multiclass_all_errors
from src.analysis.multiclass import run_multiclass_within_group
from src.plots import plot_classification_results
from src.utils.logger import get_logger

# Configure logger
logger = get_logger(__name__)

if __name__ == "__main__":

    # Set model selection
    model_selection = "paper"

    # Log experiment configuration
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info(
        f"Starting multi-class classification [Config: '{model_selection}' models]"
    )
    logger.info(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Create result directories if they don't exist
    os.makedirs("results/multiclass/all_errors/images", exist_ok=True)
    os.makedirs("results/multiclass/within_group/images", exist_ok=True)

    # =========================================================================
    # Experiment 1: All Errors (26-class classification)
    # =========================================================================
    logger.info("EXPERIMENT 1: All Errors Classification (26 classes)")
    logger.info("-" * 80)

    # Run the experiment
    all_errors_results = run_multiclass_all_errors(model_selection=model_selection)
    logger.info(
        f"All Errors experiment completed with {len(all_errors_results) if all_errors_results is not None else 0} model evaluations"
    )

    # Generate and save visualizations
    logger.info("Generating visualizations for All Errors experiment...")
    plot_classification_results(
        csv_path="results/multiclass/all_errors/results.csv",
        output_dir="results/multiclass/all_errors/images",
        is_multiclass=True,
    )
    logger.info("Visualizations saved to results/multiclass/all_errors/images/")

    # =========================================================================
    # Experiment 2: Within Group (6-class classification per group)
    # =========================================================================
    logger.info("=" * 80)
    logger.info("EXPERIMENT 2: Within Group Classification (6 classes per group)")
    logger.info("-" * 80)

    # Run the experiment
    within_group_results = run_multiclass_within_group(model_selection=model_selection)
    logger.info(
        f"Within Group experiment completed with {len(within_group_results) if within_group_results is not None else 0} model evaluations"
    )

    # Generate and save visualizations
    logger.info("Generating visualizations for Within Group experiment...")
    plot_classification_results(
        csv_path="results/multiclass/within_group/results.csv",
        output_dir="results/multiclass/within_group/images",
        is_multiclass=True,
        group_column="group",
    )
    logger.info("Visualizations saved to results/multiclass/within_group/images/")

    # =========================================================================
    # Summary and Completion
    # =========================================================================
    # Calculate and log timing information
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("-" * 80)
    logger.info(f"All experiments completed in {hours}h {minutes}m {seconds}s")
    logger.info(
        f"All Errors (26-class): {len(all_errors_results) if all_errors_results is not None else 0} evaluations"
    )
    logger.info(
        f"Within Group (6-class): {len(within_group_results) if within_group_results is not None else 0} evaluations"
    )
    logger.info(
        "All results and visualizations available in 'results/multiclass/' directory"
    )
    logger.info("Multi-class classification experiments completed successfully.")
