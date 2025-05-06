"""
Binary Classification Experiment Runner for PyScrew Project
==========================================================

This script executes comprehensive binary classification experiments for the
PyScrew project using time series classification models. The script supports
two distinct experimental approaches:

1. Reference vs Faulty (vs_ref):
   For each class, compares normal/reference samples of that specific class against
   faulty samples of the same class, treating this as a binary classification problem.

2. All Normal vs Class-Specific Faulty (vs_all):
   For each class, compares ALL normal/reference samples across ALL classes against
   the faulty samples of that specific class. This tests if models can distinguish
   between normal operation and specific fault types regardless of class context.

Results are saved as CSV files and visualized through multiple plots to
facilitate analysis and interpretation.

Usage:
    python scripts/run_binary.py [--fast|--paper|--full]
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from src.analysis.binary import run_binary_vs_all, run_binary_vs_ref
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
    logger.info(f"Starting binary classification [Config: '{model_selection}' models]")
    logger.info(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Create result directories if they don't exist
    os.makedirs("results/binary/vs_ref/images", exist_ok=True)
    os.makedirs("results/binary/vs_all/images", exist_ok=True)

    # =========================================================================
    # Experiment 1: Reference vs. Faulty (Binary)
    # =========================================================================
    logger.info("EXPERIMENT 1: Reference vs Faulty Classification")
    logger.info("-" * 80)

    # Run the experiment
    vs_ref_results = run_binary_vs_ref(model_selection=model_selection)
    logger.info(
        f"Reference vs Faulty experiment completed with {len(vs_ref_results) if vs_ref_results is not None else 0} model evaluations"
    )

    # Generate and save visualizations
    logger.info("Generating visualizations for Reference vs Faulty experiment...")
    plot_classification_results(
        csv_path="results/binary/vs_ref/results.csv",
        output_dir="results/binary/vs_ref/images",
    )
    logger.info("Visualizations saved to results/binary/vs_ref/images/")

    # =========================================================================
    # Experiment 2: All vs. All (Multi-Binary)
    # =========================================================================
    logger.info("=" * 80)
    logger.info("EXPERIMENT 2: All vs All Classification")
    logger.info("-" * 80)

    # Run the experiment
    vs_all_results = run_binary_vs_all(model_selection=model_selection)
    logger.info(
        f"All vs All experiment completed with {len(vs_all_results) if vs_all_results is not None else 0} model evaluations"
    )

    # Generate and save visualizations
    logger.info("Generating visualizations for All vs All experiment...")
    plot_classification_results(
        csv_path="results/binary/vs_all/results.csv",
        output_dir="results/binary/vs_all/images",
    )
    logger.info("Visualizations saved to results/binary/vs_all/images/")

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
        f"Reference vs Faulty: {len(vs_ref_results) if vs_ref_results is not None else 0} evaluations"
    )
    logger.info(
        f"All vs All: {len(vs_all_results) if vs_all_results is not None else 0} evaluations"
    )
    logger.info(
        "All results and visualizations available in 'results/binary/' directory"
    )
    logger.info("Binary classification experiments completed successfully.")
