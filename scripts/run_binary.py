"""
Script to run binary classification experiments for the PyScrew project.

This script runs both binary classification approaches:
1. Reference vs Faulty (vs_ref): All normal/reference samples vs all faulty samples
2. All vs All (vs_all): All samples compared against each other

Usage:
    python scripts/run_binary.py [--fast|--paper|--full]
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.binary import run_binary_vs_all, run_binary_vs_ref
from src.utils.logger import get_logger

# Configure logger
logger = get_logger(__name__)


if __name__ == "__main__":

    # Set model selection
    model_selection = "fast"  # Default

    # Log experiment start
    start_time = datetime.now()
    logger.info(f"Starting binary classification ('{model_selection}')")

    # Run experiments
    logger.info("Running reference vs faulty experiment...")
    vs_ref_results = run_binary_vs_ref(model_selection=model_selection)

    logger.info("Running all vs all experiment...")
    vs_all_results = run_binary_vs_all(model_selection=model_selection)

    # Log experiment completion
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"All experiments completed in {duration}")

    # Log a brief summary
    logger.info(
        f"Completed {len(vs_ref_results) if vs_ref_results is not None else 0} evaluations in vs_ref experiment"
    )
    logger.info(
        f"Completed {len(vs_all_results) if vs_all_results is not None else 0} evaluations in vs_all experiment"
    )
    logger.info("Done.")
