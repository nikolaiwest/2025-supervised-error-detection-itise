import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.binary import run_binary_vs_50, run_binary_vs_all

# Run the binary classification with 50 normal vs 50 faulty
run_binary_vs_50(model_selection="fast")  # Use 'fast' for quicker testing
run_binary_vs_all(model_selection="fast")  # Use 'fast' for quicker testing
