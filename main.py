import argparse
import os
from datetime import datetime


def main():
    """Run the main experiment pipeline."""
    parser = argparse.ArgumentParser(
        description="Run supervised error detection experiments"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="paper",
        choices=["paper", "full", "fast"],
        help="Model selection to use",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="both",
        choices=["binary", "multiclass", "both"],
        help="Which classification approach to run",
    )

    args = parser.parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting experiments")

    # Run binary classification experiments
    if args.approach in ["binary", "both"]:
        print("\n=== BINARY CLASSIFICATION EXPERIMENTS ===\n")

        # Import only when needed
        from src.analysis.binary.vs_50 import run_binary_vs_50
        from src.analysis.binary.vs_all import run_binary_vs_all

        print("\n1. 50 normal vs 50 faulty:")
        binary_results_1 = run_binary_vs_50(model_selection=args.models)

        print("\n2. 50 faulty vs all normal:")
        binary_results_2 = run_binary_vs_all(model_selection=args.models)

    # Run multiclass classification experiments
    if args.approach in ["multiclass", "both"]:
        print("\n=== MULTICLASS CLASSIFICATION EXPERIMENTS ===\n")

        # Placeholder - will implement in future versions
        print("Multiclass classification will be implemented in a future version.")

    print(
        f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - All experiments completed!"
    )


if __name__ == "__main__":
    main()
