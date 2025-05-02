# Supervised Error Detection in Screw Driving Operations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and experiments for our paper "Multi-class Error Detection in Industrial Screw Driving Operations Using Machine Learning", to be presented at ITISE 2025 (11th International Conference on Time Series and Forecasting) in Gran Canaria, Spain.

## Abstract

Recent advances in machine learning have significantly improved anomaly detection in industrial screw driving operations. However, most existing approaches focus on binary classification of normal versus anomalous operations or employ unsupervised methods to detect novel patterns. This paper introduces a comprehensive dataset of screw driving operations encompassing 25 distinct error types and presents a multi-tiered analysis framework for error-specific classification. Our results demonstrate varying detectability across different error types and establish the feasibility of multi-class error detection in industrial settings. The complete dataset and analysis framework are made publicly available to support future research in manufacturing quality control.

## Project Structure

```
├── data/                # Data storage (not included in repo)
├── mlruns/              # MLflow experiment tracking (not included in repo)
├── results/             # Experiment results and visualizations
├── scripts/             # Utility scripts
├── src/                 # Source code
│   ├── analysis/        # Analysis modules
│   │   ├── binary/      # Binary classification approaches
│   │   └── multiclass/  # Multiclass classification approaches
│   ├── data/            # Data loading and preprocessing
│   ├── evaluation/      # Evaluation metrics and reports
│   ├── models/          # Model definitions
│   └── tuning/          # Hyperparameter tuning
├── .gitignore           # Git ignore file
├── LICENSE              # MIT License
├── main.py              # Main execution script
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── testing.py           # Testing script for development
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/2025-supervised-error-detection-itise.git
   cd 2025-supervised-error-detection-itise
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the dataset (requires `pyscrew` package):
   ```python
   import pyscrew
   data = pyscrew.get_data("s04")
   ```

## Usage

### Running Experiments

To run all experiments:

```
python main.py
```

Options:
- `--models`: Model selection to use (`paper`, `full`, or `fast`)
- `--approach`: Which classification approach to run (`binary`, `multiclass`, or `both`)

Examples:
```
# Run binary classification with fast models (for quick testing)
python main.py --models fast --approach binary

# Run all experiments with models from the paper
python main.py --models paper --approach both
```

### Results

Experiment results are saved in the `results/` directory and include:
- CSV files with performance metrics
- Confusion matrices for each model and class
- Performance comparison visualizations

## Approaches

### Binary Classification

1. **Balanced (50 vs 50)**: Classification between 50 normal and 50 faulty samples for each error class.
2. **Imbalanced (50 vs All)**: Classification between all normal samples and 50 faulty samples for each error class.

### Multiclass Classification (Coming Soon)

1. **Within Group**: Classification between different error types within the same group.
2. **All Errors**: Classification across all 25 error classes.

## Citation

If you use this code or dataset in your research, please cite our paper:

```
@inproceedings{west2025supervised,
  title={Supervised Error Detection in Dataset S04 for Variations in Assembly Conditions},
  author={West, Nikolai},
  booktitle={11th International Conference on Time Series and Forecasting (ITISE)},
  year={2025},
  address={Gran Canaria, Spain}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by [Your Institution/Grant Information]
- We thank the PyScrew project for providing the dataset