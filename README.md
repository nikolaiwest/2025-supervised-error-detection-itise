# Supervised Error Detection in Screw Driving Operations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and results for our paper "Multi-class Error Detection in Industrial Screw Driving Operations Using Machine Learning". The paper will be presented at ITISE 2025 (11th International Conference on Time Series and Forecasting) in Gran Canaria, Spain.

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

If you just want to have a look at the dataset, you can download it using our `pyscrew` package:
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
*Will be updated with the conference proceedings.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

These datasets were collected and prepared by:
- [RIF Institute for Research and Transfer e.V.](https://www.rif-ev.de/)
- [Technical University Dortmund](https://www.tu-dortmund.de/), [Institute for Production Systems](https://ips.mb.tu-dortmund.de/)
- Feel free to contact us directly for further questions: [Nikolai West (nikolai.west@tu-dortmund.de)](nikolai.west@tu-dortmund.de)

The preparation and provision of the research was supported by:

| Organization | Role | Logo |
|-------------|------|------|
| German Ministry of Education and Research (BMBF) | Funding | <img src="https://vdivde-it.de/system/files/styles/vdivde_logo_vdivde_desktop_1_5x/private/image/BMBF_englisch.jpg?itok=6FdVWG45" alt="BMBF logo" height="150"> |
| European Union's "NextGenerationEU" | Funding | <img src="https://www.bundesfinanzministerium.de/Content/DE/Bilder/Logos/nextgenerationeu.jpg?__blob=square&v=1" alt="NextGenerationEU logo" height="150"> |
| VDIVDE | Program Support | <img src="https://vdivde-it.de/themes/custom/vdivde/images/vdi-vde-it_og-image.png" alt="Projekttraeger VDIVDE logo" height="150"> |

This research is part of the funding program ["Data competencies for early career researchers"](https://www.bmbf.de/DE/Forschung/Wissenschaftssystem/Forschungsdaten/DatenkompetenzenInDerWissenschaft/datenkompetenzeninderwissenschaft_node.html). 

More information regarding the research project is available at [prodata-projekt.de](https://prodata-projekt.de/).
