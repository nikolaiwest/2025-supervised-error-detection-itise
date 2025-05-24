# Supervised Error Detection in Screw Driving Operations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org/)

> This repository contains the code and results for our paper **"Multi-class Error Detection in Industrial Screw Driving Operations Using Machine Learning"** *(yet to be)* presented at ITISE 2025 (11th International Conference on Time Series and Forecasting) in Gran Canaria, Spain.

## 🎯 Abstract

Recent advances in machine learning have significantly improved anomaly detection in industrial screw driving operations. However, most existing approaches focus on binary classification of normal versus anomalous operations or employ unsupervised methods to detect novel patterns. This paper introduces a comprehensive dataset of screw driving operations encompassing **25 distinct error types** and presents a **multi-tiered analysis framework** for error-specific classification. Our results demonstrate varying detectability across different error types and establish the feasibility of multi-class error detection in industrial settings.

## 🏗️ Project Structure

```
├── data/                     # Data storage (downloaded via pyscrew)
├── mlruns/                   # MLflow experiment tracking
├── results/                  # Experiment results and visualizations
├── scripts/                  # Utility and plotting scripts
├── src/                      # Source code
│   ├── analysis/             # Analysis modules (legacy)
│   ├── data/                 # Data loading and preprocessing
│   │   ├── load.py          # PyScrew data interface
│   │   └── process.py       # PAA and normalization
│   ├── evaluation/           # Metrics and result containers
│   │   ├── apply_metrics.py # Sklearn metrics wrapper
│   │   └── results/         # 4-level result hierarchy
│   ├── experiments/          # Experiment orchestration
│   │   ├── experiment_runner.py  # Main experiment runner
│   │   ├── sampling.py           # Dataset generation strategies
│   │   └── training.py           # Cross-validation training
│   ├── mlflow/               # MLflow integration
│   │   ├── manager.py        # Hierarchical logging manager
│   │   └── server.py         # Server management
│   ├── models/               # Model configurations
│   │   ├── classifiers.py    # Dynamic model loading
│   │   ├── sklearn_models.yml    # Sklearn model configs
│   │   └── sktime_models.yml     # Sktime model configs
│   ├── plots/                # Visualization modules
│   └── utils/                # Logging and utilities
├── .gitignore
├── LICENSE                   # MIT License
├── main.py                   # Main entry point
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── setup.py                  # Package installation
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nikolaiwest/2025-supervised-error-detection-itise.git
   cd 2025-supervised-error-detection-itise
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Dataset Access

If you just want to explore the dataset *(or refer to our library [Pyscrew](https://github.com/nikolaiwest/pyscrew) for more info on the data)*:

```python
import pyscrew
data = pyscrew.get_data("s04")  # Downloads ~2GB of time series data
print(f"Dataset contains {len(data['torque_values'])} samples")
print(f"Classes: {set(data['class_values'])}")
```

## 🧪 Experiments

### Experiment Types

Our framework supports four complementary approaches:

| Experiment | Description | Use Case |
|------------|-------------|----------|
| `binary_vs_ref` | Each error class vs its own normal samples (50 vs 50) | **Balanced comparison** for detecting specific error types |
| `binary_vs_all` | Each error class vs ALL normal samples (50 vs 1215) | **Realistic imbalanced** scenario for anomaly detection |
| `multiclass_with_groups` | Multi-class within error groups (5 groups) | **Grouped classification** of related error mechanisms |
| `multiclass_with_all` | All 25 error classes + normal (26-class problem) | **Comprehensive classification** across all error types |

### Model Selection Options

| Selection | Models | Use Case |
|-----------|---------|----------|
| `debug` | DummyClassifier only | Quick testing |
| `fast` | 3 fast models | Rapid prototyping |
| `paper` | 5 representative models | **Paper results** (recommended) |
| `full` | 15+ comprehensive models | Exhaustive comparison |
| `sklearn` | Sklearn models only | Traditional ML focus |
| `sktime` | Time series models only | Specialized TS methods |

### Running Experiments

**Run all experiments (paper setup):**
```bash
python main.py
```

**Run specific experiment:**
```bash
python main.py --experiment binary_vs_ref --models fast
```

**Run with more cross-validation folds:**
```bash
python main.py --cv-folds 10
```

**Quiet mode for production:**
```bash
python main.py --quiet
```

### Advanced Usage

**Custom experiment runner (if you want to change more things):**
```python
from src.experiments import ExperimentRunner

runner = ExperimentRunner(
    experiment_name="binary_vs_ref",
    model_selection="paper",
    scenario_id="s04",           # Fixed dataset
    target_length=2000,          # Fixed preprocessing  
    cv_folds=5,                  # Configurable via CLI
    random_seed=42,              # Configurable via CLI
    n_jobs=-1                    # Use all cores
)

results = runner.run()
```

**Available CLI options:**
```bash
python main.py --help

# Core options:
--experiment {binary_vs_ref,binary_vs_all,multiclass_with_groups,multiclass_with_all,all}
--models {debug,fast,paper,full,sklearn,sktime}
--cv-folds N
--random-seed N
--quiet / --verbose
```

## 📊 Results & Visualization

### MLflow Tracking

Every experiment automatically logs to MLflow with 4-level hierarchy:
- **Experiment** → Overall comparison across datasets
- **Dataset** → Model performance on specific error types  
- **Model** → Cross-validation results and stability metrics
- **Fold** → Individual CV fold results and confusion matrices

**Start MLflow UI:**
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Result Files

Results are saved to `results/s04/<experiment_name>/`:
- `results.csv` - Performance metrics for all model-dataset combinations
- `images/` - Performance visualizations and confusion matrices
- MLflow artifacts - Detailed experiment tracking

### Example Results

**Binary Classification Performance (F1-Score):**
```
Model                   | 101_deformed-thread | 102_filed-screw-tip | 103_glued-screw-tip
------------------------|---------------------|---------------------|-------------------
ROCKET                  | 0.95 ± 0.03         | 0.89 ± 0.05         | 0.92 ± 0.04
TS-Forest               | 0.93 ± 0.04         | 0.87 ± 0.06         | 0.90 ± 0.05
Random Forest           | 0.87 ± 0.06         | 0.82 ± 0.07         | 0.85 ± 0.08
SVM                     | 0.83 ± 0.09         | 0.79 ± 0.11         | 0.81 ± 0.10
Random Baseline         | 0.45 ± 0.12         | 0.43 ± 0.11         | 0.47 ± 0.10
```

## 🔬 Technical Details

### Data Processing Pipeline

1. **Loading**: PyScrew interface to S04 dataset (25 error types)
2. **PAA Reduction**: 2000 → 200 time points (configurable)  
3. **Normalization**: Z-score standardization per time series
4. **Sampling**: Generate datasets based on experiment type

### Model Architecture

- **Traditional ML**: Random Forest, SVM, Logistic Regression 
- **Time Series**: ROCKET, Time Series Forest, BOSS Ensemble
- **Advanced**: Elastic Ensemble, Shapelet Transform

### Cross-Validation Strategy

- **5-fold stratified CV** (default)
- **Stratification** by class to handle imbalance
- **Reproducible** with fixed random seeds
- **Parallel execution** across models and folds

## 📋 Requirements

- **Python**: 3.8+
- **Core**: scikit-learn, sktime, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Tracking**: MLflow
- **Data**: pyscrew (auto-downloads datasets)

See `requirements.txt` for complete dependencies.

## 🎯 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{west2025supervised,
  title={Multi-class Error Detection in Industrial Screw Driving Operations Using Machine Learning},
  author={West, Nikolai},
  booktitle={11th International Conference on Time Series and Forecasting (ITISE)},
  year={2025},
  address={Gran Canaria, Spain},
  publisher={ITISE},
  note={Accepted for publication}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## 📞 Contact

- **Author**: Nikolai West
- **Email**: nikolai.west@tu-dortmund.de
- **Institution**: Technical University Dortmund, Institute for Production Systems
- **Project**: [prodata-projekt.de](https://prodata-projekt.de/)

## 🏛️ Acknowledgments

This research is supported by:

| Organization | Role | 
|-------------|------|
| **German Ministry of Education and Research (BMBF)** | Primary funding through "Data competencies for early career researchers" program |
| **European Union's NextGenerationEU** | Co-funding initiative |
| **VDIVDE Innovation + Technik GmbH** | Program administration and support |

**Research Partners:**
- [RIF Institute for Research and Transfer e.V.](https://www.rif-ev.de/) - Dataset collection and preparation
- [Technical University Dortmund - Institute for Production Systems](https://ips.mb.tu-dortmund.de/) - Research execution

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> 💡 **Tip**: Start with `python main.py --experiment binary_vs_ref --models fast` for a quick overview, then explore the MLflow UI at `http://localhost:5000` to dive into detailed results!