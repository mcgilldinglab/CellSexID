# CellSexID: Single-Cell Sex Identification Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2024.12.02.626449v2-red.svg)](https://www.biorxiv.org/content/10.1101/2024.12.02.626449v2)

![CellSexID Overview](fig1.jpg)
*Figure 1: Overview of the CellSexID workflow and validation approach.*

## Publication

**CellSexID: A Tool for Predicting Biological Sex from Single-Cell RNA-Seq Data**  
[Read the paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.02.626449v2)

---

A comprehensive toolkit for predicting biological sex from single-cell RNA-seq data using machine learning approaches with automatic feature selection capabilities.

## Main Application Scenarios

- **Sample demultiplexing**: Identify sex for mixed-sex samples without prior knowledge
- **Quality control**: Verify reported sex labels in published datasets  
- **Organ transplantation studies**: Track donor vs recipient cells by sex
- **Cross-tissue analysis**: Apply models trained on one tissue to predict sex in another
- **Species translation**: Adapt human-trained models for mouse studies

## Core Functionalities

- Machine learning models: Logistic Regression, SVM, XGBoost, Random Forest
- Automatic gene discovery with cross-validation feature selection
- Predefined sex marker genes for rapid deployment
- Cross-dataset validation and batch effect handling
- Command-line interface and Python API

## Installation

### Recommended: Conda Environment
```bash
# Create clean environment
conda create -n cellsexid python=3.9
conda activate cellsexid

# Install dependencies
conda install -c conda-forge scanpy pandas scikit-learn matplotlib
conda install -c conda-forge xgboost

# Install CellSexID
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
pip install -e .
```

### Alternative: Direct Installation
```bash
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
pip install -e .
```

### Manual Setup
```bash
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

#### Basic Prediction
```bash
python cli.py --train_data training.h5ad --test_data test.h5ad --output predictions.csv
```

#### Custom Gene Markers
```bash
python cli.py --train_data training.h5ad --test_data test.h5ad \
  --custom_genes "Xist,Ddx3y,Kdm5d,Eif2s3y" --output results.csv
```

#### Automatic Gene Discovery
```bash
python cli.py --train_data training.h5ad --test_data test.h5ad \
  --feature_selection --top_k 20 --min_models 3 --output results.csv
```

#### Command Line Parameters
```bash
--train_data FILE.h5ad     # Training data with sex labels
--test_data FILE.h5ad      # Data for prediction
--output RESULTS.csv       # Output file
--model {LR,SVM,XGB,RF}    # Model choice (default: RF)
--custom_genes "Gene1,Gene2"  # Specific gene list
--feature_selection        # Enable automatic gene discovery
--top_k N                  # Top genes per model (default: 20)
--min_models N             # Minimum model consensus (default: 3)
```

### Python API

```python
from cellsexid.sex_prediction_tool import ImprovedSexPredictionTool

# Initialize tool
tool = ImprovedSexPredictionTool(
    use_predefined_genes=True,
    custom_genes=None
)

# Process data and train model
X_train, y_train = tool.process_training_data("training.h5ad")
tool.train(X_train, y_train, model_name='RF')

# Make predictions
X_test, cell_names = tool.process_test_data("test.h5ad")
predictions = tool.predict(X_test, model_name='RF')

# Save results
tool.save_predictions(predictions, cell_names, "predictions.csv")
```

## Data Requirements

### Input Format
CellSexID accepts `.h5ad` files (AnnData format).

### Training Data Structure
```python
adata.X              # Expression matrix (cells × genes)
adata.obs["sex"]     # Sex labels: "Male"/"Female" or "M"/"F" or 0/1
adata.var_names      # Gene symbols
```

### Required Preprocessing
```python
import scanpy as sc

# Standard preprocessing pipeline
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.pct_counts_mt < 5, :]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
```

## Output Files

### Predictions
```csv
cell_id,predicted_sex
CELL001,Female
CELL002,Male
CELL003,Female
```

### Feature Selection Results
```
feature_selection_results/
├── LogisticRegression_feature_importances.csv
├── SVC_feature_importances.csv  
├── XGBClassifier_feature_importances.csv
├── RandomForestClassifier_feature_importances.csv
└── selected_genes_majority_vote.csv
```

## Tutorial Notebooks

We provide three Jupyter notebooks demonstrating CellSexID usage:

### Tutorial 1: Comprehensive Sex Prediction (`CellSexID_tutorial.ipynb`)
Demonstrates sex prediction for mouse and human single-cell datasets. This notebook covers data preprocessing, model training with four machine learning algorithms (Logistic Regression, SVM, XGBoost, Random Forest), and cross-validation evaluation using predefined sex marker genes.

**Mouse analysis**: MTX file loading, quality control, and prediction using 14 mouse sex markers  
**Human analysis**: Human dataset processing and prediction using 9 human sex markers

### Tutorial 2: Cross-Tissue Analysis - Human (`CellSexID_Human_cross_tissue.ipynb`) 
Demonstrates cross-tissue validation by training models on one human tissue type and testing on another. This notebook addresses tissue-specific expression variations and model robustness across different human tissue types.

### Tutorial 3: Cross-Tissue Analysis - Mouse (`CellSexID_Mouse_cross_tissue.ipynb`)
Demonstrates cross-tissue sex prediction using mouse single-cell data. This notebook shows model transfer capabilities across different mouse tissue types and validates species-specific marker performance.

## API Reference

### Core Classes and Methods

```python
from cellsexid.sex_prediction_tool import ImprovedSexPredictionTool

# Initialize tool
tool = ImprovedSexPredictionTool(
    use_predefined_genes: bool = True,
    custom_genes: List[str] = None
)

# Data processing
tool.process_training_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]
tool.process_test_data(filepath: str) -> Tuple[np.ndarray, List[str]]

# Model training and prediction
tool.train(X: np.ndarray, y: np.ndarray, model_name: str = 'RF') -> None
tool.predict(X: np.ndarray, model_name: str = 'RF') -> np.ndarray

# Feature selection
tool.find_optimal_genes(
    X: np.ndarray, 
    y: np.ndarray,
    top_k: int = 20,
    min_models: int = 3,
    save_results: bool = False,
    output_dir: str = None
) -> List[str]

# Output handling
tool.save_predictions(predictions: np.ndarray, cell_names: List[str], filename: str) -> None
tool.plot_prediction_distribution(predictions: np.ndarray, filename: str) -> None
```

### Input/Output Specifications

**Input Types:**
- `filepath`: String path to .h5ad file
- `X`: numpy.ndarray of shape (n_cells, n_genes) 
- `y`: numpy.ndarray of shape (n_cells,) with binary labels
- `model_name`: String from {'LR', 'SVM', 'XGB', 'RF'}

**Output Types:**
- `predictions`: numpy.ndarray of binary predictions (0=Female, 1=Male)
- `cell_names`: List[str] of cell identifiers
- `selected_genes`: List[str] of optimal gene markers

## Citation

```bibtex
@article{tai2024cellsexid,
  title={CellSexID: A Tool for Predicting Biological Sex from Single-Cell RNA-Seq Data},
  author={Tai, Huilin and Li, Qian and Wang, Jingtao and Tan, Jiahui and Lang, Ryann and Petrof, Basil J and Ding, Jun},
  journal={bioRxiv},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mcgilldinglab/CellSexID/issues)
- **Contact**: huilin.tai@mail.mcgill.ca
