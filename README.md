# CellSexID: Single-Cell Sex Identification Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2024.12.02.626449-red.svg)](https://www.biorxiv.org/content/10.1101/2024.12.02.626449v1)

![CellSexID Overview](fig1.jpg)
*Figure 1: Overview of the CellSexID workflow and validation approach.*

## 📄 Publication

**CellSexID: A Tool for Predicting Biological Sex from Single-Cell RNA-Seq Data**  
📖 [Read the paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.02.626449v1)

---

A comprehensive toolkit for predicting biological sex from single-cell RNA-seq data using machine learning approaches with automatic feature selection capabilities.

## Key Features

- **Multiple ML Models**: Logistic Regression, SVM, XGBoost, Random Forest (default)
- **Automatic Gene Discovery**: Data-driven feature selection across 4 algorithms with cross-validation
- **Predefined Gene Markers**: 14 established sex-specific genes for quick start
- **Custom Gene Lists**: Use your own curated gene markers  
- **Cross-Dataset Validation**: Robust predictions across different studies
- **Easy-to-Use CLI**: Simple command-line interface for all workflows
- **Python API**: Full programmatic access for integration

## 📓 Analysis Notebooks

### CellSexID_Human.ipynb
This comprehensive notebook demonstrates sex prediction across multiple human single-cell RNA-seq datasets, showcasing cross-dataset validation and model performance evaluation. The analysis includes four major human datasets: ATL (Adult T-cell leukemia-lymphoma, GSE294224), Kidney donor cells (GSE151671), AML/MLL bone marrow samples (GSE289435), and Thymic epithelial cells (GSE262749). The notebook covers the complete pipeline from data preprocessing and quality control to machine learning model training, feature selection, and cross-dataset validation, providing insights into sex classification robustness across different tissue types and disease contexts.

### CellSexID_Mouse.ipynb  
This notebook focuses on mouse single-cell data analysis for sex prediction, demonstrating species-specific gene marker identification and validation approaches. The analysis explores sex-specific gene expression patterns in mouse tissues, validates prediction accuracy using established mouse sex markers, and provides comparative analysis between human and mouse sex classification approaches. This notebook serves as a complementary resource for researchers working with mouse models and cross-species validation studies.

### CellSexID_Paper_Figures.ipynb
This is the comprehensive methodology and validation notebook that reproduces main figures from the research paper. The notebook provides a complete end-to-end analysis pipeline including: (1) training data setup and quality control, (2) independent test dataset processing, (3) experimental validation data analysis, (4) analysis after batch effect correction using Scanorama, (5) cross-validation performance assessment with ROC and precision-recall curves, (6) bootstrap statistical analysis for confidence intervals, (7) model performance evaluation on independent test data, and (8) detailed misclassification analysis. This notebook serves as the definitive reference implementation for reproducible research and contains all statistical analyses and visualizations presented in the publication.

## Prerequisites & Installation

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM (depends on dataset size)
- **Storage**: 1GB+ free space

### Required Packages
```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scanpy>=1.8.0
matplotlib>=3.5.0
anndata>=0.8.0
```

### Installation Options

#### Option 1: Direct Installation (Recommended)
```bash
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
pip install -e .
```

#### Option 2: Manual Setup
```bash
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
pip install -r requirements.txt
```

#### Option 3: Conda Environment
```bash
# Create clean environment
conda create -n cellsexid python=3.9
conda activate cellsexid

# Install dependencies
conda install -c conda-forge scanpy pandas scikit-learn matplotlib
pip install xgboost

# Clone and install
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
pip install -e .
```

## Data Requirements & Format

### CRITICAL: Input Data Must Be H5AD Format

CellSexID **only** accepts `.h5ad` files (AnnData format). Convert your data if needed:

```python
import scanpy as sc
import pandas as pd

# From CSV/TSV
adata = sc.read_csv("your_data.csv").T  # Transpose if genes are rows
adata.var_names_unique()

# From 10X
adata = sc.read_10x_mtx("path/to/10x/")
adata.var_names_unique()

# Save as H5AD
adata.write("your_data.h5ad")
```

### Training Data Requirements

**Training data MUST have sex labels:**

```python
import scanpy as sc
adata = sc.read("training_data.h5ad")

# REQUIRED: Gender column in adata.obs
print(adata.obs["gender"].value_counts())
# Expected output:
# Male      1500
# Female    1300
# Name: gender, dtype: int64

# Acceptable label formats:
# ✓ "Male" / "Female" 
# ✓ "M" / "F"
# ✓ "male" / "female"
# ✓ 0 / 1 (0=Female, 1=Male)
```

### Required Preprocessing Pipeline

**Your data must be preprocessed following this standard pipeline:**

```python
import scanpy as sc

# 1. Load raw data
adata = sc.read("raw_data.h5ad")

# 2. Basic filtering
sc.pp.filter_genes(adata, min_cells=3)    # Remove genes in <3 cells
sc.pp.filter_cells(adata, min_genes=200)  # Remove cells with <200 genes

# 3. Calculate QC metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # Mitochondrial genes
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# 4. Filter high MT% cells
adata = adata[adata.obs.pct_counts_mt < 5, :]  # Keep cells with <5% MT

# 5. Normalization & log transformation
sc.pp.normalize_total(adata, target_sum=1e4)   # Normalize to 10,000 counts
sc.pp.log1p(adata)                             # Log(x+1) transform

# 6. Add sex labels for training data
# adata.obs["gender"] = your_sex_labels  # REQUIRED for training

# 7. Save preprocessed data
adata.write("preprocessed_data.h5ad")
```

### Expected Data Structure

```python
# Correct H5AD structure
adata.X              # Gene expression matrix (cells × genes)
adata.obs            # Cell metadata (must include "gender" for training)
adata.var            # Gene metadata  
adata.obs_names      # Cell IDs
adata.var_names      # Gene symbols

# Example inspection:
print(f"Data shape: {adata.shape}")           # (n_cells, n_genes)
print(f"Genes: {adata.var_names[:5]}")        # ['GENE1', 'GENE2', ...]
print(f"Cells: {adata.obs_names[:5]}")        # ['CELL1', 'CELL2', ...]
print(f"Sex labels: {adata.obs['gender'].unique()}")  # ['Male', 'Female']
```

## Gene Markers & Models

### Default Gene Markers (14 genes)
```python
# Predefined sex-specific markers
["Xist", "Ddx3y", "Gm42418", "Eif2s3y", "Rps27rt", "Rpl9-ps6",
 "Kdm5d", "Uba52", "Rpl35", "Rpl36a-ps1", "Uty", "Wdr89", 
 "Lars2", "Rps27"]
```

### Available Models
| Model | Code | Description | Best For |
|-------|------|-------------|----------|
| **Random Forest** | `RF` | **Default** - Robust ensemble method | Most datasets |
| Logistic Regression | `LR` | Linear classifier with regularization | Small datasets |
| Support Vector Machine | `SVM` | Linear SVM with RBF kernel | High-dim data |
| XGBoost | `XGB` | Gradient boosting classifier | Complex patterns |

### Feature Selection Algorithm
- **Cross-validation**: 5-fold stratified CV
- **Multi-model ensemble**: LR + SVM + XGB + RF
- **Majority voting**: Genes appearing in ≥N models  
- **Importance scoring**: Normalized feature importance

## Quick Start Examples

### 1. Basic Usage (Fastest)
```bash
# Use default 14 genes with Random Forest
python cli.py \
  --train_data training.h5ad \
  --test_data test.h5ad \
  --output predictions.csv \
  --plot distribution.png
```

### 2. Custom Gene Markers
```bash
# Use specific genes of interest
python cli.py \
  --train_data training.h5ad \
  --test_data test.h5ad \
  --output results.csv \
  --custom_genes "Xist,Ddx3y,Kdm5d,Eif2s3y" \
  --model XGB
```

### 3. Automatic Gene Discovery (Most Comprehensive)
```bash
# Let algorithm find best genes
python cli.py \
  --train_data training.h5ad \
  --test_data test.h5ad \
  --output results.csv \
  --feature_selection \
  --top_k 20 \
  --min_models 3 \
  --model RF
```

### 4. Model Comparison
```bash
# Test different algorithms
for model in LR SVM XGB RF; do
  python cli.py \
    --train_data training.h5ad \
    --test_data test.h5ad \
    --output ${model}_predictions.csv \
    --model $model
done
```

## Command Line Interface

### Required Arguments
```bash
--train_data TRAIN.H5AD    # Training data with sex labels
--test_data TEST.H5AD      # Data to predict (can be same as train)
--output RESULTS.CSV       # Output predictions file
```

### Optional Arguments
```bash
--plot PLOT.PNG           # Distribution plot output
--model {LR,SVM,XGB,RF}   # ML model (default: RF)
--verbose                 # Detailed output

# Gene Selection (mutually exclusive)
--custom_genes "Gene1,Gene2"    # Use specific genes
--feature_selection             # Automatic gene discovery

# Feature Selection Parameters
--top_k 20                # Top K genes per model (default: 20)
--min_models 3            # Min models threshold (default: 3)
--fs_output DIR           # Feature selection results directory
```

### Complete Examples
```bash
# Example 1: Quick prediction with defaults
python cli.py --train_data train.h5ad --test_data test.h5ad --output pred.csv

# Example 2: Custom genes with XGBoost
python cli.py --train_data train.h5ad --test_data test.h5ad --output pred.csv \
  --custom_genes "Xist,Ddx3y,Kdm5d" --model XGB --plot dist.png

# Example 3: Automatic feature selection
python cli.py --train_data train.h5ad --test_data test.h5ad --output pred.csv \
  --feature_selection --top_k 15 --min_models 2 --verbose

# Example 4: Single gene test
python cli.py --train_data train.h5ad --test_data test.h5ad --output pred.csv \
  --custom_genes "Xist" --model SVM
```

## Python API Usage

### Basic Workflow
```python
from sex_prediction_tool import ImprovedSexPredictionTool

# 1. Initialize with predefined genes
tool = ImprovedSexPredictionTool(use_predefined_genes=True)

# 2. Process training data  
X_train, y_train = tool.process_training_data("training.h5ad")

# 3. Train model
tool.train(X_train, y_train, model_name='RF')

# 4. Process test data
X_test, cell_names = tool.process_test_data("test.h5ad")

# 5. Make predictions
predictions = tool.predict(X_test, model_name='RF')

# 6. Save results
tool.save_predictions(predictions, cell_names, "predictions.csv")
tool.plot_prediction_distribution(predictions, "distribution.png")

# 7. Check results
print(f"Predicted {sum(predictions)} males, {len(predictions)-sum(predictions)} females")
```

### Custom Genes Workflow
```python
# Use specific genes
custom_genes = ["Xist", "Ddx3y", "Kdm5d", "Eif2s3y"]
tool = ImprovedSexPredictionTool(use_predefined_genes=True, custom_genes=custom_genes)

# Continue with training...
X_train, y_train = tool.process_training_data("training.h5ad")
tool.train(X_train, y_train, model_name='XGB')
```

### Automatic Feature Selection Workflow
```python
# Initialize for feature selection
tool = ImprovedSexPredictionTool(use_predefined_genes=False)

# Load training data for feature selection
X, y = tool.load_training_data("training.h5ad")

# Discover optimal genes
selected_genes = tool.find_optimal_genes(
    X, y,
    top_k=20,                    # Consider top 20 from each model
    min_models=3,                # Must appear in ≥3/4 models  
    save_results=True,           # Save intermediate results
    output_dir="fs_results"      # Output directory
)

print(f"Selected {len(selected_genes)} genes: {selected_genes}")

# Train with discovered genes
X_train, y_train = tool.process_training_data("training.h5ad")
tool.train(X_train, y_train, model_name='RF')

# Make predictions...
```

## Output Files & Results

### Predictions CSV
```csv
cell_id,predicted_sex
AAACATACAACCAC-1,Female
AAACATTGAGCTAC-1,Male  
AAACATTGATCAGC-1,Female
AAACCGTGCTTCCG-1,Male
```

### Distribution Plot
- Bar chart showing Male/Female percentages
- Saved as PNG with high resolution (300 DPI)
- Customizable colors and labels

### Feature Selection Results
```
feature_selection_results/
├── LogisticRegression_feature_importances.csv     # LR feature scores
├── SVC_feature_importances.csv                    # SVM feature scores  
├── XGBClassifier_feature_importances.csv          # XGB feature scores
├── RandomForestClassifier_feature_importances.csv # RF feature scores
└── selected_genes_majority_vote.csv              # Final selected genes
```

### Feature Selection CSV Format
```csv
Feature,LogisticRegression_Importance,SVC_Importance,XGBClassifier_Importance,RandomForestClassifier_Importance,AppearCount,CombinedNorm
Xist,0.450,0.523,0.445,0.467,4,1.885
Ddx3y,0.234,0.289,0.278,0.301,4,1.102
Kdm5d,0.189,0.198,0.234,0.256,4,0.877
```

## Complete Examples & Use Cases

### Use Case 1: Quick Cell Sex Annotation
```bash
# You have preprocessed data and want quick sex predictions
python cli.py --train_data reference_with_labels.h5ad \
              --test_data your_new_data.h5ad \
              --output sex_predictions.csv \
              --plot sex_distribution.png
```

### Use Case 2: Cross-Species Analysis  
```bash
# Use human genes for mouse data
python cli.py --train_data human_reference.h5ad \
              --test_data mouse_data.h5ad \
              --output mouse_sex.csv \
              --custom_genes "XIST,DDX3Y,KDM5D,EIF2S3Y"
```

### Use Case 3: Method Development
```bash
# Discover new sex markers in your data
python cli.py --train_data your_data.h5ad \
              --test_data your_data.h5ad \
              --output predictions.csv \
              --feature_selection \
              --top_k 50 \
              --min_models 2 \
              --verbose
```

### Use Case 4: Quality Control Pipeline
```python
# Integrate into QC workflow
import scanpy as sc
from sex_prediction_tool import ImprovedSexPredictionTool

# Load data
adata = sc.read("dataset.h5ad")

# Predict sex if not annotated
if "gender" not in adata.obs.columns:
    tool = ImprovedSexPredictionTool(use_predefined_genes=True)
    # Use public reference for training
    X_train, y_train = tool.process_training_data("reference.h5ad")
    tool.train(X_train, y_train)
    
    X_test, cell_names = tool.process_test_data("dataset.h5ad")
    predictions = tool.predict(X_test)
    
    # Add to metadata
    adata.obs["predicted_sex"] = ['Male' if p else 'Female' for p in predictions]

# Continue with analysis...
```

## Troubleshooting

### Common Issues & Solutions

#### ERROR: `ModuleNotFoundError: No module named 'sex_prediction_tool'`
```bash
# Solution: Check file location and imports
ls -la  # Ensure sex_prediction_tool.py exists
python -c "from sex_prediction_tool import ImprovedSexPredictionTool"
```

#### ERROR: `KeyError: 'gender'`
```python
# Solution: Add sex labels to training data
import scanpy as sc
adata = sc.read("data.h5ad")
print(adata.obs.columns)  # Check available columns

# Add gender column manually
adata.obs["gender"] = your_labels  # Must be Male/Female or 0/1
adata.write("data_with_labels.h5ad")
```

#### ERROR: `ValueError: None of the selected genes found`
```python
# Solution: Check gene names in your data
adata = sc.read("data.h5ad")
print("Available genes:", adata.var_names[:20])
print("Missing genes:", [g for g in ["Xist", "Ddx3y"] if g not in adata.var_names])

# Use available genes only
available_genes = [g for g in ["Xist", "Ddx3y", "Kdm5d"] if g in adata.var_names]
```

#### ERROR: Memory Issues with Large Datasets
```python
# Solution: Subsample or use chunks
import scanpy as sc

# Option 1: Subsample cells
sc.pp.subsample(adata, n_obs=10000)

# Option 2: Select specific cell types
adata = adata[adata.obs.cell_type.isin(["T cells", "B cells"])]

# Option 3: Use feature selection first
tool = ImprovedSexPredictionTool(use_predefined_genes=False)
# This will reduce to ~10-50 genes
```

### Performance Optimization

#### For Large Datasets (>50k cells)
```bash
# Use feature selection to reduce gene number
python cli.py --feature_selection --top_k 10 --min_models 4

# Or use predefined genes (fastest)
python cli.py --custom_genes "Xist,Ddx3y"  # Just 2 genes
```

#### For Cross-Platform Analysis
```python
# Handle different gene naming conventions
gene_mapping = {
    "XIST": "Xist",    # Human -> Mouse
    "DDX3Y": "Ddx3y",
    # Add more mappings...
}

# Apply mapping to your custom genes
mapped_genes = [gene_mapping.get(g, g) for g in your_genes]
```

## Citation & References

If you use CellSexID in your research, please cite:

```bibtex
@software{cellsexid2024,
  title={CellSexID: Single-Cell Sex Identification Tool},
  author={Huilin Tai, Qian Li, Jingtao Wang, Jiahui Tan, Ryann Lang, Basil J. Petrof, Jun Ding},
  year={2024},
  url={https://github.com/mcgilldinglab/CellSexID},
  note={Version 2.0}
}
```

### Related Publications
- Relevant papers on sex determination in single-cell data
- Machine learning approaches for cell annotation
- Feature selection methods in genomics

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature-awesome`)  
3. **Commit** your changes (`git commit -am 'Add awesome feature'`)
4. **Push** to the branch (`git push origin feature-awesome`)
5. **Create** a Pull Request

### Development Setup
```bash
# Clone for development
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Check code style  
flake8 cellsexid/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support & Issues

### GitHub Issues
Found a bug or have a feature request? Please open an issue:
- **Bug reports**: Include error messages, data info, system details
- **Feature requests**: Describe use case and expected behavior  
- **Questions**: Check this README first, then ask

### Common Support Topics
- Data format conversion
- Gene naming conventions  
- Cross-species analysis
- Integration with existing pipelines

## Contact

- **Author**: Huilin
- **Email**: 2378174791@qq.com

---

**Happy cell sex identification!**
