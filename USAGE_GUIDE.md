# CellSexID Usage Guide

## 📁 Project Structure

```
CellSexID/
├── cellsexid/                    # Main package
│   ├── __init__.py              # Package initialization
│   ├── sex_prediction_tool.py   # Core ImprovedSexPredictionTool class
│   └── cli.py                   # Command-line interface
├── data/                        # Your data files (create this folder)
│   ├── training_data.h5ad       # Your training data
│   └── test_data.h5ad           # Your test data
├── run_prediction.py            # Comprehensive examples
├── quick_start.py              # Simple getting started script
├── setup.py                    # Installation script
├── requirements.txt            # Dependencies
└── README.md                   # Main documentation
```

## 🚀 Three Ways to Use CellSexID

### Method 1: Command Line Interface (Easiest)
```bash
# Basic usage
python3 cellsexid/cli.py --train_data data/train.h5ad --test_data data/test.h5ad --output results.csv

# With custom genes
python3 cellsexid/cli.py --train_data data/train.h5ad --test_data data/test.h5ad --output results.csv \
  --custom_genes "XIST,DDX3Y,KDM5D"

# Automatic feature selection
python3 cellsexid/cli.py --train_data data/train.h5ad --test_data data/test.h5ad --output results.csv \
  --feature_selection --top_k 20
```

### Method 2: Quick Start Script (Beginner-Friendly)
```bash
# Edit quick_start.py to set your file paths, then run:
python3 quick_start.py
```

### Method 3: Python API (Most Flexible)
```python
from cellsexid import ImprovedSexPredictionTool

tool = ImprovedSexPredictionTool(use_predefined_genes=True)
X_train, y_train = tool.process_training_data("data/train.h5ad")
tool.train(X_train, y_train, model_name='RF')
X_test, cell_names = tool.process_test_data("data/test.h5ad")
predictions = tool.predict(X_test, model_name='RF')
tool.save_predictions(predictions, cell_names, "results.csv")
```

## 📊 Data Preparation Checklist

### ✅ Before Using CellSexID:

1. **Convert to H5AD format**:
   ```python
   import scanpy as sc
   adata = sc.read_10x_mtx('path/to/matrix.mtx', var_names='gene_symbols')
   adata.write('data.h5ad')
   ```

2. **Add gender labels** (training data only):
   ```python
   adata.obs['gender'] = ['Male', 'Female', ...]  # or [1, 0, ...]
   ```

3. **Apply standard preprocessing**:
   ```python
   sc.pp.filter_genes(adata, min_cells=3)
   sc.pp.filter_cells(adata, min_genes=200)
   adata = adata[adata.obs["pct_counts_mt"] < 5]
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   ```

## 🛠️ Installation Options

### Option A: Development Installation
```bash
git clone https://github.com/yourusername/CellSexID.git
cd CellSexID
pip install -e .
# Now you can use: import cellsexid
```

### Option B: Simple Setup
```bash
git clone https://github.com/yourusername/CellSexID.git
cd CellSexID
pip install -r requirements.txt
# Use direct file paths: python3 cellsexid/cli.py
```

## 🎯 Common Use Cases

### 1. Cross-Study Validation
```bash
# Train on Study A, test on Study B
python3 cellsexid/cli.py --train_data studyA.h5ad --test_data studyB.h5ad --output cross_validation.csv
```

### 2. Model Comparison
```bash
# Test different algorithms
for model in LR SVM XGB RF; do
  python3 cellsexid/cli.py --train_data train.h5ad --test_data test.h5ad \
    --output ${model}_results.csv --model $model
done
```

### 3. Gene Discovery
```bash
# Find optimal genes for your specific dataset
python3 cellsexid/cli.py --train_data train.h5ad --test_data test.h5ad \
  --output results.csv --feature_selection --top_k 20 --min_models 3
```

## 🔍 Troubleshooting

### Common Errors:

1. **ImportError**: Run `pip install -e .` or check Python path
2. **KeyError: 'gender'**: Add gender column to training data
3. **No genes found**: Check gene naming (human vs mouse)
4. **Memory issues**: Use smaller datasets or reduce genes

### Getting Help:
- Check the examples in `run_prediction.py`
- Read the full documentation in `README.md`
- Open an issue on GitHub

## 📝 Example Data Format

Your H5AD files should look like this:

```python
import scanpy as sc
adata = sc.read('your_data.h5ad')

# Shape: (n_cells, n_genes)
print(adata.shape)  # e.g., (5000, 20000)

# For training data:
print(adata.obs['gender'].value_counts())
# Male      2500
# Female    2500

# Gene names should include sex markers:
sex_genes = ['XIST', 'DDX3Y', 'KDM5D', 'EIF2S3Y']
print([g for g in sex_genes if g in adata.var_names])
```

## 🎉 Quick Success Test

1. Download the repository
2. Put your H5AD files in `data/` folder
3. Run: `python3 quick_start.py`
4. Check for `predictions.csv` and `distribution.png`

Happy analyzing! 🧬 