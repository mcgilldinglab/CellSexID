# CellSexID API Documentation

## SexPredictionTool Class

### Initialization

```python
from cellsexid.sex_prediction_tool import SexPredictionTool

tool = SexPredictionTool(
    species='mouse',           # 'mouse' or 'human'
    use_predefined_genes=True, # True for predefined markers, False for discovery
    custom_genes=None,         # List of custom gene symbols
    sex_column='sex'           # Column name containing sex labels
)
```

#### Parameters
- **species** (`str`): Species for predefined markers ('mouse' or 'human')
- **use_predefined_genes** (`bool`): Whether to use predefined markers
- **custom_genes** (`List[str]`, optional): Custom gene list, overrides predefined markers
- **sex_column** (`str`): Column name in `adata.obs` containing sex labels

#### Predefined Markers
- **Mouse**: `Xist`, `Ddx3y`, `Gm42418`, `Eif2s3y`, `Rps27rt`, `Rpl9-ps6`, `Kdm5d`, `Uba52`, `Rpl35`, `Rpl36a-ps1`, `Uty`, `Wdr89`, `Lars2`, `Rps27`
- **Human**: `XIST`, `DDX3Y`, `KDM5D`, `RPS4Y1`, `EIF1AY`, `USP9Y`, `UTY`, `ZFY`, `SRY`, `TMSB4Y`, `NLGN4Y`

## Core Methods

### 2-Dataset Workflow (Predefined Markers)

#### `fit(train_data, model_name='RF')`
Train model using predefined or custom markers.

**Parameters:**
- `train_data` (`str`): Path to training .h5ad file
- `model_name` (`str`): Model type ('LR', 'SVM', 'XGB', 'RF')

**Returns:** None

**Example:**
```python
tool = SexPredictionTool(species='mouse', use_predefined_genes=True)
tool.fit('training_data.h5ad', model_name='RF')
```

### 3-Dataset Workflow (Custom Marker Discovery)

#### `discover_markers(marker_data, top_k=20, min_models=3, save_results=True, output_dir="feature_selection_results")`
Discover optimal markers using feature selection.

**Parameters:**
- `marker_data` (`str`): Path to .h5ad file for marker discovery
- `top_k` (`int`): Top K features per model (default: 20)
- `min_models` (`int`): Minimum models consensus (default: 3)
- `save_results` (`bool`): Save intermediate results (default: True)
- `output_dir` (`str`): Output directory for results

**Returns:** `List[str]` - Selected gene markers

#### `fit_with_discovered_markers(train_data, model_name='RF')`
Train model using discovered markers.

**Parameters:**
- `train_data` (`str`): Path to training .h5ad file
- `model_name` (`str`): Model type ('LR', 'SVM', 'XGB', 'RF')

**Returns:** None

### Prediction

#### `predict(test_data)`
Make predictions on test data.

**Parameters:**
- `test_data` (`str`): Path to test .h5ad file

**Returns:** 
- `predictions` (`np.ndarray`): Binary predictions (0=Female, 1=Male)
- `cell_names` (`List[str]`): Cell identifiers

### Utility Methods

#### `save_predictions(predictions, cell_names, output_file)`
Save predictions to CSV file.

**Parameters:**
- `predictions` (`np.ndarray`): Prediction array
- `cell_names` (`List[str]`): Cell identifiers  
- `output_file` (`str`): Output CSV file path

#### `plot_prediction_distribution(predictions, save_path)`
Save distribution plot of predictions.

**Parameters:**
- `predictions` (`np.ndarray`): Prediction array
- `save_path` (`str`): Output plot file path

#### `get_available_genes(h5ad_path)`
Check which predefined genes are available in dataset.

**Parameters:**
- `h5ad_path` (`str`): Path to .h5ad file

**Returns:** `Dict` with 'available' and 'missing' gene lists

#### `get_summary()`
Print current configuration summary.

## Command Line Interface

### Basic Usage

```bash
# 2-dataset workflow (predefined markers)
python cli.py --species mouse --train train.h5ad --test test.h5ad --output predictions.csv

# 3-dataset workflow (marker discovery)  
python cli.py --species mouse --marker_train discovery.h5ad train.h5ad --test test.h5ad --output predictions.csv
```

### Parameters

#### Required
- `--test`: Test data (.h5ad file)
- `--output`: Output predictions (.csv file)
- `--train` OR `--marker_train`: Training data specification

#### Optional
- `--species {mouse,human}`: Species for markers (default: mouse)
- `--model {LR,SVM,XGB,RF}`: ML model (default: RF)
- `--sex_column`: Sex column name (default: sex)
- `--custom_genes`: Comma-separated gene list (2-dataset only)
- `--top_k`: Top features per model (default: 20)
- `--min_models`: Minimum model consensus (default: 3)
- `--plot`: Save distribution plot
- `--verbose`: Detailed output

## Data Types

### Input Data Structure
```python
# AnnData object requirements
adata.X              # np.ndarray or sparse matrix (n_cells × n_genes)
adata.obs[sex_column] # pd.Series with sex labels
adata.var_names      # pd.Index with gene symbols
adata.obs_names      # pd.Index with cell identifiers
```

### Sex Label Formats
Accepted formats in `adata.obs[sex_column]`:
- String: "Male"/"Female", "M"/"F", "male"/"female"
- Numeric: 0 (Female), 1 (Male)

### Output Formats

#### Predictions CSV
```csv
cell_id,predicted_sex
CELL_001,Female
CELL_002,Male
CELL_003,Female
```

#### Feature Importance CSV (marker discovery)
```csv
Feature,Importance,Rank
Xist,0.234,1
Ddx3y,0.187,2
Kdm5d,0.156,3
```

## Error Handling

### Common Errors

#### `ValueError: Sex column 'sex' not found`
- **Cause**: Specified sex column doesn't exist in `adata.obs`
- **Solution**: Use `--sex_column` parameter with correct column name

#### `ValueError: No selected markers found in dataset`
- **Cause**: None of the predefined markers exist in the dataset
- **Solution**: Check gene naming convention (mouse vs human) or use custom genes

#### `FileNotFoundError: File not found`
- **Cause**: Input file path is incorrect
- **Solution**: Verify file paths and ensure .h5ad files exist

#### `ValueError: Model not trained yet`
- **Cause**: Calling `predict()` before training
- **Solution**: Call `fit()` or `fit_with_discovered_markers()` first

 
