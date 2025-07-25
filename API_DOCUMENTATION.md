# API Documentation for `SexPredictionTool`

The `SexPredictionTool` class provides functionalities to train machine learning models and predict the biological sex of cells from single-cell RNA-seq data.

---

## Class: `SexPredictionTool`

### Description:
This class encapsulates the entire workflow of sex prediction, from data preprocessing to model training, prediction, and visualization of results.

---

### Methods:

#### `__init__()`
Initializes the `SexPredictionTool` class and sets up the available machine learning models.

- **Available Models:**
  - `XGB` (XGBoost)
  - `LR` (Logistic Regression)
  - `SVM` (Support Vector Machine)
  - `RF` (Random Forest)

---

#### `process_training_data(h5ad_path)`
Reads preprocessed training data from an H5AD file and extracts features based on a predefined list of genes.

- **Parameters:**
  - `h5ad_path` (str): Path to the preprocessed training data file in `.h5ad` format.
- **Returns:**
  - `X` (numpy.ndarray): Feature matrix (genes as features, cells as samples).
  - `y` (numpy.ndarray): Labels array (binary values for gender).
- **Notes:**
  - Assumes the training data has a `gender` column in `adata.obs`.

---

#### `process_test_data(h5ad_path)`
Processes test data from an H5AD file and extracts features for prediction.

- **Parameters:**
  - `h5ad_path` (str): Path to the test data file in `.h5ad` format.
- **Returns:**
  - `X_test` (numpy.ndarray): Test feature matrix.
  - `cell_names` (list): List of cell identifiers.

---

#### `train(X_train, y_train, model_name='XGB')`
Trains the selected model using the given training data.

- **Parameters:**
  - `X_train` (numpy.ndarray): Feature matrix for training.
  - `y_train` (numpy.ndarray): Labels for training.
  - `model_name` (str): Model to train (`XGB`, `LR`, `SVM`, `RF`).
- **Raises:**
  - `ValueError`: If the specified `model_name` is not available.

---

#### `predict(X_test, model_name='XGB')`
Makes predictions on the test data using the trained model.

- **Parameters:**
  - `X_test` (numpy.ndarray): Test feature matrix.
  - `model_name` (str): Model to use for predictions (`XGB`, `LR`, `SVM`, `RF`).
- **Returns:**
  - `y_pred` (numpy.ndarray): Predicted labels (binary values).

---

#### `predict_proba(X_test, model_name='XGB')`
Returns prediction probabilities for the test data.

- **Parameters:**
  - `X_test` (numpy.ndarray): Test feature matrix.
  - `model_name` (str): Model to use for predictions (`XGB`, `LR`, `SVM`, `RF`).
- **Returns:**
  - `y_proba` (numpy.ndarray): Prediction probabilities.

---

#### `save_predictions(y_pred, cell_names, output_file)`
Saves predictions to a CSV file.

- **Parameters:**
  - `y_pred` (numpy.ndarray): Predicted labels.
  - `cell_names` (list): Cell identifiers corresponding to predictions.
  - `output_file` (str): Path to the output CSV file.

---

#### `plot_prediction_distribution(y_pred, save_path=None)`
Plots the distribution of predicted sexes as a bar chart.

- **Parameters:**
  - `y_pred` (numpy.ndarray): Predicted labels (binary values).
  - `save_path` (str, optional): Path to save the plot as an image file.
- **Notes:**
  - Displays the plot if `save_path` is not specified.

---

## Command-Line Tool

The `sex_prediction_tool` provides a command-line interface for running the entire workflow.

### Arguments:
- `--train_data` (required): Path to the preprocessed training H5AD file.
- `--test_data` (required): Path to the test H5AD file.
- `--model` (optional): Machine learning model to use (`XGB`, `LR`, `SVM`, `RF`). Default is `XGB`.
- `--output` (required): Path to save predictions as a CSV file.
- `--plot` (optional): Path to save the distribution plot as an image.

### Example Usage:
```bash
python run.py \
    --train_data preprocessed_data.h5ad \
    --test_data test_data.h5ad \
    --model XGB \
    --output predictions.csv \
    --plot distribution.pdf
```

---

## Example Workflow (Code)

### Python Usage:
```python
from sex_prediction_tool import SexPredictionTool

# Initialize the tool
sex_predictor = SexPredictionTool()

# Process training data
X_train, y_train = sex_predictor.process_training_data('preprocessed_data.h5ad')

# Train the model
sex_predictor.train(X_train, y_train, model_name='XGB')

# Process test data
X_test, cell_names = sex_predictor.process_test_data('test_data.h5ad')

# Make predictions
y_pred = sex_predictor.predict(X_test, model_name='XGB')

# Save predictions
sex_predictor.save_predictions(y_pred, cell_names, 'predictions.csv')

# Plot prediction distribution
sex_predictor.plot_prediction_distribution(y_pred, save_path='distribution.pdf')
```

---
 
