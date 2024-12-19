# CellSexID

CellSexID is a streamlined and user-friendly tool designed to predict the biological sex of cells based on single-cell RNA-seq data. It leverages machine learning models trained on publicly available datasets to make accurate predictions on user-provided single-cell input data.

![Figure 1: Overview of CellSexID workflow](fig1.jpg)
*Figure 1: Overview of the CellSexID workflow.*

## Overview

- **a.** Identification of sex-specific gene features for predicting cell origin in single-cell RNA sequencing data. A committee of four classifiers (Random Forest, Logistic Regression, SVM, XGBoost) selects genes deemed critical for distinguishing cell sex, based on majority voting across models.
- **b.** Experimental validation of CellSexID using chimeric mouse models. The single-cell RNA-seq data from chimeric mice are used to test classifiers pre-trained on a public dataset, with predictions based on the selected gene features. Diaphragm macrophages from female mice transplanted with male bone marrow are isolated, flow-sorted, barcoded, and sequenced, serving as ground truth for validating CellSexID predictions.
- **c.** Evaluation of the model's predictive performance using various metrics calculated against flow cytometry-derived ground truth, providing an assessment of reliability.
- **d.** Application of CellSexID for annotating cell origin in chimeric mice enables a range of single-cell analyses, supporting studies of cellular dynamics and differences between recipient and donor cells in diverse research contexts.
 
## Publication

Our research article detailing CellSexID is now available on bioRxiv:
[CellSexID: A Tool for Predicting Biological Sex from Single-Cell RNA-Seq Data](https://www.biorxiv.org/content/10.1101/2024.12.02.626449v1)

## Key Contributions and Impact

### Transformative Methodology
CellSexID eliminates the need for exogenous labeling or cross-breeding strategies by leveraging endogenous gene expression associated with biological sex. This significantly reduces cost and complexity, positioning CellSexID as a robust and scalable solution for experimental and clinical applications.

### Biological Discovery
Using sex as a natural surrogate marker, CellSexID enabled the identification of a unique subset of tissue-resident macrophages in skeletal muscle. These macrophages, derived from bone marrow-independent embryonic origins, exhibit an anti-inflammatory and pro-regenerative gene expression profile. This discovery underscores the role of cellular ontogeny in shaping immune responses and highlights CellSexID’s ability to uncover functional differences that were previously challenging to resolve.

### Real-World Validation
CellSexID demonstrated high concordance with flow cytometry ground truth data, validating its accuracy in distinguishing bone marrow-derived and bone marrow-independent macrophages. This underscores its utility for exploring cell-specific behaviors and gene expression profiles across diverse biological systems.



## Tech Features

- **Python Package Integration:** Distributed as a Python package for direct integration into your existing data analysis workflows, offering programmatic control and flexibility.
- **Simplified Workflow:** Trains on public datasets to ensure accessibility and reproducibility.
- **Single-Cell Focus:** Tailored for single-cell RNA-seq data input in H5AD format.
- **Accurate Predictions:** Utilizes advanced machine learning models (XGBoost, Logistic Regression, SVM, Random Forest).
- **Visualization:** Provides options to visualize the distribution of predicted sexes.
- **Command-Line Interface (CLI):** In addition to the Python package, offers a user-friendly CLI for seamless integration into command-line-based pipelines.
- **Detailed Tutorial:** A comprehensive tutorial notebook covers all steps, including model training and prediction.

## Getting Started

 

 

 
 
 
## Prerequisites

Before installing CellSexID, ensure that you have the following prerequisites set up:

### Anaconda Environment

Using Anaconda is recommended for managing dependencies and creating isolated environments.

1. **Install Anaconda or Miniconda:**
   - [Download Anaconda](https://www.anaconda.com/products/distribution)
   - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. **Set Up a Virtual Environment:**
   Open your terminal or command prompt and execute the following commands to create and activate a new Conda environment:

   ```bash
   conda create -n cellsexid_env python=3.8
   conda activate cellsexid_env
   ```

3. **Verify Python Installation:**
   Ensure that Python 3.8 is installed in the environment:

   ```bash
   python --version
   # Output should be Python 3.8.x
   ```

## Installation

You can install CellSexID using one of the following methods:

### Option 1: Install Directly from GitHub

This method allows you to install the latest version of CellSexID directly from the GitHub repository.

```bash
pip install git+https://github.com/mcgilldinglab/CellSexID.git
```

### Option 2: Clone the Repository

Cloning the repository gives you access to the source code, which can be useful for development or customization.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mcgilldinglab/CellSexID.git
   cd CellSexID
   ```

2. **Install Dependencies:**

   Ensure that you are in the `cellsexid_env` Conda environment before installing dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Package Locally:**

   This installs CellSexID in editable mode, allowing you to make changes to the source code if needed.

   ```bash
   pip install .
   ```

## Verification

After installation, you can verify that CellSexID is installed correctly by importing it in Python:

```python
import cellsexid
print(cellsexid.__version__)
```

If no errors occur and the version number is displayed, the installation was successful.

---

**Note:** Always ensure that you are operating within the `cellsexid_env` Conda environment to avoid conflicts with other Python packages on your system.
 
 




## Usage

### Command-Line Interface (CLI)

#### Command-Line Arguments
| Argument | Description |
|----------|-------------|
| --train_data | Path to the preprocessed training data in .h5ad format |
| --test_data | Path to the test data in .h5ad format |
| --model | (Optional) Machine learning model to use (XGB, LR, SVM, RF). Default: XGB |
| --output | Path to the output CSV file where predictions will be saved |
| --plot | (Optional) Path to save the distribution plot of predicted sexes |

#### Example Usage
```bash
cellsexid-run \
    --train_data preprocessed_data.h5ad \
    --test_data sex_chimeric_gender_9_25.h5ad \
    --model XGB \
    --output predictions.csv \
    --plot distribution.pdf
```

### Programmatic Usage

```python
from cellsexid import SexPredictionTool

# Initialize the tool
sex_predictor = SexPredictionTool()

# Process training data
X_train, y_train = sex_predictor.process_training_data("preprocessed_data.h5ad")

# Train the model
sex_predictor.train(X_train, y_train, model_name="XGB")

# Process test data
X_test, cell_names = sex_predictor.process_test_data("sex_chimeric_gender_9_25.h5ad")

# Make predictions
y_pred = sex_predictor.predict(X_test, model_name="XGB")

# Save predictions
sex_predictor.save_predictions(y_pred, cell_names, "predictions.csv")

# Save distribution plot
sex_predictor.plot_prediction_distribution(y_pred, "distribution.pdf")
```

## Data Preparation

### Input Requirements
- Format: Both training and test data should be in AnnData's .h5ad format
- Gene Expression Matrix:
  - Genes as variables (adata.var_names)
  - Cells as observations (adata.obs_names)
- Training Data:
  - Includes a gender column in adata.obs with binary labels:
    - 0 for Female
    - 1 for Male

 

### Training Data
We provide a zipped file containing preprocessed training data (`preprocessed_data.zip`). The training data has undergone the following preprocessing steps:
- Filtered to include genes present in at least 3 cells
- Normalized to 10,000 counts per cell
- Log1p transformed

Steps to prepare training data:
1. Download the training data from the repository
2. Unzip the file:
```bash
unzip preprocessed_data.zip
```

### Test Data Requirements
- Gene expression data of the cells you want to predict
- Must be in .h5ad format
- Must be preprocessed using the same steps as training data:
  - Filter genes present in at least 3 cells
  - Normalize to 10,000 counts per cell
  - Apply log1p transformation
- Must include these genes in adata.var_names:
  - 'Rpl35', 'Rps27rt', 'Rpl9-ps6', 'Rps27', 'Uba52', 'Lars2', 'Gm42418', 'Uty', 'Kdm5d', 'Eif2s3y', 'Ddx3y', 'Xist'

 
## Outputs

### Predictions CSV (predictions.csv)
| Column | Description |
|--------|-------------|
| cell_id | Cell identifiers from test data |
| predicted_sex | Predicted sex (Male or Female) |

### Distribution Plot (distribution.pdf)
A bar plot visualizing the percentage distribution of predicted sexes across the test dataset.

## Example Workflow

1. Prepare Training Data (`preprocessed_data.h5ad`):
   - Ensure the training data includes a gender column (0 for Female, 1 for Male)

2. Prepare Test Data (`sex_chimeric_gender_9_25.h5ad`):
   - Ensure the test data includes the selected genes

3. Run CLI Prediction:
```bash
cellsexid-run \
    --train_data preprocessed_data.h5ad \
    --test_data sex_chimeric_gender_9_25.h5ad \
    --model XGB \
    --output predictions.csv \
    --plot distribution.pdf
```

4. Review Outputs:
   - Open predictions.csv for detailed results
   - View the distribution plot in distribution.pdf


 

## API Documentation

### SexPredictionTool Class

The main class for performing sex prediction on single-cell RNA-seq data.

#### Constructor

```python
tool = SexPredictionTool()
```

Initializes a new SexPredictionTool instance with pre-configured models:
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- XGBoost (XGB)
- Random Forest (RF)

All models include StandardScaler preprocessing in their pipelines.

#### Properties

- `selected_genes`: List of genes used for prediction
  ```python
  ['Rpl35', 'Rps27rt', 'Rpl9-ps6', 'Rps27', 'Uba52', 'Lars2',
   'Gm42418', 'Uty', 'Kdm5d', 'Eif2s3y', 'Ddx3y', 'Xist']
  ```

#### Methods

##### process_training_data(h5ad_path)
```python
X_train, y_train = tool.process_training_data("path/to/training_data.h5ad")
```
Processes training data from an H5AD file.

**Parameters:**
- `h5ad_path` (str): Path to training data in H5AD format

**Returns:**
- `X_train` (numpy.ndarray): Feature matrix
- `y_train` (numpy.ndarray): Labels (0 for Female, 1 for Male)

**Raises:**
- `ValueError`: If none of the selected genes are found in the dataset

##### process_test_data(h5ad_path)
```python
X_test, cell_names = tool.process_test_data("path/to/test_data.h5ad")
```
Processes test data from an H5AD file.

**Parameters:**
- `h5ad_path` (str): Path to test data in H5AD format

**Returns:**
- `X_test` (numpy.ndarray): Feature matrix
- `cell_names` (pandas.Index): Cell identifiers

**Raises:**
- `ValueError`: If none of the selected genes are found in the dataset

##### train(X_train, y_train, model_name='XGB')
```python
tool.train(X_train, y_train, model_name='XGB')
```
Trains the selected model on the provided data.

**Parameters:**
- `X_train` (numpy.ndarray): Feature matrix
- `y_train` (numpy.ndarray): Labels
- `model_name` (str): Model to train ('LR', 'SVM', 'XGB', or 'RF')

**Raises:**
- `ValueError`: If model_name is not recognized

##### predict(X_test, model_name='XGB')
```python
y_pred = tool.predict(X_test, model_name='XGB')
```
Makes predictions using the trained model.

**Parameters:**
- `X_test` (numpy.ndarray): Feature matrix
- `model_name` (str): Model to use for prediction

**Returns:**
- `numpy.ndarray`: Predicted labels (0 for Female, 1 for Male)

**Raises:**
- `ValueError`: If model_name is not recognized

##### save_predictions(y_pred, cell_names, output_file)
```python
tool.save_predictions(y_pred, cell_names, "predictions.csv")
```
Saves predictions to a CSV file.

**Parameters:**
- `y_pred` (numpy.ndarray): Predicted labels
- `cell_names` (pandas.Index): Cell identifiers
- `output_file` (str): Path to save predictions

##### plot_prediction_distribution(y_pred, save_path)
```python
tool.plot_prediction_distribution(y_pred, "distribution.pdf")
```
Creates and saves a bar plot showing the distribution of predicted sexes.

**Parameters:**
- `y_pred` (numpy.ndarray): Predicted labels
- `save_path` (str): Path to save the plot

### Command-Line Interface

The package provides a command-line interface through the `cellsexid-run` command.

```bash
cellsexid-run --train_data PATH --test_data PATH --model MODEL --output PATH --plot PATH
```

**Arguments:**
- `--train_data`: Path to training data (H5AD format)
- `--test_data`: Path to test data (H5AD format)
- `--model`: Model to use (LR, SVM, XGB, or RF)
- `--output`: Path for output CSV containing predictions
- `--plot`: Path for output prediction distribution plot

All arguments are required.



## Contributing

Contributions are welcome! If you have suggestions or would like to report issues, please submit them through the [Issues](https://github.com/mcgilldinglab/CellSexID/issues) section of the repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by robust methodologies developed for biological sex prediction
- Thanks to the contributors and public datasets that make this tool possible

## Contact

For questions or support, please contact **Huilin Tai** at [2378174791@qq.com](mailto:2378174791@qq.com)

## References

- [Scanpy Documentation](https://scanpy.readthedocs.io/en/stable/)
- [Anndata Documentation](https://anndata.readthedocs.io/en/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

## Quick Links

- [API Documentation](#api-documentation)
- [Installation Guide](#getting-started)
- [Usage Instructions](#usage)
- [Example Workflow](#example-workflow)
- [Publication on bioRxiv](#publication)

 

**Happy Analyzing!**
