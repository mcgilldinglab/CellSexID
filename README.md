
---

# CellSexID

CellSexID is a streamlined and user-friendly tool designed to predict the biological sex of cells based on single-cell RNA-seq data. It leverages machine learning models trained on publicly available datasets to make accurate predictions on user-provided single-cell input data.


<div style="text-align: center; margin: 20px 0;">
  <img src="fig1.jpg" alt="Figure 1: Overview of CellSexID workflow">
  <p><em>Figure 1: Overview of the CellSexID workflow.</em></p>
</div>

- **a.** Identification of sex-specific gene features for predicting cell origin in single-cell RNA sequencing data. A committee of four classifiers (Random Forest, Logistic Regression, SVM, XGBoost) selects genes deemed critical for distinguishing cell sex, based on majority voting across models.
- **b.** Experimental validation of CellSexID using chimeric mouse models. The single-cell RNA-seq data from chimeric mice are used to test classifiers pre-trained on a public dataset, with predictions based on the selected gene features. Diaphragm macrophages from female mice transplanted with male bone marrow are isolated, flow-sorted, barcoded, and sequenced, serving as ground truth for validating CellSexID predictions.
- **c.** Evaluation of the model's predictive performance using various metrics calculated against flow cytometry-derived ground truth, providing an assessment of reliability.
- **d.** Application of CellSexID for annotating cell origin in chimeric mice enables a range of single-cell analyses, supporting studies of cellular dynamics and differences between recipient and donor cells in diverse research contexts.

---

## Key Features

- **Simplified Workflow:** Trains on public datasets to ensure accessibility and reproducibility.
- **Single-Cell Focus:** Tailored for single-cell RNA-seq data input in H5AD format.
- **Accurate Predictions:** Utilizes advanced machine learning models (XGBoost, Logistic Regression, SVM, Random Forest).
- **Visualization:** Provides options to visualize the distribution of predicted sexes.
- **Command-Line Interface:** Easy-to-use CLI for seamless integration into pipelines.
- **Detailed Tutorial:** Our notebook serves as a comprehensive tutorial covering all steps, including model training and prediction.

---

Getting Started
Prerequisites
Before using CellSexID, ensure you have the following installed and set up:

Python Environment
Python 3.8+
Anaconda or Miniconda (recommended for managing environments)
Required Libraries
Ensure the following dependencies are installed:

scanpy
anndata
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
You can install all required libraries using the requirements.txt file.

Installation
Option 1: Install Directly from GitHub
The simplest way to install CellSexID is directly from GitHub using pip:

bash
Copy code
pip install git+https://github.com/mcgilldinglab/CellSexID.git
This will fetch the latest version of CellSexID and make it available system-wide.

Option 2: Clone the Repository
If you wish to modify or explore the code:

Clone the repository:

bash
Copy code
git clone https://github.com/mcgilldinglab/CellSexID.git
cd CellSexID
Set up a virtual environment (recommended):

bash
Copy code
conda create -n cellsexid_env python=3.8
conda activate cellsexid_env
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Install the package locally:

bash
Copy code
pip install .
Usage
CellSexID can be used either through the command-line interface (CLI) or programmatically in Python.

Command-Line Interface (CLI)
Command-Line Arguments
Argument	Description
--train_data	Path to the preprocessed training data in .h5ad format.
--test_data	Path to the test data in .h5ad format.
--model	(Optional) Machine learning model to use (XGB, LR, SVM, RF). Default: XGB.
--output	Path to the output CSV file where predictions will be saved.
--plot	(Optional) Path to save the distribution plot of predicted sexes.
Example Usage
bash
Copy code
cellsexid-run \
    --train_data preprocessed_data.h5ad \
    --test_data sex_chimeric_gender_9_25.h5ad \
    --model XGB \
    --output predictions.csv \
    --plot distribution.pdf
Outputs
Predictions CSV (predictions.csv):

Contains two columns:
cell_id: Identifiers for cells in the test dataset.
predicted_sex: Predicted sex (Male or Female).
Distribution Plot (distribution.pdf):

Bar plot showing the percentage distribution of predicted sexes.
Programmatic Usage
You can use CellSexID as a Python library for greater control and integration into your custom workflows.

python
Copy code
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
Preparing Data
Input Requirements
Format: Both training and test data should be in AnnData's .h5ad format.
Gene Expression Matrix:
Genes as variables (adata.var_names).
Cells as observations (adata.obs_names).
Training Data:
Includes a gender column in adata.obs with binary labels:
0 for Female
1 for Male.
Training Data Provided
We provide a zipped file containing preprocessed training data, preprocessed_data.zip.

Steps to Prepare Training Data:
Download the training data from the repository.
Unzip the file:
bash
Copy code
unzip preprocessed_data.zip
This will extract preprocessed_data.h5ad, which you can use for training.
Test Data Requirements
Gene expression data of the cells you want to predict.
Should be in .h5ad format.
Must include the following genes in adata.var_names:
'Rpl35', 'Rps27rt', 'Rpl9-ps6', 'Rps27', 'Uba52', 'Lars2', 'Gm42418', 'Uty', 'Kdm5d', 'Eif2s3y', 'Ddx3y', 'Xist'.
Outputs
Predictions CSV (predictions.csv)
Column	Description
cell_id	Cell identifiers from test data
predicted_sex	Predicted sex (Male or Female)
Distribution Plot (distribution.pdf)
A bar plot visualizing the percentage distribution of predicted sexes across the test dataset.

Example Workflow
Prepare Training Data (preprocessed_data.h5ad):

Ensure the training data includes a gender column (0 for Female, 1 for Male).
Prepare Test Data (sex_chimeric_gender_9_25.h5ad):

Ensure the test data includes the selected genes.
Run CLI Prediction:

bash
Copy code
cellsexid-run \
    --train_data preprocessed_data.h5ad \
    --test_data sex_chimeric_gender_9_25.h5ad \
    --model XGB \
    --output predictions.csv \
    --plot distribution.pdf
Review Outputs:

Open predictions.csv for detailed results.
View the distribution plot in distribution.pdf.

---

## Publication

Our research article detailing CellSexID is now available on bioRxiv:

[CellSexID: A Tool for Predicting Biological Sex from Single-Cell RNA-Seq Data]((https://www.biorxiv.org/content/10.1101/2024.12.02.626449v1))
 

---

## Contributing

Contributions are welcome! If you have suggestions or would like to report issues, please submit them through the [Issues](https://github.com/mcgilldinglab/CellSexID/issues) section of the repository.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Inspired by robust methodologies developed for biological sex prediction.
- Thanks to the contributors and public datasets that make this tool possible.

---

## Contact

For questions or support, please contact **Huilin Tai** at [2378174791@qq.com](mailto:2378174791@qq.com).

---

## References

- [Scanpy Documentation](https://scanpy.readthedocs.io/en/stable/)
- [Anndata Documentation](https://anndata.readthedocs.io/en/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

---

**Note:** Ensure that you have the necessary permissions and licenses for any datasets used during training and prediction.

---

# Quick Links

- [API Documentation](#api-documentation)
- [Installation Guide](#getting-started)
- [Usage Instructions](#usage)
- [Example Workflow](#example-workflow)
- [Publication on bioRxiv](#publication)

---

By following this guide, you can set up and use CellSexID for your single-cell RNA-seq data analysis, leveraging its powerful prediction capabilities and integrating it into your research pipeline.

---

**Disclaimer:** While the installation and basic functionality should work, the package may contain bugs. Please report any issues on the GitHub repository so we can address them promptly.

---

**Happy Analyzing!**

---
