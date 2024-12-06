import numpy as np
import pandas as pd
import scanpy as sc
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Set matplotlib to a non-interactive backend
import matplotlib
matplotlib.use("Agg")  # Ensure plots are saved without displaying

class SexPredictionTool:
    def __init__(self):
        self.models = {
            'LR': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=551))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel='linear', probability=True, random_state=551))
            ]),
            'XGB': Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=10,
                    eval_metric='logloss',
                    random_state=551
                ))
            ]),
            'RF': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(max_depth=100, random_state=41))
            ])
        }
        self.selected_genes = [
            'Rpl35', 'Rps27rt', 'Rpl9-ps6', 'Rps27', 'Uba52', 'Lars2',
            'Gm42418', 'Uty', 'Kdm5d', 'Eif2s3y', 'Ddx3y', 'Xist'
        ]

    def train(self, X_train, y_train, model_name='XGB'):
        """Train the selected model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        self.models[model_name].fit(X_train, y_train)

    def process_training_data(self, h5ad_path):
        """Read and preprocess training data."""
        adata = sc.read(h5ad_path)
        print(f"Training data shape (adata.X): {adata.X.shape}")
        available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
        if not available_genes:
            raise ValueError("None of the selected genes found in the training dataset.")
        X = adata[:, available_genes].X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        le = LabelEncoder()
        y = le.fit_transform(adata.obs["gender"])
        return X, y

    def process_test_data(self, h5ad_path):
        """Read and preprocess test data."""
        adata = sc.read(h5ad_path)
        print(f"Test data shape (adata.X): {adata.X.shape}")
        available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
        if not available_genes:
            raise ValueError("None of the selected genes found in the test dataset.")
        X_test = adata[:, available_genes].X
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.toarray()
        return X_test, adata.obs_names

    def predict(self, X_test, model_name='XGB'):
        """Make predictions using the trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name].predict(X_test)

    def save_predictions(self, y_pred, cell_names, output_file):
        """Save predictions to a CSV file."""
        pred_df = pd.DataFrame({
            'cell_id': cell_names,
            'predicted_sex': ['Male' if p == 1 else 'Female' for p in y_pred]
        })
        pred_df.to_csv(output_file, index=False)

    def plot_prediction_distribution(self, y_pred, save_path):
        """Save distribution plot of predictions."""
        total = len(y_pred)
        male_count = np.sum(y_pred == 1)
        female_count = np.sum(y_pred == 0)
        percentages = [female_count / total * 100, male_count / total * 100]
        plt.figure(figsize=(6, 4))
        plt.bar(['Female', 'Male'], percentages, color=['lightpink', 'lightblue'])
        plt.title('Predicted Sex Distribution')
        plt.xlabel('Sex')
        plt.ylabel('Percentage')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free resources
        print(f"Plot saved to {save_path}")
