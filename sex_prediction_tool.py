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
                    eval_metric='logloss',  # Set eval_metric here
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
        """Read preprocessed training data from H5AD file."""
        # Read preprocessed data
        adata = sc.read(h5ad_path)
        
        # Debugging: Check shapes
        print("Training data shape (adata.X):", adata.X.shape)
        print("Number of genes (adata.var_names):", len(adata.var_names))
        print("Number of cells (adata.obs_names):", len(adata.obs_names))
        
        # Extract selected genes that are present in the dataset
        available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
        if not available_genes:
            raise ValueError("None of the selected genes found in the training dataset.")
        
        # Create training matrices
        X = adata[:, available_genes].X
        y = adata.obs["gender"].astype(int).values  # Assuming "gender" column exists
        
        # Convert X to dense if necessary
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        return X, y
    
    def process_test_data(self, path):
        """Process test data from H5AD file."""
        adata = sc.read(path)
        
        # Debugging: Check shapes
        print("Test data shape (adata.X):", adata.X.shape)
        print("Number of genes (adata.var_names):", len(adata.var_names))
        print("Number of cells (adata.obs_names):", len(adata.obs_names))
        
        # Extract selected genes that are present in the dataset
        available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
        if not available_genes:
            raise ValueError("None of the selected genes found in the test dataset.")
        
        X_test = adata[:, available_genes].X
        
        # Convert X_test to dense if necessary
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.toarray()
        
        return X_test, adata.obs_names
    
 
    
    def predict(self, X_test, model_name='XGB'):
        """Make predictions using the trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X_test)
    
    def predict_proba(self, X_test, model_name='XGB'):
        """Get prediction probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        return self.models[model_name].predict_proba(X_test)
    
    def save_predictions(self, y_pred, cell_names, output_file):
        """Save predictions to CSV file."""
        pred_df = pd.DataFrame({
            'cell_id': cell_names,
            'predicted_sex': ['Male' if p == 1 else 'Female' for p in y_pred]
        })
        pred_df.to_csv(output_file, index=False)
    
    def plot_prediction_distribution(self, y_pred, save_path=None):
        """Plot distribution of predicted sexes."""
        # Calculate percentages
        total = len(y_pred)
        male_count = np.sum(y_pred == 1)
        female_count = np.sum(y_pred == 0)
        male_percent = (male_count / total) * 100
        female_percent = (female_count / total) * 100
        
        # Create bar plot
        plt.figure(figsize=(5.1, 2.545))
        classes = ['Female', 'Male']
        percentages = [female_percent, male_percent]
        
        bars = plt.bar(classes, percentages, color=['lightpink', 'lightblue'])
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f'{height:.1f}%',
                     ha='center', va='bottom')
        
        plt.title('Predicted Sex Distribution', fontsize=12)
        plt.xlabel('Sex', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
