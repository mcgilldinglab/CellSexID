import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Set matplotlib to a non-interactive backend
import matplotlib
matplotlib.use("Agg")  # Ensure plots are saved without displaying


class ImprovedSexPredictionTool:
    def __init__(self, use_predefined_genes=True, custom_genes=None):
        """
        Initialize the sex prediction tool.
        
        Parameters:
        -----------
        use_predefined_genes : bool
            If True, use predefined gene markers. If False, will use feature selection.
        custom_genes : list or None
            Custom list of genes to use. If None, uses predefined markers.
        """
        # Updated model parameters as requested
        self.models = {
            'LR': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=551))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel="linear", probability=True, random_state=551))
            ]),
            'XGB': Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(
                    random_state=551,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=10
                ))
            ]),
            'RF': Pipeline([  # Random Forest as default
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(max_depth=10, random_state=41))
            ])
        }
        
        # Updated predefined gene markers
        self.predefined_genes = [
            "Xist", "Ddx3y", "Gm42418", "Eif2s3y", "Rps27rt", "Rpl9-ps6",
            "Kdm5d", "Uba52", "Rpl35", "Rpl36a-ps1", "Uty", "Wdr89",
            "Lars2", "Rps27"
        ]
        
        # Set genes to use
        if custom_genes is not None:
            self.selected_genes = custom_genes
        elif use_predefined_genes:
            self.selected_genes = self.predefined_genes
        else:
            self.selected_genes = None  # Will be determined by feature selection
            
        self.use_predefined_genes = use_predefined_genes
        self.feature_selection_results = None

    def load_training_data(self, h5ad_path):
        """Load training data and return processed X, y for feature selection or training."""
        adata = sc.read(h5ad_path)
        print(f"Training data shape: {adata.X.shape}")
        
        # If using feature selection, return all genes
        if not self.use_predefined_genes:
            X = adata.X
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            X = pd.DataFrame(X, columns=adata.var_names)
        else:
            # Use selected genes
            available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
            if not available_genes:
                raise ValueError("None of the selected genes found in the training dataset.")
            X = adata[:, available_genes].X
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            X = pd.DataFrame(X, columns=available_genes)
        
        # Encode labels
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(adata.obs["gender"]))
        
        return X, y

    def find_optimal_genes(self, X, y, top_k=20, min_models=3, save_results=True, output_dir="feature_importance_results"):
        """
        Perform feature selection using cross-validation across multiple models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (genes)
        y : pd.Series
            Target labels
        top_k : int
            Number of top features to consider from each model
        min_models : int
            Minimum number of models a feature must appear in to be selected
        save_results : bool
            Whether to save intermediate results
        output_dir : str
            Directory to save results
        
        Returns:
        --------
        selected_genes : list
            List of genes selected by majority vote
        """
        print("="*60)
        print("AUTOMATIC GENE MARKER DISCOVERY")
        print("="*60)
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Starting evaluation at: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Perform cross-validation feature importance evaluation
        aggregated_importances = self._evaluate_models_cv(X, y)
        
        # Save intermediate results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n💾 SAVING INTERMEDIATE RESULTS to '{output_dir}'...")
            for model_name, data in aggregated_importances.items():
                file_path = os.path.join(output_dir, f"{model_name}_feature_importances.csv")
                data.to_csv(file_path, index=False)
                print(f"✓ Saved: {file_path}")
        
        # Perform majority voting
        majority_df = self._majority_vote_features(aggregated_importances, top_k, min_models)
        
        # Extract selected genes
        selected_genes = majority_df['Feature'].tolist()
        
        print(f"\n🎯 SELECTED {len(selected_genes)} GENES by majority vote:")
        for i, gene in enumerate(selected_genes, 1):
            count = majority_df[majority_df['Feature'] == gene]['AppearCount'].iloc[0]
            score = majority_df[majority_df['Feature'] == gene]['CombinedNorm'].iloc[0]
            print(f"  {i:2d}. {gene:<15} (appeared in {count}/4 models, score: {score:.4f})")
        
        # Update selected genes and save results
        self.selected_genes = selected_genes
        self.feature_selection_results = majority_df
        
        if save_results:
            majority_df.to_csv(os.path.join(output_dir, "selected_genes_majority_vote.csv"), index=False)
            print(f"\n✅ Final results saved to '{output_dir}/selected_genes_majority_vote.csv'")
        
        return selected_genes

    def _evaluate_models_cv(self, X, y):
        """Internal method for cross-validation feature importance evaluation."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
        
        # Define models without pipelines for feature importance extraction
        base_models = [
            LogisticRegression(max_iter=1000, random_state=551),
            SVC(kernel="linear", probability=True, random_state=551),
            XGBClassifier(
                random_state=551,
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=100,
                learning_rate=0.05,
                max_depth=10
            ),
            RandomForestClassifier(max_depth=10, random_state=41)
        ]

        model_names = [type(model).__name__ for model in base_models]
        print(f"Models for feature selection: {', '.join(model_names)}")
        print(f"Total training iterations: {len(base_models)} models × 5 folds = {len(base_models) * 5}")
        print()

        importance_dfs = {type(model).__name__: [] for model in base_models}
        total_iterations = 0
        max_iterations = len(base_models) * 5

        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            print(f"📁 FOLD {fold_idx}/5")
            print(f"   Train samples: {len(train_index)}, Test samples: {len(test_index)}")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Scale data
            print("   📊 Scaling features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train each model and extract feature importances
            for model_idx, model in enumerate(base_models, 1):
                total_iterations += 1
                model_name = type(model).__name__
                
                print(f"   🤖 Training {model_name} ({model_idx}/{len(base_models)}) "
                      f"[Overall: {total_iterations}/{max_iterations}]")
                
                model.fit(X_train_scaled, y_train)

                # Extract feature importances
                if hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                    importance_type = "coefficients"
                elif hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_type = "feature_importances"
                else:
                    importances = np.zeros(X_train.shape[1])
                    importance_type = "zeros (fallback)"
                
                print(f"      ✓ Extracted {importance_type}, max importance: {importances.max():.4f}")
                
                feature_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                })
                importance_dfs[model_name].append(feature_df)
            
            print(f"   ✅ Fold {fold_idx} completed\n")

        print("🔄 AGGREGATING RESULTS ACROSS FOLDS...")
        
        # Aggregate results
        aggregated_importances = {}
        for model_name, dfs in importance_dfs.items():
            all_importances = pd.concat(dfs)
            feature_importance_mean = (all_importances
                                     .groupby('Feature')['Importance']
                                     .mean()
                                     .reset_index())
            feature_importance_mean = feature_importance_mean.sort_values(
                by='Importance', ascending=False
            ).reset_index(drop=True)
            feature_importance_mean['Rank'] = feature_importance_mean['Importance'].rank(
                method='dense', ascending=False
            ).astype(int)
            aggregated_importances[model_name] = feature_importance_mean

        return aggregated_importances

    def _majority_vote_features(self, aggregated_importances, top_k=20, min_models=3):
        """Internal method for majority voting on features."""
        top_features_per_model = {}
        for model_name, df in aggregated_importances.items():
            df_sorted = df.sort_values("Importance", ascending=False, ignore_index=True)
            top_k_df = df_sorted.head(top_k).copy()
            top_k_df["Importance_Norm"] = top_k_df["Importance"] / top_k_df["Importance"].sum()
            top_features_per_model[model_name] = top_k_df

        # Merge results
        merged = None
        for model_name, df_top in top_features_per_model.items():
            df_renamed = df_top[['Feature', 'Importance', 'Importance_Norm']].rename(
                columns={
                    'Importance': f"{model_name}_Importance",
                    'Importance_Norm': f"{model_name}_Norm"
                }
            )
            merged = df_renamed if merged is None else merged.merge(df_renamed, on="Feature", how="outer")

        model_names = list(top_features_per_model.keys())
        merged["AppearCount"] = sum(merged[f"{m}_Importance"].notna() for m in model_names)
        
        # Filter by minimum models requirement
        merged = merged[merged["AppearCount"] >= min_models].copy()
        
        # Calculate combined score
        for m in model_names:
            merged[f"{m}_Norm"] = merged[f"{m}_Norm"].fillna(0.0)
        merged["CombinedNorm"] = merged[[f"{m}_Norm" for m in model_names]].sum(axis=1)
        
        merged.sort_values("CombinedNorm", ascending=False, inplace=True)
        return merged.reset_index(drop=True)

    def train(self, X_train, y_train, model_name='RF'):  # RF as default
        """Train the selected model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        print(f"Training {model_name} model...")
        self.models[model_name].fit(X_train, y_train)
        print(f"✅ {model_name} model trained successfully!")

    def process_training_data(self, h5ad_path):
        """Read and preprocess training data using selected genes."""
        if self.selected_genes is None:
            raise ValueError("No genes selected. Run find_optimal_genes() first or set use_predefined_genes=True")
        
        adata = sc.read(h5ad_path)
        print(f"Training data shape: {adata.X.shape}")
        available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
        
        if not available_genes:
            raise ValueError("None of the selected genes found in the training dataset.")
        
        print(f"Using {len(available_genes)}/{len(self.selected_genes)} selected genes")
        
        X = adata[:, available_genes].X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        le = LabelEncoder()
        y = le.fit_transform(adata.obs["gender"])
        
        return X, y

    def process_test_data(self, h5ad_path):
        """Read and preprocess test data using selected genes."""
        if self.selected_genes is None:
            raise ValueError("No genes selected. Run find_optimal_genes() first or set use_predefined_genes=True")
        
        adata = sc.read(h5ad_path)
        print(f"Test data shape: {adata.X.shape}")
        available_genes = [gene for gene in self.selected_genes if gene in adata.var_names]
        
        if not available_genes:
            raise ValueError("None of the selected genes found in the test dataset.")
        
        print(f"Using {len(available_genes)}/{len(self.selected_genes)} selected genes for prediction")
        
        X_test = adata[:, available_genes].X
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.toarray()
        
        return X_test, adata.obs_names

    def predict(self, X_test, model_name='RF'):  # RF as default
        """Make predictions using the trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        print(f"Making predictions with {model_name} model...")
        predictions = self.models[model_name].predict(X_test)
        print(f"✅ Predictions completed!")
        
        return predictions

    def save_predictions(self, y_pred, cell_names, output_file):
        """Save predictions to a CSV file."""
        pred_df = pd.DataFrame({
            'cell_id': cell_names,
            'predicted_sex': ['Male' if p == 1 else 'Female' for p in y_pred]
        })
        pred_df.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")

    def plot_prediction_distribution(self, y_pred, save_path):
        """Save distribution plot of predictions."""
        total = len(y_pred)
        male_count = np.sum(y_pred == 1)
        female_count = np.sum(y_pred == 0)
        percentages = [female_count / total * 100, male_count / total * 100]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Female', 'Male'], percentages, color=['lightpink', 'lightblue'])
        plt.title('Predicted Sex Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Sex', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Distribution plot saved to {save_path}")

    def get_selected_genes(self):
        """Return the currently selected genes."""
        return self.selected_genes

    def print_summary(self):
        """Print a summary of the current configuration."""
        print("="*50)
        print("SEX PREDICTION TOOL SUMMARY")
        print("="*50)
        print(f"Available models: {list(self.models.keys())}")
        print(f"Default model: RF (Random Forest)")
        if self.selected_genes:
            print(f"Selected genes ({len(self.selected_genes)}): {', '.join(self.selected_genes[:5])}{'...' if len(self.selected_genes) > 5 else ''}")
        else:
            print("No genes selected yet - run find_optimal_genes() or use predefined genes")
        print("="*50)


# Example usage functions
def example_with_predefined_genes():
    """Example: Using predefined gene markers"""
    print("🧬 EXAMPLE: Using predefined gene markers")
    
    # Initialize with predefined genes
    tool = ImprovedSexPredictionTool(use_predefined_genes=True)
    tool.print_summary()
    
    # Load and process training data
    # X_train, y_train = tool.process_training_data("path/to/training.h5ad")
    
    # Train model (RF is default)
    # tool.train(X_train, y_train, model_name='RF')
    
    # Process test data and predict
    # X_test, cell_names = tool.process_test_data("path/to/test.h5ad")
    # predictions = tool.predict(X_test, model_name='RF')
    
    # Save results
    # tool.save_predictions(predictions, cell_names, "predictions.csv")
    # tool.plot_prediction_distribution(predictions, "distribution_plot.png")


def example_with_feature_selection():
    """Example: Using automatic feature selection"""
    print("🔍 EXAMPLE: Using automatic feature selection")
    
    # Initialize for feature selection
    tool = ImprovedSexPredictionTool(use_predefined_genes=False)
    
    # Load training data for feature selection
    # X, y = tool.load_training_data("path/to/training.h5ad")
    
    # Find optimal genes
    # selected_genes = tool.find_optimal_genes(X, y, top_k=20, min_models=3)
    
    # Now use selected genes for training
    # X_train, y_train = tool.process_training_data("path/to/training.h5ad")
    # tool.train(X_train, y_train, model_name='RF')
    
    # Make predictions
    # X_test, cell_names = tool.process_test_data("path/to/test.h5ad")
    # predictions = tool.predict(X_test, model_name='RF')
    
    # Save results
    # tool.save_predictions(predictions, cell_names, "predictions.csv")
    # tool.plot_prediction_distribution(predictions, "distribution_plot.png")


if __name__ == "__main__":
    print("🚀 Improved Sex Prediction Tool")
    print("Choose your workflow:")
    print("1. Use predefined gene markers (quick)")
    print("2. Automatic gene discovery (comprehensive)")
    print()
    print("See example functions for usage patterns.")