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

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use("Agg")

class SexPredictionTool:
    def __init__(self, species='mouse', use_predefined_genes=True, custom_genes=None, sex_column='sex'):
        """
        Initialize the sex prediction tool.
        
        Parameters:
        -----------
        species : str
            'mouse' or 'human' for appropriate gene markers
        use_predefined_genes : bool
            True: Use predefined markers (2-dataset workflow)
            False: Discover markers via feature selection (3-dataset workflow)
        custom_genes : list or None
            Custom list of genes to use. Overrides predefined genes.
        sex_column : str
            Column name containing sex labels
        """
        # Species-specific gene markers
        self.mouse_markers = [
            "Xist", "Ddx3y", "Gm42418", "Eif2s3y", "Rps27rt", "Rpl9-ps6",
            "Kdm5d", "Uba52", "Rpl35", "Rpl36a-ps1", "Uty", "Wdr89",
            "Lars2", "Rps27"
        ]
        
        self.human_markers = [
            "XIST", "DDX3Y", "KDM5D", "RPS4Y1", "EIF1AY", "USP9Y",
            "UTY", "ZFY", "SRY", "TMSB4Y", "NLGN4Y"
        ]
        
        if species not in ['mouse', 'human']:
            raise ValueError("Species must be 'mouse' or 'human'")
        
        self.species = species
        self.sex_column = sex_column
        self.use_predefined_genes = use_predefined_genes
        
        # Set initial markers
        if custom_genes is not None:
            self.selected_markers = custom_genes
        elif use_predefined_genes:
            self.selected_markers = self.human_markers if species == 'human' else self.mouse_markers
        else:
            self.selected_markers = None  # Will be discovered
        
        # Model storage
        self.models = self._initialize_models()
        self.trained_model = None
        self.is_fitted = False
        self.feature_selection_results = None

    def _initialize_models(self):
        """Initialize ML models."""
        return {
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
            'RF': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(max_depth=10, random_state=41))
            ])
        }

    def _load_h5ad(self, filepath):
        """Load h5ad file and return AnnData object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return sc.read(filepath)

    def _extract_features_and_labels(self, adata, include_labels=True, use_all_genes=False):
        """Extract features and optionally labels from AnnData object."""
        
        if use_all_genes:
            # For feature selection, use ALL genes
            X = adata.X
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            X = pd.DataFrame(X, columns=adata.var_names)
        else:
            # Use selected markers
            if self.selected_markers is None:
                raise ValueError("No markers selected. Run discover_markers() first or use predefined markers.")
            
            # Check available markers
            available_markers = [gene for gene in self.selected_markers if gene in adata.var_names]
            if not available_markers:
                missing = set(self.selected_markers) - set(adata.var_names)
                raise ValueError(f"No selected markers found in dataset. Missing: {list(missing)[:5]}")
            
            print(f"Using {len(available_markers)}/{len(self.selected_markers)} markers")
            
            # Extract expression data
            X = adata[:, available_markers].X
            if not isinstance(X, np.ndarray):
                X = X.toarray()
        
        if include_labels:
            if self.sex_column not in adata.obs.columns:
                available_cols = list(adata.obs.columns)
                raise ValueError(f"Sex column '{self.sex_column}' not found. Available: {available_cols}")
            
            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(adata.obs[self.sex_column])
            return X, y, adata.obs_names
        else:
            return X, adata.obs_names

    # =============================================================================
    # WORKFLOW 1: SIMPLE 2-DATASET (PREDEFINED MARKERS)
    # =============================================================================
    
    def fit(self, train_data, model_name='RF'):
        """
        Train model using predefined markers (2-dataset workflow).
        
        Parameters:
        -----------
        train_data : str
            Path to training .h5ad file
        model_name : str
            Model to use ('LR', 'SVM', 'XGB', 'RF')
        """
        if not self.use_predefined_genes:
            raise ValueError("fit() is for predefined markers. Use discover_markers() + fit_with_discovered_markers() instead.")
        
        print(f"🧬 TRAINING WITH PREDEFINED {self.species.upper()} MARKERS")
        print(f"Markers: {', '.join(self.selected_markers[:5])}{'...' if len(self.selected_markers) > 5 else ''}")
        
        # Load training data
        adata = self._load_h5ad(train_data)
        X_train, y_train, _ = self._extract_features_and_labels(adata, include_labels=True)
        
        # Train model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        print(f"Training {model_name} model with {X_train.shape[0]} samples...")
        self.models[model_name].fit(X_train, y_train)
        self.trained_model = model_name
        self.is_fitted = True
        
        print(f"✅ {model_name} model trained successfully!")

    # =============================================================================
    # WORKFLOW 2: ADVANCED 3-DATASET (CUSTOM MARKER DISCOVERY)  
    # =============================================================================
    
    def discover_markers(self, marker_data, top_k=20, min_models=3, save_results=True, 
                        output_dir="feature_selection_results"):
        """
        Discover optimal markers using feature selection (3-dataset workflow step 1).
        
        Parameters:
        -----------
        marker_data : str
            Path to .h5ad file for marker discovery (can be same as train_data)
        top_k : int
            Top K features per model
        min_models : int  
            Minimum models consensus
        save_results : bool
            Save intermediate results
        output_dir : str
            Output directory for results
        """
        if self.use_predefined_genes:
            raise ValueError("discover_markers() is for custom marker discovery. Set use_predefined_genes=False.")
        
        print(f"🔍 DISCOVERING OPTIMAL MARKERS FROM {marker_data}")
        
        # Load data with ALL genes for feature selection
        adata = self._load_h5ad(marker_data)
        X, y, _ = self._extract_features_and_labels(adata, include_labels=True, use_all_genes=True)
        
        print(f"Dataset shape: {X.shape}")
        y_series = pd.Series(y)
        print(f"Target distribution: {y_series.value_counts().to_dict()}")
        
        # Perform feature selection using your existing logic
        selected_markers = self.find_optimal_genes(X, y_series, top_k, min_models, save_results, output_dir)
        
        # Update selected markers
        self.selected_markers = selected_markers
        
        print(f"🎯 DISCOVERED {len(selected_markers)} OPTIMAL MARKERS:")
        for i, marker in enumerate(selected_markers[:10], 1):
            print(f"  {i:2d}. {marker}")
        if len(selected_markers) > 10:
            print(f"  ... and {len(selected_markers)-10} more")
        
        return selected_markers

    def fit_with_discovered_markers(self, train_data, model_name='RF'):
        """
        Train model using discovered markers (3-dataset workflow step 2).
        
        Parameters:
        -----------
        train_data : str
            Path to training .h5ad file (can be same as marker_data)
        model_name : str
            Model to use
        """
        if self.use_predefined_genes:
            raise ValueError("fit_with_discovered_markers() is for discovered markers. Use discover_markers() first.")
        
        if self.selected_markers is None:
            raise ValueError("No markers discovered yet. Run discover_markers() first.")
        
        print(f"🧬 TRAINING WITH {len(self.selected_markers)} DISCOVERED MARKERS")
        
        # Load training data  
        adata = self._load_h5ad(train_data)
        X_train, y_train, _ = self._extract_features_and_labels(adata, include_labels=True)
        
        # Train model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        print(f"Training {model_name} model with {X_train.shape[0]} samples...")
        self.models[model_name].fit(X_train, y_train)
        self.trained_model = model_name
        self.is_fitted = True
        
        print(f"✅ {model_name} model trained successfully!")

    # =============================================================================
    # COMMON PREDICTION METHOD
    # =============================================================================
    
    def predict(self, test_data):
        """
        Make predictions on test data.
        
        Parameters:
        -----------
        test_data : str
            Path to test .h5ad file
            
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0=Female, 1=Male)
        cell_names : list
            Cell identifiers
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Run fit() or fit_with_discovered_markers() first.")
        
        print(f"🔮 MAKING PREDICTIONS WITH {self.trained_model} MODEL")
        
        # Load test data
        adata = self._load_h5ad(test_data)
        X_test, cell_names = self._extract_features_and_labels(adata, include_labels=False)
        
        # Make predictions
        predictions = self.models[self.trained_model].predict(X_test)
        
        print(f"✅ Predictions completed for {len(predictions)} cells!")
        
        return predictions, cell_names

    # =============================================================================
    # FEATURE SELECTION METHODS (YOUR EXISTING LOGIC)
    # =============================================================================
    
    def find_optimal_genes(self, X, y, top_k=20, min_models=3, save_results=True, output_dir="feature_selection_results"):
        """
        Perform feature selection using cross-validation across multiple models.
        (Your existing logic integrated)
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
        
        # Update results and save
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

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def save_predictions(self, predictions, cell_names, output_file):
        """Save predictions to CSV file."""
        pred_df = pd.DataFrame({
            'cell_id': cell_names,
            'predicted_sex': ['Male' if p == 1 else 'Female' for p in predictions]
        })
        pred_df.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")

    def plot_prediction_distribution(self, predictions, save_path):
        """Save distribution plot of predictions."""
        total = len(predictions)
        male_count = np.sum(predictions == 1)
        female_count = np.sum(predictions == 0)
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

    def get_available_genes(self, h5ad_path):
        """Check which predefined genes are available in dataset."""
        adata = self._load_h5ad(h5ad_path)
        if self.selected_markers is None:
            print("No markers selected yet.")
            return {'available': [], 'missing': []}
        
        available = [gene for gene in self.selected_markers if gene in adata.var_names]
        missing = [gene for gene in self.selected_markers if gene not in adata.var_names]
        
        print(f"Species: {self.species}")
        print(f"Available genes ({len(available)}): {available}")
        print(f"Missing genes ({len(missing)}): {missing}")
        
        return {'available': available, 'missing': missing}

    def get_summary(self):
        """Print current configuration summary."""
        print("="*60)
        print("SEX PREDICTION TOOL SUMMARY")
        print("="*60)
        print(f"Species: {self.species}")
        print(f"Mode: {'Predefined markers' if self.use_predefined_genes else 'Custom marker discovery'}")
        print(f"Sex column: {self.sex_column}")
        if self.selected_markers:
            print(f"Selected markers ({len(self.selected_markers)}): {', '.join(self.selected_markers[:5])}{'...' if len(self.selected_markers) > 5 else ''}")
        print(f"Model trained: {self.is_fitted} ({self.trained_model if self.is_fitted else 'None'})")
        print("="*60)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_2_dataset_workflow():
    """Example: Simple 2-dataset workflow with predefined markers."""
    print("📋 EXAMPLE: 2-Dataset Workflow (Predefined Markers)")
    
    # Initialize for predefined markers
    tool = SexPredictionTool(species='mouse', use_predefined_genes=True)
    
    # Train and predict (2 datasets only)
    tool.fit(train_data='train.h5ad', model_name='RF')
    predictions, cell_names = tool.predict(test_data='test.h5ad')
    tool.save_predictions(predictions, cell_names, 'predictions.csv')

def example_3_dataset_workflow():
    """Example: Advanced 3-dataset workflow with marker discovery."""
    print("📋 EXAMPLE: 3-Dataset Workflow (Custom Marker Discovery)")
    
    # Initialize for marker discovery
    tool = SexPredictionTool(species='mouse', use_predefined_genes=False)
    
    # Step 1: Discover markers (can use same data as training)
    tool.discover_markers(marker_data='marker_discovery.h5ad', top_k=20, min_models=3)
    
    # Step 2: Train with discovered markers
    tool.fit_with_discovered_markers(train_data='train.h5ad', model_name='RF')
    
    # Step 3: Predict
    predictions, cell_names = tool.predict(test_data='test.h5ad')
    tool.save_predictions(predictions, cell_names, 'predictions.csv')

def example_human_species():
    """Example: Using human markers."""
    print("📋 EXAMPLE: Human Species Workflow")
    
    tool = SexPredictionTool(species='human', use_predefined_genes=True, sex_column='gender')
    tool.fit(train_data='human_train.h5ad', model_name='RF')
    predictions, cell_names = tool.predict(test_data='human_test.h5ad')
    tool.save_predictions(predictions, cell_names, 'human_predictions.csv')


if __name__ == "__main__":
    print("🚀 Complete Sex Prediction Tool")
    print("Features:")
    print("- Human and mouse species support")
    print("- 2-dataset workflow (predefined markers)")
    print("- 3-dataset workflow (custom marker discovery)")
    print("- Configurable sex column name")
    print()
    print("See example functions for usage patterns.")