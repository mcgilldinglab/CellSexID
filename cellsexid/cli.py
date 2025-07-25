#!/usr/bin/env python3
"""
Simple run example - direct replacement for the original script
Demonstrates the enhanced ImprovedSexPredictionTool with all use cases
"""

# Suppress all warnings at the very beginning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import argparse
import sys
import os

# Try different import methods to handle various project structures
try:
    from sex_prediction_tool import ImprovedSexPredictionTool
except ImportError:
    try:
        from .sex_prediction_tool import ImprovedSexPredictionTool
    except ImportError:
        # If running from cellsexid directory, try parent directory
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from sex_prediction_tool import ImprovedSexPredictionTool
        except ImportError:
            print("❌ Error: Cannot find 'sex_prediction_tool.py'")
            print("Please ensure 'sex_prediction_tool.py' is in the same directory as this script")
            print("Available files in current directory:")
            for f in os.listdir('.'):
                if f.endswith('.py'):
                    print(f"  - {f}")
            sys.exit(1)


def main():
    # Parse command-line arguments with all new options
    parser = argparse.ArgumentParser(
        description='Run sex prediction with enhanced features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (predefined genes, RF model - now default)
  python run_example.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png

  # Use custom genes
  python run_example.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png \\
    --custom_genes "Xist,Ddx3y,Kdm5d,Eif2s3y"

  # Automatic feature selection
  python run_example.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png \\
    --feature_selection --top_k 20 --min_models 3

  # Different model
  python run_example.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png \\
    --model XGB
        """
    )
    
    # Required arguments
    parser.add_argument('--train_data', required=True, 
                       help='Path to preprocessed training h5ad file')
    parser.add_argument('--test_data', required=True, 
                       help='Path to test h5ad file')
    parser.add_argument('--output', required=True, 
                       help='Output file for predictions')
    parser.add_argument('--plot', help='Output file for distribution plot')
    
    # Model selection (RF is now default)
    parser.add_argument('--model', default='RF', 
                       choices=['LR', 'SVM', 'XGB', 'RF'],
                       help='Model to use (default: RF)')
    
    # Gene selection options
    gene_group = parser.add_mutually_exclusive_group()
    gene_group.add_argument('--feature_selection', action='store_true',
                           help='Use automatic feature selection')
    gene_group.add_argument('--custom_genes', type=str,
                           help='Comma-separated custom genes (e.g., "Xist,Ddx3y,Kdm5d")')
    
    # Feature selection parameters
    parser.add_argument('--top_k', type=int, default=20,
                       help='Top K features per model for feature selection (default: 20)')
    parser.add_argument('--min_models', type=int, default=3,
                       help='Minimum models threshold for feature selection (default: 3)')
    parser.add_argument('--fs_output', type=str, default='feature_selection_results',
                       help='Feature selection output directory')
    
    # Additional options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path, name in [(args.train_data, 'training'), (args.test_data, 'test')]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {name} data file not found: {file_path}")
            sys.exit(1)
    
    try:
        print("🧬 ENHANCED SEX PREDICTION TOOL")
        print("=" * 50)
        print(f"Training: {args.train_data}")
        print(f"Test: {args.test_data}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output}")
        if args.plot:
            print(f"Plot: {args.plot}")
        
        # Initialize the tool based on mode
        if args.feature_selection:
            print("Mode: Automatic feature selection")
            print(f"  - Top K: {args.top_k}")
            print(f"  - Min models: {args.min_models}")
            print(f"  - Results dir: {args.fs_output}")
            
            # Feature selection workflow
            sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=False)
            
            # Load data for feature selection
            print("\n📚 Loading training data for feature selection...")
            X, y = sex_predictor.load_training_data(args.train_data)
            
            # Perform feature selection
            print("\n🧠 Discovering optimal genes...")
            selected_genes = sex_predictor.find_optimal_genes(
                X, y, 
                top_k=args.top_k, 
                min_models=args.min_models,
                save_results=True,
                output_dir=args.fs_output
            )
            
        elif args.custom_genes:
            custom_genes = [gene.strip() for gene in args.custom_genes.split(',') if gene.strip()]
            print(f"Mode: Custom genes ({len(custom_genes)} genes)")
            print(f"  - Genes: {', '.join(custom_genes)}")
            
            # Custom genes workflow
            sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=True, custom_genes=custom_genes)
            
        else:
            print("Mode: Predefined gene markers")
            
            # Default predefined genes workflow
            sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=True)
        
        print("=" * 50)
        
        if args.verbose:
            sex_predictor.print_summary()
        
        # Process training data
        print("\n📚 Processing training data...")
        X_train, y_train = sex_predictor.process_training_data(args.train_data)
        
        # Train the model
        print(f"\n🤖 Training {args.model} model...")
        sex_predictor.train(X_train, y_train, model_name=args.model)
        
        # Process test data
        print("\n🔮 Processing test data...")
        X_test, cell_names = sex_predictor.process_test_data(args.test_data)
        
        # Make predictions
        print("\n🎯 Making predictions...")
        y_pred = sex_predictor.predict(X_test, model_name=args.model)
        
        # Save predictions
        print("\n💾 Saving results...")
        sex_predictor.save_predictions(y_pred, cell_names, args.output)
        
        # Plot distribution if requested
        if args.plot:
            sex_predictor.plot_prediction_distribution(y_pred, args.plot)
        
        # Print results summary
        male_count = sum(y_pred == 1)
        female_count = sum(y_pred == 0)
        total = len(y_pred)
        
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"   Total cells: {total}")
        print(f"   Female: {female_count} ({female_count/total*100:.1f}%)")
        print(f"   Male: {male_count} ({male_count/total*100:.1f}%)")
        
        # Show selected genes
        selected_genes = sex_predictor.get_selected_genes()
        print(f"\n🧬 Used {len(selected_genes)} genes:")
        print(f"   {', '.join(selected_genes[:8])}{'...' if len(selected_genes) > 8 else ''}")
        
        if args.feature_selection:
            print(f"\n📁 Feature selection results saved in: {args.fs_output}")
        
        print("\n✅ SUCCESS!")
        print(f"📄 Predictions: {args.output}")
        if args.plot:
            print(f"📊 Plot: {args.plot}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()