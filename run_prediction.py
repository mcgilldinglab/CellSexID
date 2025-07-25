#!/usr/bin/env python3
"""
Comprehensive examples demonstrating all use cases of the ImprovedSexPredictionTool
"""

import argparse
import sys
import os
from pathlib import Path

# Fixed import - use the cellsexid package
try:
    from cellsexid.sex_prediction_tool import ImprovedSexPredictionTool
except ImportError:
    # Fallback for different directory structures
    import sys
    sys.path.append('cellsexid')
    from sex_prediction_tool import ImprovedSexPredictionTool


def example_1_predefined_genes():
    """
    Example 1: Basic usage with predefined gene markers (default mode)
    This is the quickest way to get predictions using established sex-specific genes.
    """
    print("=" * 60)
    print("EXAMPLE 1: PREDEFINED GENE MARKERS (QUICK START)")
    print("=" * 60)
    
    # Command line equivalent:
    # python cli.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png
    
    # Initialize with predefined genes (default behavior)
    sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=True)
    sex_predictor.print_summary()
    
    # Process training data
    print("\n📚 Processing training data...")
    X_train, y_train = sex_predictor.process_training_data("train.h5ad")
    
    # Train the model (RF is now default)
    print("\n🤖 Training Random Forest model...")
    sex_predictor.train(X_train, y_train, model_name='RF')
    
    # Process test data
    print("\n🔮 Processing test data...")
    X_test, cell_names = sex_predictor.process_test_data("test.h5ad")
    
    # Make predictions
    print("\n🎯 Making predictions...")
    y_pred = sex_predictor.predict(X_test, model_name='RF')
    
    # Save results
    print("\n💾 Saving results...")
    sex_predictor.save_predictions(y_pred, cell_names, "example1_predictions.csv")
    sex_predictor.plot_prediction_distribution(y_pred, "example1_distribution.png")
    
    print("\n✅ Example 1 completed!")
    return sex_predictor


def example_2_custom_genes():
    """
    Example 2: Using custom gene markers
    Useful when you have specific genes of interest or domain knowledge.
    """
    print("=" * 60)
    print("EXAMPLE 2: CUSTOM GENE MARKERS")
    print("=" * 60)
    
    # Command line equivalent:
    # python cli.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png \
    #   --custom_genes "Xist,Ddx3y,Kdm5d,Eif2s3y,Uty"
    
    # Define custom genes
    custom_genes = ["Xist", "Ddx3y", "Kdm5d", "Eif2s3y", "Uty"]
    print(f"Using custom genes: {', '.join(custom_genes)}")
    
    # Initialize with custom genes
    sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=True, custom_genes=custom_genes)
    sex_predictor.print_summary()
    
    # Process training data
    print("\n📚 Processing training data...")
    X_train, y_train = sex_predictor.process_training_data("train.h5ad")
    
    # Train with XGBoost this time
    print("\n🤖 Training XGBoost model...")
    sex_predictor.train(X_train, y_train, model_name='XGB')
    
    # Process test data
    print("\n🔮 Processing test data...")
    X_test, cell_names = sex_predictor.process_test_data("test.h5ad")
    
    # Make predictions
    print("\n🎯 Making predictions...")
    y_pred = sex_predictor.predict(X_test, model_name='XGB')
    
    # Save results
    print("\n💾 Saving results...")
    sex_predictor.save_predictions(y_pred, cell_names, "example2_predictions.csv")
    sex_predictor.plot_prediction_distribution(y_pred, "example2_distribution.png")
    
    print("\n✅ Example 2 completed!")
    return sex_predictor


def example_3_feature_selection():
    """
    Example 3: Automatic feature selection (most comprehensive)
    Discovers optimal gene markers automatically from your data.
    """
    print("=" * 60)
    print("EXAMPLE 3: AUTOMATIC FEATURE SELECTION")
    print("=" * 60)
    
    # Command line equivalent:
    # python cli.py --train_data train.h5ad --test_data test.h5ad --output results.csv --plot dist.png \
    #   --feature_selection --top_k 20 --min_models 3
    
    # Initialize for feature selection
    sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=False)
    
    # Load training data for feature selection
    print("\n📚 Loading training data for feature selection...")
    X, y = sex_predictor.load_training_data("train.h5ad")
    
    # Perform feature selection
    print("\n🧠 Discovering optimal gene markers...")
    selected_genes = sex_predictor.find_optimal_genes(
        X, y, 
        top_k=20,           # Consider top 20 features from each model
        min_models=3,       # Gene must appear in at least 3/4 models
        save_results=True,  # Save intermediate results
        output_dir="feature_selection_results"
    )
    
    # Now process training data with selected genes
    print("\n📚 Processing training data with selected genes...")
    X_train, y_train = sex_predictor.process_training_data("train.h5ad")
    
    # Train with SVM
    print("\n🤖 Training SVM model...")
    sex_predictor.train(X_train, y_train, model_name='SVM')
    
    # Process test data
    print("\n🔮 Processing test data...")
    X_test, cell_names = sex_predictor.process_test_data("test.h5ad")
    
    # Make predictions
    print("\n🎯 Making predictions...")
    y_pred = sex_predictor.predict(X_test, model_name='SVM')
    
    # Save results
    print("\n💾 Saving results...")
    sex_predictor.save_predictions(y_pred, cell_names, "example3_predictions.csv")
    sex_predictor.plot_prediction_distribution(y_pred, "example3_distribution.png")
    
    print(f"\n📊 Feature selection results saved in: feature_selection_results")
    print("\n✅ Example 3 completed!")
    return sex_predictor


def example_4_model_comparison():
    """
    Example 4: Compare all available models
    Test different algorithms to find the best performer for your data.
    """
    print("=" * 60)
    print("EXAMPLE 4: MODEL COMPARISON")
    print("=" * 60)
    
    models_to_test = ['LR', 'SVM', 'XGB', 'RF']
    results = {}
    
    # Initialize with predefined genes
    sex_predictor = ImprovedSexPredictionTool(use_predefined_genes=True)
    
    # Process training data once
    print("\n📚 Processing training data...")
    X_train, y_train = sex_predictor.process_training_data("train.h5ad")
    
    # Process test data once
    print("\n🔮 Processing test data...")
    X_test, cell_names = sex_predictor.process_test_data("test.h5ad")
    
    # Test each model
    for model_name in models_to_test:
        print(f"\n🤖 Testing {model_name} model...")
        
        # Train model
        sex_predictor.train(X_train, y_train, model_name=model_name)
        
        # Make predictions
        y_pred = sex_predictor.predict(X_test, model_name=model_name)
        
        # Save results for this model
        output_file = f"example4_{model_name}_predictions.csv"
        plot_file = f"example4_{model_name}_distribution.png"
        
        sex_predictor.save_predictions(y_pred, cell_names, output_file)
        sex_predictor.plot_prediction_distribution(y_pred, plot_file)
        
        # Store results for comparison
        male_count = sum(y_pred == 1)
        female_count = sum(y_pred == 0)
        results[model_name] = {
            'total': len(y_pred),
            'male': male_count,
            'female': female_count,
            'male_pct': (male_count / len(y_pred)) * 100
        }
        
        print(f"   Results: {female_count} Female, {male_count} Male")
    
    # Print comparison summary
    print("\n📊 MODEL COMPARISON SUMMARY:")
    print("-" * 50)
    for model, stats in results.items():
        print(f"{model:>3}: {stats['female']:>4} Female ({100-stats['male_pct']:>5.1f}%) | "
              f"{stats['male']:>4} Male ({stats['male_pct']:>5.1f}%)")
    
    print("\n✅ Example 4 completed!")
    return results


def example_5_cli_simulation():
    """
    Example 5: Simulate CLI usage programmatically
    Shows how to replicate CLI functionality in code.
    """
    print("=" * 60)
    print("EXAMPLE 5: CLI SIMULATION")
    print("=" * 60)
    
    # Simulate different CLI commands
    cli_examples = [
        {
            'name': 'Basic CLI usage',
            'args': {
                'train_data': 'train.h5ad',
                'test_data': 'test.h5ad',
                'output': 'cli_basic_predictions.csv',
                'plot': 'cli_basic_distribution.png',
                'model': 'RF'
            }
        },
        {
            'name': 'Custom genes CLI',
            'args': {
                'train_data': 'train.h5ad',
                'test_data': 'test.h5ad',
                'output': 'cli_custom_predictions.csv',
                'plot': 'cli_custom_distribution.png',
                'model': 'XGB',
                'custom_genes': 'Xist,Ddx3y,Kdm5d'
            }
        },
        {
            'name': 'Feature selection CLI',
            'args': {
                'train_data': 'train.h5ad',
                'test_data': 'test.h5ad',
                'output': 'cli_fs_predictions.csv',
                'plot': 'cli_fs_distribution.png',
                'model': 'SVM',
                'feature_selection': True,
                'top_k': 15,
                'min_models': 2
            }
        }
    ]
    
    for example in cli_examples:
        print(f"\n🔧 Simulating: {example['name']}")
        args = example['args']
        
        # Print equivalent CLI command
        cmd_parts = ['python cli.py']
        for key, value in args.items():
            if key == 'feature_selection' and value:
                cmd_parts.append('--feature_selection')
            elif key == 'custom_genes':
                cmd_parts.append(f'--custom_genes "{value}"')
            elif key != 'feature_selection':
                cmd_parts.append(f'--{key} {value}')
        
        print(f"CLI equivalent: {' '.join(cmd_parts)}")
        
        # Simulate the execution logic
        if args.get('feature_selection'):
            tool = ImprovedSexPredictionTool(use_predefined_genes=False)
            # Would perform feature selection here
            print("   → Would run feature selection workflow")
        elif args.get('custom_genes'):
            custom_genes = [g.strip() for g in args['custom_genes'].split(',')]
            tool = ImprovedSexPredictionTool(use_predefined_genes=True, custom_genes=custom_genes)
            print(f"   → Would use custom genes: {custom_genes}")
        else:
            tool = ImprovedSexPredictionTool(use_predefined_genes=True)
            print("   → Would use predefined genes")
        
        print(f"   → Would train {args['model']} model")
        print(f"   → Would save results to {args['output']}")
        print(f"   → Would save plot to {args['plot']}")
    
    print("\n✅ Example 5 completed!")


def main():
    """
    Main function to run all examples
    """
    print("🧬 CELLSEXID - COMPREHENSIVE USAGE EXAMPLES")
    print("🚀 Demonstrating all features of the ImprovedSexPredictionTool")
    print()
    
    # Check if data files exist (for demonstration purposes)
    if not os.path.exists("train.h5ad") or not os.path.exists("test.h5ad"):
        print("⚠️  Note: This example assumes train.h5ad and test.h5ad exist")
        print("   Replace with your actual data file paths")
        print()
    
    try:
        # Run all examples
        print("Running all examples...\n")
        
        # Example 1: Basic predefined genes
        example_1_predefined_genes()
        
        # Example 2: Custom genes
        example_2_custom_genes()
        
        # Example 3: Feature selection
        example_3_feature_selection()
        
        # Example 4: Model comparison
        example_4_model_comparison()
        
        # Example 5: CLI simulation
        example_5_cli_simulation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("📄 Predictions: example1_predictions.csv, example2_predictions.csv, etc.")
        print("📊 Plots: example1_distribution.png, example2_distribution.png, etc.")
        print("🔍 Feature selection: feature_selection_results/")
        print("\nYou can now:")
        print("1. Use the CLI directly: python cli.py --help")
        print("2. Import and use ImprovedSexPredictionTool in your scripts")
        print("3. Customize any of these examples for your specific needs")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure your H5AD files exist before running examples")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # You can also parse command line arguments to run specific examples
    parser = argparse.ArgumentParser(description='Run sex prediction examples')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5], 
                       help='Run specific example (1-5)')
    parser.add_argument('--train_data', default='train.h5ad', 
                       help='Path to training data')
    parser.add_argument('--test_data', default='test.h5ad', 
                       help='Path to test data')
    
    args = parser.parse_args()
    
    if args.example:
        examples = {
            1: example_1_predefined_genes,
            2: example_2_custom_genes,
            3: example_3_feature_selection,
            4: example_4_model_comparison,
            5: example_5_cli_simulation
        }
        examples[args.example]()
    else:
        main()