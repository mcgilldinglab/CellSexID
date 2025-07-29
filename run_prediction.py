#!/usr/bin/env python3
"""
Comprehensive examples demonstrating all use cases of the SexPredictionTool
Updated to match current implementation and package structure
"""

import argparse
import sys
import os
from pathlib import Path

# Fixed import - use the cellsexid package
try:
    from cellsexid import SexPredictionTool
except ImportError:
    # Fallback for different directory structures
    try:
        from cellsexid.sex_prediction_tool import SexPredictionTool
    except ImportError:
        print("❌ Error: Cannot find SexPredictionTool")
        print("Please ensure cellsexid package is installed: pip install -e .")
        sys.exit(1)


def example_1_predefined_genes():
    """
    Example 1: Basic usage with predefined gene markers (2-dataset workflow)
    This is the quickest way to get predictions using established sex-specific genes.
    """
    print("=" * 60)
    print("EXAMPLE 1: PREDEFINED GENE MARKERS (2-DATASET WORKFLOW)")
    print("=" * 60)
    
    # Command line equivalent:
    # cellsexid --species mouse --train train.h5ad --test test.h5ad --output results.csv --plot dist.png
    
    # Initialize with predefined genes (default behavior)
    sex_predictor = SexPredictionTool(species='mouse', use_predefined_genes=True)
    sex_predictor.get_summary()
    
    print("\n📚 WORKFLOW: 2-Dataset with predefined markers")
    print("Step 1: Training model with predefined markers...")
    
    # Train the model using predefined markers
    sex_predictor.fit(train_data='train.h5ad', model_name='RF')
    
    print("\nStep 2: Making predictions on test data...")
    
    # Make predictions
    predictions, cell_names = sex_predictor.predict(test_data='test.h5ad')
    
    print("\nStep 3: Saving results...")
    
    # Save results
    sex_predictor.save_predictions(predictions, cell_names, "example1_predictions.csv")
    sex_predictor.plot_prediction_distribution(predictions, "example1_distribution.png")
    
    print("\n✅ Example 1 completed!")
    print(f"📄 Predictions saved: example1_predictions.csv")
    print(f"📊 Plot saved: example1_distribution.png")
    
    return sex_predictor


def example_2_custom_genes():
    """
    Example 2: Using custom gene markers (2-dataset workflow)
    Useful when you have specific genes of interest or domain knowledge.
    """
    print("=" * 60)
    print("EXAMPLE 2: CUSTOM GENE MARKERS (2-DATASET WORKFLOW)")
    print("=" * 60)
    
    # Command line equivalent:
    # cellsexid --species mouse --train train.h5ad --test test.h5ad --output results.csv \
    #   --custom_genes "Xist,Ddx3y,Kdm5d,Eif2s3y,Uty" --model XGB
    
    # Define custom genes
    custom_genes = ["Xist", "Ddx3y", "Kdm5d", "Eif2s3y", "Uty"]
    print(f"Using custom genes: {', '.join(custom_genes)}")
    
    # Initialize with custom genes
    sex_predictor = SexPredictionTool(
        species='mouse', 
        use_predefined_genes=True,  # Still True because we're providing custom genes
        custom_genes=custom_genes
    )
    sex_predictor.get_summary()
    
    print("\n📚 WORKFLOW: 2-Dataset with custom genes")
    print("Step 1: Training model with custom genes...")
    
    # Train with XGBoost this time
    sex_predictor.fit(train_data='train.h5ad', model_name='XGB')
    
    print("\nStep 2: Making predictions...")
    
    # Make predictions
    predictions, cell_names = sex_predictor.predict(test_data='test.h5ad')
    
    print("\nStep 3: Saving results...")
    
    # Save results
    sex_predictor.save_predictions(predictions, cell_names, "example2_predictions.csv")
    sex_predictor.plot_prediction_distribution(predictions, "example2_distribution.png")
    
    print("\n✅ Example 2 completed!")
    print(f"📄 Predictions saved: example2_predictions.csv")
    print(f"📊 Plot saved: example2_distribution.png")
    
    return sex_predictor


def example_3_feature_selection():
    """
    Example 3: Automatic feature selection (3-dataset workflow)
    Discovers optimal gene markers automatically from your data.
    """
    print("=" * 60)
    print("EXAMPLE 3: AUTOMATIC FEATURE SELECTION (3-DATASET WORKFLOW)")
    print("=" * 60)
    
    # Command line equivalent:
    # cellsexid --species mouse --marker_train marker.h5ad train.h5ad --test test.h5ad \
    #   --output results.csv --top_k 20 --min_models 3 --model SVM
    
    # Initialize for feature selection (3-dataset workflow)
    sex_predictor = SexPredictionTool(species='mouse', use_predefined_genes=False)
    sex_predictor.get_summary()
    
    print("\n📚 WORKFLOW: 3-Dataset with marker discovery")
    print("Step 1: Discovering optimal gene markers...")
    
    # Perform feature selection (use training data for marker discovery)
    selected_genes = sex_predictor.discover_markers(
        marker_data='train.h5ad',  # Can use same file as training
        top_k=20,                  # Consider top 20 features from each model
        min_models=3,              # Gene must appear in at least 3/4 models
        save_results=True,         # Save intermediate results
        output_dir="feature_selection_results"
    )
    
    print(f"\nDiscovered {len(selected_genes)} optimal markers")
    
    print("\nStep 2: Training with discovered markers...")
    
    # Train with discovered markers
    sex_predictor.fit_with_discovered_markers(train_data='train.h5ad', model_name='SVM')
    
    print("\nStep 3: Making predictions...")
    
    # Make predictions
    predictions, cell_names = sex_predictor.predict(test_data='test.h5ad')
    
    print("\nStep 4: Saving results...")
    
    # Save results
    sex_predictor.save_predictions(predictions, cell_names, "example3_predictions.csv")
    sex_predictor.plot_prediction_distribution(predictions, "example3_distribution.png")
    
    print(f"\n📊 Feature selection results saved in: feature_selection_results/")
    print("\n✅ Example 3 completed!")
    print(f"📄 Predictions saved: example3_predictions.csv")
    print(f"📊 Plot saved: example3_distribution.png")
    
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
    
    print("Testing all models with predefined markers...")
    
    # Test each model
    for model_name in models_to_test:
        print(f"\n🤖 Testing {model_name} model...")
        
        # Initialize fresh tool for each model
        sex_predictor = SexPredictionTool(species='mouse', use_predefined_genes=True)
        
        # Train model
        sex_predictor.fit(train_data='train.h5ad', model_name=model_name)
        
        # Make predictions
        predictions, cell_names = sex_predictor.predict(test_data='test.h5ad')
        
        # Save results for this model
        output_file = f"example4_{model_name}_predictions.csv"
        plot_file = f"example4_{model_name}_distribution.png"
        
        sex_predictor.save_predictions(predictions, cell_names, output_file)
        sex_predictor.plot_prediction_distribution(predictions, plot_file)
        
        # Store results for comparison
        male_count = sum(predictions == 1)
        female_count = sum(predictions == 0)
        results[model_name] = {
            'total': len(predictions),
            'male': male_count,
            'female': female_count,
            'male_pct': (male_count / len(predictions)) * 100
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


def example_5_human_species():
    """
    Example 5: Using human markers with different sex column
    Demonstrates species switching and custom column names.
    """
    print("=" * 60)
    print("EXAMPLE 5: HUMAN SPECIES WITH CUSTOM SEX COLUMN")
    print("=" * 60)
    
    # Command line equivalent:
    # cellsexid --species human --train human_train.h5ad --test human_test.h5ad \
    #   --output human_results.csv --sex_column gender --model RF
    
    # Initialize for human data with custom sex column
    sex_predictor = SexPredictionTool(
        species='human', 
        use_predefined_genes=True,
        sex_column='gender'  # Different column name
    )
    sex_predictor.get_summary()
    
    print("\n📚 WORKFLOW: Human markers with custom sex column")
    print("Step 1: Training with human markers...")
    
    try:
        # Train model (adjust file names for your data)
        sex_predictor.fit(train_data='human_train.h5ad', model_name='RF')
        
        print("\nStep 2: Making predictions...")
        
        # Make predictions
        predictions, cell_names = sex_predictor.predict(test_data='human_test.h5ad')
        
        print("\nStep 3: Saving results...")
        
        # Save results
        sex_predictor.save_predictions(predictions, cell_names, "example5_human_predictions.csv")
        sex_predictor.plot_prediction_distribution(predictions, "example5_human_distribution.png")
        
        print("\n✅ Example 5 completed!")
        print(f"📄 Predictions saved: example5_human_predictions.csv")
        print(f"📊 Plot saved: example5_human_distribution.png")
        
    except FileNotFoundError:
        print("⚠️  Human data files not found (human_train.h5ad, human_test.h5ad)")
        print("   This example requires human-specific data files")
        print("   Skipping execution, but code structure is demonstrated")
    
    return sex_predictor


def example_6_cli_simulation():
    """
    Example 6: Simulate CLI usage programmatically
    Shows how to replicate CLI functionality in code.
    """
    print("=" * 60)
    print("EXAMPLE 6: CLI COMMAND SIMULATION")
    print("=" * 60)
    
    # Simulate different CLI commands
    cli_examples = [
        {
            'name': 'Basic 2-dataset workflow',
            'cmd': 'cellsexid --species mouse --train train.h5ad --test test.h5ad --output basic_results.csv --model RF',
            'workflow': '2-dataset',
            'description': 'Predefined markers with Random Forest'
        },
        {
            'name': 'Custom genes 2-dataset',
            'cmd': 'cellsexid --species mouse --train train.h5ad --test test.h5ad --output custom_results.csv --custom_genes "Xist,Ddx3y,Kdm5d" --model XGB',
            'workflow': '2-dataset',
            'description': 'Custom genes with XGBoost'
        },
        {
            'name': 'Feature selection 3-dataset',
            'cmd': 'cellsexid --species mouse --marker_train train.h5ad train.h5ad --test test.h5ad --output fs_results.csv --top_k 15 --min_models 2 --model SVM',
            'workflow': '3-dataset',
            'description': 'Marker discovery with SVM'
        },
        {
            'name': 'Human with custom column',
            'cmd': 'cellsexid --species human --train human_train.h5ad --test human_test.h5ad --output human_results.csv --sex_column gender --plot human_dist.png',
            'workflow': '2-dataset',
            'description': 'Human markers with plot output'
        }
    ]
    
    for i, example in enumerate(cli_examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Workflow: {example['workflow']}")
        print(f"   Description: {example['description']}")
        print(f"   Command: {example['cmd']}")
        print("   " + "─" * 50)
    
    print("\n💡 To run any of these commands:")
    print("   1. Copy the command above")
    print("   2. Ensure you have the required data files")
    print("   3. Run in terminal after installing: pip install -e .")
    print("\n✅ Example 6 completed!")


def check_data_files():
    """Check if required data files exist."""
    required_files = ['train.h5ad', 'test.h5ad']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️  Missing data files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n   These examples assume train.h5ad and test.h5ad exist")
        print("   Replace with your actual data file paths")
        return False
    return True


def main():
    """
    Main function to run all examples
    """
    print("🧬 CELLSEXID - COMPREHENSIVE USAGE EXAMPLES")
    print("🚀 Demonstrating all features of the SexPredictionTool")
    print("📦 Updated for current package structure and API")
    print()
    
    # Check if package is installed
    try:
        from cellsexid import SexPredictionTool
        print("✅ CellSexID package found")
    except ImportError:
        print("❌ CellSexID package not found")
        print("   Please install: pip install -e .")
        return
    
    # Check data files
    has_data = check_data_files()
    print()
    
    try:
        # Run all examples
        print("Running all examples...\n")
        
        if has_data:
            # Example 1: Basic predefined genes
            example_1_predefined_genes()
            
            # Example 2: Custom genes
            example_2_custom_genes()
            
            # Example 3: Feature selection
            example_3_feature_selection()
            
            # Example 4: Model comparison
            example_4_model_comparison()
        
        # Example 5: Human species (may skip if no data)
        example_5_human_species()
        
        # Example 6: CLI simulation (always works)
        example_6_cli_simulation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED!")
        print("=" * 60)
        
        if has_data:
            print("\nGenerated files:")
            print("📄 Predictions: example*_predictions.csv")
            print("📊 Plots: example*_distribution.png")
            print("🔍 Feature selection: feature_selection_results/")
        
        print("\nNext steps:")
        print("1. Use the console script: cellsexid --help")
        print("2. Import and use SexPredictionTool in your scripts:")
        print("   from cellsexid import SexPredictionTool")
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
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5, 6], 
                       help='Run specific example (1-6)')
    parser.add_argument('--train_data', default='train.h5ad', 
                       help='Path to training data')
    parser.add_argument('--test_data', default='test.h5ad', 
                       help='Path to test data')
    
    args = parser.parse_args()
    
    if args.example:
        # Check package installation
        try:
            from cellsexid import SexPredictionTool
        except ImportError:
            print("❌ CellSexID package not found. Please install: pip install -e .")
            sys.exit(1)
        
        examples = {
            1: example_1_predefined_genes,
            2: example_2_custom_genes,
            3: example_3_feature_selection,
            4: example_4_model_comparison,
            5: example_5_human_species,
            6: example_6_cli_simulation
        }
        
        print(f"Running Example {args.example}...")
        examples[args.example]()
    else:
        main()
