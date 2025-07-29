#!/usr/bin/env python3
"""
Complete CLI for Sex Prediction Tool
Supports both 2-dataset and 3-dataset workflows with human/mouse species
"""

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import argparse
import sys
import os

# Import the tool
try:
    from sex_prediction_tool import SexPredictionTool
except ImportError:
    print("❌ Error: Cannot find 'sex_prediction_tool.py'")
    print("Please ensure 'sex_prediction_tool.py' is in the same directory as this script")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Sex Prediction Tool - Complete Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW 1: Simple 2-Dataset (Predefined Markers)
  # Mouse with predefined markers
  python cli.py --species mouse --train train.h5ad --test test.h5ad --output results.csv
  
  # Human with predefined markers
  python cli.py --species human --train train.h5ad --test test.h5ad --output results.csv
  
  # Custom genes
  python cli.py --species mouse --train train.h5ad --test test.h5ad --output results.csv \\
    --custom_genes "Xist,Ddx3y,Kdm5d,Eif2s3y"

WORKFLOW 2: Advanced 3-Dataset (Custom Marker Discovery)  
  # Discover markers + train + test (3 separate files)
  python cli.py --species human --marker marker.h5ad --train train.h5ad --test test.h5ad --output results.csv
  
  # Same data for marker discovery and training (most common)
  python cli.py --species mouse --marker train.h5ad --train train.h5ad --test test.h5ad --output results.csv

Additional Options:
  --model RF              # Choose model (LR, SVM, XGB, RF)
  --sex_column gender     # If sex labels are in 'gender' column instead of 'sex'
  --plot dist.png         # Save prediction distribution plot
  --top_k 15             # Top K features for marker discovery
  --min_models 2         # Minimum model consensus for marker discovery
        """
    )
    
    # Required arguments
    parser.add_argument('--test', required=True, help='Test data (.h5ad file)')
    parser.add_argument('--output', required=True, help='Output predictions (.csv file)')
    
    # Species selection
    parser.add_argument('--species', choices=['mouse', 'human'], default='mouse',
                       help='Species for gene markers (default: mouse)')
    
    # Data workflow selection
    workflow_group = parser.add_mutually_exclusive_group(required=True)
    workflow_group.add_argument('--train', 
                               help='Training data for 2-dataset workflow (predefined markers)')
    workflow_group.add_argument('--marker_train', nargs=2, metavar=('MARKER', 'TRAIN'),
                               help='Marker discovery data + training data for 3-dataset workflow')
    
    # Gene selection options (only for 2-dataset workflow)
    parser.add_argument('--custom_genes', type=str,
                       help='Comma-separated custom genes (e.g., "Xist,Ddx3y,Kdm5d") - only for 2-dataset workflow')
    
    # Model options
    parser.add_argument('--model', choices=['LR', 'SVM', 'XGB', 'RF'], default='RF',
                       help='ML model to use (default: RF)')
    parser.add_argument('--sex_column', default='sex',
                       help='Column name containing sex labels (default: sex)')
    
    # Feature selection options (only for 3-dataset workflow)
    parser.add_argument('--top_k', type=int, default=20,
                       help='Top K features per model for marker discovery (default: 20)')
    parser.add_argument('--min_models', type=int, default=3,
                       help='Minimum models consensus for marker discovery (default: 3)')
    
    # Output options
    parser.add_argument('--plot', help='Save prediction distribution plot')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine workflow
    if args.train:
        workflow = "2-dataset"
        use_predefined = True
        files_to_check = [args.train, args.test]
        
        # Handle custom genes
        custom_genes = None
        if args.custom_genes:
            custom_genes = [gene.strip() for gene in args.custom_genes.split(',') if gene.strip()]
            
    else:
        workflow = "3-dataset"
        use_predefined = False
        marker_data, train_data = args.marker_train
        files_to_check = [marker_data, train_data, args.test]
        custom_genes = None
        
        if args.custom_genes:
            print("⚠️ Warning: --custom_genes is ignored in 3-dataset workflow (markers will be discovered)")
    
    # Validate input files
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found: {filepath}")
            sys.exit(1)
    
    try:
        # Print workflow info
        print("🧬 SEX PREDICTION TOOL")
        print("=" * 60)
        print(f"Species: {args.species}")
        print(f"Workflow: {workflow}")
        print(f"Model: {args.model}")
        print(f"Sex column: {args.sex_column}")
        print(f"Test data: {args.test}")
        print(f"Output: {args.output}")
        
        if workflow == "2-dataset":
            print(f"Train data: {args.train}")
            if custom_genes:
                print(f"Mode: Custom genes ({len(custom_genes)} provided)")
                print(f"Genes: {', '.join(custom_genes)}")
            else:
                print("Mode: Predefined markers")
        else:
            marker_data, train_data = args.marker_train
            print(f"Marker data: {marker_data}")
            print(f"Train data: {train_data}")
            print(f"Mode: Custom marker discovery (top_k={args.top_k}, min_models={args.min_models})")
        
        print("=" * 60)
        
        # Initialize tool
        tool = SexPredictionTool(
            species=args.species,
            use_predefined_genes=use_predefined,
            custom_genes=custom_genes,
            sex_column=args.sex_column
        )
        
        if args.verbose:
            tool.get_summary()
        
        # Execute workflow
        if workflow == "2-dataset":
            # WORKFLOW 1: Simple 2-dataset with predefined/custom markers
            print(f"\n📚 WORKFLOW 1: Training with {'custom' if custom_genes else 'predefined'} markers...")
            tool.fit(train_data=args.train, model_name=args.model)
            
        else:
            # WORKFLOW 2: Advanced 3-dataset with marker discovery
            marker_data, train_data = args.marker_train
            
            print("\n🔍 WORKFLOW 2 - Step 1: Discovering optimal markers...")
            tool.discover_markers(
                marker_data=marker_data,
                top_k=args.top_k,
                min_models=args.min_models,
                save_results=True
            )
            
            print("\n📚 WORKFLOW 2 - Step 2: Training with discovered markers...")
            tool.fit_with_discovered_markers(train_data=train_data, model_name=args.model)
        
        # Make predictions
        print("\n🔮 Making predictions...")
        predictions, cell_names = tool.predict(test_data=args.test)
        
        # Save results
        print("\n💾 Saving results...")
        tool.save_predictions(predictions, cell_names, args.output)
        
        # Plot distribution if requested
        if args.plot:
            tool.plot_prediction_distribution(predictions, args.plot)
        
        # Print summary
        male_count = sum(predictions == 1)
        female_count = sum(predictions == 0)
        total = len(predictions)
        
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"   Total cells: {total}")
        print(f"   Female: {female_count} ({female_count/total*100:.1f}%)")
        print(f"   Male: {male_count} ({male_count/total*100:.1f}%)")
        
        markers_used = tool.selected_markers
        print(f"\n🧬 Used {len(markers_used)} markers:")
        print(f"   {', '.join(markers_used[:8])}{'...' if len(markers_used) > 8 else ''}")
        
        if workflow == "3-dataset":
            print(f"\n📁 Feature selection results saved in: feature_selection_results/")
        
        print(f"\n✅ SUCCESS!")
        print(f"📄 Predictions saved: {args.output}")
        if args.plot:
            print(f"📊 Plot saved: {args.plot}")
            
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