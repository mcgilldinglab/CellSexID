#!/usr/bin/env python3
"""
CellSexID Quick Start Example

This script demonstrates the most basic usage of CellSexID.
Perfect for new users who want to get started immediately.
"""

import os
from cellsexid import ImprovedSexPredictionTool

def main():
    print("🧬 CellSexID Quick Start Example")
    print("=" * 40)
    
    # Example file paths (replace with your actual files)
    TRAIN_FILE = "data/training_data.h5ad"     # Replace with your training data
    TEST_FILE = "data/test_data.h5ad"          # Replace with your test data
    OUTPUT_FILE = "predictions.csv"
    PLOT_FILE = "distribution.png"
    
    # Check if files exist
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Training file not found: {TRAIN_FILE}")
        print("Please update TRAIN_FILE path with your actual training data")
        return
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        print("Please update TEST_FILE path with your actual test data")
        return
    
    try:
        # Step 1: Initialize the tool
        print("Step 1: Initializing CellSexID with predefined genes...")
        tool = ImprovedSexPredictionTool(use_predefined_genes=True)
        
        # Step 2: Process training data
        print("Step 2: Processing training data...")
        X_train, y_train = tool.process_training_data(TRAIN_FILE)
        
        # Step 3: Train the model (Random Forest is default)
        print("Step 3: Training Random Forest model...")
        tool.train(X_train, y_train, model_name='RF')
        
        # Step 4: Process test data
        print("Step 4: Processing test data...")
        X_test, cell_names = tool.process_test_data(TEST_FILE)
        
        # Step 5: Make predictions
        print("Step 5: Making predictions...")
        predictions = tool.predict(X_test, model_name='RF')
        
        # Step 6: Save results
        print("Step 6: Saving results...")
        tool.save_predictions(predictions, cell_names, OUTPUT_FILE)
        tool.plot_prediction_distribution(predictions, PLOT_FILE)
        
        # Summary
        male_count = sum(predictions == 1)
        female_count = sum(predictions == 0)
        total = len(predictions)
        
        print("\n✅ SUCCESS! Results:")
        print(f"   📄 Predictions saved: {OUTPUT_FILE}")
        print(f"   📊 Plot saved: {PLOT_FILE}")
        print(f"   🔢 Total cells: {total}")
        print(f"   ♀️  Female: {female_count} ({female_count/total*100:.1f}%)")
        print(f"   ♂️  Male: {male_count} ({male_count/total*100:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your data files are in H5AD format")
        print("2. Training data must have 'gender' column in adata.obs")
        print("3. Data should be preprocessed (normalized, log-transformed)")

if __name__ == "__main__":
    main() 