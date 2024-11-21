import argparse
from sex_prediction_tool import SexPredictionTool  # Import the class

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run sex prediction')
parser.add_argument('--train_data', required=True, help='Path to preprocessed training h5ad file')
parser.add_argument('--test_data', required=True, help='Path to test h5ad file')
parser.add_argument('--model', default='XGB', help='Model to use (XGB, LR, SVM, RF)')
parser.add_argument('--output', required=True, help='Output file for predictions')
parser.add_argument('--plot', help='Output file for distribution plot')
args = parser.parse_args()

# Initialize the tool
sex_predictor = SexPredictionTool()

# Process training data
X_train, y_train = sex_predictor.process_training_data(args.train_data)

# Train the model
sex_predictor.train(X_train, y_train, model_name=args.model)

# Process test data
X_test, cell_names = sex_predictor.process_test_data(args.test_data)

# Make predictions
y_pred = sex_predictor.predict(X_test, model_name=args.model)

# Save predictions
sex_predictor.save_predictions(y_pred, cell_names, args.output)

# Plot distribution if a plot path is provided
if args.plot:
    sex_predictor.plot_prediction_distribution(y_pred, save_path=args.plot)
