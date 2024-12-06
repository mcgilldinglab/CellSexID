import argparse
from .sex_prediction_tool import SexPredictionTool

def main():
    parser = argparse.ArgumentParser(description="Run CellSexID predictions.")
    parser.add_argument("--train_data", required=True, help="Path to training data (H5AD format).")
    parser.add_argument("--test_data", required=True, help="Path to test data (H5AD format).")
    parser.add_argument("--model", required=True, choices=["LR", "SVM", "XGB", "RF"], help="Model to use.")
    parser.add_argument("--output", required=True, help="Output CSV for predictions.")
    parser.add_argument("--plot", required=True, help="Output file for prediction distribution plot.")
    args = parser.parse_args()

    tool = SexPredictionTool()
    X_train, y_train = tool.process_training_data(args.train_data)
    tool.train(X_train, y_train, model_name=args.model)
    X_test, cell_names = tool.process_training_data(args.test_data)
    y_pred = tool.models[args.model].predict(X_test)
    tool.save_predictions(y_pred, cell_names, args.output)
    tool.plot_prediction_distribution(y_pred, args.plot)

if __name__ == "__main__":
    main()
