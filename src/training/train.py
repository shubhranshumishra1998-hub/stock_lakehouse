import os
import argparse
import pandas as pd
import xgboost as xgb
import joblib

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load training data
    train_path = os.path.join(args.train, "train.csv")
    df = pd.read_csv(train_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X, y)

    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
