import os
import io
import json
import pickle
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def prep_with_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date if string
    if "date" in df.columns and df["date"].dtype == object:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Extract features then drop
    if "date" in df.columns and is_datetime(df["date"]):
        df["year"] = df["date"].dt.year.astype("int16")
        df["month"] = df["date"].dt.month.astype("int8")
        df["day"] = df["date"].dt.day.astype("int8")
        df["dow"] = df["date"].dt.dayofweek.astype("int8")
        df = df.drop(columns=["date"])

    # Cast objects → category
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category")

    # Fill NA
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].astype("category").cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

    return df

def model_fn(model_dir):
    path = os.path.join(model_dir, "xgb_model_bundle.pkl")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "instances" in data:
            df = pd.DataFrame(data["instances"])
        else:
            df = pd.DataFrame(data)
    else:
        df = pd.read_csv(io.StringIO(request_body))
    return df

def predict_fn(df, model_bundle):
    model = model_bundle["model"]
    cat_cols = model_bundle["cat_cols"]
    train_cats = model_bundle["train_categories"]

    X = prep_with_date_features(df)

    # align categories
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
            allowed = list(train_cats[c]) + ["__UNSEEN__"]
            X[c] = X[c].where(X[c].isin(train_cats[c]), "__UNSEEN__")
            X[c] = X[c].cat.set_categories(allowed)

    preds = model.predict(X)
    return preds

def output_fn(predictions, accept):
    if accept == "application/json":
        return json.dumps({"predictions": predictions.tolist()})
    return "\n".join(map(str, predictions))
