import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "xgb_stock_model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    raise ValueError("Unsupported content type")


def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())
