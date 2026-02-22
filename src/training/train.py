import xgboost as xgb
import joblib

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)
