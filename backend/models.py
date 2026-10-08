from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


class ModelEvaluator:
    @staticmethod
    def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Prediction shape {y_pred.shape} does not match target shape {y_true.shape}.")
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1, y_true))) * 100
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
        }


def build_models() -> List[Dict[str, Any]]:
    models = [
        {"name": "Linear Regression", "model": LinearRegression()},
        {"name": "Random Forest Regressor", "model": RandomForestRegressor(n_estimators=200, random_state=42)},
        {"name": "Gradient Boosting Regressor", "model": GradientBoostingRegressor(random_state=42)},
    ]
    if XGBRegressor is not None:
        models.append({"name": "XGBoost Regressor", "model": XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42, objective="reg:squarederror")})
    return models


def train_and_evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> List[Dict[str, Any]]:
    results = []
    for entry in build_models():
        model = entry["model"]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = ModelEvaluator.compute_metrics(y_test, predictions)
        results.append({
            "name": entry["name"],
            "metrics": metrics,
            "model": model,
        })
    return results
