from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.data_loader import dataset_summary, find_primary_dataset, load_dataset
from backend.eda import get_eda_summary
from backend.models import train_and_evaluate_models
from backend.preprocessing import preprocess_dataframe

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "powerbi_data"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(title="AI Water Demand Forecasting", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_CACHE: Dict[str, Any] = {}
MODEL_CACHE: Dict[str, Any] = {}


class TrainRequest(BaseModel):
    target_column: Optional[str] = Field(default=None)
    forecast_horizon: int = Field(default=7, ge=1, le=30)


def _dataset_context(target_column: str | None = None) -> dict:
    dataset_file = find_primary_dataset(DATASET_DIR)
    if dataset_file is None:
        raise HTTPException(status_code=404, detail="No dataset found in dataset folder.")
    processed = preprocess_dataframe(load_dataset(dataset_file), target_column)
    frame = processed["processed_df"].copy()
    target = processed["target_column"]
    if not target or target not in frame.columns:
        raise HTTPException(status_code=400, detail="Unable to detect a numeric target column.")
    frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=[target])
    categorical = [col for col in processed["categorical_columns"] if col in frame.columns]
    feature_columns = [col for col in frame.columns if col != target and col != processed["date_column"]]
    encoded = pd.get_dummies(frame[feature_columns], columns=categorical, dtype=float)
    encoded = encoded.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    return {"dataset_file": dataset_file, "frame": frame, "processed": processed, "X": encoded, "y": pd.to_numeric(frame[target]), "target": target}


@app.get("/api/dataset")
def get_dataset() -> dict:
    dataset_file = find_primary_dataset(DATASET_DIR)
    if dataset_file is None:
        raise HTTPException(status_code=404, detail="No dataset found in dataset folder.")

    df = load_dataset(dataset_file)
    summary = dataset_summary(df)
    summary["dataset_name"] = dataset_file.name
    summary["dataset_path"] = str(dataset_file)
    summary["total_rows"] = int(len(df))
    summary["columns"] = list(df.columns)
    summary["data"] = df.head(100).to_dict(orient="records")
    summary["missing_values"] = df.isna().sum().to_dict()
    summary["dtypes"] = df.dtypes.astype(str).to_dict()

    target = summary["target_candidates"][0] if summary["target_candidates"] else None
    if target:
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        summary["statistics"] = {
            "target_column": target,
            "average": float(values.mean()),
            "maximum": float(values.max()),
            "minimum": float(values.min()),
            "country_count": int(df["Country"].nunique()) if "Country" in df.columns else 0,
            "year_min": int(df["Year"].min()) if "Year" in df.columns else None,
            "year_max": int(df["Year"].max()) if "Year" in df.columns else None,
        }
    return summary


@app.get("/api/dataset-summary")
def get_dataset_summary() -> dict:
    dataset_file = find_primary_dataset(DATASET_DIR)
    if dataset_file is None:
        raise HTTPException(status_code=404, detail="No dataset found in dataset folder.")

    df = load_dataset(dataset_file)
    summary = dataset_summary(df)
    summary["dataset_name"] = dataset_file.name
    summary["dataset_path"] = str(dataset_file)
    target = summary["target_candidates"][0] if summary["target_candidates"] else None
    if target:
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        summary["statistics"] = {
            "target_column": target,
            "average": float(values.mean()),
            "maximum": float(values.max()),
            "minimum": float(values.min()),
            "country_count": int(df["Country"].nunique()) if "Country" in df.columns else 0,
            "year_min": int(df["Year"].min()) if "Year" in df.columns else None,
            "year_max": int(df["Year"].max()) if "Year" in df.columns else None,
        }
    return summary


@app.get("/api/statistics")
def get_statistics() -> dict:
    summary = get_dataset_summary()
    return summary.get("statistics", {})


@app.get("/api/columns")
def get_columns() -> dict:
    dataset_file = find_primary_dataset(DATASET_DIR)
    if dataset_file is None:
        raise HTTPException(status_code=404, detail="No dataset found in dataset folder.")
    df = load_dataset(dataset_file)
    return {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


@app.get("/api/eda")
def get_eda() -> dict:
    dataset_file = find_primary_dataset(DATASET_DIR)
    if dataset_file is None:
        raise HTTPException(status_code=404, detail="No dataset found in dataset folder.")
    df = load_dataset(dataset_file)
    processed = preprocess_dataframe(df)
    result = get_eda_summary(processed["processed_df"], processed["target_column"], processed["date_column"])
    return {
        "dataset_name": dataset_file.name,
        "date_column": processed["date_column"],
        "target_column": processed["target_column"],
        "eda": result,
    }


@app.post("/api/train")
def train_models(payload: TrainRequest):
    context = _dataset_context(payload.target_column)
    dataset_file, processed, X, y, target = context["dataset_file"], context["processed"], context["X"], context["y"], context["target"]
    date_col = processed["date_column"]
    if len(X) < 2:
        raise HTTPException(status_code=400, detail="Dataset is too small to train a model.")

    split_index = min(max(1, int(len(X) * 0.8)), len(X) - 1)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best = sorted(results, key=lambda item: (item["metrics"]["mae"], item["metrics"]["rmse"]))[0]

    predictions = best["model"].predict(X_test)
    prediction_rows = pd.DataFrame({"Index": X_test.index, "Actual Demand": y_test.values, "Predicted Demand": predictions})
    prediction_rows.to_csv(OUTPUT_DIR / "water_demand_predictions.csv", index=False)
    pd.DataFrame({"model": [item["name"] for item in results], **{metric: [item["metrics"][metric] for item in results] for metric in ["mae", "rmse", "r2", "mape"]}}).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    processed["processed_df"].to_csv(OUTPUT_DIR / "water_demand_cleaned.csv", index=False)
    eda_result = get_eda_summary(processed["processed_df"], target, date_col)
    pd.DataFrame(eda_result["yearly_summary"]).to_csv(OUTPUT_DIR / "monthly_demand_summary.csv", index=False)

    MODEL_CACHE["latest"] = {
        "target": target,
        "date_column": date_col,
        "results": results,
        "best_model": best["name"],
        "X": X,
        "y": y,
        "split_index": split_index,
        "feature_columns": list(X.columns),
        "last_features": X.iloc[-1].to_dict(),
        "last_year": int(context["frame"]["Year"].max()) if "Year" in context["frame"].columns else None,
    }

    return {
        "target_column": target,
        "date_column": date_col,
        "results": [{"model": item["name"], **item["metrics"]} for item in results],
        "best_model": best["name"],
        "predictions": prediction_rows.to_dict(orient="records"),
    }


@app.get("/api/model-comparison")
def get_model_comparison() -> dict:
    if "latest" in MODEL_CACHE:
        latest = MODEL_CACHE["latest"]
        results = [
            {
                "model": item["name"],
                "mae": item["metrics"]["mae"],
                "mse": item["metrics"].get("mse", item["metrics"]["rmse"] ** 2),
                "rmse": item["metrics"]["rmse"],
                "r2": item["metrics"]["r2"],
                "mape": item["metrics"]["mape"],
            }
            for item in latest["results"]
        ]
        return {"results": results, "best_model": latest["best_model"]}

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        return {"results": []}
    return {"results": pd.read_csv(comparison_path).to_dict(orient="records")}


@app.get("/api/predictions")
def get_predictions() -> dict:
    predictions_path = OUTPUT_DIR / "water_demand_predictions.csv"
    if not predictions_path.exists():
        return {"predictions": []}
    return {"predictions": pd.read_csv(predictions_path).to_dict(orient="records")}


@app.get("/api/forecast")
def get_forecast() -> dict:
    if "latest" not in MODEL_CACHE:
        forecast_path = OUTPUT_DIR / "water_demand_forecast.csv"
        if not forecast_path.exists():
            return {"actual": [], "forecasted": [], "years": [], "model": "Not trained"}
        forecast_df = pd.read_csv(forecast_path)
        return {
            "forecast": forecast_df.to_dict(orient="records"),
            "model": "Saved forecast",
        }
    latest = MODEL_CACHE["latest"]
    forecast_rows = []
    for row in latest.get("forecast_rows", []):
        forecast_rows.append({
            "Year": row.get("Year"),
            "Actual": row.get("Actual"),
            "Forecasted": row.get("Forecasted"),
            "Model": row.get("Model"),
        })
    return {
        "actual": [row["Actual"] for row in forecast_rows if row.get("Actual") is not None],
        "forecasted": [row["Forecasted"] for row in forecast_rows if row.get("Forecasted") is not None],
        "years": [row["Year"] for row in forecast_rows if row.get("Year") is not None],
        "model": latest["best_model"],
    }


@app.post("/api/forecast")
def generate_forecast(payload: TrainRequest):
    dataset_file = find_primary_dataset(DATASET_DIR)
    if dataset_file is None:
        raise HTTPException(status_code=404, detail="No dataset found in dataset folder.")

    context = _dataset_context(payload.target_column)
    target = context["target"]
    if target is None:
        raise HTTPException(status_code=400, detail="Target column not detected. Please select one.")

    if "latest" not in MODEL_CACHE:
        raise HTTPException(status_code=400, detail="Train a model before generating forecast.")

    latest = MODEL_CACHE["latest"]
    model_name = latest["best_model"]
    target_model = next(item for item in latest["results"] if item["name"] == model_name)
    model = target_model["model"]

    future = pd.DataFrame([latest["last_features"]] * payload.forecast_horizon)
    if latest.get("last_year") is not None and "Year" in context["frame"].columns:
        future["Year"] = [latest["last_year"] + offset for offset in range(1, payload.forecast_horizon + 1)]
    future = future.reindex(columns=latest["feature_columns"], fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0)
    future_values = model.predict(future)

    forecast_years = list(range(latest.get("last_year", 2024) + 1, latest.get("last_year", 2024) + 1 + payload.forecast_horizon))
    forecast_df = pd.DataFrame({
        "Date": pd.to_datetime([f"{year}-01-01" for year in forecast_years]),
        "Year": forecast_years,
        "Actual Demand": [None] * len(future_values),
        "Predicted Demand": future_values,
        "Forecast Status": ["Forecast"] * len(future_values),
    })
    latest["forecast_rows"] = [{"Year": int(year), "Actual": None, "Forecasted": float(value), "Model": model_name} for year, value in zip(forecast_years, future_values)]

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    forecast_path = OUTPUT_DIR / "water_demand_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    return {"forecast": forecast_df.to_dict(orient="records"), "model": model_name, "years": forecast_years, "forecasted": list(future_values)}


@app.get("/api/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=file_path.name)


@app.get("/api/files")
def list_files() -> dict:
    files = [p.name for p in sorted(OUTPUT_DIR.glob("*.csv"))]
    return {"files": files}


@app.get("/")
def home() -> dict:
    return {"message": "Water Demand Forecasting API is running."}
