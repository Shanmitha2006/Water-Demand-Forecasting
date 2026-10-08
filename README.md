# AI-Driven Water Demand Forecasting and Consumption Analytics

## Project Title

AI-Driven Water Demand Forecasting and Consumption Analytics Using Data Mining and Power BI

## Project Objective

This project analyzes water consumption or water demand data using data mining, machine learning, and dashboard visualization to forecast future water demand and support resource planning.

## Problem Statement

Water demand fluctuates due to temporal, seasonal, operational, and environmental factors. Accurate forecasting helps utilities, planners, and decision makers allocate supply, reduce losses, and plan infrastructure investments more effectively.

## Proposed Solution

A dynamic web application is built with FastAPI and React to load the dataset, clean it, perform EDA, train multiple forecasting models, compare results, and generate Power BI-ready CSV exports.

## System Architecture

- Data collection from CSV/Excel source
- Cleaning and preprocessing
- Feature engineering and automatic column detection
- Machine learning model training and evaluation
- Forecast generation
- Dashboard visualization and Power BI export

## Technologies Used

- Python
- FastAPI
- Pandas, NumPy
- Scikit-learn
- XGBoost (if installed)
- Joblib
- Matplotlib / Plotly
- React + Vite
- Recharts

## Dataset

The main dataset is automatically discovered from the dataset folder. The application does not assume hard-coded columns and instead inspects the actual file schema.

## Data Preprocessing

- Missing value detection and treatment
- Duplicate removal
- Data type correction
- Date/time detection and conversion
- Outlier handling
- Categorical encoding support
- Chronological sorting for time-series data

## Feature Engineering

- Date-derived features such as year, month, day, week, and season
- Lag and moving average features when appropriate
- Environmental and demand-related variables, if present

## Machine Learning

The project trains and compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost if installed
- LSTM is included as optional if the dataset is suitable

## Model Evaluation

The app calculates:

- MAE
- RMSE
- R²
- MAPE

## Forecasting

The best model is selected based on the evaluated metrics and then used to forecast future demand over a configurable horizon.

## Power BI Integration

The app exports clean CSV files ready for Microsoft Power BI import, including demand, prediction, summary, and model comparison files.

## How to Install

Windows commands:

```powershell
cd c:\Users\User\Desktop\dmt
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

## How to Run Backend

```powershell
cd c:\Users\User\Desktop\dmt
venv\Scripts\activate
uvicorn backend.main:app --reload
```

## How to Run Frontend

```powershell
cd c:\Users\User\Desktop\dmt\frontend
npm install
npm run dev
```

## How to Generate Predictions

Use the frontend forecasting page or call the backend forecasting route after training the model.

## How to Export Power BI Data

The Power BI-ready CSVs are created in the powerbi_data directory. They can be imported into Microsoft Power BI Desktop.

## Future Scope

- IoT smart meters
- Real-time weather and demand integration
- Cloud deployment
- Alerts and anomaly detection

## Important Note

The project is designed to adapt to the actual dataset schema. It will inspect the real CSV/Excel file and not assume fixed column names.
