# Power BI Guide

## 1. Open Power BI Desktop

Open Microsoft Power BI Desktop and select Get Data.

## 2. Import the generated CSV files

Import files from the powerbi_data folder:

- water_demand_cleaned.csv
- water_demand_predictions.csv
- model_comparison.csv
- monthly_demand_summary.csv
- yearly_demand_summary.csv
- feature_importance.csv

## 3. Recommended relationships

- Link a date column between cleaned data and prediction tables when present.
- Link model comparison to related summary tables if a common date or model field exists.

## 4. Recommended visuals

- Line chart for demand trend
- Bar chart for monthly demand
- Scatter or line chart for actual vs predicted
- KPI cards for total demand, average demand, max, min, forecast demand
- Matrix or table for model comparison

## 5. Recommended slicers

- Year
- Month
- Region/Location (if available)
- Model name

## 6. Recommended KPIs

- Total Water Consumption
- Average Demand
- Maximum Demand
- Minimum Demand
- Forecast Demand

## 7. Recommended dashboard layout

Page 1: Executive Overview
Page 2: Demand Analysis
Page 3: Forecast Analytics
Page 4: Factor Analysis
Page 5: Model Performance

## 8. How to refresh the data

After updating the dataset or retraining the model, regenerate the CSV exports in the project and use Refresh in Power BI Desktop.

## 9. How to export or update predictions

Use the project’s forecasting workflow to generate new prediction CSV outputs, then import the updated files into Power BI.
