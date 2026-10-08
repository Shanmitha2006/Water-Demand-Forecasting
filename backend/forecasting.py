from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def create_forecast_index(last_date: pd.Timestamp, periods: int, freq: str = "D") -> pd.DatetimeIndex:
    if freq == "M":
        return pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periods, freq="M")
    if freq == "H":
        return pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=periods, freq="H")
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")


def generate_forecast_dataframe(last_date: pd.Timestamp, forecast_values: List[float], freq: str = "D") -> pd.DataFrame:
    indices = create_forecast_index(last_date, len(forecast_values), freq=freq)
    df = pd.DataFrame({
        "date": indices,
        "forecast_value": forecast_values,
        "forecast_status": ["Forecast"] * len(forecast_values),
    })
    return df
