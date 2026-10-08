from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def add_datetime_features(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    if date_col is None:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).reset_index(drop=True)
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_col].dt.quarter
    df["weekend"] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    df["season"] = df[date_col].dt.month.map({
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    })
    return df


def generate_lag_features(df: pd.DataFrame, target_col: str | None, lags: List[int] | None = None) -> pd.DataFrame:
    if target_col is None:
        return df

    df = df.copy()
    lags = lags or [1, 7, 14, 30]
    for lag in lags:
        if len(df) > lag:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

    rolling_windows = [7, 14, 30]
    for window in rolling_windows:
        if len(df) > window:
            df[f"rolling_mean_{window}"] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f"rolling_std_{window}"] = df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
    return df


def prepare_model_features(df: pd.DataFrame, target_col: str | None, date_col: str | None) -> Dict[str, Any]:
    df = df.copy()
    if date_col is not None:
        df = add_datetime_features(df, date_col)
    if target_col is not None:
        df = generate_lag_features(df, target_col)

    exclude_cols = {target_col, date_col}
    if target_col is not None:
        exclude_cols.update({col for col in df.columns if col.startswith("lag_")})
        exclude_cols.update({col for col in df.columns if col.startswith("rolling_")})
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
    ]

    feature_frame = df[numeric_cols].copy()
    if target_col and target_col in df.columns:
        feature_frame[target_col] = df[target_col]
    return {
        "model_df": feature_frame,
        "feature_names": numeric_cols,
    }
