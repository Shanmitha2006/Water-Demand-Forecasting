from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from backend.data_loader import is_datetime_like_column


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        try:
            if is_datetime_like_column(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
    return df


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0 and pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
    return numeric


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in df.columns
        if col not in detect_numeric_columns(df) and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]


def fill_missing_values(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if date_col == col:
                df[col] = df[col].sort_values().reset_index(drop=True)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            if pd.isna(med):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(med)
        else:
            mode = df[col].mode(dropna=True)
            if len(mode) == 0:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(mode.iloc[0])
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.lower().endswith("date") or "date" in col.lower() or "time" in col.lower():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0:
                df[col] = coerced
        except Exception:
            pass
    return df


def detect_and_clean_outliers(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_columns:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def auto_select_target(df: pd.DataFrame) -> str | None:
    keywords = [
        "water_demand",
        "water_consumption",
        "demand",
        "consumption",
        "water_usage",
        "usage",
        "flow",
        "volume",
        "total_demand",
        "daily_demand",
    ]

    for column in df.columns:
        normalized = str(column).lower().replace(" ", "_").replace("-", "_")
        if any(word in normalized for word in keywords):
            return column

    numeric_candidates = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in {"year", "month", "day"}
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    return None


def detect_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if is_datetime_like_column(df[col]):
            return col
    return None


def preprocess_dataframe(df: pd.DataFrame, target_column: str | None = None) -> Dict[str, Any]:
    df = normalize_column_names(remove_duplicate_rows(df))
    df = coerce_numeric_columns(df)
    df = convert_datetime_columns(df)

    date_col = detect_datetime_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(by=date_col).reset_index(drop=True)

    if target_column is None:
        target_column = auto_select_target(df)

    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col != target_column
    ]

    df = fill_missing_values(df, date_col=date_col)
    df = detect_and_clean_outliers(df, numeric_cols)

    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        df[target_column] = df[target_column].fillna(df[target_column].median())

    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and col != target_column and col != date_col and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].astype(str)

    return {
        "processed_df": df,
        "date_column": date_col,
        "target_column": target_column,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
