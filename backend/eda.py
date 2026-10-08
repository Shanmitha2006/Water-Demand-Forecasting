from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def summarize_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    return {
        "missing_count": missing.to_dict(),
        "missing_percentage": missing_pct.to_dict(),
    }


def detect_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    outlier_summary = {}
    for col in numeric_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((s < lower) | (s > upper)).sum()
        outlier_summary[col] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(outliers),
        }
    return outlier_summary


def get_correlation_summary(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    selected = df[numeric_cols].copy()
    corr = selected.corr(numeric_only=True)
    corr = corr.round(3)
    return corr.to_dict()


def get_monthly_summary(df: pd.DataFrame, date_col: str | None, target_col: str | None) -> List[dict]:
    if target_col is None:
        return []
    if date_col is None and "Year" in df.columns:
        yearly = get_yearly_summary(df, None, target_col)
        return [{"date_label": str(item["year"]), "demand": item[target_col], "year": item["year"], target_col: item[target_col]} for item in yearly]
    if date_col is None:
        return []
    temp = df[[date_col, target_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, target_col])
    if temp.empty:
        return []
    temp["year"] = temp[date_col].dt.year
    temp["month"] = temp[date_col].dt.month
    grouped = temp.groupby(["year", "month"], as_index=False)[target_col].sum()
    grouped["date_label"] = grouped["year"].astype(str) + "-" + grouped["month"].astype(str).str.zfill(2)
    grouped["demand"] = grouped[target_col]
    return grouped.to_dict(orient="records")


def get_yearly_summary(df: pd.DataFrame, date_col: str | None, target_col: str | None) -> List[dict]:
    if target_col is None:
        return []
    if date_col is None and "Year" in df.columns:
        temp = df[["Year", target_col]].copy()
        temp["Year"] = pd.to_numeric(temp["Year"], errors="coerce")
        temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
        grouped = temp.dropna().groupby("Year", as_index=False)[target_col].sum()
        grouped["year"] = grouped["Year"]
        grouped["demand"] = grouped[target_col]
        return grouped.to_dict(orient="records")
    if date_col is None:
        return []
    temp = df[[date_col, target_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, target_col])
    if temp.empty:
        return []
    grouped = temp.assign(year=temp[date_col].dt.year).groupby("year", as_index=False)[target_col].sum()
    grouped["demand"] = grouped[target_col]
    return grouped.to_dict(orient="records")


def get_eda_summary(df: pd.DataFrame, target_col: str | None, date_col: str | None) -> Dict[str, Any]:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    missing_summary = summarize_missing_values(df)
    outlier_summary = detect_outliers(df, numeric_cols)
    corr_summary = get_correlation_summary(df, numeric_cols) if numeric_cols else {}
    monthly_summary = get_monthly_summary(df, date_col, target_col)
    yearly_summary = get_yearly_summary(df, date_col, target_col)

    return {
        "numeric_columns": numeric_cols,
        "missing_summary": missing_summary,
        "outlier_summary": outlier_summary,
        "correlation_summary": corr_summary,
        "monthly_summary": monthly_summary,
        "yearly_summary": yearly_summary,
    }
