from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"


def get_project_root() -> Path:
    return ROOT_DIR


def find_dataset_files(directory: Path | str | None = None) -> List[Path]:
    target_dir = Path(directory) if directory else DATASET_DIR
    if not target_dir.exists():
        return []

    extensions = {".csv", ".xls", ".xlsx"}
    files = [
        path
        for path in target_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files)


def find_primary_dataset(directory: Path | str | None = None) -> Optional[Path]:
    files = find_dataset_files(directory)
    if not files:
        return None

    preferred = []
    for p in files:
        name = p.name.lower()
        if "water" in name or "demand" in name or "consumption" in name:
            preferred.append(p)
    if preferred:
        return preferred[0]
    return files[0]


def load_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    if file_path is None:
        file_path = find_primary_dataset()

    if file_path is None:
        raise FileNotFoundError("No CSV or Excel dataset found in the dataset folder.")

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    df.columns = [str(col).strip() for col in df.columns]
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    duplicate_rows = int(df.duplicated().sum())
    missing_values = df.isnull().sum().to_dict()

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols and not is_datetime_like_column(df[col])
    ]
    datetime_cols = [
        col for col in df.columns if is_datetime_like_column(df[col])
    ]

    target_candidates = infer_target_candidates(df)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "target_candidates": target_candidates,
    }


def infer_target_candidates(df: pd.DataFrame) -> List[str]:
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
    normalized = {str(col).lower().replace(" ", "_").replace("-", "_") for col in df.columns}
    matches = []
    for column in df.columns:
        key = str(column).lower().replace(" ", "_").replace("-", "_")
        if any(keyword in key for keyword in keywords):
            matches.append(column)
    return matches


def is_datetime_like_column(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False

    try:
        converted = pd.to_datetime(sample, errors="raise")
        if len(converted) > 0:
            return True
    except Exception:
        return False
    return False
