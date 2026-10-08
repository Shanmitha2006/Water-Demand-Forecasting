from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def rank_models(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for item in results:
        metrics = item["metrics"]
        score = (metrics["mae"] * 0.35) + (metrics["rmse"] * 0.35) + (max(0.0, 1.0 - metrics["r2"]) * 0.2) + (metrics["mape"] * 0.1)
        ranked.append({**item, "score": score})
    ranked.sort(key=lambda x: x["score"])
    return ranked


def best_model_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("No model results available.")
    return min(results, key=lambda x: (x["metrics"]["mae"], x["metrics"]["rmse"]))
