from __future__ import annotations

from typing import Any


def format_predict_response(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a payload compatible with multiple Label Studio ML API variants."""
    return {
        "results": predictions,
        "predictions": predictions,
    }


def extract_predictions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read predictions regardless of whether backend uses results or predictions."""
    raw = payload.get("results")
    if raw is None:
        raw = payload.get("predictions")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []
