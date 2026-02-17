from __future__ import annotations

import logging
import pathlib
import sys
import urllib.parse
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.config import load_app_env
from birdsys.logging import configure_logging

from .predictors.model_a_yolo import YoloDetector
from .predictors.model_b_attributes import AttributePredictor
from .predictors.model_c_image_status import ImageStatusPredictor
from .serializers import to_label_studio_prediction


class PredictRequest(BaseModel):
    tasks: list[dict[str, Any]] | None = None


app = FastAPI(title="BirdSys ML Backend", version="0.1.0")
logger = logging.getLogger("birdsys.ml_backend")


env = load_app_env()
configure_logging(env.log_level)

try:
    model_a = YoloDetector(weights=env.model_a_weights)
except Exception as exc:  # noqa: BLE001
    logger.exception("Failed to load Model A: %s", exc)
    model_a = None

model_b = AttributePredictor(checkpoint_path=env.model_b_checkpoint)
model_c = ImageStatusPredictor(checkpoint_path=env.model_c_checkpoint)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_a_loaded": model_a is not None,
        "model_a_weights": str(env.model_a_weights),
        "model_b_checkpoint": str(env.model_b_checkpoint) if env.model_b_checkpoint else None,
        "model_c_checkpoint": str(env.model_c_checkpoint) if env.model_c_checkpoint else None,
    }


def _task_image_path(task: dict[str, Any]) -> pathlib.Path:
    def _safe_path(value: str) -> pathlib.Path:
        path = pathlib.Path(value).expanduser()
        try:
            return path.resolve()
        except OSError:
            if path.is_absolute():
                return path
            return (pathlib.Path.cwd() / path).absolute()

    data = task.get("data") or {}
    candidates = [data.get("image"), data.get("filepath"), data.get("path"), task.get("image")]
    for item in candidates:
        if not item:
            continue
        value = str(item)

        if "?d=" in value:
            parsed = urllib.parse.urlparse(value)
            query = urllib.parse.parse_qs(parsed.query)
            local = query.get("d", [""])[0]
            if local:
                return _safe_path(local)

        if value.startswith("http://") or value.startswith("https://"):
            continue

        return _safe_path(value)

    raise ValueError("Task does not include a usable local image path")


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if tasks is None and isinstance(payload, list):
        tasks = payload

    if not tasks:
        raise HTTPException(status_code=400, detail="No tasks provided")

    if model_a is None:
        raise HTTPException(status_code=500, detail="Model A is not loaded")

    predictions = []
    for task in tasks:
        task_id = task.get("id", "unknown")
        try:
            image_path = _task_image_path(task)
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            detections = model_a.predict(image_path)
            attrs = model_b.predict(detections)
            image_status_label, image_status_conf = model_c.predict_label(detections)

            pred = to_label_studio_prediction(
                task_id=task_id,
                detections=detections,
                attributes=attrs,
                image_status=image_status_label,
                image_status_conf=image_status_conf,
                model_version="birdsys-backend-v0.1",
            )
            predictions.append(pred)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed for task %s: %s", task_id, exc)
            predictions.append(
                {
                    "task": task_id,
                    "model_version": "birdsys-backend-v0.1",
                    "score": 0.0,
                    "result": [],
                    "error": str(exc),
                }
            )

    return {
        "predictions": predictions,
        "results": predictions,
    }
