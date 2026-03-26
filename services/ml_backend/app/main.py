from __future__ import annotations

import json
import logging
import os
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
from .response_contract import format_predict_response
from .serializers import to_label_studio_prediction

MODEL_VERSION = "birdsys-backend-v0.1"


class PredictRequest(BaseModel):
    tasks: list[dict[str, Any]] | None = None


app = FastAPI(title="BirdSys ML Backend", version="0.1.0")
logger = logging.getLogger("birdsys.ml_backend")


env = load_app_env()
configure_logging(env.log_level)

try:
    model_a = YoloDetector(
        weights=env.model_a_weights,
        conf=float(os.getenv("MODEL_A_CONF", "0.25")),
        iou=float(os.getenv("MODEL_A_IOU", "0.45")),
        imgsz=int(os.getenv("MODEL_A_IMGSZ", "1280")),
        max_det=int(os.getenv("MODEL_A_MAX_DET", "300")),
        device=os.getenv("MODEL_A_DEVICE", "auto"),
    )
except Exception as exc:  # noqa: BLE001
    logger.exception("Failed to load Model A: %s", exc)
    model_a = None

model_b = AttributePredictor(checkpoint_path=env.model_b_checkpoint)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        **_model_health_payload(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    # Label Studio ML backend validation calls /health.
    return {
        "status": "UP",
        **_model_health_payload(),
    }


@app.post("/setup")
def setup(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    # Label Studio validates ML backends by calling /setup after /health.
    return {
        "status": "UP",
        "model_version": MODEL_VERSION,
        **_model_health_payload(),
    }


@app.post("/validate")
def validate(payload: dict[str, Any]) -> dict[str, Any]:
    # Optional endpoint used by some ML backend flows.
    return {"valid": True}


def _model_a_promotion_payload() -> dict[str, Any]:
    metadata_path = env.model_a_weights.parent / "promotion.json"
    if not metadata_path.exists():
        return {
            "model_a_release_id": None,
            "model_a_source_path": None,
            "model_a_source_sha256": None,
        }

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # noqa: PERF203
        logger.warning("Failed to parse promotion metadata at %s", metadata_path)
        return {
            "model_a_release_id": None,
            "model_a_source_path": None,
            "model_a_source_sha256": None,
        }

    return {
        "model_a_release_id": payload.get("release_id"),
        "model_a_source_path": payload.get("source_path"),
        "model_a_source_sha256": payload.get("source_sha256"),
    }


def _model_health_payload() -> dict[str, Any]:
    return {
        "model_a_loaded": model_a is not None,
        "model_b_loaded": model_b.model is not None,
        "model_a_weights": str(env.model_a_weights),
        "model_a_device": model_a.device if model_a is not None else None,
        "model_a_imgsz": model_a.imgsz if model_a is not None else None,
        "model_a_max_det": model_a.max_det if model_a is not None else None,
        "model_b_checkpoint": str(env.model_b_checkpoint) if env.model_b_checkpoint else None,
        "model_b_schema_version": model_b.schema_version,
        "model_b_supported_labels": {head: sorted(labels) for head, labels in model_b.supported_labels.items()},
        **_model_a_promotion_payload(),
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

    def _resolve_local_files_query_path(value: str) -> pathlib.Path:
        # Label Studio local-files URLs commonly use query values like:
        # - birds_project/raw_images/...
        # - lines_project/labelstudio/images_compressed/...
        # - raw_images/...
        # - /data/birds_project/raw_images/...
        decoded = urllib.parse.unquote(value).strip()
        if not decoded:
            raise ValueError("Label Studio local-files URL has empty d= query value")

        data_root_parent = env.birds_data_root.parent

        if decoded == "/data":
            return _safe_path(str(data_root_parent))

        if decoded == "/data/birds_project":
            return _safe_path(str(env.birds_data_root))

        if decoded.startswith("/data/"):
            rel = decoded.removeprefix("/data/").lstrip("/")
            return _safe_path(str(data_root_parent / rel))

        if decoded.startswith("/data/birds_project/"):
            rel = decoded.removeprefix("/data/birds_project/").lstrip("/")
            return _safe_path(str(env.birds_data_root / rel))

        if decoded.startswith("/"):
            return _safe_path(decoded)

        if decoded.startswith("data/"):
            if decoded.startswith("data/birds_project/"):
                rel = decoded.removeprefix("data/birds_project/").lstrip("/")
                return _safe_path(str(env.birds_data_root / rel))
            rel = decoded.removeprefix("data/").lstrip("/")
            return _safe_path(str(data_root_parent / rel))

        if decoded.startswith("birds_project/"):
            rel = decoded.removeprefix("birds_project/").lstrip("/")
            return _safe_path(str(env.birds_data_root / rel))

        dataset, sep, remainder = decoded.partition("/")
        if sep and dataset:
            return _safe_path(str(data_root_parent / dataset / remainder))

        return _safe_path(str(env.birds_data_root / decoded))

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
                return _resolve_local_files_query_path(local)

        if value.startswith("http://") or value.startswith("https://"):
            continue

        return _safe_path(value)

    raise ValueError("Task does not include a usable local image path")


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if tasks is None and isinstance(payload, list):
        tasks = payload

    logger.info("predict request tasks=%d payload_type=%s", len(tasks or []), type(payload).__name__)

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
            attrs = model_b.predict(detections, image_path=image_path)
            logger.info(
                "predict task=%s image=%s detections=%d",
                task_id,
                image_path.name,
                len(detections),
            )
            for idx, attr in enumerate(attrs):
                logger.debug(
                    "predict task=%s det_idx=%d readability=%s specie=%s behavior=%s substrate=%s stance=%s",
                    task_id,
                    idx,
                    attr.readability,
                    attr.specie,
                    attr.behavior,
                    attr.substrate,
                    attr.stance,
                )

            pred = to_label_studio_prediction(
                task_id=task_id,
                detections=detections,
                attributes=attrs,
                model_version=MODEL_VERSION,
            )
            predictions.append(pred)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed for task %s: %s", task_id, exc)
            predictions.append(
                {
                    "task": task_id,
                    "model_version": MODEL_VERSION,
                    "score": 0.0,
                    "result": [],
                    "error": str(exc),
                }
            )

    response = format_predict_response(predictions)
    first_result_len = 0
    if predictions:
        first_result_len = len(predictions[0].get("result") or [])
    logger.info(
        "predict response tasks=%d keys=%s first_prediction_result_count=%d",
        len(predictions),
        ",".join(sorted(response.keys())),
        first_result_len,
    )
    return response
