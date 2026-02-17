from __future__ import annotations

from .predictors.model_a_yolo import Detection
from .predictors.model_b_attributes import AttributePrediction


def to_label_studio_prediction(
    *,
    task_id: int | str,
    detections: list[Detection],
    attributes: list[AttributePrediction],
    image_status: str,
    image_status_conf: float,
    model_version: str,
) -> dict:
    results = []

    for idx, (det, attr) in enumerate(zip(detections, attributes)):
        region_id = f"r_{task_id}_{idx:03d}"
        x = det.x * 100.0
        y = det.y * 100.0
        w = det.w * 100.0
        h = det.h * 100.0

        results.append(
            {
                "id": region_id,
                "from_name": "bird_bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": float(det.score),
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "rectanglelabels": [det.label],
                },
            }
        )

        results.append(
            {
                "id": region_id,
                "from_name": "readability",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.readability_conf),
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "choices": [attr.readability],
                },
            }
        )

        results.append(
            {
                "id": region_id,
                "from_name": "activity",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.activity_conf),
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "choices": [attr.activity],
                },
            }
        )

        results.append(
            {
                "id": region_id,
                "from_name": "support",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.support_conf),
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "choices": [attr.support],
                },
            }
        )

        results.append(
            {
                "id": region_id,
                "from_name": "legs",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.legs_conf),
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "choices": [attr.legs],
                },
            }
        )

        results.append(
            {
                "id": region_id,
                "from_name": "resting_back",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.resting_back_conf),
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "choices": [attr.resting_back],
                },
            }
        )

    results.append(
        {
            "id": f"img_{task_id}_image_status",
            "from_name": "image_status",
            "to_name": "image",
            "type": "choices",
            "score": float(image_status_conf),
            "value": {"choices": [image_status]},
        }
    )

    return {
        "task": task_id,
        "score": float(image_status_conf),
        "model_version": model_version,
        "result": results,
    }
