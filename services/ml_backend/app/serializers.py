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

        results.append(
            {
                "id": region_id,
                "from_name": "bird_bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": float(det.score),
                "value": {
                    "x": det.x * 100.0,
                    "y": det.y * 100.0,
                    "width": det.w * 100.0,
                    "height": det.h * 100.0,
                    "rotation": 0,
                    "rectanglelabels": [det.label],
                },
            }
        )

        results.append(
            {
                "id": f"{region_id}_readability",
                "from_name": "readability",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.readability_conf),
                "value": {"choices": [attr.readability]},
            }
        )

        results.append(
            {
                "id": f"{region_id}_activity",
                "from_name": "activity",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.activity_conf),
                "value": {"choices": [attr.activity]},
            }
        )

        results.append(
            {
                "id": f"{region_id}_support",
                "from_name": "support",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.support_conf),
                "value": {"choices": [attr.support]},
            }
        )

        results.append(
            {
                "id": f"{region_id}_legs",
                "from_name": "legs",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.legs_conf),
                "value": {"choices": [attr.legs]},
            }
        )

        results.append(
            {
                "id": f"{region_id}_resting_back",
                "from_name": "resting_back",
                "to_name": "image",
                "type": "choices",
                "parentID": region_id,
                "score": float(attr.resting_back_conf),
                "value": {"choices": [attr.resting_back]},
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
