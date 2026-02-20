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

        def append_choice(from_name: str, score: float, choice: str) -> None:
            results.append(
                {
                    "id": region_id,
                    "from_name": from_name,
                    "to_name": "image",
                    "type": "choices",
                    "parentID": region_id,
                    "score": float(score),
                    "value": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "rotation": 0,
                        "choices": [choice],
                    },
                }
            )

        # Always present core fields.
        append_choice("readability", attr.readability_conf, attr.readability)
        append_choice("specie", attr.specie_conf, attr.specie)

        # Strict LS config branch:
        # behavior/substrate visible only if readability != unreadable and specie != incorrect.
        can_show_context = attr.readability != "unreadable" and attr.specie != "incorrect"
        if can_show_context:
            append_choice("behavior", attr.behavior_conf, attr.behavior)
            append_choice("substrate", attr.substrate_conf, attr.substrate)

            # Stance (legs) visible only for resting/backresting on ground/water.
            can_show_stance = (
                attr.behavior in {"resting", "backresting"}
                and attr.substrate in {"ground", "water"}
            )
            if can_show_stance:
                append_choice("legs", attr.legs_conf, attr.legs)

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
