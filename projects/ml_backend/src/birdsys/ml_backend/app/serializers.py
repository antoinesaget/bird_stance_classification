from __future__ import annotations

from .predictors.model_a_yolo import Detection
from .predictors.model_b_attributes import AttributePrediction


def to_label_studio_prediction(
    *,
    task_id: int | str,
    detections: list[Detection],
    attributes: list[AttributePrediction],
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
        append_choice("isbird", float(det.score), "yes")
        append_choice("readability", attr.readability_conf, attr.readability)
        append_choice("specie", attr.specie_conf, attr.specie)

        # Strict LS config branch:
        # behavior/substrate visible only if readability != unreadable and specie != incorrect.
        can_show_context = attr.readability != "unreadable" and attr.specie != "incorrect"
        if can_show_context:
            append_choice("behavior", attr.behavior_conf, attr.behavior)
            append_choice("substrate", attr.substrate_conf, attr.substrate)

            # Stance visible only for resting/backresting on bare_ground/water/unsure.
            can_show_stance = (
                attr.behavior in {"resting", "backresting"}
                and attr.substrate in {"bare_ground", "water", "unsure"}
            )
            if can_show_stance:
                append_choice("stance", attr.stance_conf, attr.stance)

    return {
        "task": task_id,
        "score": float(max((det.score for det in detections), default=0.0)),
        "model_version": model_version,
        "result": results,
    }
