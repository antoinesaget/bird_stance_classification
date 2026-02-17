from __future__ import annotations

from services.ml_backend.app.predictors.model_a_yolo import Detection
from services.ml_backend.app.predictors.model_b_attributes import AttributePrediction
from services.ml_backend.app.serializers import to_label_studio_prediction


def test_serializer_shapes_prediction() -> None:
    det = Detection(x=0.1, y=0.2, w=0.3, h=0.4, score=0.88)
    attr = AttributePrediction(
        readability="readable",
        readability_conf=0.9,
        specie="correct",
        specie_conf=0.85,
        behavior="resting",
        behavior_conf=0.8,
        substrate="ground",
        substrate_conf=0.8,
        legs="two",
        legs_conf=0.7,
    )

    out = to_label_studio_prediction(
        task_id=123,
        detections=[det],
        attributes=[attr],
        image_status="has_usable_birds",
        image_status_conf=0.77,
        model_version="test-v1",
    )

    assert out["task"] == 123
    assert out["model_version"] == "test-v1"
    assert isinstance(out["result"], list)
    assert any(item["type"] == "rectanglelabels" for item in out["result"])
    assert any(item["from_name"] == "specie" for item in out["result"])
    assert any(item["from_name"] == "behavior" for item in out["result"])
    assert any(item["from_name"] == "substrate" for item in out["result"])
    assert any(item["from_name"] == "image_status" for item in out["result"])
